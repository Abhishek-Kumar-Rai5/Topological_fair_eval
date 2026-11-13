"""Plan A-lite helpers (EXPLORATORY, docs/plan_A_lite.md). Nothing here is part of the prereg.

L1  extra_metrics, positive_runs, random_scores   metrics from per-window scores (paper style next to ours)
L2  synthetic_machine, WholeWindowH1              H1 sanity check with a chosen delay tau
    tau_ac, PerChannelPH                          "H1-friendly" per-channel clouds with an autocorrelation tau
L4  mst_h0_deaths, h0_deaths_ripser               MST edge lengths against ripser's H0 deaths
"""
import numpy as np
from ripser import ripser
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from src.features_topo import BETTI_MULTIPLES, N_HALF, delay_embed, diagram_summary, zscore_params

N_THRESHOLDS = 400
WINDOW_STRIDE = 6          # SMD windows are 60 long with step 10, so every 6th window does not overlap the last


# ------------------------------------------------------------------------------------------ L1
def _f1(y, alarm):
    tp = int((y & alarm).sum())
    return 2 * tp / (2 * tp + int((~y & alarm).sum()) + int((y & ~alarm).sum())) if tp > 0 else 0.0


def positive_runs(labels):
    """[(start, stop), ...] of runs of consecutive windows with label 1. A normal (0) or excluded (-1) window ends a run."""
    pos = np.asarray(labels) == 1
    starts = np.flatnonzero(pos & ~np.r_[False, pos[:-1]])
    stops = np.flatnonzero(pos & ~np.r_[pos[1:], False]) + 1
    return list(zip(starts, stops))


def extra_metrics(scores, labels, threshold, n_thresholds=N_THRESHOLDS):
    """F1 and accuracy at OUR validation threshold; best F1 over thresholds chosen on TEST labels (paper style); the same
    with window-level point adjustment. Excluded windows (-1) are dropped from every count.

    best-F1 thresholds: n_thresholds evenly spaced values from the lowest to the highest test score.
    Point adjustment: a run of consecutive positive windows counts as fully alarmed if any window in it alarms.
    The best-F1 numbers use test labels on purpose: they show what the papers' protocol would report. They decide nothing.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    keep, y_all = labels != -1, labels == 1
    y = y_all[keep]
    alarm = scores > threshold
    runs = positive_runs(labels)
    best, best_pa = 0.0, 0.0
    for t in np.linspace(scores[keep].min(), scores[keep].max(), n_thresholds):
        a = scores > t
        best = max(best, _f1(y, a[keep]))
        adj = a.copy()
        for lo, hi in runs:
            if a[lo:hi].any():
                adj[lo:hi] = True
        best_pa = max(best_pa, _f1(y, adj[keep]))
    return {"f1_at_thr": _f1(y, alarm[keep]), "accuracy_at_thr": float((alarm[keep] == y).mean()),
            "best_f1_test_thr": best, "best_f1_point_adjusted": best_pa}


def random_scores(n_val, n_test, seed=0):
    """Uniform scores for the RANDOM row: the floor every detector should beat."""
    rng = np.random.default_rng(seed)
    return rng.uniform(size=n_val), rng.uniform(size=n_test)


# ------------------------------------------------------------------------------------------ L2 (a)
def synthetic_machine(seed, n_channels=10, n_train=2400, n_test=2400, noise=0.1, at=1200, length=60):
    """10 channels of sinusoids (periods 24 and 37, alternating) plus noise.

    Test anomaly: channel 0 (period 24) runs at period 48 for `length` samples. Returns (train, test, point_labels).
    """
    rng = np.random.default_rng(seed)
    periods = np.where(np.arange(n_channels) % 2 == 0, 24.0, 37.0)
    phase = rng.uniform(0, 2 * np.pi, n_channels)

    def series(n):
        t = np.arange(n)[:, None]
        return np.sin(2 * np.pi * t / periods + phase) + rng.normal(0, noise, (n, n_channels))

    train, test = series(n_train), series(n_test)
    t = np.arange(at, at + length)
    test[at:at + length, 0] = np.sin(2 * np.pi * t / (2 * periods[0]) + phase[0]) + rng.normal(0, noise, length)
    labels = np.zeros(n_test, dtype=int)
    labels[at:at + length] = 1
    return train, test, labels


def _median_distance(clouds):
    return float(np.median(np.concatenate([pdist(c) for c in clouds])))


class WholeWindowH1:
    """F3's H1 half (16 numbers) with a chosen delay tau: the whole z-scored window is one cloud (as F3).

    Used only for the synthetic check; max_life (column 2) is the maximum H1 persistence of the window.
    """

    def __init__(self, tau):
        self.tau = tau

    def _cloud(self, w):
        return delay_embed((np.asarray(w, float) - self.mean) / self.std, 3, self.tau)

    def fit(self, train_windows):
        self.mean, self.std = zscore_params(train_windows)
        self.ref = _median_distance(self._cloud(w) for w in train_windows)
        return self

    def transform(self, windows):
        scales = BETTI_MULTIPLES * self.ref
        out = []
        for w in windows:
            h1 = ripser(squareform(pdist(self._cloud(w))), maxdim=1, distance_matrix=True)["dgms"][1]
            out.append(diagram_summary(h1, scales))
        return np.array(out)


# ------------------------------------------------------------------------------------------ L2 (b)
def tau_ac(train_windows, stride=WINDOW_STRIDE, lo=2, hi=15):
    """Delay from TRAINING data only: first local minimum in [lo, hi] of the mean autocorrelation across channels.

    Non-overlapping training windows (every `stride`-th) are joined per channel; each channel is centred, its
    autocorrelation at lags 0..hi+1 is computed, and channels with zero variance are skipped. Local minimum at k:
    acf[k] < acf[k-1] and acf[k] <= acf[k+1]. If there is none, the lag of the lowest autocorrelation in [lo, hi].
    If every channel is constant: 1.
    """
    x = np.asarray(train_windows)[::stride]
    series = x.reshape(-1, x.shape[2])
    series = series - series.mean(axis=0)
    energy = (series ** 2).sum(axis=0)
    ok = energy > 1e-12
    if not ok.any():
        return 1
    s, e = series[:, ok], energy[ok]
    acf = np.array([((s[:len(s) - k] * s[k:]).sum(axis=0) / e).mean() for k in range(hi + 2)])
    for k in range(lo, hi + 1):
        if acf[k] < acf[k - 1] and acf[k] <= acf[k + 1]:
            return k
    return lo + int(np.argmin(acf[lo:hi + 1]))


class PerChannelPH:
    """H0 or H1 summaries (16 numbers, as F3) of PER-CHANNEL clouds, averaged over channels.

    Per channel: z-score with the training mean and std (std floor as F3), delay-embed that channel alone
    (m = 3, tau = tau_ac of the machine) into 3-D points, ripser, summarise with the same 16 numbers; the Betti scales are
    multiples (0.2 .. 2.0) of that channel's training median pairwise distance (1.0 if it is 0). The 38 vectors are averaged.
    dims = 0 gives H0 only (maxdim 0 is faster), dims = 1 gives H1.
    """

    def __init__(self, dims):
        assert dims in (0, 1)
        self.dims = dims

    def _clouds(self, w):
        z = (np.asarray(w, float) - self.mean) / self.std
        return [delay_embed(z[:, c:c + 1], 3, self.tau) for c in range(z.shape[1])]

    def fit(self, train_windows):
        self.mean, self.std = zscore_params(train_windows)
        self.tau = tau_ac(train_windows)
        per_channel = [[] for _ in range(train_windows.shape[2])]
        for w in train_windows:
            for c, cloud in enumerate(self._clouds(w)):
                per_channel[c].append(pdist(cloud))
        self.ref = np.array([np.median(np.concatenate(d)) for d in per_channel])
        self.ref[self.ref <= 0] = 1.0
        return self

    def transform(self, windows):
        out = []
        for w in windows:
            rows = []
            for c, cloud in enumerate(self._clouds(w)):
                dgm = ripser(squareform(pdist(cloud)), maxdim=self.dims, distance_matrix=True)["dgms"][self.dims]
                rows.append(diagram_summary(dgm, BETTI_MULTIPLES * self.ref[c]))
            out.append(np.mean(rows, axis=0))
        return np.array(out)


def pc_names(dims):
    from src.features_topo import f3_names
    names = f3_names()[dims * N_HALF:(dims + 1) * N_HALF]
    return [n.replace("F3_", "PC_") for n in names]


# ------------------------------------------------------------------------------------------ L4
ZERO = 1e-9      # distances below this = "the same point" (ripser reports no bar for them)


def mst_h0_deaths(dist):
    """Edge lengths of the minimum spanning tree of a distance matrix, sorted, without zero-length edges.

    scipy reads a 0 as "no edge", and a DENSE input also drops values below about 1e-8, so duplicate points get a tiny
    distance and the matrix is passed as a sparse one (which keeps every stored value).
    """
    d = np.array(dist, float)
    off = ~np.eye(len(d), dtype=bool)
    d[off & (d == 0)] = ZERO / 10
    edges = np.sort(minimum_spanning_tree(csr_matrix(d)).data)
    return edges[edges > ZERO]


def h0_deaths_ripser(dist):
    """Finite H0 death times from ripser, sorted, without zero-length bars."""
    dgm = ripser(dist, maxdim=0, distance_matrix=True)["dgms"][0]
    deaths = np.sort(dgm[np.isfinite(dgm[:, 1]), 1])
    return deaths[deaths > ZERO]
