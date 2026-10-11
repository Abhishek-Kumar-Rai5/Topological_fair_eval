"""F5 Fluent Bit-style features (prereg §4, exploratory): beta0, beta1, beta2 per window.

A Python copy of the logic in Fluent Bit's `tda` processor, pinned at commit 25ea7e1
(`plugins/processor_tda/tda.c`, plus `src/ripser/flb_ripser_wrapper.cpp`), defaults kept:
    embed_dim = 3, embed_delay = 1, threshold = 0 (automatic scan), min_points = 10.

Steps, each marked with where it comes from in tda.c / the wrapper:
 1. first difference of EVERY metric, then signed log1p (tda_build_vector_from_cmt, L561-569):
    rate = (x_t - x_(t-1)) / dt,  v = sign(rate) * log1p(|rate|).  The C code does this for
    counters, gauges and untyped alike. SMD has no timestamps, so dt = 1 sample.
    A 60-sample window gives 59 vectors.
 2. delay embedding m = 3, tau = 1 (tda_window_run_ripser): point = the 3 latest vectors
    stacked; Euclidean distances, stored as float32 like the C code.
 3. scan 9 distance thresholds, the 0.1, 0.2, ..., 0.9 quantiles of the pairwise distances
    (tda_choose_threshold_from_dist): sort the n(n-1)/2 distances, take sorted[floor(q * (k-1))].
 4. for each threshold run ripser (Rips complex up to that distance, homology dimensions 0-2)
    and count bars (flb_ripser_compute_betti_from_dense_distance):
        counted = FINITE bars with death > birth and death - birth >= 1e-3, per dimension.
    Bars still alive at the threshold have no finite death, so the wrapper skips them.
    (The wrapper's own comment says "beta_k = number of infinite bars". The code does the
    opposite. We copy the code, not the comment. See docs/code_verification.md note.)
 5. keep the best scale (the selection rules, copied exactly):
        if the scale has dimension-1 bars and more of them than the best so far: take its
            (beta0, beta1, beta2);
        else if no scale has had a beta1 yet and its beta0 is larger than the best beta0:
            take its beta0 and beta2 (beta1 stays 0).
 Output: [beta0, beta1, beta2]. Nothing is fitted.
"""
import numpy as np
from ripser import ripser

EMBED_DIM = 3                      # tda.c default embed_dim
EMBED_DELAY = 1                    # tda.c default embed_delay
MIN_POINTS = 10                    # tda.c default min_points
Q_CANDIDATES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MIN_PERSISTENCE = 1e-3             # flb_ripser_wrapper.cpp MIN_PERSIST
MAX_DIM = 2                        # beta0, beta1, beta2
DT = 1.0                           # one sample = 1 time unit (SMD has no timestamps)


def signed_log1p_rate(window, dt=DT):
    """(time, channels) raw values -> (time - 1, channels): sign(rate) * log1p(|rate|), rate = diff / dt."""
    rate = np.diff(np.asarray(window, dtype=float), axis=0) / dt
    return np.sign(rate) * np.log1p(np.abs(rate))   # as in tda.c: sign by the rate, magnitude log1p(|rate|)


def delay_distances(vectors, m=EMBED_DIM, tau=EMBED_DELAY):
    """Euclidean distance matrix (float32) of the delay-embedded points.

    Point p stacks the vectors at p + (m-1)tau, p + (m-2)tau, ..., p; so n - (m-1)tau points.
    """
    n = len(vectors) - (m - 1) * tau
    points = np.hstack([vectors[(m - 1 - lag) * tau : (m - 1 - lag) * tau + n] for lag in range(m)])
    sq = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(sq).astype(np.float32)


def quantile_threshold(dist, q):
    """tda_choose_threshold_from_dist: sorted lower-triangle distances, index floor(q * (k - 1))."""
    n = len(dist)
    if n < 2:
        return np.float32(0.0)
    values = np.sort(dist[np.tril_indices(n, k=-1)])
    k = len(values)
    index = 0 if k == 1 else int(q * (k - 1))
    return np.float32(values[index])


def count_bars(dist, threshold):
    """Wrapper's Betti numbers: per dimension 0..2, finite bars with persistence >= 1e-3.

    Returns (betti list of 3, num_dims) where num_dims = highest counted dimension + 1.
    threshold <= 0 means no limit (the C wrapper then uses the largest float).
    """
    thresh = float(threshold) if threshold > 0 else np.inf
    diagrams = ripser(dist, maxdim=MAX_DIM, thresh=thresh, distance_matrix=True)["dgms"]
    betti = [0, 0, 0]
    for dim, dgm in enumerate(diagrams):
        for birth, death in dgm:
            if not np.isfinite(death) or death <= birth:
                continue
            if np.float32(death) - np.float32(birth) < MIN_PERSISTENCE:
                continue
            betti[dim] += 1
    num_dims = max([d + 1 for d in range(3) if betti[d] > 0], default=0)
    return betti, num_dims


def window_betti(window, dt=DT):
    """The 3 F5 features of one raw window: [beta0, beta1, beta2]."""
    vectors = signed_log1p_rate(window, dt)
    if len(vectors) < MIN_POINTS:
        return np.zeros(3)
    dist = delay_distances(vectors)
    best_b0 = best_b1 = best_b2 = 0
    for q in Q_CANDIDATES:
        betti, num_dims = count_bars(dist, quantile_threshold(dist, q))
        if num_dims > 1 and betti[1] > best_b1:
            best_b0, best_b1, best_b2 = betti[0], betti[1], (betti[2] if num_dims > 2 else 0)
        elif best_b1 == 0 and betti[0] > best_b0:
            best_b0 = betti[0]
            best_b2 = betti[2] if num_dims > 2 else 0
    return np.array([best_b0, best_b1, best_b2], dtype=float)


class F5Features:
    """Fits nothing; same interface as the other feature groups."""

    def fit(self, train_windows):
        return self

    def transform(self, windows):
        return np.array([window_betti(w) for w in windows])


def f5_names(n_channels=None):
    return ["F5_beta0", "F5_beta1", "F5_beta2"]
