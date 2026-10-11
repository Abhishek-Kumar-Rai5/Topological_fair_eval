"""F3 topology features (prereg §4): 32 per window, no tuning.

Fitted on TRAINING windows only (prereg §4: scalers are fitted on training data):
    mean and std of each channel  -> z-score (std floored at STD_FLOOR so near-constant
                                      channels do not blow up)
    reference distance            -> median pairwise distance between embedded points,
                                      pooled over all training windows of the machine
For each window (60 time steps x 38 channels):
 1. Z-score every channel with the training mean and std.
 2. Delay embedding, m = 3, tau = 1: point t = [x_t, x_(t+1), x_(t+2)], where x_t is
    the 38-channel vector at time t. A 60-step window gives 58 points in 3 x 38 = 114 dims.
    (One point cloud per window, so one diagram per window -> 32 features in total.)
 3. Vietoris-Rips persistent homology with ripser, H0 (connected components) and
    H1 (loops), on the Euclidean distances between those points.
 4. Summarise each diagram (H0 and H1 separately) with 16 numbers, FINITE bars only:
        number of bars                                        1
        total, max, mean, std of bar lifetimes (death-birth)  4
        persistent entropy                                    1
        Betti curve at 10 fixed multiples (0.2, 0.4, ..., 2.0) of the
        TRAINING median pairwise distance (the same for every window)   10
    2 diagrams x 16 = 32 features.
Why the reference distance comes from training and not from each window: a per-window
scale would divide away changes in overall size (volume), which would rig the
volume-vs-shape test.
The one H0 bar that never dies (death = inf) is dropped everywhere, including the Betti curve.
"""
import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

EMBED_DIM = 3                                   # m
EMBED_DELAY = 1                                 # tau
BETTI_MULTIPLES = np.arange(1, 11) / 5          # 0.2, 0.4, ..., 2.0 x training median distance
STD_FLOOR = 0.01                                # SMD is scaled to [0, 1]; 1% of the range
STATS = ["n_bars", "total_life", "max_life", "mean_life", "std_life", "entropy"]


def delay_embed(window, m=EMBED_DIM, tau=EMBED_DELAY):
    """(time, channels) -> (time - (m-1)*tau, m * channels) point cloud."""
    n = len(window) - (m - 1) * tau
    return np.hstack([window[i * tau : i * tau + n] for i in range(m)])


def persistent_entropy(lifetimes):
    """-sum p_i log p_i with p_i = lifetime_i / total lifetime. 0 if there are no bars.

    High = many bars of similar length; low = one bar dominates.
    """
    total = lifetimes.sum()
    if len(lifetimes) == 0 or total == 0:
        return 0.0
    p = lifetimes / total
    return float(-(p * np.log(p)).sum())


def betti_curve(diagram, scales):
    """Number of FINITE bars alive at each scale: birth <= scale < death."""
    diagram = diagram[np.isfinite(diagram[:, 1])]
    births, deaths = diagram[:, 0], diagram[:, 1]
    return np.array([np.sum((births <= s) & (s < deaths)) for s in scales], dtype=float)


def diagram_summary(diagram, scales):
    """16 numbers for one diagram (one homology dimension)."""
    finite = diagram[np.isfinite(diagram[:, 1])]
    life = finite[:, 1] - finite[:, 0]
    if len(life) == 0:
        stats = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        stats = [len(life), life.sum(), life.max(), life.mean(), life.std(), persistent_entropy(life)]
    return np.r_[stats, betti_curve(diagram, scales)]


def zscore_params(train_windows):
    """Per-channel mean and std over all time points of the training windows."""
    points = train_windows.reshape(-1, train_windows.shape[2])
    return points.mean(axis=0), np.maximum(points.std(axis=0), STD_FLOOR)


def embedded_distances(window, mean, std):
    """Z-score a window, delay-embed it, return the (58, 58) Euclidean distance matrix."""
    points = delay_embed((np.asarray(window, dtype=float) - mean) / std)
    return squareform(pdist(points))


def training_median_distance(train_windows, mean, std):
    """Median pairwise distance, pooled over all training windows (one number per machine)."""
    pooled = []
    for w in train_windows:
        dist = embedded_distances(w, mean, std)
        pooled.append(dist[np.triu_indices(len(dist), k=1)])
    return float(np.median(np.concatenate(pooled)))


def window_topology(window, mean, std, ref_distance):
    """The 32 F3 features of one window."""
    dist = embedded_distances(window, mean, std)
    scales = BETTI_MULTIPLES * ref_distance
    h0, h1 = ripser(dist, maxdim=1, distance_matrix=True)["dgms"]
    return np.r_[diagram_summary(h0, scales), diagram_summary(h1, scales)]


class F3Features:
    """Fits the z-score and the reference distance on training windows only."""

    def fit(self, train_windows):
        self.mean, self.std = zscore_params(train_windows)
        self.ref_distance = training_median_distance(train_windows, self.mean, self.std)
        return self

    def transform(self, windows):
        return np.array([window_topology(w, self.mean, self.std, self.ref_distance)
                         for w in windows])


def f3_names(n_channels=None):
    names = []
    for dim in ("H0", "H1"):
        names += [f"F3_{dim}_{s}" for s in STATS]
        names += [f"F3_{dim}_betti_{m:.1f}xmed" for m in BETTI_MULTIPLES]
    return names
