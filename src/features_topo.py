"""F3 topology features (prereg §4): 32 per window, no tuning.

For each window (60 time steps x 38 channels):
 1. Delay embedding, m = 3, tau = 1: point t = [x_t, x_(t+1), x_(t+2)], where x_t is
    the 38-channel vector at time t. A 60-step window gives 58 points in 3 x 38 = 114 dims.
    (One point cloud per window, so one diagram per window -> 32 features in total.)
 2. Vietoris-Rips persistent homology with ripser, H0 (connected components) and
    H1 (loops), on the Euclidean distances between those points.
 3. Summarise each diagram (H0 and H1 separately) with 16 numbers:
        number of bars                                        1
        total, max, mean, std of bar lifetimes (death-birth)  4
        persistent entropy                                    1
        Betti curve at 10 fixed multiples of the window's
        median pairwise distance                             10
    2 diagrams x 16 = 32 features.
The one H0 bar that never dies (death = inf) is left out of the bar count and the
lifetime statistics (every window has exactly one, so it carries no information),
but it is counted in the Betti curve, because it is a real component.
Nothing is fitted: the data are used as given (SMD is already scaled to [0, 1]).
"""
import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

EMBED_DIM = 3                                   # m
EMBED_DELAY = 1                                 # tau
BETTI_MULTIPLES = np.arange(1, 11) / 10         # 0.1, 0.2, ..., 1.0 x median distance
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
    """Number of bars alive at each scale: birth <= scale < death."""
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


def window_topology(window):
    """The 32 F3 features of one window."""
    points = delay_embed(np.asarray(window, dtype=float))
    dist = squareform(pdist(points))                       # Euclidean, (58, 58)
    median = np.median(dist[np.triu_indices(len(dist), k=1)])
    scales = BETTI_MULTIPLES * median
    h0, h1 = ripser(dist, maxdim=1, distance_matrix=True)["dgms"]
    return np.r_[diagram_summary(h0, scales), diagram_summary(h1, scales)]


class F3Features:
    """Fits nothing (no tuning); same interface as the other feature groups."""

    def fit(self, train_windows):
        return self

    def transform(self, windows):
        return np.array([window_topology(w) for w in windows])


def f3_names(n_channels=None):
    names = []
    for dim in ("H0", "H1"):
        names += [f"F3_{dim}_{s}" for s in STATS]
        names += [f"F3_{dim}_betti_{m:.1f}xmed" for m in BETTI_MULTIPLES]
    return names
