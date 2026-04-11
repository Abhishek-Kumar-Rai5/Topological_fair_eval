"""95% cluster bootstrap confidence interval (prereg §8).

A "cluster" is one machine (SMD) or one scenario (CTU-13).
Steps:
  1. For each cluster, average the metric over the 5 seeds  -> one number per cluster.
  2. For a comparison (e.g. PR-AUC(F1+F2+F3) - PR-AUC(F1+F2)), take the per-cluster
     difference -> one Δ per cluster.
  3. Resample the clusters WITH replacement 2,000 times; each time take the mean.
  4. The CI is the 2.5th and 97.5th percentile of those 2,000 means.
Resampling whole machines (not windows) keeps windows from the same machine
together, because they are not independent.
"""
import numpy as np
import pandas as pd


def average_over_seeds(df, cluster_col, value_col):
    """One value per cluster: the mean over seeds. Returns a Series indexed by cluster."""
    return df.groupby(cluster_col)[value_col].mean()


def cluster_bootstrap_ci(per_cluster, n_resamples=2000, ci=0.95, seed=0):
    """Mean over clusters and its percentile bootstrap CI.

    per_cluster: one value per cluster (already averaged over seeds).
    Returns (mean, ci_low, ci_high).
    """
    v = np.asarray(per_cluster, dtype=float)
    assert len(v) > 0 and not np.isnan(v).any(), "need one finite value per cluster"
    rng = np.random.default_rng(seed)
    # each row = one bootstrap sample of cluster indices (drawn with replacement)
    idx = rng.integers(0, len(v), size=(n_resamples, len(v)))
    means = v[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(v.mean()), float(lo), float(hi)


def machines_with_positives(df, cluster_col="machine"):
    """Clusters that have at least one positive test window in this cell.

    df: one row per (machine, seed) for ONE cell (dataset x anomaly type), with an
    `n_positive` column from src.metrics.evaluate. Machines with no positive window
    have no defined PR-AUC, so they are skipped for this cell (implementation_notes #5).
    """
    has_pos = df.groupby(cluster_col)["n_positive"].max() > 0
    return list(has_pos[has_pos].index)


def pooled_window_counts(df, cluster_col="machine"):
    """Positive and normal test windows in the cell, pooled over machines.

    Counted once per machine (the windows are the same for every seed).
    """
    per_machine = df.groupby(cluster_col)[["n_positive", "n_normal"]].first()
    return int(per_machine["n_positive"].sum()), int(per_machine["n_normal"].sum())


def cell_has_enough_windows(df, min_positive=20, min_normal=20, cluster_col="machine"):
    """Prereg §8: a cell with < 20 positive or < 20 normal windows (pooled) is Inconclusive."""
    n_pos, n_norm = pooled_window_counts(df, cluster_col)
    return n_pos >= min_positive and n_norm >= min_normal


def paired_delta_ci(a, b, **kwargs):
    """CI for mean(a - b), where a and b are per-cluster Series (same clusters).

    Example: a = PR-AUC of F1+F2+F3 per machine, b = PR-AUC of F1+F2 per machine.
    """
    a, b = pd.Series(a), pd.Series(b)
    assert set(a.index) == set(b.index), "a and b must cover the same clusters"
    delta = (a - b.loc[a.index]).to_numpy()
    return cluster_bootstrap_ci(delta, **kwargs)
