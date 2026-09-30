"""Cluster bootstrap CI over machines (prereg §8), on toy data."""
import numpy as np
import pandas as pd
import pytest

from src.bootstrap import average_over_seeds, cluster_bootstrap_ci, paired_delta_ci


def test_constant_values_give_zero_width_ci():
    mean, lo, hi = cluster_bootstrap_ci([0.3] * 28)
    assert mean == lo == hi == pytest.approx(0.3)


def test_ci_contains_mean_and_is_reproducible():
    v = np.random.default_rng(0).normal(0.05, 0.02, size=28)
    a = cluster_bootstrap_ci(v, seed=1)
    b = cluster_bootstrap_ci(v, seed=1)
    assert a == b                                     # same seed -> same CI
    mean, lo, hi = a
    assert lo < mean < hi
    assert lo > 0                                     # clear positive effect excludes 0


def test_ci_width_close_to_normal_theory():
    # for a mean of n values, the 95% CI is about ±1.96 * sd / sqrt(n)
    v = np.random.default_rng(2).normal(0, 1, size=200)
    mean, lo, hi = cluster_bootstrap_ci(v, n_resamples=4000, seed=0)
    expected_half = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    assert (hi - lo) / 2 == pytest.approx(expected_half, rel=0.15)


def test_average_over_seeds_then_paired_delta():
    # 3 machines x 2 seeds; the topology set beats the baseline by exactly 0.1 on every machine
    rows = []
    for m in ["m1", "m2", "m3"]:
        for seed in [0, 1]:
            base = {"m1": 0.5, "m2": 0.6, "m3": 0.7}[m] + 0.01 * seed
            rows += [{"machine": m, "seed": seed, "set": "F1+F2", "pr_auc": base},
                     {"machine": m, "seed": seed, "set": "F1+F2+F3", "pr_auc": base + 0.1}]
    df = pd.DataFrame(rows)
    a = average_over_seeds(df[df.set == "F1+F2+F3"], "machine", "pr_auc")
    b = average_over_seeds(df[df.set == "F1+F2"], "machine", "pr_auc")
    assert a["m1"] == pytest.approx(0.605)            # mean of 0.6 and 0.61
    mean, lo, hi = paired_delta_ci(a, b)
    assert mean == lo == hi == pytest.approx(0.1)


def test_nan_is_rejected():
    with pytest.raises(AssertionError):
        cluster_bootstrap_ci([0.1, np.nan, 0.2])
