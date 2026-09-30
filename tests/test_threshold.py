"""Threshold at 1% false alarms on validation windows (prereg §5), on toy data."""
import numpy as np

from src.threshold import alarms, threshold_at_fpr


def test_exactly_one_percent_alarm():
    val = np.arange(1000, dtype=float)               # 1,000 distinct validation scores
    thr = threshold_at_fpr(val, 0.01)
    assert alarms(val, thr).sum() == 10              # 1% of 1,000


def test_never_more_than_allowed():
    rng = np.random.default_rng(0)
    for n in [50, 99, 150, 1234]:
        val = rng.normal(size=n)
        thr = threshold_at_fpr(val, 0.01)
        assert alarms(val, thr).sum() <= int(np.floor(0.01 * n))


def test_ties_do_not_add_alarms():
    val = np.r_[np.zeros(990), np.ones(10) * 5.0]    # 10 tied top scores
    thr = threshold_at_fpr(val, 0.005)               # allows floor(5) = 5 alarms
    assert alarms(val, thr).sum() <= 5               # ties at the threshold do not alarm


def test_small_validation_set_allows_zero_alarms():
    val = np.arange(50, dtype=float)                 # floor(0.5) = 0 alarms allowed
    thr = threshold_at_fpr(val, 0.01)
    assert thr == 49 and alarms(val, thr).sum() == 0
