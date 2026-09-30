"""Decision threshold at a 1% false-alarm rate on VALIDATION windows (prereg §5).

Convention used everywhere: higher score = more anomalous, and a window raises
an alarm when  score > threshold.
SMD validation windows are all normal, so every validation alarm is a false
alarm. We choose the threshold so that at most 1% of them alarm.
Test data is never used here.
"""
import numpy as np


def threshold_at_fpr(val_scores, fpr=0.01):
    """Threshold such that at most floor(fpr * n) validation windows score above it.

    Example: 1,000 validation windows, fpr = 0.01 -> we allow k = 10 false alarms.
    Sort the scores from high to low; the threshold is the 11th highest score
    (index k), so only the 10 scores strictly above it can alarm (fewer if tied).
    """
    assert 0 <= fpr < 1
    s = np.sort(np.asarray(val_scores, dtype=float))[::-1]  # largest first
    assert len(s) > 0, "no validation windows"
    k = int(np.floor(fpr * len(s)))  # number of false alarms we allow
    return float(s[k])


def alarms(scores, threshold):
    """Boolean alarm per window."""
    return np.asarray(scores) > threshold
