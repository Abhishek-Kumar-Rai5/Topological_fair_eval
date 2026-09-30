"""Metrics on one machine's TEST windows (prereg §6). No point adjustment.

Excluded windows (label -1) are dropped from PR-AUC, ROC-AUC and recall.
    PR-AUC   average precision of the scores (primary metric)
    ROC-AUC  for comparison with TAME's published numbers
    recall   share of positive windows that alarm at the validation threshold
    false alarms per day
             alarms on NORMAL windows, with a run of consecutive false-alarm
             windows counted as ONE event, divided by the test length in days
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.windows import NORMAL, POSITIVE, EXCLUDED

MINUTES_PER_DAY = 24 * 60


def count_alarm_events(is_false_alarm):
    """Number of runs of consecutive True values, e.g. [1,1,0,1,0,0,1,1,1] -> 3.

    Windows must be in time order. A window that is not a false alarm
    (no alarm, a positive window, or an excluded window) ends a run.
    """
    f = np.asarray(is_false_alarm, dtype=bool)
    if len(f) == 0:
        return 0
    # a run starts wherever f is True and the previous window was not
    starts = f & ~np.r_[False, f[:-1]]
    return int(starts.sum())


def evaluate(scores, labels, threshold, n_test_points, minutes_per_point=1.0):
    """All §6 metrics for one machine.

    scores:        detector score per test window, in time order (higher = more anomalous)
    labels:        window labels 1 / 0 / -1 (src.windows), same order
    threshold:     from src.threshold.threshold_at_fpr on validation windows
    n_test_points: length of the test series (to convert to days)
    minutes_per_point: SMD sampling interval (1 minute; see data/README.md)
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    assert len(scores) == len(labels)

    keep = labels != EXCLUDED
    y = labels[keep] == POSITIVE
    s = scores[keep]
    n_pos, n_norm = int(y.sum()), int((~y).sum())
    both_classes = n_pos > 0 and n_norm > 0

    alarm = scores > threshold
    false_alarm = alarm & (labels == NORMAL)
    test_days = n_test_points * minutes_per_point / MINUTES_PER_DAY

    return {
        "pr_auc": float(average_precision_score(y, s)) if both_classes else np.nan,
        "roc_auc": float(roc_auc_score(y, s)) if both_classes else np.nan,
        "recall_at_thr": float(alarm[keep][y].mean()) if n_pos > 0 else np.nan,
        "false_alarm_events": count_alarm_events(false_alarm),
        "false_alarms_per_day": count_alarm_events(false_alarm) / test_days,
        "false_alarm_windows": int(false_alarm.sum()),
        "n_positive": n_pos,
        "n_normal": n_norm,
        "n_excluded": int((~keep).sum()),
    }
