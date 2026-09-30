"""Metrics (prereg §6), on toy data."""
import numpy as np
import pytest

from src.metrics import count_alarm_events, evaluate
from src.windows import EXCLUDED, NORMAL, POSITIVE


def test_alarm_events_merge_consecutive():
    assert count_alarm_events([1, 1, 0, 1, 0, 0, 1, 1, 1]) == 3
    assert count_alarm_events([0, 0, 0]) == 0
    assert count_alarm_events([1]) == 1
    assert count_alarm_events([]) == 0


def test_perfect_scores():
    labels = np.array([NORMAL, NORMAL, POSITIVE, POSITIVE])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(scores, labels, threshold=0.5, n_test_points=1440)
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0
    assert m["recall_at_thr"] == 1.0
    assert m["false_alarms_per_day"] == 0.0


def test_excluded_windows_are_ignored():
    labels = np.array([NORMAL, EXCLUDED, POSITIVE])
    scores = np.array([0.1, 99.0, 0.9])              # a huge score on the excluded window
    m = evaluate(scores, labels, threshold=0.5, n_test_points=1440)
    assert m["pr_auc"] == 1.0                        # excluded window does not hurt PR-AUC
    assert m["false_alarm_events"] == 0              # and is not a false alarm
    assert (m["n_positive"], m["n_normal"], m["n_excluded"]) == (1, 1, 1)


def test_false_alarms_per_day():
    # 2 days of test data at 1 minute per point = 2,880 points
    labels = np.array([NORMAL] * 6 + [POSITIVE])
    scores = np.array([0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9])
    m = evaluate(scores, labels, threshold=0.5, n_test_points=2880, minutes_per_point=1.0)
    assert m["false_alarm_windows"] == 3
    assert m["false_alarm_events"] == 2              # [0.9, 0.9] merged into one event
    assert m["false_alarms_per_day"] == pytest.approx(1.0)


def test_positive_window_breaks_a_run():
    labels = np.array([NORMAL, POSITIVE, NORMAL])
    scores = np.array([0.9, 0.9, 0.9])               # all alarm
    m = evaluate(scores, labels, threshold=0.5, n_test_points=1440)
    assert m["false_alarm_events"] == 2


def test_recall_at_threshold():
    labels = np.array([NORMAL, POSITIVE, POSITIVE, POSITIVE, POSITIVE])
    scores = np.array([0.0, 0.2, 0.6, 0.7, 0.8])
    m = evaluate(scores, labels, threshold=0.5, n_test_points=1440)
    assert m["recall_at_thr"] == pytest.approx(0.75)


def test_matches_sklearn_on_random_data():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(1)
    labels = rng.choice([NORMAL, POSITIVE, EXCLUDED], size=300, p=[0.8, 0.15, 0.05])
    scores = rng.normal(size=300) + (labels == POSITIVE)
    m = evaluate(scores, labels, threshold=1.0, n_test_points=3000)
    keep = labels != EXCLUDED
    assert m["pr_auc"] == pytest.approx(average_precision_score(labels[keep] == POSITIVE, scores[keep]))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(labels[keep] == POSITIVE, scores[keep]))


def test_no_positives_gives_nan_not_crash():
    m = evaluate(np.array([0.1, 0.2]), np.array([NORMAL, NORMAL]), threshold=0.5, n_test_points=1440)
    assert np.isnan(m["pr_auc"]) and np.isnan(m["recall_at_thr"])
