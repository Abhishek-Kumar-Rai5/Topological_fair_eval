"""Windowing and the window-labelling rule (prereg §2), on toy data."""
import numpy as np

from src.splits import split_train_val
from src.windows import EXCLUDED, NORMAL, POSITIVE, label_windows, make_windows, window_starts


def test_starts_full_windows_only():
    # 100 points, size 60, step 10 -> starts 0, 10, 20, 30, 40 (40 + 60 = 100 is the last full one)
    assert window_starts(100, 60, 10).tolist() == [0, 10, 20, 30, 40]
    assert window_starts(59, 60, 10).tolist() == []


def test_make_windows_content_and_shape():
    X = np.arange(100 * 2).reshape(100, 2)          # 100 time steps, 2 channels
    w, starts = make_windows(X, 60, 10)
    assert w.shape == (5, 60, 2)
    assert np.array_equal(w[1], X[10:70])           # window 1 starts at 10


def test_label_rule():
    size, tail = 60, 10
    starts = np.array([0])
    lab = np.zeros(60, int)
    assert label_windows(lab, starts, size, tail)[0] == NORMAL        # no anomaly

    lab = np.zeros(60, int); lab[59] = 1
    assert label_windows(lab, starts, size, tail)[0] == POSITIVE      # last sample
    lab = np.zeros(60, int); lab[50] = 1
    assert label_windows(lab, starts, size, tail)[0] == POSITIVE      # first of the final 10

    lab = np.zeros(60, int); lab[49] = 1
    assert label_windows(lab, starts, size, tail)[0] == EXCLUDED      # just before the final 10
    lab = np.zeros(60, int); lab[0] = 1
    assert label_windows(lab, starts, size, tail)[0] == EXCLUDED      # anomaly only at the start


def test_labels_along_a_series():
    # one anomalous point at t = 75; windows start 0, 10, 20, 30, 40
    lab = np.zeros(100, int); lab[75] = 1
    starts = window_starts(100, 60, 10)
    # window [0,60): no anomaly -> normal
    # window [10,70): no anomaly -> normal
    # window [20,80): final 10 = [70,80) holds 75 -> positive
    # window [30,90): 75 is in [30,80), not in final 10 [80,90) -> excluded
    # window [40,100): final 10 = [90,100) has none, but 75 is inside -> excluded
    assert label_windows(lab, starts).tolist() == [NORMAL, NORMAL, POSITIVE, EXCLUDED, EXCLUDED]


def test_split_last_20_percent_is_validation():
    X = np.arange(100)[:, None]
    fit, val = split_train_val(X, 0.2)
    assert len(fit) == 80 and len(val) == 20
    assert val[0, 0] == 80 and val[-1, 0] == 99        # the LAST 20%, in time order
