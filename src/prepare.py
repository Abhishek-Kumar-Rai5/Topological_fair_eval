"""Glue: one SMD machine -> train, validation and test windows (prereg §2).

    train file --split--> fit part (first 80%)  -> training windows   (all normal)
                      \-> validation (last 20%) -> validation windows (all normal)
    test file  ------------------------------->  test windows + labels 1 / 0 / -1
Windows are cut separately inside each part, so no window spans two parts.
"""
from dataclasses import dataclass

import numpy as np

from src.smd import DEFAULT_ROOT, load_machine
from src.splits import split_train_val
from src.windows import label_windows, make_windows


@dataclass
class MachineWindows:
    name: str
    train: np.ndarray        # (n_train_windows, size, channels)
    val: np.ndarray          # (n_val_windows, size, channels)
    test: np.ndarray         # (n_test_windows, size, channels)
    test_labels: np.ndarray  # (n_test_windows,) 1 / 0 / -1
    test_starts: np.ndarray  # (n_test_windows,) start index of each test window
    n_test_points: int       # length of the test series (for false alarms per day)


def prepare_machine(name, root=DEFAULT_ROOT, size=60, step=10, tail=10, val_frac=0.2):
    train, test, test_point_labels = load_machine(name, root)
    fit_part, val_part = split_train_val(train, val_frac)
    train_w, _ = make_windows(fit_part, size, step)
    val_w, _ = make_windows(val_part, size, step)
    test_w, starts = make_windows(test, size, step)
    labels = label_windows(test_point_labels, starts, size, tail)
    return MachineWindows(name, train_w, val_w, test_w, labels, starts, len(test))
