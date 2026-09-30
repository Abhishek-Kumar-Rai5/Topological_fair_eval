"""Sliding windows and the window-labelling rule (prereg §2, SMD).

Windows: 60 samples long, a new one every 10 samples.
Labels for a TEST window, from the point labels inside it:
    POSITIVE (1)  at least one anomalous point among the window's final 10 samples
    NORMAL   (0)  no anomalous point anywhere in the window
    EXCLUDED (-1) anything else (an anomaly only in the first 50 samples);
                  these windows are left out of every metric
Why the "final 10" rule: the window is scored at its end, so it counts as a hit
only if the anomaly is recent; windows that merely contain an old anomaly are
neither a fair hit nor a fair false alarm, so they are dropped.
"""
import numpy as np

POSITIVE, NORMAL, EXCLUDED = 1, 0, -1


def window_starts(n_points, size=60, step=10):
    """Start index of every full window: 0, step, 2*step, ...

    Only full windows are kept, so the last few points may not start a window.
    """
    if n_points < size:
        return np.array([], dtype=int)
    return np.arange(0, n_points - size + 1, step)


def make_windows(X, size=60, step=10):
    """Cut a (time, channels) array into windows.

    Returns (windows, starts):
        windows: array (n_windows, size, channels); windows[i] = X[starts[i] : starts[i] + size]
        starts:  array (n_windows,) of start indices
    """
    X = np.asarray(X)
    starts = window_starts(len(X), size, step)
    # index matrix: row i = [s_i, s_i + 1, ..., s_i + size - 1]
    idx = starts[:, None] + np.arange(size)[None, :]
    return X[idx], starts


def label_windows(point_labels, starts, size=60, tail=10):
    """Apply the window-labelling rule to every window. Returns int array of 1 / 0 / -1."""
    anomalous = np.asarray(point_labels).astype(bool)
    out = np.empty(len(starts), dtype=int)
    for i, s in enumerate(starts):
        window = anomalous[s : s + size]
        if window[-tail:].any():        # anomaly in the final `tail` samples
            out[i] = POSITIVE
        elif not window.any():          # no anomaly at all
            out[i] = NORMAL
        else:                           # anomaly only earlier in the window
            out[i] = EXCLUDED
    return out
