"""Load the Server Machine Dataset (SMD), one machine at a time.

Files (see data/README.md): data/SMD/{train,test,test_label}/machine-<g>-<i>.txt
- train, test: one row per time step, 38 comma-separated metrics.
- test_label:  one 0/1 label per test row.
Training data has no label file: the prereg treats it as all normal (labels = 0).
"""
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("data/SMD")


def list_machines(root=DEFAULT_ROOT):
    """All machine names, in natural order (machine-1-1, ..., machine-3-11)."""
    names = [p.stem for p in Path(root, "train").glob("machine-*.txt")]
    # sort by the numbers in the name, so machine-3-10 comes after machine-3-9
    return sorted(names, key=lambda n: [int(x) for x in n.split("-")[1:]])


def load_machine(name, root=DEFAULT_ROOT):
    """Return (train, test, test_labels) for one machine.

    train:       float array (n_train, 38)
    test:        float array (n_test, 38)
    test_labels: int array (n_test,), 1 = anomalous point
    """
    root = Path(root)
    train = np.loadtxt(root / "train" / f"{name}.txt", delimiter=",", dtype=float)
    test = np.loadtxt(root / "test" / f"{name}.txt", delimiter=",", dtype=float)
    labels = np.loadtxt(root / "test_label" / f"{name}.txt", delimiter=",", dtype=int)
    assert len(test) == len(labels), f"{name}: test and label lengths differ"
    assert train.shape[1] == test.shape[1], f"{name}: channel counts differ"
    return train, test, labels
