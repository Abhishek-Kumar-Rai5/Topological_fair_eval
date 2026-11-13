"""Combine feature groups into the feature sets named in the prereg ("F1", "F1+F2", ...).

Usage:
    fs = FeatureSet("F1+F2").fit(train_windows)     # fits only what needs fitting (F2's PCA)
    X_train = fs.transform(train_windows)
    X_test  = fs.transform(test_windows)
Groups are concatenated side by side, in the order written in the name.
F5 (Fluent Bit-style, exploratory) is one group of 3 features.
"""
import numpy as np

from src.features_simple import F2Features, f1_features, f1_names, f2_names
from src.features_fluentbit import F5Features, f5_names
from src.alite import PerChannelPH, pc_names
from src.features_topo import F3Features, F3H0Features, F3H1Features, f3_names, f3h0_names, f3h1_names


class _F1:
    def fit(self, train_windows):
        return self                       # nothing to learn

    def transform(self, windows):
        return f1_features(windows)


GROUPS = {"F1": (_F1, f1_names), "F2": (F2Features, f2_names), "F3": (F3Features, f3_names),
          "F5": (F5Features, f5_names),
          "F3-H0": (F3H0Features, f3h0_names),      # exploratory (E3x): F3's H0 half only
          "F3-H1": (F3H1Features, f3h1_names),      # exploratory (E3x): F3's H1 half only
          "PC-H0": (lambda: PerChannelPH(0), lambda n=None: pc_names(0)),   # exploratory (L2b): per-channel clouds, tau_ac
          "PC-H1": (lambda: PerChannelPH(1), lambda n=None: pc_names(1))}


class FeatureSet:
    def __init__(self, name):
        self.name = name
        self.parts = name.split("+")
        unknown = [p for p in self.parts if p not in GROUPS]
        if unknown:
            raise ValueError(f"unknown feature group(s) {unknown}; known: {list(GROUPS)}")

    def fit(self, train_windows):
        """Fit every group on TRAINING windows only."""
        self.groups = [GROUPS[p][0]().fit(train_windows) for p in self.parts]
        self.n_channels = train_windows.shape[2]
        return self

    def transform(self, windows):
        return np.hstack([g.transform(windows) for g in self.groups])

    def names(self):
        return [n for p in self.parts for n in GROUPS[p][1](self.n_channels)]
