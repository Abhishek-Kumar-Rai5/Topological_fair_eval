"""The one detector used for every feature set (prereg §5).

Isolation Forest with fixed defaults: 300 trees, max_samples="auto", seed varied.
No tuning (SMD validation has no anomalies to tune on).
A StandardScaler is fitted on the TRAINING features only, then applied to
validation and test features.
"""
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IFDetector:
    def __init__(self, seed, n_estimators=300, max_samples="auto"):
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, random_state=seed
        )

    def fit(self, train_features):
        """Fit on training-window features only (all normal)."""
        self.forest.fit(self.scaler.fit_transform(train_features))
        return self

    def score(self, features):
        """Anomaly score per window; higher = more anomalous.

        sklearn's score_samples is higher for NORMAL points, so we negate it.
        """
        return -self.forest.score_samples(self.scaler.transform(features))
