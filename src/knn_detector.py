"""kNN distance detector (exploratory L3): score = mean Euclidean distance to the k nearest TRAINING windows.

Same interface as IFDetector (fit on training features, score any features; higher = more anomalous).
The scaler is fitted on the training features only. Deterministic: no seed.
"""
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class KNNDetector:
    def __init__(self, k=5):
        self.k, self.scaler = k, StandardScaler()

    def fit(self, train_features):
        Z = self.scaler.fit_transform(train_features)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(Z))).fit(Z)
        return self

    def score(self, features):
        dist, _ = self.nn.kneighbors(self.scaler.transform(features))
        return dist.mean(axis=1)
