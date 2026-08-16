"""k-Nearest Neighbours from scratch: vectorised Euclidean distance, majority vote."""

import numpy as np

from evaluate import evaluate_model, report

NAME = "kNN"


class KNNScratch:
    """k-Nearest Neighbours classifier with Euclidean distance."""

    def __init__(self, n_neighbors=7):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """Memorise the training set; kNN has no parameters to learn."""
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y).astype(int)
        self.classes_ = np.unique(self.y_train)
        return self

    def _squared_distances(self, X):
        """Distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab, as one matrix multiply."""
        train_sq = np.sum(self.X_train**2, axis=1)
        query_sq = np.sum(X**2, axis=1)[:, np.newaxis]
        cross = X @ self.X_train.T
        dists = query_sq + train_sq - 2.0 * cross
        # Rounding can push near-zero distances slightly negative.
        return np.maximum(dists, 0.0)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        dists = self._squared_distances(X)

        # argpartition finds the k smallest without sorting the whole row.
        k = min(self.n_neighbors, self.X_train.shape[0])
        nearest = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        neighbour_labels = self.y_train[nearest]

        proba = np.zeros((X.shape[0], len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            proba[:, idx] = np.mean(neighbour_labels == cls, axis=1)
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build():
    """Estimator for this dataset; 7 neighbours, odd so a binary vote cannot tie."""
    return KNNScratch(n_neighbors=7)


def train(data=None):
    """Train and score. Returns the metrics dict."""
    from data_prep import prepare_data

    data = data or prepare_data()
    model = build()
    model.fit(data["X_train"], data["y_train"])

    return evaluate_model(NAME, model, data["X_test"], data["y_test"])


if __name__ == "__main__":
    report(train())
