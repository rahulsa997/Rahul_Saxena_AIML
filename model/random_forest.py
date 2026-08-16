"""Random Forest from scratch: bagged decision trees with random feature subsets."""

import numpy as np

from decision_tree import DecisionTreeScratch
from evaluate import evaluate_model, report

NAME = "Random Forest (Ensemble)"


class RandomForestScratch:
    """Random forest classifier: bagged decision trees, combined by soft voting."""

    def __init__(
        self,
        n_estimators=50,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        """Grow each tree on a bootstrap sample, splitting on random feature subsets."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n_samples = X.shape[0]

        self.classes_ = np.unique(y)
        self.trees = []

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            # Bootstrap: n rows drawn with replacement.
            indices = rng.integers(0, n_samples, size=n_samples)

            # Per-tree seed drawn from the forest's generator, keeping it reproducible.
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        """Soft voting: average the per-tree probabilities."""
        X = np.asarray(X, dtype=float)
        summed = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.trees:
            summed += tree.predict_proba(X)
        return summed / len(self.trees)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build():
    """Estimator for this dataset; 50 trees, past the point where more stops helping."""
    return RandomForestScratch(
        n_estimators=50, max_depth=8, max_features="sqrt", random_state=42
    )


def train(data=None):
    """Train and score. Returns the metrics dict."""
    from data_prep import prepare_data

    data = data or prepare_data()
    model = build()
    model.fit(data["X_train"], data["y_train"])

    return evaluate_model(NAME, model, data["X_test"], data["y_test"])


if __name__ == "__main__":
    report(train())
