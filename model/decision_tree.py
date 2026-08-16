"""CART Decision Tree from scratch: greedy Gini-impurity splits, grown recursively."""

import numpy as np

from evaluate import evaluate_model, report

NAME = "Decision Tree"

# Candidate thresholds per feature; sampling beats testing every midpoint for speed.
MAX_THRESHOLDS = 32


class _Node:
    """A split (feature/threshold/children) or a leaf (class probabilities)."""

    __slots__ = ("feature", "threshold", "left", "right", "proba")

    def __init__(self, feature=None, threshold=None, left=None, right=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.proba = proba

    @property
    def is_leaf(self):
        return self.proba is not None


class DecisionTreeScratch:
    """CART decision tree classifier using Gini impurity."""

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        # None = every feature; "sqrt" = a random sqrt(d) subset per split.
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.classes_ = None
        self.n_classes_ = 0

    def _class_counts(self, y):
        return np.bincount(y, minlength=self.n_classes_)

    @staticmethod
    def _gini_from_counts(counts):
        """Gini impurity: 0 when pure, 0.5 when a two-class node is evenly split."""
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return 1.0 - np.sum(p**2)

    def _feature_subset(self, n_features, rng):
        """Which features this node may consider."""
        if self.max_features is None:
            return np.arange(n_features)
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        else:
            k = max(1, min(int(self.max_features), n_features))
        return rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X, y, rng):
        """Return the (feature, threshold, gain) that most reduces impurity."""
        n_samples, n_features = X.shape
        parent_counts = self._class_counts(y)
        parent_gini = self._gini_from_counts(parent_counts)

        best_gain, best_feature, best_threshold = 0.0, None, None

        for feature in self._feature_subset(n_features, rng):
            values = X[:, feature]
            unique_values = np.unique(values)
            if unique_values.size < 2:
                continue

            # Only midpoints between distinct values can change the partition.
            midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
            if midpoints.size > MAX_THRESHOLDS:
                midpoints = np.quantile(midpoints, np.linspace(0, 1, MAX_THRESHOLDS))
                midpoints = np.unique(midpoints)

            for threshold in midpoints:
                left_mask = values <= threshold
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_counts = self._class_counts(y[left_mask])
                right_counts = parent_counts - left_counts

                weighted = (
                    n_left * self._gini_from_counts(left_counts)
                    + n_right * self._gini_from_counts(right_counts)
                ) / n_samples

                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        return best_feature, best_threshold, best_gain

    def _grow(self, X, y, depth, rng):
        """Recursively split until a stopping rule fires, then emit a leaf."""
        counts = self._class_counts(y)
        proba = counts / counts.sum()

        stop = (
            depth >= self.max_depth
            or y.shape[0] < self.min_samples_split
            or np.count_nonzero(counts) == 1
        )
        if stop:
            return _Node(proba=proba)

        feature, threshold, gain = self._best_split(X, y, rng)
        if feature is None or gain <= 0.0:
            return _Node(proba=proba)

        left_mask = X[:, feature] <= threshold
        left = self._grow(X[left_mask], y[left_mask], depth + 1, rng)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1, rng)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.max()) + 1

        rng = np.random.default_rng(self.random_state)
        self.root = self._grow(X, y, depth=0, rng=rng)
        return self

    def _predict_row(self, row):
        """Walk the tree to the leaf this row lands in."""
        node = self.root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.proba

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.vstack([self._predict_row(row) for row in X])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def depth(self):
        """Actual depth reached, for confirming the cap did something."""

        def _d(node):
            return 0 if node.is_leaf else 1 + max(_d(node.left), _d(node.right))

        return _d(self.root)


def build():
    """Estimator for this dataset; depth capped at 5 to limit overfitting."""
    return DecisionTreeScratch(max_depth=5, random_state=42)


def train(data=None):
    """Train and score. Returns the metrics dict."""
    from data_prep import prepare_data

    data = data or prepare_data()
    model = build()
    model.fit(data["X_train"], data["y_train"])

    return evaluate_model(NAME, model, data["X_test"], data["y_test"])


if __name__ == "__main__":
    report(train())
