"""Gaussian Naive Bayes from scratch: per-class bell curves, combined in log space."""

import numpy as np

from evaluate import evaluate_model, report

NAME = "Naive Bayes"


class GaussianNaiveBayesScratch:
    """Gaussian Naive Bayes classifier."""

    def __init__(self, var_smoothing=1e-9):
        # Added to every variance so a near-constant feature cannot divide by 0.
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.means_ = None
        self.variances_ = None
        self.log_priors_ = None

    def fit(self, X, y):
        """Measure each feature's mean and variance within each class."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.log_priors_ = np.zeros(n_classes)

        # Smoothing scaled to the data, matching sklearn's convention.
        epsilon = self.var_smoothing * X.var(axis=0).max()

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.means_[idx] = X_cls.mean(axis=0)
            self.variances_[idx] = X_cls.var(axis=0) + epsilon
            self.log_priors_[idx] = np.log(X_cls.shape[0] / X.shape[0])

        return self

    def _joint_log_likelihood(self, X):
        """Log prior plus summed Gaussian log-densities; logs avoid underflow."""
        n_samples = X.shape[0]
        jll = np.zeros((n_samples, len(self.classes_)))

        for idx in range(len(self.classes_)):
            mean, var = self.means_[idx], self.variances_[idx]
            log_pdf = -0.5 * (np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var)
            jll[:, idx] = self.log_priors_[idx] + log_pdf.sum(axis=1)

        return jll

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        jll = self._joint_log_likelihood(X)

        # Log-sum-exp: subtract the row max before exponentiating so nothing overflows.
        max_log = jll.max(axis=1, keepdims=True)
        exp_shifted = np.exp(jll - max_log)
        return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


def build():
    """Gaussian rather than Multinomial, since all 30 features are continuous."""
    return GaussianNaiveBayesScratch()


def train(data=None):
    """Train and score. Returns the metrics dict."""
    from data_prep import prepare_data

    data = data or prepare_data()
    model = build()
    model.fit(data["X_train"], data["y_train"])

    return evaluate_model(NAME, model, data["X_test"], data["y_test"])


if __name__ == "__main__":
    report(train())
