"""Logistic Regression from scratch: sigmoid + gradient descent on log-loss with L2."""

import numpy as np

from evaluate import evaluate_model, report

NAME = "Logistic Regression"


class LogisticRegressionScratch:
    """Binary logistic regression trained by batch gradient descent."""

    def __init__(self, learning_rate=0.1, n_iters=2000, C=1.0, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        # Inverse regularization strength, as in sklearn: smaller C shrinks harder.
        self.C = C
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = 0.0
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        """Sigmoid evaluated branch-wise so large negative z cannot overflow."""
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iters):
            probs = self._sigmoid(X @ self.weights + self.bias)

            # Log-loss gradient: the error projected back onto the inputs.
            error = probs - y
            grad_w = (X.T @ error) / n_samples
            grad_b = error.mean() if self.fit_intercept else 0.0

            # L2 penalty; intercept left unpenalised as sklearn does.
            if np.isfinite(self.C):
                grad_w += self.weights / (self.C * n_samples)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Clipped so a saturated prediction cannot give log(0).
            safe = np.clip(probs, 1e-12, 1 - 1e-12)
            log_loss = -np.mean(y * np.log(safe) + (1 - y) * np.log(1 - safe))
            if np.isfinite(self.C):
                log_loss += np.sum(self.weights**2) / (2.0 * self.C * n_samples)
            self.loss_history_.append(log_loss)

        return self

    def predict_proba(self, X):
        """Return (n, 2) array of [P(class 0), P(class 1)], as sklearn does."""
        X = np.asarray(X, dtype=float)
        p1 = self._sigmoid(X @ self.weights + self.bias)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build():
    """Estimator for this dataset; lr/iters chosen so the convex loss converges."""
    return LogisticRegressionScratch(learning_rate=0.5, n_iters=5000, C=1.0)


def train(data=None):
    """Train and score. Returns the metrics dict."""
    from data_prep import prepare_data

    data = data or prepare_data()
    model = build()
    model.fit(data["X_train"], data["y_train"])

    return evaluate_model(NAME, model, data["X_test"], data["y_test"])


if __name__ == "__main__":
    report(train())
