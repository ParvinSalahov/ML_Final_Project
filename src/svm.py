"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""

from __future__ import annotations
import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        random_state: int = 42,
        n_epochs: int = 20,
        learning_rate_scale: float = 1.0,
        use_rff: bool = True,
        n_components: int = 256,
        gamma: float = 1.0,
        
    ):
        if not np.isfinite(lambda_) or lambda_ <= 0:
            raise ValueError("lambda_ must be a positive finite number")

        if (
        isinstance(n_epochs, bool)
        or not isinstance(n_epochs, (int, np.integer))
        or n_epochs < 1):
            raise ValueError("n_epochs must be a positive integer")

        if not np.isfinite(learning_rate_scale) or learning_rate_scale <= 0:
            raise ValueError("learning_rate_scale must be a positive finite number")

        if not isinstance(use_rff, bool):
            raise ValueError("use_rff must be a boolean")
        if (
            isinstance(n_components, bool)
            or not isinstance(n_components, (int, np.integer))
            or n_components < 1):
            raise ValueError("n_components must be a positive integer")

        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be a positive finite number")
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.learning_rate_scale = learning_rate_scale
        self.random_state = random_state
        self.use_rff = use_rff
        self.n_components = n_components
        self.gamma = gamma

        self.w: np.ndarray | None = None
        self.b = 0.0
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.rff_weights_: np.ndarray | None = None
        self.rff_offsets_: np.ndarray | None = None

        self.objective_history_: list[float] = []
        self.support_vector_indices_: np.ndarray | None = None
        self.training_margins_: np.ndarray | None = None
        self.weight_norm_: float | None = None
        self.margin_width_: float | None = None
    
    def _to_pm1(self,y):
        labels = np.asarray(y)

        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("y must be a non-empty one-dimensional array")

        if labels.dtype.kind not in "biuf":
            raise ValueError("y must contain numeric class labels")

        if not np.all(np.isfinite(labels)):
            raise ValueError("y must contain only finite labels")

        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError("y must contain exactly two classes")

        self.classes_ = classes
        return np.where(labels == classes[0], -1.0, 1.0)

    def _transform(self, X):
        if not self.use_rff:
            return X

        if self.rff_weights_ is None or self.rff_offsets_ is None:
            raise RuntimeError("Random Fourier features are not initialized")

        return np.sqrt(2.0 / self.n_components) * np.cos(
            X @ self.rff_weights_ + self.rff_offsets_
        )
    def _objective(self, X_transformed, y_pm1, weights, bias):
        scores = X_transformed @ weights + bias
        hinge_loss = np.maximum(0.0, 1.0 - y_pm1 * scores)

        return float(
            0.5 * self.lambda_ * np.dot(weights, weights)
            + np.mean(hinge_loss)
        )
    def fit(self, X, y):
        features = np.asarray(X, dtype=float)

        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] == 0
        ):
            raise ValueError("X must be a non-empty two-dimensional array")

        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values")

        labels = self._to_pm1(y)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self.n_features_in_ = features.shape[1]

        sample_rng = np.random.default_rng(self.random_state)

        if self.use_rff:
            feature_rng = np.random.default_rng(self.random_state + 1)
            self.rff_weights_ = feature_rng.normal(
                scale=np.sqrt(2.0 * self.gamma),
                size=(self.n_features_in_, self.n_components),
            )
            self.rff_offsets_ = feature_rng.uniform(
                0.0,
                2.0 * np.pi,
                self.n_components,
            )
        else:
            self.rff_weights_ = None
            self.rff_offsets_ = None

        transformed = self._transform(features)
        weights = np.zeros(transformed.shape[1], dtype=float)
        bias = 0.0

        self.objective_history_ = [
            self._objective(transformed, labels, weights, bias)
        ]

        step = 0

        for _ in range(self.n_epochs):
            for index in sample_rng.permutation(transformed.shape[0]):
                step += 1
                x_i = transformed[index]
                y_i = labels[index]

                eta = self.learning_rate_scale / (self.lambda_ * step)
                margin = y_i * (np.dot(weights, x_i) + bias)

                weights *= 1.0 - eta * self.lambda_
                if margin < 1.0:
                    weights += eta * y_i * x_i
                    bias += eta * y_i

            self.objective_history_.append(
                self._objective(transformed, labels, weights, bias)
            )

        self.w = weights
        self.b = bias
        scores = transformed @ self.w + self.b
        self.training_margins_ = labels * scores

        self.support_vector_indices_ = np.flatnonzero(
            self.training_margins_ <= 1.0
        )

        self.weight_norm_ = float(np.linalg.norm(self.w))

        if self.weight_norm_ > 0:
            self.margin_width_ = 2.0 / self.weight_norm_
        else:
            self.margin_width_ = float("inf")

        return self

    def decision_function(self, X):
        if (
            self.w is None
            or self.classes_ is None
            or self.n_features_in_ is None
            or self.rff_weights_ is None
            or self.rff_offsets_ is None
        ):
            raise RuntimeError("fit must be called before making predictions")

        features = np.asarray(X, dtype=float)

        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same feature count used in fit")

        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values")

        transformed = self._transform(features)
        return transformed @ self.w + self.b

    def predict(self, X):
        margins = self.decision_function(X)
        return self.classes_[(margins >= 0).astype(int)]
