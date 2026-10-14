"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS and random
Fourier features (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . phi(x_i) + b))

where phi maps the standardized input to random Fourier features that
approximate an RBF kernel.

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
    """Pegasos linear SVM trained on random Fourier features of the inputs."""

    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
        n_components: int = 256,
        gamma: float = 1.0,
    ):
        if not np.isfinite(lambda_) or lambda_ <= 0:
            raise ValueError("lambda_ must be a positive finite number")
        if (
            isinstance(n_iters, bool)
            or not isinstance(n_iters, (int, np.integer))
            or n_iters < 1
        ):
            raise ValueError("n_iters must be a positive integer")
        if (
            isinstance(n_components, bool)
            or not isinstance(n_components, (int, np.integer))
            or n_components < 1
        ):
            raise ValueError("n_components must be a positive integer")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be a positive finite number")

        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.n_components = n_components
        self.gamma = gamma
        self.w: np.ndarray | None = None
        self.b = 0.0               # bias
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.rff_weights_: np.ndarray | None = None
        self.rff_offsets_: np.ndarray | None = None

    def _to_pm1(self, y):
        """Map {0,1} (or any 2-class labels) to {-1, +1}. Store the mapping."""
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("y must be a non-empty one-dimensional array")
        if np.issubdtype(labels.dtype, np.number) and not np.all(np.isfinite(labels)):
            raise ValueError("y must not contain missing or non-finite labels")

        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError("y must contain exactly two distinct classes")

        encoded = np.where(labels == classes[0], -1.0, 1.0)
        if np.unique(encoded).size != 2:
            raise ValueError("y must not contain missing or invalid class labels")

        self.classes_ = classes
        return encoded

    def fit(self, X, y):
        """
        Run Pegasos. Sample one (or a mini-batch of) example(s) per step,
        use eta_t = 1/(lambda*t) on random Fourier features, and return self.
        """
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values")

        labels = self._to_pm1(y)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        feature_seed, sample_seed = np.random.SeedSequence(self.random_state).spawn(2)
        feature_rng = np.random.default_rng(feature_seed)
        sample_rng = np.random.default_rng(sample_seed)
        self.n_features_in_ = features.shape[1]
        self.rff_weights_ = feature_rng.normal(
            scale=np.sqrt(2.0 * self.gamma),
            size=(self.n_features_in_, self.n_components),
        )
        self.rff_offsets_ = feature_rng.uniform(0.0, 2.0 * np.pi, self.n_components)
        features = np.sqrt(2.0 / self.n_components) * np.cos(
            features @ self.rff_weights_ + self.rff_offsets_
        )

        weights = np.zeros(self.n_components, dtype=float)
        bias = 0.0

        for step in range(1, self.n_iters + 1):
            index = sample_rng.integers(features.shape[0])
            x_i = features[index]
            y_i = labels[index]
            eta = 1.0 / (self.lambda_ * step)
            margin = y_i * (np.dot(weights, x_i) + bias)

            weights *= 1.0 - eta * self.lambda_
            if margin < 1.0:
                weights += eta * y_i * x_i
                bias += eta * y_i

        self.w = weights
        self.b = bias
        return self

    def decision_function(self, X):
        """Return the signed margin in the random Fourier feature space."""
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
            raise ValueError(
                "X must be a two-dimensional array with the training feature count"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values")

        transformed = np.sqrt(2.0 / self.n_components) * np.cos(
            features @ self.rff_weights_ + self.rff_offsets_
        )
        return transformed @ self.w + self.b

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""
        margins = self.decision_function(X)
        return self.classes_[(margins >= 0).astype(int)]
