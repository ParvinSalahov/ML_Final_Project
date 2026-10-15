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
from pyexpat import features

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
        
    ):
        if not np.isfinite(lambda_) or lambda_ <= 0:
            raise ValueError("lambda_ must be a positive finite number")

        if (
             isinstance(n_iters, bool)
            or not isinstance(n_iters, (int, np.integer))
            or n_iters < 1
):
            raise ValueError("n_iters must be a positive integer")
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.w = None              # weight vector
        self.b = 0.0  
        self.classes_: np.ndarray | None = None

    
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

        rng = np.random.default_rng(self.random_state)
        weights = np.zeros(features.shape[1], dtype=float)
        bias = 0.0

        for step in range(1, self.n_iters + 1):
            index = rng.integers(features.shape[0])
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
        """Return the signed margin  w . x + b  for each row."""
        raise NotImplementedError("TODO: decision_function")

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""
        raise NotImplementedError("TODO: predict")
