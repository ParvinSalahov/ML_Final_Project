"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""

    def __init__(self):
        self.feature: int | None = None  # feature index to split on (internal)
        self.threshold: float | None = None  # split threshold (internal)
        self.left: Node | None = None  # left child
        self.right: Node | None = None  # right child
        self.value: int | float | None = None  # leaf prediction (class label or mean)
        self.probs: np.ndarray | None = (
            None  # leaf class probabilities (for classification)
        )

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",  # "classification" | "regression"
        criterion: str = "gini",  # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        if task not in ("classification", "regression"):
            raise ValueError(
                f"Unknown task: {task}. Must be 'classification' or 'regression'."
            )

        # Auto-adjust default criterion if task is regression and criterion was not overridden
        if task == "regression" and criterion == "gini":
            criterion = "mse"

        if task == "classification" and criterion not in ("gini", "entropy"):
            raise ValueError(
                f"Classification criterion must be 'gini' or 'entropy', got {criterion}"
            )
        if task == "regression" and criterion != "mse":
            raise ValueError(f"Regression criterion must be 'mse', got {criterion}")

        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if min_impurity_decrease < 0.0:
            raise ValueError("min_impurity_decrease must be >= 0.0")

        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        y_arr = np.asarray(y)
        n = len(y_arr)
        if n == 0:
            return 0.0

        if self.task == "regression":
            mean_y = np.mean(y_arr)
            return float(np.mean((y_arr - mean_y) ** 2))

        # Classification
        _, counts = np.unique(y_arr, return_counts=True)
        p = counts / float(n)

        if self.criterion == "gini":
            return float(1.0 - np.sum(p**2))
        elif self.criterion == "entropy":
            p = p[p > 0]
            return float(-np.sum(p * np.log2(p)))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None

        parent_impurity = self._impurity(y)
        if parent_impurity <= 1e-12:
            return None

        best_decrease = 0.0
        best_feature = None
        best_threshold = None

        if self.task == "regression":
            total_sum = float(np.sum(y))
            total_sum_sq = float(np.sum(y**2))
        else:
            class_map = {c: i for i, c in enumerate(self.classes_)}
            y_indices = np.array([class_map[val] for val in y], dtype=int)
            n_classes = len(self.classes_)
            total_counts = np.bincount(y_indices, minlength=n_classes).astype(float)

        for j in range(n_features):
            xj = X[:, j]
            sort_idx = np.argsort(xj)
            xj_sorted = xj[sort_idx]

            split_mask = xj_sorted[:-1] != xj_sorted[1:]
            split_indices = np.flatnonzero(split_mask) + 1
            if len(split_indices) == 0:
                continue

            # Enforce min_samples_leaf on both sides
            valid = (split_indices >= self.min_samples_leaf) & (
                (n_samples - split_indices) >= self.min_samples_leaf
            )
            split_indices = split_indices[valid]
            if len(split_indices) == 0:
                continue

            left_n = split_indices.astype(float)
            right_n = (n_samples - split_indices).astype(float)

            if self.task == "regression":
                yj_sorted = y[sort_idx]
                cumsum_y = np.cumsum(yj_sorted)
                cumsum_y2 = np.cumsum(yj_sorted**2)

                left_sum = cumsum_y[split_indices - 1]
                right_sum = total_sum - left_sum

                left_sum2 = cumsum_y2[split_indices - 1]
                right_sum2 = total_sum_sq - left_sum2

                left_var = np.maximum(
                    0.0, left_sum2 / left_n - (left_sum / left_n) ** 2
                )
                right_var = np.maximum(
                    0.0, right_sum2 / right_n - (right_sum / right_n) ** 2
                )

                decreases = (
                    parent_impurity
                    - (left_n / n_samples) * left_var
                    - (right_n / n_samples) * right_var
                )
            else:
                yj_indices = y_indices[sort_idx]
                y_one_hot = np.zeros((n_samples, n_classes), dtype=float)
                y_one_hot[np.arange(n_samples), yj_indices] = 1.0
                cumsum_classes = np.cumsum(y_one_hot, axis=0)

                left_counts = cumsum_classes[split_indices - 1]
                right_counts = total_counts[None, :] - left_counts

                p_left = left_counts / left_n[:, None]
                p_right = right_counts / right_n[:, None]

                if self.criterion == "gini":
                    left_imp = 1.0 - np.sum(p_left**2, axis=1)
                    right_imp = 1.0 - np.sum(p_right**2, axis=1)
                else:  # entropy
                    eps = 1e-15
                    p_l_safe = np.maximum(p_left, eps)
                    p_r_safe = np.maximum(p_right, eps)
                    left_imp = -np.sum(
                        np.where(p_left > 0, p_left * np.log2(p_l_safe), 0.0),
                        axis=1,
                    )
                    right_imp = -np.sum(
                        np.where(p_right > 0, p_right * np.log2(p_r_safe), 0.0),
                        axis=1,
                    )

                decreases = (
                    parent_impurity
                    - (left_n / n_samples) * left_imp
                    - (right_n / n_samples) * right_imp
                )

            best_idx_in_valid = int(np.argmax(decreases))
            feat_decrease = float(decreases[best_idx_in_valid])

            if (
                feat_decrease > best_decrease
                and feat_decrease > self.min_impurity_decrease
            ):
                best_decrease = feat_decrease
                best_feature = j
                k = split_indices[best_idx_in_valid]
                best_threshold = float((xj_sorted[k - 1] + xj_sorted[k]) / 2.0)

        if best_feature is None:
            return None

        return (best_feature, best_threshold, best_decrease)

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        node = Node()
        n_samples = len(y)

        # Precompute leaf value / probabilities
        if self.task == "regression":
            node.value = float(np.mean(y))
        else:
            counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
            node.probs = (
                counts / float(n_samples)
                if n_samples > 0
                else np.zeros(len(self.classes_))
            )
            node.value = self.classes_[np.argmax(counts)]

        # Stopping criteria:
        # 1. max_depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        # 2. fewer samples than min_samples_split
        if n_samples < self.min_samples_split:
            return node

        # 3. pure node
        if self.task == "regression":
            if np.all(y == y[0]):
                return node
        else:
            if len(np.unique(y)) <= 1:
                return node

        # 4. find best split
        split = self._best_split(X, y)
        if split is None:
            return node

        best_feature, best_threshold, best_decrease = split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Fallback if split fails to divide samples
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return node

        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")

        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be a 1D array with the same length as X")

        self.n_features_in_ = X.shape[1]

        if self.task == "classification":
            self.classes_ = np.unique(y)
            if len(self.classes_) == 0:
                raise ValueError("y cannot be empty")
        else:
            self.classes_ = None

        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def _predict_proba_one(self, x, node: Node):
        if node.is_leaf():
            return node.probs
        if x[node.feature] <= node.threshold:
            return self._predict_proba_one(x, node.left)
        else:
            return self._predict_proba_one(x, node.right)

    def predict(self, X):
        if self.root is None:
            raise RuntimeError("fit must be called before predict")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        preds = [self._predict_one(x, self.root) for x in X]
        if self.task == "regression":
            return np.array(preds, dtype=float)
        return np.array(preds, dtype=self.classes_.dtype)

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification task")
        if self.root is None:
            raise RuntimeError("fit must be called before predict_proba")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        probs = [self._predict_proba_one(x, self.root) for x in X]
        return np.array(probs, dtype=float)
