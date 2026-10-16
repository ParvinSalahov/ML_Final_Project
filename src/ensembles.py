"""
ensembles.py — optional bonus: ensembles built from scratch on top of OUR OWN DecisionTree.

  - RandomForestScratch      : bootstrap rows + random feature subset at EVERY split
                               (DecisionTree(max_features=...)), average of trees.
  - GradientBoostingScratch  : squared-loss boosting, each shallow tree fits the current residuals.

`tree_cls` is the project's DecisionTree class, used UNMODIFIED. If it has no `max_features` argument,
a thin subclass defined below adds per-split feature subsampling without touching decision_tree.py.
Use the UNSTANDARDISED feature matrices (the ones the tree uses). Target is log1p(price) (Task A).
n_estimators is chosen on the validation split only; test error is reported at that choice, and
the full test curve is plotted to show the error dropping as learners are added.
"""

from __future__ import annotations

import inspect
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "report/figures"


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def _n_candidates(max_features, n_features):
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, float):
        return max(1, min(n_features, int(round(max_features * n_features))))
    return max(1, min(n_features, int(max_features)))


def _forest_tree_class(tree_cls):
    if "max_features" in inspect.signature(tree_cls.__init__).parameters:
        return tree_cls

    class SubspaceTree(tree_cls):
        def __init__(self, *args, max_features=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_features = max_features
            self._rng = None

        def fit(self, X, y):
            self._rng = np.random.default_rng(self.random_state)
            return super().fit(X, y)

        def _best_split(self, X, y):
            d = X.shape[1]
            k = _n_candidates(self.max_features, d)
            if k >= d:
                return super()._best_split(X, y)
            order = self._rng.permutation(d)
            for start in range(0, d, k):
                chosen = order[start:start + k]
                X_masked = np.zeros_like(X)
                X_masked[:, chosen] = X[:, chosen]
                split = super()._best_split(X_masked, y)
                if split is not None:
                    return split
            return None

    return SubspaceTree


class RandomForestScratch:
    def __init__(self, tree_cls, n_estimators=100, max_depth=12, min_samples_leaf=10,
                 max_features=0.5, random_state=42):
        self.tree_cls = tree_cls
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        forest_tree_cls = _forest_tree_class(self.tree_cls)
        self.trees_ = []
        for i in range(self.n_estimators):
            rows = rng.integers(0, n, n)
            tree = forest_tree_cls(
                task="regression", criterion="mse", max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                random_state=self.random_state + i,
            )
            tree.fit(X[rows], y[rows])
            self.trees_.append(tree)
        return self

    def staged_predict(self, X):
        X = np.asarray(X, dtype=float)
        total = np.zeros(len(X))
        for k, tree in enumerate(self.trees_, start=1):
            total += tree.predict(X)
            yield total / k

    def predict(self, X):
        out = None
        for out in self.staged_predict(X):
            pass
        return out


class GradientBoostingScratch:
    def __init__(self, tree_cls, n_estimators=200, learning_rate=0.1, max_depth=4,
                 min_samples_leaf=10, subsample=1.0, random_state=42):
        self.tree_cls = tree_cls
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.init_ = 0.0
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        self.init_ = float(y.mean())
        F = np.full(n, self.init_)
        self.trees_ = []
        for i in range(self.n_estimators):
            resid = y - F
            if self.subsample < 1.0:
                rows = rng.choice(n, size=max(2, int(self.subsample * n)), replace=False)
                Xs, rs = X[rows], resid[rows]
            else:
                Xs, rs = X, resid
            tree = self.tree_cls(
                task="regression", criterion="mse", max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, random_state=self.random_state + i,
            )
            tree.fit(Xs, rs)
            F = F + self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        return self

    def staged_predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), self.init_)
        for tree in self.trees_:
            out = out + self.learning_rate * tree.predict(X)
            yield out.copy()

    def predict(self, X):
        out = None
        for out in self.staged_predict(X):
            pass
        return out


def _evaluate_stages(name, val_stages, test_stages, y_va, y_te, fit_time):
    val_curve = np.array([_rmse(y_va, p) for p in val_stages])
    test_curve = np.array([_rmse(y_te, p) for p in test_stages])
    best = int(np.argmin(val_curve))
    p = test_stages[best]
    row = {"RMSE": _rmse(y_te, p), "MAE": _mae(y_te, p), "R2": _r2(y_te, p), "Fit Time (s)": fit_time}
    print(f"  - {name}: best n_estimators by validation = {best + 1} "
          f"(val RMSE = {val_curve[best]:.4f}, test RMSE = {row['RMSE']:.4f})")
    return row, val_curve, test_curve, best + 1


def run_ensembles(tree_cls, X_tr, y_tr, X_va, y_va, X_te, y_te, figures_dir=FIG_DIR, quick=False,
                  seed=42, tree_test_rmse=None, rf_params=None, gb_params=None):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    X_tr = np.asarray(X_tr, dtype=float)
    X_va = np.asarray(X_va, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    y_tr = np.asarray(y_tr, dtype=float)
    y_va = np.asarray(y_va, dtype=float)
    y_te = np.asarray(y_te, dtype=float)
    os.makedirs(figures_dir, exist_ok=True)

    rf_cfg = dict(n_estimators=30 if quick else 100, max_depth=12, min_samples_leaf=10, max_features=0.5)
    gb_cfg = dict(n_estimators=50 if quick else 200, learning_rate=0.1, max_depth=4, min_samples_leaf=10, subsample=1.0)
    rf_cfg.update(rf_params or {})
    gb_cfg.update(gb_params or {})

    rows = {}
    curves = {}

    t0 = time.time()
    rf = RandomForestScratch(tree_cls, random_state=seed, **rf_cfg).fit(X_tr, y_tr)
    rf_fit = time.time() - t0
    row, v, t, n_best = _evaluate_stages(
        "Ours RandomForest", list(rf.staged_predict(X_va)), list(rf.staged_predict(X_te)), y_va, y_te, rf_fit)
    rows[f"Ours: RandomForest (n={n_best})"] = row
    curves["rf_ours"] = (v, t)

    t0 = time.time()
    sk_rf = RandomForestRegressor(
        n_estimators=rf_cfg["n_estimators"], max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"], max_features=rf_cfg["max_features"],
        bootstrap=True, n_jobs=-1, random_state=seed,
    ).fit(X_tr, y_tr)
    sk_rf_fit = time.time() - t0
    v_st = np.cumsum([e.predict(X_va) for e in sk_rf.estimators_], axis=0) / np.arange(1, len(sk_rf.estimators_) + 1)[:, None]
    t_st = np.cumsum([e.predict(X_te) for e in sk_rf.estimators_], axis=0) / np.arange(1, len(sk_rf.estimators_) + 1)[:, None]
    row, v, t, n_best = _evaluate_stages("sklearn RandomForest", list(v_st), list(t_st), y_va, y_te, sk_rf_fit)
    rows[f"scikit-learn RandomForest (n={n_best})"] = row
    curves["rf_sk"] = (v, t)

    t0 = time.time()
    gb = GradientBoostingScratch(tree_cls, random_state=seed, **gb_cfg).fit(X_tr, y_tr)
    gb_fit = time.time() - t0
    row, v, t, n_best = _evaluate_stages(
        "Ours GradientBoosting", list(gb.staged_predict(X_va)), list(gb.staged_predict(X_te)), y_va, y_te, gb_fit)
    rows[f"Ours: GradientBoosting (n={n_best})"] = row
    curves["gb_ours"] = (v, t)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk_gb = GradientBoostingRegressor(
            n_estimators=gb_cfg["n_estimators"], learning_rate=gb_cfg["learning_rate"],
            max_depth=gb_cfg["max_depth"], min_samples_leaf=gb_cfg["min_samples_leaf"],
            subsample=gb_cfg["subsample"], random_state=seed,
        ).fit(X_tr, y_tr)
    sk_gb_fit = time.time() - t0
    row, v, t, n_best = _evaluate_stages(
        "sklearn GradientBoosting", list(sk_gb.staged_predict(X_va)), list(sk_gb.staged_predict(X_te)),
        y_va, y_te, sk_gb_fit)
    rows[f"scikit-learn GradientBoosting (n={n_best})"] = row
    curves["gb_sk"] = (v, t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, key, title in [(axes[0], "rf", "Random Forest"), (axes[1], "gb", "Gradient Boosting")]:
        v, t = curves[key + "_ours"]
        x = np.arange(1, len(v) + 1)
        ax.plot(x, t, color="C0", label="ours: test")
        ax.plot(x, v, color="C0", linestyle="--", label="ours: val")
        v2, t2 = curves[key + "_sk"]
        x2 = np.arange(1, len(v2) + 1)
        ax.plot(x2, t2, color="C1", label="sklearn: test")
        ax.plot(x2, v2, color="C1", linestyle="--", label="sklearn: val")
        if tree_test_rmse is not None:
            ax.axhline(tree_test_rmse, color="gray", linestyle=":", label="single tree: test")
        ax.set_xlabel("number of learners")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMSE (log1p price)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "ensemble_error_vs_learners.png"), dpi=150)
    plt.close(fig)

    return rows