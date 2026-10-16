"""
linear_baselines.py — Problem 8 (bonus): from-scratch linear baselines for Task A.

  - RidgeRegressionScratch : closed-form normal equations,
        w = (Xc^T Xc + lambda I)^-1 Xc^T yc   (intercept left unpenalised via centering)
  - SVRScratch             : linear epsilon-insensitive SVR trained with sub-gradient descent,
        J(w, b) = (lambda / 2) ||w||^2 + mean(max(0, |y - (w.x + b)| - epsilon))

Both expect STANDARDISED features (use the train-statistics standardisation used for the SVM)
and the same log1p(price) target as the tree regressor, so metrics are directly comparable.
Hyper-parameters are chosen on the validation split only; the test split is scored once.
"""

from __future__ import annotations

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


class RidgeRegressionScratch:
    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_
        self.w_ = None
        self.b_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - x_mean
        yc = y - y_mean
        d = X.shape[1]
        A = Xc.T @ Xc + self.lambda_ * np.eye(d)
        rhs = Xc.T @ yc
        try:
            self.w_ = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            self.w_ = np.linalg.lstsq(A, rhs, rcond=None)[0]
        self.b_ = float(y_mean - x_mean @ self.w_)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w_ + self.b_


class SVRScratch:
    def __init__(self, lambda_: float = 1e-3, epsilon: float = 0.1, n_epochs: int = 1000, lr0: float = 0.5):
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.n_epochs = n_epochs
        self.lr0 = lr0
        self.w_ = None
        self.b_ = 0.0
        self.history_ = []

    def objective(self, X, y, w, b):
        excess = np.abs(y - (X @ w + b)) - self.epsilon
        return float(0.5 * self.lambda_ * (w @ w) + np.maximum(0.0, excess).mean())

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        w = np.zeros(d)
        b = float(np.median(y))
        best_obj = np.inf
        best_w, best_b = w.copy(), b
        self.history_ = []
        for t in range(1, self.n_epochs + 1):
            res = y - (X @ w + b)
            viol = np.abs(res) > self.epsilon
            s = np.sign(res[viol])
            grad_w = self.lambda_ * w - (X[viol].T @ s) / n
            grad_b = -s.sum() / n
            lr = self.lr0 / np.sqrt(t)
            w = w - lr * grad_w
            b = b - lr * grad_b
            obj = self.objective(X, y, w, b)
            self.history_.append(obj)
            if obj < best_obj:
                best_obj = obj
                best_w, best_b = w.copy(), b
        self.w_, self.b_ = best_w, best_b
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w_ + self.b_


def print_task_a_table(rows):
    cols = ["RMSE", "MAE", "R2", "Fit Time (s)"]
    print(f"  {'Model':<30}" + "".join(f"{c:>14}" for c in cols))
    print("  " + "-" * (30 + 14 * len(cols)))
    for name, r in rows.items():
        print(f"  {name:<30}" + "".join(f"{r[c]:>14.4f}" for c in cols))


def run_linear_baselines(X_tr, y_tr, X_va, y_va, X_te, y_te, figures_dir=FIG_DIR, quick=False, seed=42):
    from sklearn.linear_model import Ridge
    from sklearn.svm import LinearSVR

    X_tr = np.asarray(X_tr, dtype=float)
    X_va = np.asarray(X_va, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    y_tr = np.asarray(y_tr, dtype=float)
    y_va = np.asarray(y_va, dtype=float)
    y_te = np.asarray(y_te, dtype=float)
    os.makedirs(figures_dir, exist_ok=True)
    results = {}

    lambdas = np.logspace(-4, 4, 17)
    tr_curve, va_curve = [], []
    for lam in lambdas:
        m = RidgeRegressionScratch(lam).fit(X_tr, y_tr)
        tr_curve.append(_rmse(y_tr, m.predict(X_tr)))
        va_curve.append(_rmse(y_va, m.predict(X_va)))
    best_i = int(np.argmin(va_curve))
    ridge_lam = float(lambdas[best_i])
    print(f"  - Ridge sweep: best lambda = {ridge_lam:g} (val RMSE = {va_curve[best_i]:.4f})")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(lambdas, tr_curve, linestyle="--", marker="o", color="C0", label="train RMSE")
    ax.plot(lambdas, va_curve, linestyle="-", marker="o", color="C1", label="val RMSE")
    ax.axvline(ridge_lam, color="gray", linestyle=":", label=f"selected lambda={ridge_lam:g}")
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("RMSE (log1p price)")
    ax.set_title("Ridge (normal equations): lambda sweep")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "ridge_lambda_sweep.png"), dpi=150)
    plt.close(fig)

    t0 = time.time()
    ridge = RidgeRegressionScratch(ridge_lam).fit(X_tr, y_tr)
    ridge_fit = time.time() - t0
    p = ridge.predict(X_te)
    results["Ours: Ridge (normal eq.)"] = {
        "RMSE": _rmse(y_te, p), "MAE": _mae(y_te, p), "R2": _r2(y_te, p), "Fit Time (s)": ridge_fit,
    }

    t0 = time.time()
    sk_ridge = Ridge(alpha=ridge_lam).fit(X_tr, y_tr)
    sk_ridge_fit = time.time() - t0
    p = sk_ridge.predict(X_te)
    results["scikit-learn Ridge"] = {
        "RMSE": _rmse(y_te, p), "MAE": _mae(y_te, p), "R2": _r2(y_te, p), "Fit Time (s)": sk_ridge_fit,
    }
    max_dw = float(np.max(np.abs(ridge.w_ - sk_ridge.coef_)))
    print(f"  - Ridge check: max |w_ours - w_sklearn| = {max_dw:.2e}, |b_ours - b_sklearn| = {abs(ridge.b_ - sk_ridge.intercept_):.2e}")

    if quick:
        grid = [(1e-3, 0.1)]
    else:
        grid = [(lam, eps) for lam in (1e-4, 1e-3, 1e-2, 1e-1) for eps in (0.05, 0.1, 0.25, 0.5)]
    best = None
    for lam, eps in grid:
        m = SVRScratch(lam, eps).fit(X_tr, y_tr)
        v = _rmse(y_va, m.predict(X_va))
        if best is None or v < best[0]:
            best = (v, lam, eps)
    _, svr_lam, svr_eps = best
    print(f"  - SVR sweep ({len(grid)} configs): best lambda = {svr_lam:g}, epsilon = {svr_eps:g} (val RMSE = {best[0]:.4f})")

    t0 = time.time()
    svr = SVRScratch(svr_lam, svr_eps).fit(X_tr, y_tr)
    svr_fit = time.time() - t0
    p = svr.predict(X_te)
    results["Ours: SVR (sub-gradient)"] = {
        "RMSE": _rmse(y_te, p), "MAE": _mae(y_te, p), "R2": _r2(y_te, p), "Fit Time (s)": svr_fit,
    }

    n = len(X_tr)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk_svr = LinearSVR(
            epsilon=svr_eps, C=1.0 / (n * svr_lam), loss="epsilon_insensitive",
            intercept_scaling=10.0, max_iter=20000, dual=True, random_state=seed,
        ).fit(X_tr, y_tr)
    sk_svr_fit = time.time() - t0
    p = sk_svr.predict(X_te)
    results["scikit-learn LinearSVR"] = {
        "RMSE": _rmse(y_te, p), "MAE": _mae(y_te, p), "R2": _r2(y_te, p), "Fit Time (s)": sk_svr_fit,
    }
    ref_obj = svr.objective(X_tr, y_tr, sk_svr.coef_, float(np.ravel(sk_svr.intercept_)[0]))
    final_obj = svr.objective(X_tr, y_tr, svr.w_, svr.b_)
    print(f"  - SVR objective: ours = {final_obj:.5f} | LinearSVR solution = {ref_obj:.5f}")

    h = np.asarray(svr.history_)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(np.arange(1, len(h) + 1), h, color="C0", label="sub-gradient descent")
    ax.axhline(ref_obj, color="C3", linestyle="--", label="LinearSVR solution")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective J(w, b)")
    ax.set_title(f"SVR training objective (lambda={svr_lam:g}, epsilon={svr_eps:g})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "svr_convergence.png"), dpi=150)
    plt.close(fig)

    results["_params"] = {"ridge_lambda": ridge_lam, "svr_lambda": svr_lam, "svr_epsilon": svr_eps}
    return results