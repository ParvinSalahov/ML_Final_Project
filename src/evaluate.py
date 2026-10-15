"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_se = y_true - y_pred
    se = np.dot(y_se, y_se)
    mse = se / y_se.size
    rmse_val = np.sqrt(mse)
    return rmse_val


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_e = y_true - y_pred
    y_ae = np.abs(y_e)
    mae_val = np.mean(y_ae)
    return mae_val


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_se = y_true - y_pred
    res = np.dot(y_se, y_se)
    mean = np.mean(y_true)
    tot = np.dot((y_true - mean), (y_true - mean))
    r2_val = 1 - (res / tot)
    return r2_val


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y = y_pred + y_true
    TP = np.count_nonzero(y == 2)
    TN = np.count_nonzero(y == 0)
    PP = np.count_nonzero(y_pred == 1)
    PN = np.count_nonzero(y_pred == 0)
    FP = PP - TP
    FN = PN - TN
    matris = np.array([[TN, FP], [FN, TP]])
    return matris


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y = y_pred + y_true
    TP = np.count_nonzero(y == 2)
    PP = np.count_nonzero(y_pred == 1)
    if PP == 0:
        prec = 0
    else:
        prec = TP / PP
    AP = np.count_nonzero(y_true == 1)
    if AP == 0:
        rec = 0
    else:
        rec = TP / AP
    sum = prec + rec
    if sum == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / sum
    return prec, rec, f1


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    if n1 == 0 or n0 == 0:
        return 0.5

    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_y = y_true[sorted_indices]

    _, inverse, counts = np.unique(
        sorted_scores, return_inverse=True, return_counts=True
    )
    cum_counts = np.cumsum(counts)

    avg_ranks = cum_counts - (counts - 1) / 2.0
    ranks = avg_ranks[inverse]

    r1 = np.sum(ranks[sorted_y == 1])

    u1 = r1 - (n1 * (n1 + 1)) / 2.0

    auc = u1 / (n1 * n0)

    return float(auc)


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.
    """
    if not name_to_metrics:
        print("No metrics provided to compare.")
        return

    all_metrics = []
    for m_dict in name_to_metrics.values():
        for k in m_dict.keys():
            if k not in all_metrics:
                all_metrics.append(k)

    col_widths = {m: max(len(m), 10) for m in all_metrics}
    name_width = max(max(len(str(name)) for name in name_to_metrics.keys()), 24)

    header = f"{'Model':<{name_width}} | " + " | ".join(
        f"{m:>{col_widths[m]}}" for m in all_metrics
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    for name, m_dict in name_to_metrics.items():
        row_str = f"{str(name):<{name_width}} | "
        vals = []
        for m in all_metrics:
            val = m_dict.get(m, "---")
            if isinstance(val, float):
                vals.append(f"{val:>{col_widths[m]}.4f}")
            else:
                vals.append(f"{str(val):>{col_widths[m]}}")
        row_str += " | ".join(vals)
        print(row_str)

    print(separator + "\n")


# --- Feature importances --------------------------------------
def plot_feature_importances(
    importances: np.ndarray,
    feature_names: list,
    title: str = "Decision Tree Feature Importances",
    top_n: int = 15,
    save_path: str = None,
) -> None:
    """
    Horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    importances : 1-D array of shape (n_features,)
        Normalized impurity-decrease importances (sum to 1.0).
        Comes from DecisionTree.feature_importances_ after fit().
    feature_names : list of str
        Column names matching the feature order used during fit().
    title : str
        Chart title.
    top_n : int
        How many top features to display.
    save_path : str or None
        If given, save the figure here; otherwise show interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping feature importance plot.")
        return

    importances = np.asarray(importances, dtype=float)
    feature_names = list(feature_names)

    if len(importances) != len(feature_names):
        raise ValueError(
            f"importances length ({len(importances)}) != "
            f"feature_names length ({len(feature_names)})"
        )

    # Sort descending, take top N, then reverse for horizontal bar chart
    order = np.argsort(importances)[::-1][:top_n]
    top_imp = importances[order][::-1]
    top_names = [feature_names[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4 + 1)), dpi=150)
    bars = ax.barh(top_names, top_imp, color="#4C72B0", edgecolor="white", height=0.7)

    for bar, val in zip(bars, top_imp):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Normalized impurity decrease (sum = 1.0)")
    ax.set_title(title)
    ax.set_xlim(0, top_imp.max() * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved feature importance plot: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_svm_feature_weights(
    svm_model,
    feature_names: list,
    title: str = "Pegasos SVM: Top Feature Contributions",
    top_n: int = 15,
    save_path: str = None,
) -> None:
    """
    Approximate per-input-feature importance for the Pegasos RFF SVM.

    Because Pegasos + RFF maps X -> phi(X) before learning w, we cannot
    directly read off input-feature weights.  Instead we project the learned
    weight vector back through the RFF projection matrix Omega:

        importance_j = |w|^T |Omega[:, j]|

    where Omega has shape (n_features, n_components), so each column j gives
    the RFF projection coefficients for input feature j.  This is a proxy
    for how much each original feature drives the decision boundary.

    Parameters
    ----------
    svm_model : fitted PegasosSVM instance
    feature_names : list of str
    title, top_n, save_path : same as plot_feature_importances
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping SVM weight plot.")
        return

    if svm_model.w is None:
        raise RuntimeError("SVM must be fitted before plotting weights.")

    if not svm_model.use_rff:
        # Linear SVM: weights map directly to input features
        raw = np.abs(svm_model.w)
    else:
        # RFF SVM: rff_weights_ has shape (n_features, n_components)
        # importance_j = sum_k  |w_k| * |Omega[j, k]|
        Omega = svm_model.rff_weights_          # (n_features, n_components)
        raw = Omega.dot(np.abs(svm_model.w))    # (n_features,)
        raw = np.abs(raw)

    total = raw.sum()
    importances = raw / total if total > 0.0 else raw

    feature_names = list(feature_names)
    if len(importances) != len(feature_names):
        raise ValueError(
            f"importances length ({len(importances)}) != "
            f"feature_names length ({len(feature_names)})"
        )

    order = np.argsort(importances)[::-1][:top_n]
    top_imp = importances[order][::-1]
    top_names = [feature_names[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4 + 1)), dpi=150)
    bars = ax.barh(top_names, top_imp, color="#DD8452", edgecolor="white", height=0.7)

    for bar, val in zip(bars, top_imp):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Normalized |w|·|Omega| contribution (proxy, sum = 1.0)")
    ax.set_title(title)
    ax.set_xlim(0, top_imp.max() * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved SVM weight plot: {save_path}")
    else:
        plt.show()
    plt.close(fig)
