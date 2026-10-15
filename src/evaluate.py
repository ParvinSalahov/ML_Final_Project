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
