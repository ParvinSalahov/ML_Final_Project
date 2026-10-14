"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_se = y_true-y_pred
    se = np.dot(y_se, y_se)
    mse = se/y_se.size
    rmse_val = np.sqrt(mse)
    return rmse_val


def mae(y_true, y_pred) -> float:
    y_e = y_true-y_pred
    y_ae = np.abs(y_e)
    mae_val = np.mean(y_ae)
    return mae_val


def r2(y_true, y_pred) -> float:
    y_se = y_true-y_pred
    res = np.dot(y_se, y_se)
    mean=np.mean(y_true)
    tot=np.dot((y_true-mean),(y_true-mean))
    r2_val=1-(res/tot)
    return r2_val


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    raise NotImplementedError("TODO: confusion matrix")


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    raise NotImplementedError("TODO: precision/recall/f1")


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.
    """
    raise NotImplementedError("TODO: ROC AUC")


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.
    """
    raise NotImplementedError("TODO: print comparison table")
