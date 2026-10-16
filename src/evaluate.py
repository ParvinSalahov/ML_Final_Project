"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def pr_curve(y_true, scores):
    """
    Compute Precision-Recall pairs for different probability/margin thresholds.
    Returns (precision, recall, thresholds).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([])

    desc_order = np.argsort(scores)[::-1]
    y_true_sorted = y_true[desc_order]
    scores_sorted = scores[desc_order]

    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    precision = tps / (tps + fps)
    recall = tps / n_pos

    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    thresholds = scores_sorted[threshold_idxs][::-1]
    return precision, recall, thresholds


def pr_auc(y_true, scores) -> float:
    """
    Compute Area Under the Precision-Recall Curve (Average Precision).
    Uses the uninterpolated step-function integral:
        AP = sum_n (R_n - R_{n-1}) * P_n
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if np.sum(y_true == 1) == 0:
        return 0.0

    precision, recall, _ = pr_curve(y_true, scores)
    return float(np.sum(-np.diff(recall) * precision[:-1]))


def plot_pr_curve(
    curves: dict,
    title: str = "Precision-Recall Curves (Task B: Price Tier)",
    save_path: str = None,
) -> None:
    """Plot Precision-Recall curves for one or more models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping PR curve plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for name, item in curves.items():
        if isinstance(item, tuple) and len(item) == 2:
            y_t, sc = item
            prec, rec, _ = pr_curve(y_t, sc)
            auc_val = pr_auc(y_t, sc)
        elif isinstance(item, tuple) and len(item) == 3:
            prec, rec, auc_val = item
        else:
            continue
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {auc_val:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  - Saved PR curve plot: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_residuals(
    y_true,
    y_pred,
    title: str = "Task A Regression: Predicted vs Actual",
    save_path: str = None,
) -> None:
    """Plot Predicted vs Actual scatter and residual distribution for Task A."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping residual plot.")
        return

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    res = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    n_pts = len(y_true)
    if n_pts > 4000:
        idx = np.random.default_rng(42).choice(n_pts, size=4000, replace=False)
        y_t_plot, y_p_plot = y_true[idx], y_pred[idx]
    else:
        y_t_plot, y_p_plot = y_true, y_pred

    ax1.scatter(
        y_t_plot, y_p_plot, alpha=0.25, s=12, color="#4C72B0", edgecolors="none"
    )
    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=1.5,
        label="Perfect Fit (y = x)",
    )
    ax1.set_xlabel("Actual log(price)")
    ax1.set_ylabel("Predicted log(price)")
    ax1.set_title("Predicted vs Actual")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    ax2.hist(res, bins=60, color="#55A868", edgecolor="white", alpha=0.85, density=True)
    ax2.axvline(0, color="red", linestyle="--", lw=1.5)
    ax2.set_xlabel("Residual (Actual - Predicted)")
    ax2.set_ylabel("Density")
    ax2.set_title(
        f"Residual Distribution (Mean: {res.mean():.3f}, Std: {res.std():.3f})"
    )
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  - Saved residual plot: {save_path}")
    else:
        plt.show()
    plt.close(fig)


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
    """Horizontal bar chart of the top-N most important features from DecisionTree.feature_importances_."""

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
        print(f"  - Saved feature importance plot: {save_path}")
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
    Projects the learned weight vector back through the RFF matrix Omega:
        importance_j = |Omega[j, :]| . |w|
    This is a proxy — direct weight reading is not possible with RFF.
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
        Omega = svm_model.rff_weights_  # (n_features, n_components)
        raw = Omega.dot(np.abs(svm_model.w))  # (n_features,)
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
        print(f"  - Saved SVM weight plot: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# --- Problem 6: Complete Evaluation Runner ----------------------------------
def evaluate_all(save_dir: str = "report/figures") -> dict:
    """
    Run full Problem 6 evaluation pipeline:
      1. Train Decision Tree (Regression & Classification) and Pegasos SVM.
      2. Generate feature importance plots for Tree and SVM.
      3. Evaluate ROC-AUC, PR-AUC and plot PR curves.
      4. Plot residuals and analyze failure modes (luxury outliers, listing anomalies, sparse districts).
      5. Output the required Consolidated Results Table comparing all models on both tasks.
    """
    os.makedirs(save_dir, exist_ok=True)
    from . import data_prep as dp
    from .decision_tree import DecisionTree
    from .svm import PegasosSVM
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

    print("\n[Evaluate] Loading data...")
    tr, va, te = dp.prepare_data(seed=dp.SEED)
    X_tr_df, y_tr_price, feat_names, fit_info = dp.make_features(tr)
    X_va_df, y_va_price, _, _ = dp.make_features(va, fit_info=fit_info)
    X_te_df, y_te_price, _, _ = dp.make_features(te, fit_info=fit_info)

    y_tr_cls, threshold = dp.make_tier_label(y_tr_price)
    y_va_cls, _ = dp.make_tier_label(y_va_price, threshold=threshold)
    y_te_cls, _ = dp.make_tier_label(y_te_price, threshold=threshold)

    X_tr = X_tr_df.to_numpy(dtype=float)
    X_va = X_va_df.to_numpy(dtype=float)
    X_te = X_te_df.to_numpy(dtype=float)

    # 1. Decision Tree Classification
    print("\n[Evaluate] Fitting decision tree (entropy, depth=12, leaf=10)...")
    tree_cls = DecisionTree(
        task="classification",
        criterion="entropy",
        max_depth=12,
        min_samples_leaf=10,
        random_state=dp.SEED,
    ).fit(X_tr, y_tr_cls.to_numpy(dtype=int))

    plot_feature_importances(
        tree_cls.feature_importances_,
        feat_names,
        title="Decision Tree Feature Importances (entropy, depth=12, leaf=10)",
        top_n=15,
        save_path=os.path.join(save_dir, "tree_feature_importances.png"),
    )

    # 2. Pegasos SVM Classification
    print("\n[Evaluate] Fitting Pegasos SVM (lambda=1e-4, gamma=0.1, 30 epochs)...")
    X_tr_scaled, X_va_scaled = dp.standardize(X_tr_df, X_va_df)
    _, X_te_scaled = dp.standardize(X_tr_df, X_te_df)
    X_tr_svm = X_tr_scaled.to_numpy(dtype=float)
    X_va_svm = X_va_scaled.to_numpy(dtype=float)
    X_te_svm = X_te_scaled.to_numpy(dtype=float)

    svm = PegasosSVM(
        lambda_=1e-4,
        gamma=0.1,
        n_epochs=30,
        n_components=256,
        use_rff=True,
        random_state=dp.SEED,
    ).fit(X_tr_svm, y_tr_cls.to_numpy(dtype=int))

    plot_svm_feature_weights(
        svm,
        feat_names,
        title="Pegasos SVM: Top Feature Contributions (proxy via RFF weights)",
        top_n=15,
        save_path=os.path.join(save_dir, "svm_feature_weights.png"),
    )

    # 3. PR-AUC and ROC-AUC
    print("\n[Evaluate] Evaluating PR-AUC & ROC-AUC for task b (price tier)...")
    tree_va_scores = tree_cls.predict_proba(X_va)[:, 1]
    tree_te_scores = tree_cls.predict_proba(X_te)[:, 1]
    svm_va_scores = svm.decision_function(X_va_svm)
    svm_te_scores = svm.decision_function(X_te_svm)

    val_test_auc_table = {
        "Decision Tree (Val)": {
            "ROC-AUC": roc_auc(y_va_cls, tree_va_scores),
            "PR-AUC": pr_auc(y_va_cls, tree_va_scores),
        },
        "Decision Tree (Test)": {
            "ROC-AUC": roc_auc(y_te_cls, tree_te_scores),
            "PR-AUC": pr_auc(y_te_cls, tree_te_scores),
        },
        "Pegasos SVM (Val)": {
            "ROC-AUC": roc_auc(y_va_cls, svm_va_scores),
            "PR-AUC": pr_auc(y_va_cls, svm_va_scores),
        },
        "Pegasos SVM (Test)": {
            "ROC-AUC": roc_auc(y_te_cls, svm_te_scores),
            "PR-AUC": pr_auc(y_te_cls, svm_te_scores),
        },
    }
    compare(val_test_auc_table)

    plot_pr_curve(
        {
            "Decision Tree": (y_te_cls, tree_te_scores),
            "Pegasos SVM": (y_te_cls, svm_te_scores),
        },
        title="Precision-Recall Curves on Test Set (Task B: Price Tier)",
        save_path=os.path.join(save_dir, "pr_curve.png"),
    )

    # 4. Task A Regression & Residuals
    print(
        "\n[Evaluate] Fitting task a decision tree regressor (mse, depth=12, leaf=10)..."
    )
    tree_reg = DecisionTree(
        task="regression",
        criterion="mse",
        max_depth=12,
        min_samples_leaf=10,
        random_state=dp.SEED,
    ).fit(X_tr, y_tr_price.to_numpy(dtype=float))

    sk_tree_reg = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=12,
        min_samples_leaf=10,
        random_state=dp.SEED,
    ).fit(X_tr, y_tr_price.to_numpy(dtype=float))

    y_te_arr = y_te_price.to_numpy(dtype=float)
    pred_reg = tree_reg.predict(X_te)
    pred_sk_reg = sk_tree_reg.predict(X_te)

    plot_residuals(
        y_te_arr,
        pred_reg,
        title="Decision Tree (Task A: Price Regression) Residual Analysis",
        save_path=os.path.join(save_dir, "residuals_task_a.png"),
    )

    # 5. Error Analysis (Failure Mode Breakdown)
    print("\n[Problem 6] Error analysis (failure mode breakdown)")

    residuals = y_te_arr - pred_reg
    print("\nTask a regression failure modes:")
    print("  - Top luxury outliers (under-predictions: actual >> predicted):")
    worst_under = np.argsort(residuals)[::-1][:5]
    for idx in worst_under:
        act_azn = np.expm1(y_te_arr[idx])
        prd_azn = np.expm1(pred_reg[idx])
        orig = te.iloc[idx]
        loc = orig.get("location", "?")
        cat = orig.get("category", "?")
        print(
            f"   Actual: {act_azn:12,.0f} AZN | Pred: {prd_azn:10,.0f} AZN | "
            f"Area: {orig.get('area', '?')} | Cat: {cat} | Loc: {loc}"
        )
    print("  - Why it fails: massive luxury properties (>3M AZN) or Torpaq (land plots")
    print("    where living area=0) suffer from tree bounded leaf means. Trees cannot")
    print("    extrapolate beyond the highest training leaf average.")

    print(
        "\n  - Top low-end listing anomalies (over-predictions: predicted >> actual):"
    )
    worst_over = np.argsort(residuals)[:5]
    for idx in worst_over:
        act_azn = np.expm1(y_te_arr[idx])
        prd_azn = np.expm1(pred_reg[idx])
        orig = te.iloc[idx]
        loc = orig.get("location", "?")
        cat = orig.get("category", "?")
        print(
            f"   Actual: {act_azn:12,.0f} AZN | Pred: {prd_azn:10,.0f} AZN | "
            f"Area: {orig.get('area', '?')} | Cat: {cat} | Loc: {loc}"
        )
    print("  - Why it fails: data noise / listing entry errors (e.g., monthly rental")
    print("    or down payments posted under sale listings, e.g., 3,000-10,000 AZN).")

    print("\nTask b classification failure modes & confusion matrices:")
    pred_tree_cls = tree_cls.predict(X_te)
    pred_svm_cls = svm.predict(X_te_svm)
    y_te_cls_arr = y_te_cls.to_numpy(dtype=int)

    cm_tree = confusion_matrix(y_te_cls_arr, pred_tree_cls)
    cm_svm = confusion_matrix(y_te_cls_arr, pred_svm_cls)

    print(
        f"Decision Tree Confusion Matrix: TN={cm_tree[0,0]}, FP={cm_tree[0,1]}, FN={cm_tree[1,0]}, TP={cm_tree[1,1]}"
    )
    print(
        f"Pegasos SVM   Confusion Matrix: TN={cm_svm[0,0]}, FP={cm_svm[0,1]}, FN={cm_svm[1,0]}, TP={cm_svm[1,1]}"
    )

    fp_tree = np.where((pred_tree_cls == 1) & (y_te_cls_arr == 0))[0]
    fn_tree = np.where((pred_tree_cls == 0) & (y_te_cls_arr == 1))[0]

    print(
        f"  - Tree false positives (predicted premium, actually standard: {len(fp_tree)} cases):"
    )
    print(
        "    - Typical cause: large properties in distant suburbs (Absheron/Khirdalan/Masazir)"
    )
    print(
        "      or border properties whose log price sits just barely below the median threshold."
    )

    print(
        f"  - Tree false negatives (predicted standard, actually premium: {len(fn_tree)} cases):"
    )
    print(
        "    - Typical cause: small/1-room apartments in ultra-prime districts (Sabayil, Nizami St)"
    )
    print(
        "      where high square-meter price compensates for low area, but low area triggers standard prediction."
    )

    # 6. Consolidated Results Table
    print("\n[Problem 6] Consolidated results table")

    sk_tree_cls = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=12,
        min_samples_leaf=10,
        random_state=dp.SEED,
    ).fit(X_tr, y_tr_cls.to_numpy(dtype=int))
    pred_sk_cls = sk_tree_cls.predict(X_te)
    sk_cls_proba = sk_tree_cls.predict_proba(X_te)[:, 1]

    all_models_table = {
        "Custom Decision Tree": {
            "Task A RMSE": rmse(y_te_arr, pred_reg),
            "Task A MAE": mae(y_te_arr, pred_reg),
            "Task A R2": r2(y_te_arr, pred_reg),
            "Task B Acc": float(np.mean(pred_tree_cls == y_te_cls_arr)),
            "Task B F1": precision_recall_f1(y_te_cls_arr, pred_tree_cls)[2],
            "Task B ROC-AUC": roc_auc(y_te_cls_arr, tree_te_scores),
            "Task B PR-AUC": pr_auc(y_te_cls_arr, tree_te_scores),
        },
        "Sklearn Decision Tree": {
            "Task A RMSE": rmse(y_te_arr, pred_sk_reg),
            "Task A MAE": mae(y_te_arr, pred_sk_reg),
            "Task A R2": r2(y_te_arr, pred_sk_reg),
            "Task B Acc": float(np.mean(pred_sk_cls == y_te_cls_arr)),
            "Task B F1": precision_recall_f1(y_te_cls_arr, pred_sk_cls)[2],
            "Task B ROC-AUC": roc_auc(y_te_cls_arr, sk_cls_proba),
            "Task B PR-AUC": pr_auc(y_te_cls_arr, sk_cls_proba),
        },
        "Custom Pegasos SVM": {
            "Task A RMSE": "---",
            "Task A MAE": "---",
            "Task A R2": "---",
            "Task B Acc": float(np.mean(pred_svm_cls == y_te_cls_arr)),
            "Task B F1": precision_recall_f1(y_te_cls_arr, pred_svm_cls)[2],
            "Task B ROC-AUC": roc_auc(y_te_cls_arr, svm_te_scores),
            "Task B PR-AUC": pr_auc(y_te_cls_arr, svm_te_scores),
        },
    }
    compare(all_models_table)

    print("\nDone")
    return all_models_table


if __name__ == "__main__":
    evaluate_all()
