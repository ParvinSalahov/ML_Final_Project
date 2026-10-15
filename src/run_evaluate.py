"""
_run_evaluate.py -- Runner for Problem 6b feature importance plots.

Usage:
    python -m src.run_evaluate

Generates:
    report/figures/tree_feature_importances.png
    report/figures/svm_feature_weights.png
"""

import os
import numpy as np
from . import data_prep as dp
from .decision_tree import DecisionTree
from .svm import PegasosSVM
from . import evaluate as ev

os.makedirs("report/figures", exist_ok=True)

print("Loading data...")
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

# --- Decision Tree feature importances ---
print("\nFitting Decision Tree (entropy, depth=12, leaf=10)...")
tree = DecisionTree(
    task="classification",
    criterion="entropy",
    max_depth=12,
    min_samples_leaf=10,
    random_state=dp.SEED,
).fit(X_tr, y_tr_cls.to_numpy(dtype=int))

print("Top 5 features by impurity decrease:")
top5 = np.argsort(tree.feature_importances_)[::-1][:5]
for i in top5:
    print(f"  {feat_names[i]:30s}  {tree.feature_importances_[i]:.4f}")

ev.plot_feature_importances(
    tree.feature_importances_,
    feat_names,
    title="Decision Tree Feature Importances (entropy, depth=12, leaf=10)",
    top_n=15,
    save_path="report/figures/tree_feature_importances.png",
)

# --- Pegasos SVM feature weights ---
print("\nFitting Pegasos SVM (lambda=1e-4, gamma=0.1, 30 epochs)...")
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

ev.plot_svm_feature_weights(
    svm,
    feat_names,
    title="Pegasos SVM: Top Feature Contributions (proxy via RFF weights)",
    top_n=15,
    save_path="report/figures/svm_feature_weights.png",
)

# --- Problem 6a: PR-AUC Evaluation & Curves ---
print("\nEvaluating PR-AUC & ROC-AUC for Task B (Price Tier)...")
tree_va_scores = tree.predict_proba(X_va)[:, 1]
tree_te_scores = tree.predict_proba(X_te)[:, 1]

svm_va_scores = svm.decision_function(X_va_svm)
svm_te_scores = svm.decision_function(X_te_svm)

metrics = {
    "Decision Tree (Val)": {
        "ROC-AUC": ev.roc_auc(y_va_cls, tree_va_scores),
        "PR-AUC": ev.pr_auc(y_va_cls, tree_va_scores),
    },
    "Decision Tree (Test)": {
        "ROC-AUC": ev.roc_auc(y_te_cls, tree_te_scores),
        "PR-AUC": ev.pr_auc(y_te_cls, tree_te_scores),
    },
    "Pegasos SVM (Val)": {
        "ROC-AUC": ev.roc_auc(y_va_cls, svm_va_scores),
        "PR-AUC": ev.pr_auc(y_va_cls, svm_va_scores),
    },
    "Pegasos SVM (Test)": {
        "ROC-AUC": ev.roc_auc(y_te_cls, svm_te_scores),
        "PR-AUC": ev.pr_auc(y_te_cls, svm_te_scores),
    },
}
ev.compare(metrics)

ev.plot_pr_curve(
    {
        "Decision Tree": (y_te_cls, tree_te_scores),
        "Pegasos SVM": (y_te_cls, svm_te_scores),
    },
    title="Precision-Recall Curves on Test Set (Task B: Price Tier)",
    save_path="report/figures/pr_curve.png",
)

print("\nDone. All Problem 6 plots saved to report/figures/")
