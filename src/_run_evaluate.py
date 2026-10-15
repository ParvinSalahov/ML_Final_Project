"""
_run_evaluate.py -- Runner for Problem 6b feature importance plots.

Usage:
    python -m src._run_evaluate

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

y_tr_cls, threshold = dp.make_tier_label(y_tr_price)
X_tr = X_tr_df.to_numpy(dtype=float)

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
X_tr_scaled, _ = dp.standardize(X_tr_df, X_va_df)
X_tr_svm = X_tr_scaled.to_numpy(dtype=float)

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

print("\nDone. Plots saved to report/figures/")
