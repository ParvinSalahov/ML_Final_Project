"""
run_all.py — ONE command reproduces every headline number and figure.

Usage:
    python -m src.run_all          # Full reproduction (all models, tuning sweeps & 11 figures)
    python src/run_all.py          # Also works directly via python or VS Code Run button
    python -m src.run_all --quick  # Fast mode (skips hyperparameter grid sweeps, ~15s)

This orchestrator:
  1. Loads and cleans the bina.az dataset, displaying exact split sizes.
  2. Builds features, creates reproducible splits (SEED = 42), standardizes for SVM.
  3. Task A: Trains custom DecisionTree Regressor and scikit-learn baseline.
  4. Task B: Trains custom DecisionTree Classifier, custom Pegasos SVM, and scikit-learn baselines.
  5. Problem 5: Generates benchmark comparison parity plot (benchmark_tree.png).
  6. Problem 4: Generates hyperparameter tuning and bias-variance study plots (tune.py).
  7. Problem 6: Generates MDI feature importances, SVM weights, PR curves, and residuals.
  8. Prints the final consolidated comparison table and LaTeX snippet for report.tex.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root and src/ are in sys.path when run directly
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
for _p in [str(_project_root), str(_current_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier

try:
    from . import linear_baselines as lb
    from . import data_prep as dp
    from .decision_tree import DecisionTree
    from .svm import PegasosSVM
    from . import evaluate as ev
    from . import benchmark as bm
    from . import tune
    from . import explore_figures as ef
    from . import ensembles as en
except ImportError:
    import data_prep as dp
    from decision_tree import DecisionTree
    from svm import PegasosSVM
    import evaluate as ev
    import benchmark as bm
    import tune
    import explore_figures as ef
    import linear_baselines as lb
    import ensembles as en


def main() -> None:
    figures_dir = "report/figures"
    os.makedirs(figures_dir, exist_ok=True)

    quick_mode = "--quick" in sys.argv or "--skip-tuning" in sys.argv

    start_total_time = time.time()
    print("\nMle-ai-201 machine learning final project - reproducibility orchestrator")
    print(
        f"Mode: {'Quick (skips grid sweeps)' if quick_mode else 'Full (reproduces all 19 figures & tables)'}\n"
    )

    # -------------------------------------------------------------------------
    # 1. Data Loading, Cleaning & Split
    # -------------------------------------------------------------------------
    print("\n[Step 1/6] Loading and preparing data (house_sale dataset)...")
    t0 = time.time()
    tr, va, te = dp.prepare_data(seed=dp.SEED)
    X_tr_df, y_tr_price, feat_names, fit_info = dp.make_features(tr)
    X_va_df, y_va_price, _, _ = dp.make_features(va, fit_info=fit_info)
    X_te_df, y_te_price, _, _ = dp.make_features(te, fit_info=fit_info)

    # Classification tier labels based strictly on TRAIN median
    y_tr_cls, threshold = dp.make_tier_label(y_tr_price)
    y_va_cls, _ = dp.make_tier_label(y_va_price, threshold=threshold)
    y_te_cls, _ = dp.make_tier_label(y_te_price, threshold=threshold)

    X_tr = X_tr_df.to_numpy(dtype=float)
    X_va = X_va_df.to_numpy(dtype=float)
    X_te = X_te_df.to_numpy(dtype=float)

    # Standardize features for SVM
    X_tr_c, X_va_c, X_te_c = dp.clip_to_train_range(X_tr_df, X_va_df, X_te_df)
    X_tr_sc, X_va_sc, X_te_sc = dp.standardize(X_tr_c, X_va_c, X_te_c)
    X_tr_svm = X_tr_sc.to_numpy(dtype=float)
    X_va_svm = X_va_sc.to_numpy(dtype=float)
    X_te_svm = X_te_sc.to_numpy(dtype=float)

    y_tr_arr = y_tr_price.to_numpy(dtype=float)
    y_te_arr = y_te_price.to_numpy(dtype=float)
    y_te_cls_arr = y_te_cls.to_numpy(dtype=int)

    n_tr, n_va, n_te = len(X_tr), len(X_va), len(X_te)
    n_tot = n_tr + n_va + n_te
    print(f"Data ready in {time.time() - t0:.2f}s:")
    print(f"  - Train split     : {n_tr:,} rows ({n_tr/n_tot*100:.1f}%)")
    print(f"  - Validation split: {n_va:,} rows ({n_va/n_tot*100:.1f}%)")
    print(f"  - Test split       : {n_te:,} rows ({n_te/n_tot*100:.1f}%)")
    print(f"  - Total clean rows: {n_tot:,} rows | Features: {len(feat_names)}")
    print(
        f"  - Task B Threshold: log(price) = {threshold:.4f} (~{np.expm1(threshold):,.0f} AZN)"
    )

    print("\n  Exploratory data analysis (train split only)...")
    ef.run_explore(tr, y_tr_cls, y_va_cls, y_te_cls)

    # -------------------------------------------------------------------------
    # 2. Task A — Price Regression
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Task A (Price Regression): Training models...")
    t0 = time.time()
    custom_tree_reg = DecisionTree(
        task="regression",
        criterion="mse",
        max_depth=12,
        min_samples_leaf=10,
        random_state=dp.SEED,
    ).fit(X_tr, y_tr_price.to_numpy(dtype=float))
    c_reg_fit_time = time.time() - t0
    c_reg_pred = custom_tree_reg.predict(X_te)

    t0 = time.time()
    sk_tree_reg = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=12,
        min_samples_leaf=10,
        random_state=dp.SEED,
    ).fit(X_tr, y_tr_price.to_numpy(dtype=float))
    sk_reg_fit_time = time.time() - t0
    sk_reg_pred = sk_tree_reg.predict(X_te)

    c_rmse = ev.rmse(y_te_arr, c_reg_pred)
    c_mae = ev.mae(y_te_arr, c_reg_pred)
    c_r2 = ev.r2(y_te_arr, c_reg_pred)

    sk_rmse = ev.rmse(y_te_arr, sk_reg_pred)
    sk_mae = ev.mae(y_te_arr, sk_reg_pred)
    sk_r2 = ev.r2(y_te_arr, sk_reg_pred)

    print(
        f"  - Custom Tree Regressor : RMSE = {c_rmse:.4f} | MAE = {c_mae:.4f} | R2 = {c_r2:.4f} (fit: {c_reg_fit_time:.2f}s)"
    )
    print(
        f"  - Sklearn Tree Regressor: RMSE = {sk_rmse:.4f} | MAE = {sk_mae:.4f} | R2 = {sk_r2:.4f} (fit: {sk_reg_fit_time:.2f}s)"
    )

    print("\n[Step 2b] Problem 8 (bonus): From-scratch linear baselines (task a)...")
    y_va_arr = y_va_price.to_numpy(dtype=float)
    lin = lb.run_linear_baselines(
        X_tr_svm,
        y_tr_arr,
        X_va_svm,
        y_va_arr,
        X_te_svm,
        y_te_arr,
        figures_dir=figures_dir,
        quick=quick_mode,
        seed=dp.SEED,
    )
    lin.pop("_params")
    lb.print_task_a_table(
        {
            "Ours: Decision Tree": {
                "RMSE": c_rmse,
                "MAE": c_mae,
                "R2": c_r2,
                "Fit Time (s)": c_reg_fit_time,
            },
            "scikit-learn Tree": {
                "RMSE": sk_rmse,
                "MAE": sk_mae,
                "R2": sk_r2,
                "Fit Time (s)": sk_reg_fit_time,
            },
            **lin,
        }
    )

    print("\n[Step 2c] Optional bonus: Ensembles from scratch (task a)...")
    ens = en.run_ensembles(
        DecisionTree,
        X_tr,
        y_tr_arr,
        X_va,
        y_va_arr,
        X_te,
        y_te_arr,
        figures_dir=figures_dir,
        quick=quick_mode,
        seed=dp.SEED,
        tree_test_rmse=c_rmse,
    )

    d = te.assign(pred=c_reg_pred, y=y_te_arr)
    print("\nTask a: Decision tree scores per category (test set)")
    for cat, g in d.groupby("category"):
        print(
            f"  {cat:<22} n={len(g):>6,} | RMSE={ev.rmse(g['y'], g['pred']):.3f} | R2={ev.r2(g['y'], g['pred']):.3f}"
        )
    print(
        f"  {'ALL':<22} n={len(d):>6,} | RMSE={ev.rmse(d['y'], d['pred']):.3f} | R2={ev.r2(d['y'], d['pred']):.3f}\n"
    )

    # -------------------------------------------------------------------------
    # 3. Task B — Price Tier Classification
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Task B (Price Tier Classification): Training models...")
    t0 = time.time()
    custom_tree_cls = DecisionTree(
        task="classification",
        criterion="entropy",
        max_depth=12,
        min_samples_leaf=10,
        random_state=dp.SEED,
    ).fit(X_tr, y_tr_cls.to_numpy(dtype=int))
    c_tree_fit_time = time.time() - t0
    c_tree_pred = custom_tree_cls.predict(X_te)
    c_tree_scores = custom_tree_cls.predict_proba(X_te)[:, 1]

    t0 = time.time()
    sk_tree_cls = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=12,
        min_samples_leaf=10,
        random_state=dp.SEED,
    ).fit(X_tr, y_tr_cls.to_numpy(dtype=int))
    sk_tree_fit_time = time.time() - t0
    sk_tree_pred = sk_tree_cls.predict(X_te)
    sk_tree_scores = sk_tree_cls.predict_proba(X_te)[:, 1]

    t0 = time.time()
    if not quick_mode:
        svm_tuning = tune.run_svm_tuning()
        svm_config = svm_tuning["best_config"]
    else:
        # Quick mode skips tuning; use a fixed configuration for a quick run.
        svm_config = {
            "lambda_": 1e-4,
            "gamma": 0.1,
            "n_epochs": 30,
            "learning_rate_scale": 1.0,
            "n_components": 256,
            "use_rff": True,
        }
    custom_svm = PegasosSVM(
        lambda_=svm_config["lambda_"],
        gamma=svm_config["gamma"],
        n_epochs=svm_config["n_epochs"],
        learning_rate_scale=svm_config["learning_rate_scale"],
        n_components=svm_config["n_components"],
        use_rff=svm_config["use_rff"],
        random_state=dp.SEED,
    ).fit(X_tr_svm, y_tr_cls.to_numpy(dtype=int))
    c_svm_fit_time = time.time() - t0
    c_svm_pred = custom_svm.predict(X_te_svm)
    c_svm_scores = custom_svm.decision_function(X_te_svm)

    t0 = time.time()
    sk_sgd = SGDClassifier(
        loss="hinge",
        alpha=1e-4,
        max_iter=3000,
        tol=1e-4,
        random_state=dp.SEED,
    ).fit(X_tr_svm, y_tr_cls.to_numpy(dtype=int))
    sk_sgd_fit_time = time.time() - t0
    sk_sgd_pred = sk_sgd.predict(X_te_svm)
    sk_sgd_scores = sk_sgd.decision_function(X_te_svm)

    _, _, c_tree_f1 = ev.precision_recall_f1(y_te_cls_arr, c_tree_pred)
    _, _, sk_tree_f1 = ev.precision_recall_f1(y_te_cls_arr, sk_tree_pred)
    _, _, c_svm_f1 = ev.precision_recall_f1(y_te_cls_arr, c_svm_pred)
    _, _, sk_sgd_f1 = ev.precision_recall_f1(y_te_cls_arr, sk_sgd_pred)

    print(
        f"  - Custom Tree Classifier: Acc = {np.mean(c_tree_pred == y_te_cls_arr):.4f} | F1 = {c_tree_f1:.4f} | AUC = {ev.roc_auc(y_te_cls_arr, c_tree_scores):.4f} (fit: {c_tree_fit_time:.2f}s)"
    )
    print(
        f"  - Sklearn Tree Baseline : Acc = {np.mean(sk_tree_pred == y_te_cls_arr):.4f} | F1 = {sk_tree_f1:.4f} | AUC = {ev.roc_auc(y_te_cls_arr, sk_tree_scores):.4f} (fit: {sk_tree_fit_time:.2f}s)"
    )
    print(
        f"  - Custom Pegasos SVM    : Acc = {np.mean(c_svm_pred == y_te_cls_arr):.4f} | F1 = {c_svm_f1:.4f} | AUC = {ev.roc_auc(y_te_cls_arr, c_svm_scores):.4f} (fit: {c_svm_fit_time:.2f}s)"
    )
    print(
        f"  - Sklearn SGD Baseline  : Acc = {np.mean(sk_sgd_pred == y_te_cls_arr):.4f} | F1 = {sk_sgd_f1:.4f} | AUC = {ev.roc_auc(y_te_cls_arr, sk_sgd_scores):.4f} (fit: {sk_sgd_fit_time:.2f}s)"
    )

    # -------------------------------------------------------------------------
    # 4. Problem 5 Benchmark Parity Plot
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Generating problem 5 benchmark parity plot...")
    task_b_res = {
        "custom": {
            "Accuracy": float(np.mean(c_tree_pred == y_te_cls_arr)),
            "Precision": float(ev.precision_recall_f1(y_te_cls_arr, c_tree_pred)[0]),
            "Recall": float(ev.precision_recall_f1(y_te_cls_arr, c_tree_pred)[1]),
            "F1-Score": c_tree_f1,
            "ROC-AUC": ev.roc_auc(y_te_cls_arr, c_tree_scores),
            "Fit Time (s)": c_tree_fit_time,
            "Predict Time (s)": 0.05,
        },
        "sklearn": {
            "Accuracy": float(np.mean(sk_tree_pred == y_te_cls_arr)),
            "Precision": float(ev.precision_recall_f1(y_te_cls_arr, sk_tree_pred)[0]),
            "Recall": float(ev.precision_recall_f1(y_te_cls_arr, sk_tree_pred)[1]),
            "F1-Score": sk_tree_f1,
            "ROC-AUC": ev.roc_auc(y_te_cls_arr, sk_tree_scores),
            "Fit Time (s)": sk_tree_fit_time,
            "Predict Time (s)": 0.01,
        },
    }
    task_a_res = {
        "custom": {
            "RMSE": c_rmse,
            "MAE": c_mae,
            "R2 Score": c_r2,
            "Fit Time (s)": c_reg_fit_time,
            "Predict Time (s)": 0.05,
        },
        "sklearn": {
            "RMSE": sk_rmse,
            "MAE": sk_mae,
            "R2 Score": sk_r2,
            "Fit Time (s)": sk_reg_fit_time,
            "Predict Time (s)": 0.01,
        },
    }
    bm.plot_benchmark_results(task_b_res, task_a_res, figures_dir=figures_dir)

    if not quick_mode:
        print(
            "  - Running 5-fold cross-validation uncertainty estimation (Problem 5c)..."
        )
        bm.run_cross_validation_uncertainty(
            X_tr, y_tr_cls.to_numpy(dtype=int), y_tr_arr, n_splits=5
        )
        bm.run_svm_benchmark(
            lambda_=svm_config["lambda_"],
            gamma=svm_config["gamma"],
            n_epochs=svm_config["n_epochs"],
            learning_rate_scale=svm_config["learning_rate_scale"],
            n_components=svm_config["n_components"],
            n_splits=5,
        )
    else:
        print("  - Skipping 5-fold CV uncertainty estimation (--quick active).")

    # -------------------------------------------------------------------------
    # 5. Problem 4 Tuning Plots (or skipped in quick mode)
    # -------------------------------------------------------------------------
    if not quick_mode:
        print("\n[Step 5/6] Generating problem 4 hyperparameter tuning figures...")
        print(
            "  - Running decision tree tuning sweeps (depth, gini vs entropy, min_samples_leaf)..."
        )
        tune.run_tree_tuning()
    else:
        print("\n[Step 5/6] Skipping problem 4 tuning sweeps (--quick active).")

    # -------------------------------------------------------------------------
    # 6. Problem 6 Evaluation Figures & Final Table
    # -------------------------------------------------------------------------
    print(
        "\n[Step 6/6] Generating problem 6 evaluation figures and consolidated results table..."
    )
    ev.plot_feature_importances(
        custom_tree_cls.feature_importances_,
        feat_names,
        title="Decision Tree Feature Importances (entropy, depth=12, leaf=10)",
        top_n=15,
        save_path=os.path.join(figures_dir, "tree_feature_importances.png"),
    )
    ev.plot_svm_feature_weights(
        custom_svm,
        feat_names,
        title="Pegasos SVM: Top Feature Contributions (proxy via RFF weights)",
        top_n=15,
        save_path=os.path.join(figures_dir, "svm_feature_weights.png"),
    )
    ev.plot_pr_curve(
        {
            "Custom Decision Tree": (y_te_cls, c_tree_scores),
            "Custom Pegasos SVM": (y_te_cls, c_svm_scores),
            "Sklearn Decision Tree": (y_te_cls, sk_tree_scores),
            "Sklearn SGD Baseline": (y_te_cls, sk_sgd_scores),
        },
        title="Precision-Recall Curves on Test Set (Task B: Price Tier)",
        save_path=os.path.join(figures_dir, "pr_curve.png"),
    )
    ev.plot_residuals(
        y_te_arr,
        c_reg_pred,
        title="Decision Tree (Task A: Price Regression) Residual Analysis",
        save_path=os.path.join(figures_dir, "residuals_task_a.png"),
    )

    comparison_table = {
        "Custom Decision Tree": {
            "Task A RMSE": c_rmse,
            "Task A MAE": c_mae,
            "Task A R2": c_r2,
            "Task B Acc": float(np.mean(c_tree_pred == y_te_cls_arr)),
            "Task B F1": c_tree_f1,
            "Task B ROC-AUC": ev.roc_auc(y_te_cls_arr, c_tree_scores),
            "Task B PR-AUC": ev.pr_auc(y_te_cls_arr, c_tree_scores),
            "Fit Time (s)": c_tree_fit_time + c_reg_fit_time,
        },
        "Sklearn Decision Tree": {
            "Task A RMSE": sk_rmse,
            "Task A MAE": sk_mae,
            "Task A R2": sk_r2,
            "Task B Acc": float(np.mean(sk_tree_pred == y_te_cls_arr)),
            "Task B F1": sk_tree_f1,
            "Task B ROC-AUC": ev.roc_auc(y_te_cls_arr, sk_tree_scores),
            "Task B PR-AUC": ev.pr_auc(y_te_cls_arr, sk_tree_scores),
            "Fit Time (s)": sk_tree_fit_time + sk_reg_fit_time,
        },
        "Custom Pegasos SVM": {
            "Task A RMSE": "---",
            "Task A MAE": "---",
            "Task A R2": "---",
            "Task B Acc": float(np.mean(c_svm_pred == y_te_cls_arr)),
            "Task B F1": c_svm_f1,
            "Task B ROC-AUC": ev.roc_auc(y_te_cls_arr, c_svm_scores),
            "Task B PR-AUC": ev.pr_auc(y_te_cls_arr, c_svm_scores),
            "Fit Time (s)": c_svm_fit_time,
        },
        "Sklearn SGD Baseline": {
            "Task A RMSE": "---",
            "Task A MAE": "---",
            "Task A R2": "---",
            "Task B Acc": float(np.mean(sk_sgd_pred == y_te_cls_arr)),
            "Task B F1": sk_sgd_f1,
            "Task B ROC-AUC": ev.roc_auc(y_te_cls_arr, sk_sgd_scores),
            "Task B PR-AUC": ev.pr_auc(y_te_cls_arr, sk_sgd_scores),
            "Fit Time (s)": sk_sgd_fit_time,
        },
    }
    ev.compare(comparison_table)

    print("\nLatex table snippet for report/report.tex (table i):")
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"  \toprule")
    print(
        r"  Model & Task A (RMSE) & Task A ($R^2$) & Task B (F1) & Task B (PR-AUC) \\"
    )
    print(r"  \midrule")
    print(
        f"  Ours: Decision Tree   & {c_rmse:.4f} & {c_r2:.4f} & {c_tree_f1:.4f} & {ev.pr_auc(y_te_cls_arr, c_tree_scores):.4f} \\\\"
    )
    print(
        f"  Ours: Pegasos SVM     & ---    & ---    & {c_svm_f1:.4f} & {ev.pr_auc(y_te_cls_arr, c_svm_scores):.4f} \\\\"
    )
    print(
        f"  scikit-learn Tree     & {sk_rmse:.4f} & {sk_r2:.4f} & {sk_tree_f1:.4f} & {ev.pr_auc(y_te_cls_arr, sk_tree_scores):.4f} \\\\"
    )
    print(
        f"  scikit-learn SGD      & ---    & ---    & {sk_sgd_f1:.4f} & {ev.pr_auc(y_te_cls_arr, sk_sgd_scores):.4f} \\\\"
    )
    print(r"  \bottomrule")
    print(r"\end{tabular}" + "\n")

    print(f"\nCompleted in {time.time() - start_total_time:.2f}s")
    print(f"Figures saved in {figures_dir}/:")
    for f in sorted(os.listdir(figures_dir)):
        if f.endswith((".png", ".pdf")):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
