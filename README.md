# MLE-AI-201 Machine Learning — Final Project (Starter Pack)

Build **two models from scratch** (a decision tree and an SVM) on the
**bina.az** real-estate dataset, then benchmark honestly against
scikit-learn. Read the full brief in the project statement PDF handed out
with this pack.

- **Released:** Sun 04 Oct 2026
- **Due:** **Sun 18 Oct 2026, 23:59 (Baku)** — two weeks, no late submissions.
- **Demo:** defended in the session after the deadline.

## What you submit (on Moodle)

Exactly **three** things; **one member submits** for the whole team:

1. **GitHub link** to your repo (tagged `v1.0-final`).
2. **Report PDF** (IEEE two-column — use `report/report.tex`).
3. **Slides PDF**.

Keep `contribution_report.md` in the repo root (checked at the demo).

## The two tasks

- **Task A — Regression:** predict `price` (consider `log(price)`).
- **Task B — Classification:** predict a derived **price tier**
  (premium vs standard, median split on the training set).

## Golden rule

Your **decision tree** and your **SVM** must be **your own NumPy code**.
scikit-learn is for the comparison baselines (and the optional bonuses)
only. For the SVM you do **not** need a QP/SMO solver — use the
**Pegasos** sub-gradient method (see the statement and `src/svm.py`).

## Repo layout

```
.
├── README.md               # this file
├── requirements.txt        # pin your versions
├── .gitignore
├── contribution_report.md  # who did what (keep in repo root)
├── data/                   # dataset lives here (not committed — see data/README.md)
├── src/                    # YOUR from-scratch code
│   ├── data_prep.py        #   loading, cleaning, split, tier label
│   ├── decision_tree.py    #   DecisionTree (Gini/entropy/MSE)
│   ├── svm.py              #   PegasosSVM (hinge-loss sub-gradient)
│   ├── evaluate.py         #   metrics + comparison helpers
│   └── run_all.py          #   ONE command reproduces every number/figure
├── report/                 # IEEE report (self-contained, no IEEEtran.cls)
│   ├── report.tex
│   └── figures/
└── presentation/           # your slides PDF
```

## Reproducibility (graded)

- Pin dependencies in `requirements.txt`; fix all random seeds.
- `python -m src.run_all` must reproduce **every** headline number and
  figure from a clean checkout.
- Do **not** commit the raw data archive (see `.gitignore`).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Download dataset from Kaggle and place in data/ (see below), then:
python -m src.run_all
```

## Dataset (download required - not in repo)

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`).

**Where to get it:** Kaggle dataset "binaaz-sale-project" by sehriyarmemmedli
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

**How to place it:**

1. Download the dataset archive from Kaggle
2. Extract it so the CSV lives directly in the `data/` folder as `data/house_sale.csv`
3. The code expects the file at `data/house_sale.csv` (set in `src/data_prep.py`)

**Important:** The dataset uses Azerbaijani column names (UTF-8 encoding). Any column derived from `price` (e.g., unit price) must be dropped to avoid leakage.
