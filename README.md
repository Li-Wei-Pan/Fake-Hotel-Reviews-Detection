# Fake-Hotel-Reviews-Detection

# Detection of AI-Generated Hotel Reviews

Code for our Data Mining assignment: classifying English hotel reviews as human-written
(Booking.com, label `0`) or AI-generated (GPT-4, label `1`) with five classifiers:
multinomial naive Bayes, lasso logistic regression, a single classification tree,
random forest, and gradient boosting.

## Requirements

- Python 3.12 (the exact version is pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/) for installing the environment

All package versions are pinned in `pyproject.toml` and `uv.lock`. A `requirements.txt`
is also included for use with plain `pip`.

Main libraries: `numpy`, `pandas`, `scikit-learn`, `nltk`, `regex`, `scipy`,
`matplotlib`, `joblib`.

## Setup

**With uv (recommended)**, from the project root:

```bash
uv sync
```

This creates a virtual environment with the exact package versions we used.

**With pip**, as an alternative:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The NLTK English stop-word list is downloaded automatically the first time a script runs.

## Data

Place `all_data.csv` from
[MichiganNLP/multilingual_reviews_deception](https://github.com/MichiganNLP/multilingual_reviews_deception)
in the `data/` folder:

```
data/all_data.csv
```

## How to run

Run the scripts **in this order** from the project root. If you installed with pip,
replace `uv run python` with `python`.

```bash
# 1. Preprocessing: filter English reviews, clean text, stratified 75/25 split
uv run python preprocessing.py

# 2. Models: tune with 5-fold stratified CV, refit on the training set, evaluate on the test set
uv run python multinomial_nb.py
uv run python logistic_regression.py
uv run python classification_tree.py
uv run python random_forest.py
uv run python gradient_boosting.py

# 3. Statistical tests (needs the predictions saved in step 2)
uv run python mcnemar_test.py
```

The preprocessing step must run first, because all model scripts read `train.csv` and
`test.csv`. The statistical tests must run last, because they read the saved predictions.

## What each script produces

| Script | Output | Used in the report |
|---|---|---|
| `preprocessing.py` | `train.csv`, `test.csv`, review-length and vocabulary figures | Data and Preprocessing sections, Figures 1–2 |
| `multinomial_nb.py` | test metrics, top-5 features per class, `models/mnb.joblib`, `models/mnb_cv_results.csv`, `data/y_pred_mnb.csv` | Model performance table, Q1, Q4 |
| `logistic_regression.py` | test metrics, top-5 features per class, `models/lg.joblib`, `models/lr_cv_results.csv`, `data/y_pred_lg.csv` | Model performance table, Q1, Q4 |
| `classification_tree.py` | test metrics, top-5 features, `models/ct.joblib`, `models/ct_cv_results.csv`, `data/y_pred_ct.csv` | Model performance table, Q2, Q4 |
| `random_forest.py` | test metrics, top-5 features, accuracy on positive vs. negative reviews, `data/y_pred_rf.csv` | Model performance table, Q2, Q3, Q4 |
| `gradient_boosting.py` | test metrics, top-5 features, `data/y_pred_gb.csv` | Model performance table, Q2, Q4 |
| `mcnemar_test.py` | McNemar 2×2 tables and p-values | Q1 and Q2 statistical tests |

The cross-validation results (`models/*_cv_results.csv`) contain the full
hyperparameter grids and are reported in the appendix.

## Reproducibility

- All randomness is fixed with `random_state=42`: the train/test split, the CV folds,
  and every model.
- The test set is used only once per model, for the final evaluation.
- Running the scripts in the order above reproduces all numbers in the report.

## Project structure

```
.
├── data/
│   └── all_data.csv          # input data (not included, see "Data")
├── models/                   # saved models and CV results (created by the scripts)
├── preprocessing.py
├── multinomial_nb.py
├── logistic_regression.py
├── classification_tree.py
├── random_forest.py
├── gradient_boosting.py
├── mcnemar_test.py
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```