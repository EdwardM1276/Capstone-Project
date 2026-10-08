# Mobile Money Fraud Detection

This project implements the Deliverable 3 modelling plan for ITDPA3-34:
classifying fraudulent mobile-money transactions in the PaySim synthetic
dataset. It provides reusable preprocessing and feature engineering, an
automated model-comparison workflow, saved estimators, evaluation reports,
and a Streamlit application for demonstration.

## Project structure

```text
app/                     Streamlit application
data/raw/                Local PaySim CSV (not committed)
data/processed/          Optional generated data
examples/                Example transaction inputs
models/                  Trained estimators and deployment model
notebooks/               Optional exploratory notebooks
reports/                 Data checks, metrics, and figures
src/fraud_detection/     Reusable data, feature, model, and evaluation code
tests/                   Unit tests for data checks and features
```

Data validation and loading are in `data.py`; feature generation is in
`features.py`; reusable encoders and scalers are in `preprocessing.py`;
estimators and imbalance samplers are in `models.py`; the training and search
workflow is in `training.py`; metric calculations are in `evaluation.py`; and
model loading and transaction scoring are in `prediction.py`.

## Requirements and installation

Use Python 3.10 or later. From the repository root, create an environment and
install the package and its dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The project uses pandas and NumPy for data handling, scikit-learn and
imbalanced-learn for modelling and resampling, XGBoost and LightGBM for
boosted-tree models, Matplotlib and Seaborn for figures, and Streamlit for
the application.

## Dataset

Place the PaySim CSV at `data/raw/paysim.csv`. The expected input columns are:

```text
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,
oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
```

The supplied CSV is about 493 MB and is excluded from Git to keep the
repository within normal hosting limits. The pipeline reads the dataset from
the path above, validates the complete file in chunks, and does not use the
sender or recipient identifiers as model inputs. If the file has a different
name or location, pass it with `--input`.

## Reproducible workflow

### 1. Validate and preprocess

```powershell
python -m fraud_detection.cli preprocess
```

This validates all rows before modelling and writes `reports/data_quality.json`
and `reports/figures/class_distribution.png`. The checks cover the required
schema, missing values, numeric conversions, non-finite numeric values,
negative amounts and balances, invalid target labels, negative time steps,
transaction types, and duplicate records. A failed validation stops the
workflow and reports the counts; records are not silently discarded or
modified.

### 2. Train and compare models

For a quick, deterministic demonstration run on a stratified sample of 100,000
rows, including all three imbalance strategies:

```powershell
python -m fraud_detection.cli train --max-rows 100000 --imbalance-strategies class_weight smote smote_tomek --tune-rows 20000 --search-iterations 2
```

To train on the complete dataset, omit `--max-rows` (or set it to `0`):

```powershell
python -m fraud_detection.cli train --imbalance-strategies class_weight smote smote_tomek
```

The default training configuration compares all five models using class
weighting. Add the three strategies shown above to compare class weighting,
SMOTE, and SMOTE with Tomek Links. Resampling occurs inside each training fold
and only after the stratified train/test split. Full-data SMOTE runs need
substantially more memory and time than a sample run; use a stratified sample
for a machine with limited resources.

The split is 80% training and 20% holdout testing, stratified by `isFraud`,
with random seed 42. A reproducible, stratified sample is selected from the
full input when `--max-rows` is set; the quality report still covers every
source row. RandomizedSearchCV tunes Random Forest and XGBoost using
three-fold stratified cross-validation and average precision. Set
`--tune-rows` and `--search-iterations` to adjust the tuning workload.

The workflow trains Logistic Regression, Random Forest, XGBoost, a
scikit-learn feedforward neural network, and LightGBM. Numeric inputs are
scaled for Logistic Regression and the neural network; transaction type is
one-hot encoded for every estimator. Missing categorical/numeric values have
mode/median imputers in the preprocessing pipeline, although source validation
fails on missing values in required columns. The final model is selected by
holdout PR-AUC, with recall and F1 used to break ties; accuracy is not used as
the main success criterion.

### 3. Results and generated files

Training writes:

- `reports/model_comparison.csv` with precision, recall, F1, ROC-AUC, PR-AUC,
  threshold, and test-set size for each model and imbalance strategy.
- `reports/data_quality.json` and `reports/training_summary.json`.
- Confusion matrices, ROC curves, precision-recall curves, class distribution,
  and supported model feature-importance figures in `reports/figures/`.
- Individual fitted estimators and `models/best_model.joblib`, selected for
  application use, plus `models/best_model.json`.

The included comparison files and deployment model are from the 100,000-row
stratified demonstration command above. Its 20,000-row holdout contains 26 fraud
cases, so its scores demonstrate the workflow and must not be treated as a
stable estimate of performance on the full dataset. Rerun training without
`--max-rows` for full-dataset evaluation before drawing conclusions.

### 4. Run the application

From the repository root, after running training:

```powershell
streamlit run app/streamlit_app.py
```

The app includes a project overview, an interactive transaction form, the
model-comparison table, and generated visualisations. The form displays the
predicted class and the saved model's fraud probability at a 0.50 threshold.
It loads the exported model and does not retrain at launch. The
`examples/sample_transactions.csv` file contains sample input values.

## Engineered features

| Feature | Definition |
| --- | --- |
| `origin_balance_change` | Originator balance before minus balance after the transaction |
| `destination_balance_change` | Destination balance after minus balance before the transaction |
| `hour` | `step % 24` |
| `log_amount` | `log1p(amount)` |
| `origin_balance_error` | Originator balance before minus transaction amount minus balance after |
| `destination_balance_error` | Destination balance after minus balance before minus transaction amount |
| `transaction_to_origin_balance` | Amount divided by originator's starting balance plus one |
| `account_drained` | One when a positive originator balance falls to zero; otherwise zero |

The balance-error features retain accounting discrepancies as potential
predictive signals. Customer identifiers and the target are excluded from
predictive features.

## Evaluation and interpretation

Precision measures the share of predicted fraud cases that are labelled fraud;
recall measures the share of fraud cases detected; F1 balances precision and
recall; ROC-AUC measures ranking across false-positive rates; and PR-AUC
summarises precision-recall performance and is the primary selection metric
for the rare fraud class. The confusion matrix and threshold-based class
predictions use a fixed 0.50 probability threshold for consistent comparisons.

This is an academic demonstration using synthetic PaySim data. Model scores
are not a production payment decision and should not be treated as a substitute
for operational review.

## Tests

```powershell
pytest
```
