# Mobile Money Fraud Detection

This project implements the Deliverable 4 modelling plan for ITDPA3-34:
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
estimators and imbalance samplers are in `models.py`; the training workflow is
in `training.py`; metric calculations are in `evaluation.py`; and
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

The default local run evaluates all five models against all three imbalance
strategies on a deterministic, class-stratified sample of 200,000 rows:

```powershell
python -m fraud_detection.cli train
```

To use the complete 6.36 million-row dataset, explicitly set `--max-rows 0`
on a machine with sufficient memory:

```powershell
python -m fraud_detection.cli train --max-rows 0
```

All 15 pairings receive the same sampled rows, chronological partitions, and
fixed baseline configurations. Partitions are made by complete `step` groups:
the earliest 60% of distinct hourly steps for fitting, the next 20% of steps
for threshold selection, and the latest 20% of steps for final testing.
Transaction volume varies substantially by hour, so row counts need not match
these time percentages. Resampling is fit only inside the training
pipeline. No test metric is used to select the winning pairing. Each validation
threshold is chosen to maximize precision subject to at least 90% recall; the
winner is selected by validation cost, using 10 per false negative and 1 per
false positive. Those policy values are configurable. The 90% recall floor is
this project's operating target, not a universal industry threshold; review it
against measured review capacity and fraud-loss costs. Per-pair fit time and
test throughput are reported, with the PR-AUC/throughput frontier computed on
the final period for comparison only.

The 200,000-row local cap is a conservative working estimate for this 7.7 GiB
machine, based on the previously generated 100,000-row 15-pair run; it is not a
guarantee if other applications consume memory. Close memory-heavy applications
before running. At least 30 fraud cases are required in both validation and test
windows; the pipeline stops if the sampled timeline cannot meet that minimum.
Full-data SMOTE can synthesize millions of rows, so `--max-rows 0` is intended
for a higher-memory machine. The sample remains a pilot, not a substitute for
full-data conclusions.

The sample is selected reproducibly and stratified by label; temporal
boundaries are applied afterward by `step`, keeping each hour in exactly one
partition. The full input is validated even when modeling a sample. Models use
the same fixed baseline settings, avoiding an asymmetric search budget.

The workflow trains Logistic Regression, Random Forest, XGBoost, a
scikit-learn feedforward neural network, and LightGBM. Numeric inputs are
scaled for Logistic Regression and the neural network; transaction type is
one-hot encoded for every estimator. Missing categorical/numeric values have
mode/median imputers in the preprocessing pipeline, although source validation
fails on missing values in required columns. Features are limited to information
available before transaction completion: transaction type, amount, pre-event
balances, and time-of-day derived from `step`. Absolute step, post-event
balances, identifiers, and `isFlaggedFraud` are excluded from model inputs.

### 3. Results and generated files

Training writes:

- `reports/model_comparison.csv` with precision, recall, F1, ROC-AUC, PR-AUC,
  Brier score, validation threshold metrics, weighted false-positive/negative
  cost, hourly-bootstrap confidence intervals for precision/recall/PR-AUC,
  fit/prediction duration, scoring
  throughput, and Pareto status for each pairing.
- `reports/imbalance_strategy_comparison.csv` with average model performance
  and runtime for each imbalance strategy.
- `reports/probability_bands.csv` with the number of transactions and observed
  fraud rate in each model's low/medium/high probability band.
- `reports/data_quality.json` and `reports/training_summary.json`.
- A six-metric grouped performance dashboard, two-panel Pareto dashboard,
  algorithm-by-treatment PR-AUC heatmap, high-score observed-risk map, and class
  distribution in `reports/figures/`. Obsolete generated PNGs are removed after
  a successful run so the figures directory represents only the latest run.
- Individual fitted estimators and `models/best_model.joblib`, selected for
  application use, plus `models/best_model.json`.

The included comparison files and deployment model predate this methodology
update. Rerun training before using them. Check `reports/training_summary.json`
for the sampled row count, time boundaries, threshold policy, and fraud counts.

### 4. Run the application

From the repository root, after running training:

```powershell
streamlit run app/streamlit_app.py
```

The app includes a project overview, an interactive transaction form, the
model-comparison table, and generated visualisations. The form requests only
pre-transaction information and applies the selected model's validation-derived
threshold.
It loads the exported model and does not retrain at launch. The
`examples/sample_transactions.csv` file contains sample input values.

## Engineered features

| Feature | Definition |
| --- | --- |
| `hour` | `step % 24` |
| `log_amount` | `log1p(amount)` |
| `transaction_to_origin_balance` | Amount divided by originator's starting balance plus one |

Post-event balance changes and simulator fraud flags are excluded because they
may reveal transaction outcomes unavailable when a live decision is made.
Customer identifiers and the target are excluded from predictive features.

## Evaluation and interpretation

Precision measures the share of predicted fraud cases that are labelled fraud;
recall measures the share of fraud cases detected; F1 balances precision and
recall; ROC-AUC measures ranking across false-positive rates; and PR-AUC
summarises precision-recall performance and is the primary selection metric
for the rare fraud class. Brier score is included as a probability scoring
metric (lower is better), and calibration diagrams compare predicted scores
with observed fraud rates. Thresholds are selected separately for each pairing
on the validation period to meet the configured recall floor while maximizing
precision. The final temporal test window is not used for threshold or model
selection.

The cost metric is `10 × false negatives + 1 × false positives` by default.
These are explicit project policy assumptions, not universal industry cost
ratios; replace them with measured operational costs when available. Precision
and recall intervals use a bootstrap that resamples whole hourly `step` groups
to account for within-hour dependence.

The reports group scores as Low (<1%), Medium (1% to <5%), and High (>=5%).
These are configurable starting bands for triage, not research-established
fraud cutoffs or validated operational actions; change them with
`--low-risk-upper-bound` and `--high-risk-lower-bound` when a documented policy
requires different limits. The reviewed guidance does not define universal
low/medium/high fraud probability boundaries. It instead recommends choosing
classification thresholds for the intended use and checking whether predicted
probabilities are calibrated. SMOTE-based training can particularly affect raw
probability interpretation, so consult the per-model calibration plots and
observed fraud rates before treating a score as an actual likelihood.

Research and technical guidance:

- [scikit-learn: Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

This is an academic demonstration using synthetic PaySim data. Model scores
are not a production payment decision and should not be treated as a substitute
for operational review.

## Tests

```powershell
pytest
```
