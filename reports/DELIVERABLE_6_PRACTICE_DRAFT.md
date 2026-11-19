# Deliverable 6: Conclusion and Recommendations


**Project:** Mobile Money Fraud Detection with PaySim
**Module / group:** ITDPA3-34 / Group 3
**Evidence version used:** repository evidence run `a0c5812936ac` (recorded 9 October 2026)
**Prepared:** 18 October 2026

> This practice draft synthesizes the available planning document, implementation report, saved evaluation artifacts, and repository. The figures below are transcribed from the named artifacts. In particular, Deliverable 4 describes a different earlier run; results from that run are not combined with the saved Deliverable 5 baseline. Group members should verify all facts and adapt any reflections before using this material for any purpose.

## 1. Project Summary — 10 marks

### 1.1 Scope and objectives

This project investigated whether supervised machine-learning models could help identify potentially fraudulent mobile-money transactions using transaction information in PaySim. The original problem was to distinguish fraudulent from legitimate transactions despite fraud being rare, while balancing the cost of missed fraud against the cost and customer impact of false alerts. The proposed use was a reproducible model comparison and a transaction-review aid, not an autonomous payment-blocking system.

The main objectives carried through the planning and implementation documents were to:

1. prepare and examine the PaySim transaction data;
2. engineer usable transaction features while avoiding information unavailable at the intended scoring point;
3. compare classification models and ways of handling class imbalance;
4. evaluate fraud detection and false-alarm performance using metrics suitable for a rare positive class;
5. communicate results and demonstrate scoring through an interactive application; and
6. keep the modelling workflow reproducible and examine model explanations.

The project met these objectives to the extent of an academic prototype and saved, sampled evaluation. It did not establish that a model is suitable for live customer transactions.

### 1.2 Work completed

**Data collection and understanding.** The project used the PaySim synthetic mobile-money dataset. The recorded data-quality report describes 6,362,620 rows and 11 expected columns, with 8,213 fraud cases (about 0.129%). Its validation counts report zero missing values, invalid numeric values, negative amounts or balances, invalid labels or types, and duplicate records. These checks are reported as passing for that data snapshot. The model run did not fit on all source rows.

**Preprocessing and feature preparation.** For run `a0c5812936ac`, a reproducible, label-stratified sample of 200,000 rows was modelled. The recorded split has 187,968 training rows, 6,058 validation rows, and 5,974 test rows; the windows are ordered by PaySim step. The feature pipeline uses transaction type, amount, originator starting balance, recipient starting balance, hour derived from `step`, `log_amount`, and `transaction_to_origin_balance`. It one-hot encodes transaction type and scales numeric variables for Logistic Regression and the feedforward neural network. Sender and recipient identifiers, absolute step as a predictor, post-transaction balances, and `isFlaggedFraud` are excluded from the intended predictor set. Resampling is kept in the training pipeline rather than applied to held-out data.

**Modelling.** The saved comparison covers five algorithms—Logistic Regression, Random Forest, XGBoost, a feedforward neural network, and LightGBM—each paired with class weighting, SMOTE, and SMOTE with Tomek links. This produces 15 recorded model-treatment comparisons. The project selects an operating threshold on the validation period to satisfy a configured recall floor of 90% while maximizing precision, then selects the pairing using validation weighted cost. The recorded policy assigns cost 10 to a false negative and cost 1 to a false positive. These are project assumptions, not measured bank costs.

**Evaluation.** The saved baseline reserves the latest chronological window for final evaluation and reports precision, recall, F1, ROC-AUC, PR-AUC, Brier score, error counts, cost, throughput, and bootstrap confidence intervals. It also records a model comparison, imbalance-strategy comparison, probability bands, a confusion matrix, and visual summaries. The evidence run used 60 fraud cases in its 5,974-row final window.

**Application and communication.** The repository contains a Streamlit application for overview, scoring, model comparison, visual analysis, and explainability, alongside sample transactions and generated figures. This demonstrates a usable academic interface in code. A saved screenshot, usability study, stakeholder feedback, or independently verified live deployment is not established by the artifacts reviewed for this draft.

### 1.3 Accomplishments and evidence

| Accomplishment                                                         | Evidence                                                                                       | Significance and qualification                                                                                                                                                                               |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Full-file data checks were recorded for 6,362,620 rows                 | `evidence_deliverable_5/data_quality.json`                                                   | Provides evidence of basic schema and value validation for the recorded source snapshot; it does not establish that PaySim represents live payments.                                                         |
| A repeatable 15-pair model comparison was recorded                     | `evidence_deliverable_5/model_comparison.csv` and `training_summary.json`                  | Enables comparison across algorithms and imbalance treatments under the same saved run; results are based on a 200,000-row sample.                                                                           |
| A threshold-selected XGBoost/class-weight operating point was recorded | `training_summary.json`; `confusion_matrix_counts.json`                                    | The reported test point detected 56 of 60 fraud cases with four false alarms; the limited number of positive test cases makes uncertainty material.                                                          |
| A runnable scoring and reporting interface is present                  | `app/streamlit_app.py`; `examples/sample_transactions.csv`                                 | Shows an integration path for analyst review; code availability alone is not evidence of operational validation or adoption.                                                                                 |
| Automated testing and explanation artifacts are present                | `evidence_deliverable_5/pytest_results.txt`; `interpretability_summary.json`; `figures/` | The saved test log records 20 passed tests; it is a historical log from a different machine/path, not a test run performed for this practice draft. Interpretability evidence is limited as discussed below. |

No stakeholder feedback artifact was identified in the reviewed repository materials. This draft therefore makes no claim that users tested or approved the interface.

### 1.4 Planned versus actual outcomes

| Plan in Deliverable 3 / earlier documents                                                                                                       | Recorded implementation                                                                                                                                        | Difference and treatment                                                                                                                                                                                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Compare Logistic Regression, Random Forest, XGBoost, and a feedforward neural network; other Deliverable 3 sections also mention Decision Tree. | The saved baseline compares Logistic Regression, Random Forest, XGBoost, feedforward neural network, and LightGBM; it does not include Decision Tree.          | LightGBM appears in the implementation but was not consistently named in the plan; Decision Tree was not in the saved comparison. The final conclusion should describe the actual five-model experiment, not claim all proposed algorithms were tested. |
| Examine SMOTE, NearMiss, and SMOTE with Tomek links.                                                                                            | The saved comparison uses class weighting, SMOTE, and SMOTE with Tomek links.                                                                                  | NearMiss was replaced by class weighting. The comparison therefore does not establish how NearMiss would perform.                                                                                                                                       |
| Explore balance changes, time, outliers, and other transaction patterns; proposed features include pre/post-balance changes.                    | The saved predictor set uses pre-transaction balances and derived hour, log amount, and transaction-to-origin-balance ratio; post-event balances are excluded. | The implemented feature set is narrower and avoids post-outcome information. The original proposed balance-change features were not implemented as predictors in this baseline.                                                                         |
| Consider stratified cross-validation and broader hyperparameter search.                                                                         | The recorded evaluation uses chronological train, validation, and test periods; the Deliverable 4 report describes a bounded candidate search.                 | Temporal holdout better mirrors prediction into a later period than random folds, but the evidence is one saved sampled run, not repeated temporal validation or exhaustive tuning.                                                                     |
| Deliver a comparative model and demonstrate its intended use.                                                                                   | A Streamlit application and example input data are present in the repository.                                                                                  | This fulfils a prototype demonstration objective; the artifacts do not establish production integration, a controlled pilot, or documented stakeholder feedback.                                                                                        |

## 2. Evaluation of Project Outcomes — 10 marks

### 2.1 Achievement against objectives

The data preparation, multi-model comparison, threshold selection, evaluation reporting, and dashboard objectives were substantially demonstrated in the code and saved artifacts. Leakage safeguards are described in the repository README and covered by tests. However, the conclusion must be bounded: model fitting used a stratified 200,000-row sample, not all 6.36 million source records; the test contains only 60 fraud cases; the data are synthetic; and no independent live dataset or controlled operational pilot is evidenced.

The saved run reports chronological boundaries of steps 1–407 for training, 408–563 for validation, and 564–743 for testing. The model threshold and winner were selected using validation data, while the last interval was reserved for reporting. That is a meaningful safeguard against using the final test results for selection. It is not equivalent to demonstrating stable performance over many future periods, organizations, or fraud patterns.

### 2.2 Performance and error analysis

The saved summary identifies **XGBoost with class weighting** as the selected pairing under the project’s validation-cost policy. Its saved threshold is approximately **0.06693**. On the final window, the confusion-matrix artifact records:

| Measure                          | Saved result |
| -------------------------------- | -----------: |
| Test transactions                |        5,974 |
| Fraud cases                      |           60 |
| True negatives                   |        5,910 |
| False positives                  |            4 |
| False negatives                  |            4 |
| True positives                   |           56 |
| Precision                        |        93.3% |
| Recall                           |        93.3% |
| F1 score                         |        93.3% |
| ROC-AUC                          |       0.9999 |
| PR-AUC                           |       0.9862 |
| Brier score                      |      0.00155 |
| Weighted test error cost at 10:1 |           44 |

The precision and recall values are strong for this saved test sample: most flagged transactions were fraud and 56 of the 60 test fraud cases were detected. The four false negatives still represent missed fraud in the sample; four false positives represent legitimate transactions sent for review. A false-negative cost of 10 and false-positive cost of 1 makes this run’s weighted cost 44, but these weights encode a project policy, not measured monetary impact or a validated operational preference.

Uncertainty matters. The saved CSV reports hourly-bootstrap 95% intervals for the selected pairing of approximately **85.1%–100% precision**, **86.1%–98.6% recall**, and **96.3%–99.9% PR-AUC**. The test’s 60 fraud cases are a small basis for claims about future fraud detection. Results should therefore be described as promising evidence from the saved benchmark, not as a guaranteed detection rate.

The selection criterion also matters. The XGBoost/class-weight pairing had validation weighted cost 43 and validation recall about 92.3%, meeting the 90% recall target. Some other pairings had lower cost on this particular final test window—for example, Random Forest with SMOTE reports zero false positives and two false negatives in the saved comparison—but its validation cost was 50. Choosing a model based on which happened to look best on the final test would leak test information into selection. The recorded selection method appropriately relies on validation rather than retrospectively choosing the test winner.

Across treatments, the saved strategy summary reports mean test PR-AUC of approximately 0.880 for class weighting and 0.890 for each SMOTE treatment across the five algorithms. Mean recall was highest for class weighting (0.887), while mean test weighted cost was lowest for SMOTE with Tomek links (159) in the aggregate summary. These averages do not mean that one treatment is best for every algorithm or policy. The final selection depends on the stated validation objective.

Probability bands are exploratory groupings, not approved review policy. For the selected XGBoost/class-weight run, the recorded bands contain 5,908 transactions below 1% with observed fraud rate 0%, six transactions from 1% to below 5% with observed fraud rate 66.7%, and 60 transactions at or above 5% with observed fraud rate 93.3%. The medium band’s rate is based on only six transactions and is unstable. A model score should not be interpreted as a calibrated real-world probability without additional calibration assessment on representative data.

### 2.3 Interpretability, fairness, and limitations

The interpretability summary reports a SHAP sample size of 60. The highest mean absolute attributions in that sample were `transaction_to_origin_balance` (about 6.16), `amount` (2.76), `type_TRANSFER` (2.14), and `hour` (1.73). These values suggest which transformed inputs influenced model output in the analyzed sample; they do not establish causality, universal fraud rules, or stable importance across other data. The summary’s local example is labeled legitimate (`true_label` 0) but has predicted fraud probability about 0.928, far above the saved threshold of about 0.067. This illustrates a high-score legitimate case worthy of error review, though the available summary alone does not establish how it maps to a particular test row. A SHAP summary and a LIME example are aids for investigation, not explanations that prove a decision is correct.

No demographic or other appropriate group-level fairness evaluation is recorded. PaySim is synthetic and the saved artifacts do not provide evidence of protected-group outcomes. The project therefore cannot claim fairness or rule out disparate impact. In a real deployment, lawful and privacy-conscious subgroup evaluation would need an approved data basis, suitable group definitions, statistical review, and human accountability.

The dataset’s synthetic nature, the single sampled temporal evaluation, the small fraud count in the test window, and the absence of live-system integration limit real-world conclusions. The test results do not measure customer harm, investigator capacity, prevented financial loss, or business return. Similarly, reported prediction throughput on a local benchmark is not a production service-level guarantee; it excludes the wider payment integration, network, queue, logging, and analyst workflow unless separately measured.

### 2.4 Dashboard and visual evidence

The saved performance dashboard communicates differences between algorithm-treatment pairs, and the heatmap shows high observed fraud rates in some tree-model probability bands. These are useful comparative summaries, but some bands contain very few transactions; the accompanying probability-band table should be shown so viewers can assess denominators. The Pareto figure has crowded, overlapping point labels, making several model-treatment points difficult to distinguish. The performance figure title refers to Deliverable 4, although it is included with the later saved baseline. The LIME local plot also has dense feature text and is not stakeholder-ready in its current presentation. The dashboard code provides interactive tables and charts, but the static-figure issues should be addressed before a briefing. Redesigns should preserve the values and identify the run, evaluation window, and selection rule clearly.

### 2.5 Reconciliation of prior results

The Deliverable 4 implementation report describes an earlier run, `4b0b05a1c9a6`, with Random Forest plus SMOTE, threshold approximately 0.795, and 58 detected fraud cases out of 60 with two false negatives and no false positives. The later repository summary and comparison artifacts identify run `a0c5812936ac`, with XGBoost plus class weighting, threshold approximately 0.06693, and 56 detected cases out of 60 with four false positives and four false negatives. These are different runs, selection outcomes, and thresholds; their results must not be combined into one confusion matrix or presented as if they came from the same model snapshot. This practice draft uses `a0c5812936ac` as its primary quantitative baseline because that is the run identified by the checked-in evidence files and README. The earlier Deliverable 4 result is retained only as a version discrepancy that a project owner should reconcile against its original commit and model files.

The README further states that the checked-in baseline predates the current training methodology. Accordingly, these metrics characterize the recorded historical baseline, not necessarily results reproducible by running the current training code without a new experiment.

### 2.6 Real-world contribution

The work contributes an end-to-end demonstration: data checks, leakage-conscious features, temporal model evaluation, an explicit validation threshold policy, comparative reports, model explanations, tests, and a transaction-scoring interface. For an analyst, a future version could help prioritize a queue for human review. It does not yet establish that the model should block, delay, or approve payments, nor that it reduces losses without unacceptable customer friction. Those claims require representative real data, operational cost evidence, fairness and calibration analysis, and an approved monitored pilot.

## 3. Lessons Learned — 10 marks

| Challenge or finding                                                                                                 | Evidence-based response in the project                                                                                                                                              | Lesson for subsequent work                                                                                                                                                           |
| -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Fraud is extremely rare: 8,213 of 6,362,620 source records (about 0.129%).                                           | The implementation compares class weighting and two resampling approaches and reports precision, recall, PR-AUC, costs, and confusion counts rather than relying on accuracy alone. | For rare-event tasks, the majority-class accuracy can conceal failure to find positives. The team should define a review-capacity and error-cost policy before choosing a threshold. |
| Random splitting could allow future transaction patterns to inform evaluation of earlier ones.                       | The saved run splits complete time steps chronologically and selects thresholds using a validation window.                                                                          | Evaluation design should reflect intended use. A single chronological holdout is a useful baseline, but repeated rolling-origin windows are needed to assess temporal stability.     |
| Features observed after a transaction or simulator flags may expose outcome information unavailable at scoring time. | The documented feature policy excludes post-event balances, IDs, `isFlaggedFraud`, and absolute `step` as a predictor; a leakage-audit test file is present.                    | Feature availability should be specified at the decision point and tested automatically, not judged only by predictive performance.                                                  |
| Deliverable 3, Deliverable 4, and the saved baseline do not describe one perfectly consistent experiment.            | The saved evidence uses XGBoost/class weighting, while Deliverable 4 describes an earlier Random Forest/SMOTE result; candidate model and imbalance plans also changed.             | Run IDs, source commits, model files, thresholds, and reports should be versioned together. A final report needs a declared source of truth and should identify superseded results.  |
| Excellent point metrics can still be uncertain when the test set has few fraud cases.                                | The saved test includes 60 fraud cases and the artifacts provide hourly-bootstrap intervals.                                                                                        | Report the positive sample count and uncertainty alongside point estimates; expand testing across more time periods before operational decisions.                                    |
| Explanations can reveal unexpected high scores on legitimate examples.                                               | The interpretability summary contains one legitimate-labeled example with a high predicted score and reports feature attributions on a 60-case sample.                              | Review false positives with domain experts, check explanations for usefulness and stability, and do not treat attribution as causal proof.                                           |
| Plans for NearMiss, Decision Tree, broad tuning, and other features differ from the saved implementation.            | The completed comparison instead uses class weighting, SMOTE, SMOTE/Tomek, and adds LightGBM; the final feature set excludes post-event balances.                                   | Record deliberate scope changes and their reasons when they occur. The final evaluation should state what was actually implemented, not repeat the original plan as fact.            |
| The available evidence does not include measured stakeholder feedback or a fairness study.                           | The repository contains a dashboard and technical artifacts but no reviewed feedback report or subgroup evaluation.                                                                 | Treat stakeholder usability, fairness, and live operational value as unanswered questions until evaluated directly.                                                                  |

The artifacts support these technical and methodological lessons. They do not document enough about internal team coordination to make specific claims about which teamwork practices helped or hindered; that reflection should be supplied by the group rather than invented from code.

## 4. Recommendations — 20 marks

### 4.1 Proposed improvements

1. **Reconcile and freeze the experiment record before further comparison.**Maintain a single manifest linking each run ID to its commit, data snapshot and hash, sampling seed, split boundaries, feature schema, model and threshold, configuration, dependency versions, metrics, and exported model. Re-run the documented evaluation from a clean environment and compare outputs with the declared baseline. Success means the team can trace every reported number to one run and explain differences from the Deliverable 4 report without mixing results.
2. **Measure temporal robustness on multiple untouched windows.**Evaluate the selected pipeline on several chronological windows and, where available, a later independent dataset. Keep model and threshold selection separate from each final test period. Report per-window fraud counts, precision, recall, PR-AUC, calibration, and confidence intervals. Success criteria should be agreed with the intended fraud-review operation before examining results; a single unusually favorable window should not suffice for promotion.
3. **Revisit sampling and data coverage.**The 200,000-row stratified sample makes the multi-model pilot tractable, but it is not a full-source training experiment. Compare the existing baseline with appropriately resourced larger or full-data runs and document how sampling affects time-specific prevalence and transaction coverage. Add real, approved transaction data only under suitable governance and privacy controls. Measure performance by transaction type and time window where sample sizes support meaningful estimates.
4. **Calibrate scores and set thresholds from operational costs and capacity.**Replace the assumed 10:1 error-cost ratio and 90% recall floor only after stakeholders quantify fraud losses, investigation cost, acceptable customer friction, and analyst capacity. Evaluate calibration on untouched representative data, especially after resampling, and assess expected review volume at candidate thresholds. Success means the threshold is approved against documented service and harm constraints, not simply selected because it maximizes a metric.
5. **Investigate model errors and explanation stability.**Review the four false negatives and four false positives in the saved primary test result, including the high-scoring legitimate example in the interpretability summary. Verify row identity and context before drawing conclusions. Examine whether `log_amount` adds value given its reported zero mean absolute SHAP attribution in the limited explanation sample; test feature removal only in a controlled, validation-based experiment. Evaluate SHAP/LIME explanation stability across windows and with fraud analysts before treating explanations as operational guidance.
6. **Expand fairness and customer-impact assessment.**Before real-world use, define appropriate fairness questions with legal, privacy, and domain reviewers. Where lawful and ethically justified data are available, assess error rates and review outcomes for suitable groups with uncertainty and privacy safeguards. If those data are not available or cannot lawfully be used, state that limitation and add compensating oversight; do not infer fairness from synthetic data or from the absence of sensitive fields.
7. **Improve visual clarity and provenance.**Rework the Pareto layout to prevent overlapping labels, replace the stale “Deliverable 4” figure title with a run-specific title, and simplify the local explanation figure so feature names remain legible. Show counts beside observed rates, particularly for small probability bands. Check every static figure against its source CSV and provide a readable accessible equivalent. Success means a reviewer can identify the run, compare treatments, and interpret denominators without guessing.
8. **Add end-to-end application and workflow testing.**
   Test valid and invalid transaction inputs, model artifact mismatch, missing or stale reports, app startup, analyst interpretation, accessibility, and error handling. Conduct a documented usability exercise with representative reviewers using synthetic examples. Measure task completion, comprehension of thresholds and uncertainty, and appropriate escalation behavior. Do not infer a production service level from a local throughput figure.

### 4.2 Future research or development

**Research question 1: How stable are the selected model and threshold across future time windows?**
Use a rolling-origin evaluation with several later windows and, if permitted, a separate institution or independently sourced dataset. Compare ranking, calibration, false-positive burden, and fraud recall at a policy-approved threshold. This would answer whether the baseline generalizes across temporal variation better than one sampled test period can.

**Research question 2: Do explanations help reviewers make better and more consistent decisions?**
Run a controlled usability study with synthetic or properly approved cases. Compare reviewer understanding and decision consistency with and without explanations, and check explanation stability for similar examples. The study should test whether explanations assist review, not assume they do.

**Research question 3: Can a monitored human-in-the-loop review process reduce harm at acceptable cost?**
Simulate a queue using representative, authorized data and realistic analyst-capacity assumptions. Measure alert volume, investigation yield, time to review, escalation, false-positive customer impact, and missed-fraud cost. Use results to revise the workflow and policy before considering any controlled pilot.

### 4.3 Implementation considerations

The prototype should remain a decision-support demonstration unless and until the following are independently established:

- **Readiness gate:** Obtain representative-data validation, stable temporal performance, documented threshold and cost approval, privacy/security assessment, fairness review, operational ownership, and a tested human-review process before any live pilot.
- **Integration:** Specify the source and timing of each pre-transaction field, schema validation, response-time budget, secure API boundary, transaction identifiers, audit logging, and how the score reaches a reviewer. Reject or safely route missing, stale, or out-of-range inputs.
- **Human oversight:** Treat a score as a prompt for authorized review, not as an unquestionable decision. Define who can override a flag, how a customer can seek review or remedy, and how decisions are recorded.
- **Capacity and performance:** Benchmark the complete service—including feature preparation, model loading, API and network overhead, concurrency, logging, and queueing—against peak transaction volume and latency requirements. Establish a throughput and availability target from the actual service owner; the local artifact alone cannot set one.
- **Monitoring:** Track input schema and missingness, transaction and score distributions, alert volume, calibration, precision/recall when labels mature, subgroup outcomes where lawful, processing latency, failures, and analyst workload. Assign owners, alert thresholds, review cadence, and escalation actions before launch.
- **Fallback and incident response:** Provide a documented safe mode for model, data, or service failure; decide whether cases route to existing controls or manual review. Test pause, rollback, incident escalation, and customer communication procedures.
- **Release management:** Version the dataset snapshot, code, environment, fitted model, threshold, and approval record together. Compare every candidate release against the frozen baseline on the same evaluation protocol, require sign-off, and retain an immediate rollback path.
- **Infrastructure and cost:** Estimate secure hosting, storage, monitoring, audit retention, backup, scaling, access management, and human review costs using measured workload. Select infrastructure only after requirements and a threat model are defined.

### 4.4 Ethical and responsible AI practices

Keep a human accountable for adverse actions and provide an accessible route for review or appeal. Communicate that a score is a model estimate, explain its limits, and avoid presenting it as proof of fraud. Review false positives for customer inconvenience or exclusion and false negatives for financial harm.

Before deployment, conduct a documented impact assessment with relevant domain, privacy, security, and legal reviewers. Define permitted uses, prohibited uses, reviewer responsibilities, retention of explanations and decisions, and the process for contesting outcomes. Test explanation usefulness and stability; explanations must not be represented as causal reasons unless supported by an appropriate causal analysis.

Fairness must be evaluated rather than assumed. Synthetic benchmark data and the absence of protected attributes do not establish equal treatment. If lawful and appropriate data permit subgroup assessment, report sample counts, uncertainty, and relevant error/selection rates, and investigate disparities. Restrict access to sensitive attributes and derived explanations; use them only under approved governance. Record model limitations and require human approval for material changes to inputs, thresholds, or intended use.

### 4.5 Data governance and compliance

The evidence analyzed here is described as synthetic PaySim data. A future system using real payment information would handle sensitive financial and potentially personal data and requires a separate legal and governance assessment. Apply data minimization; collect only fields needed for an approved purpose; document the lawful authority for processing; define purpose limitation, retention and deletion schedules, role-based access, encryption in transit and at rest, secure secrets handling, and audit-log protections.

Determine applicable obligations for the deployment jurisdiction and organization with qualified privacy and legal reviewers. POPIA may be relevant in South Africa; GDPR may be relevant where its territorial and material scope applies. This draft is not legal advice and does not establish that either law applies to a particular deployment. Assess data-subject rights, breach notification, processor arrangements, cross-border transfers, and impact-assessment duties where applicable.

Maintain lineage for source data, transformations, model versions, thresholds, evaluation results, approvals, incidents, and retraining decisions. Assign named organizational roles—not merely a development team—to data stewardship, access approval, model risk, incident response, customer redress, and periodic review. Define criteria for retraining, independent revalidation, suspension, and retirement; never replace a model solely because a newer run has a better single-test score.

### 4.6 Prioritized action plan

| Phase                                  | Action                                                                                                                                                            | Accountable role                                  | Exit evidence                                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Immediate project close-out            | Reconcile run `a0c5812936ac` with the Deliverable 4 run; verify the source commit, model artifact, threshold, and metrics; mark superseded results clearly.     | Project lead / model owner                        | One approved run manifest and consistent report figures.                                      |
| Offline research                       | Reproduce the baseline and evaluate multiple later temporal windows; review errors, calibration, and score thresholds against documented operational assumptions. | Data science lead with domain reviewer            | Reproducible results, uncertainty analysis, and a documented model-risk decision.             |
| Governance and design                  | Specify lawful data access, privacy/security controls, fairness assessment, review/appeal workflow, monitoring, fallback, and service requirements.               | Data protection, security, compliance, operations | Approved impact and operational design before any live-data pilot.                            |
| Controlled pilot, only if approved     | Run in shadow or human-reviewed mode with representative authorized data; do not automate adverse payment decisions.                                              | Service owner / operations                        | Predefined pilot metrics, incident log, customer safeguards, and independent go/no-go review. |
| Ongoing operation, only after approval | Monitor, audit, revalidate, control versions, handle appeals, and suspend or roll back when limits are exceeded.                                                  | Model owner and designated control functions      | Periodic signed review against pre-agreed thresholds and governance obligations.              |

## Conclusion

The repository documents a substantial academic prototype: a leakage-conscious feature pipeline, a comparison of five classifiers and three imbalance treatments, a validation-selected threshold, a chronological sampled test, interpretability artifacts, automated tests, and a Streamlit scoring interface. In the saved baseline, XGBoost with class weighting achieved 93.3% precision and 93.3% recall on a final window containing 60 fraud cases. Those findings support further offline investigation, not a claim of production readiness. The synthetic data, sampled single-run evidence, small positive test count, policy assumptions, run-history discrepancy, and absence of live-data fairness and operational studies remain material constraints. The next priority is to reconcile the run record and test a reproducible baseline across additional time windows before considering a governed human-reviewed pilot.

## Sources reviewed

- Group 3, *Deliverable 3: Modelling*, supplied document: `../../ITDPA3-34-Group3-Deliverable3.docx`.
- *Deliverable 4 Implementation Report*, supplied document: `../../Deliverable 4 Implementation Report.docx`.
- Project workflow and methodology: `../README.md`.
- Saved baseline: `../evidence_deliverable_5/training_summary.json`, `run_status.json`, `data_quality.json`, `model_comparison.csv`, `model_comparison_key_metrics.csv`, `imbalance_strategy_comparison.csv`, `probability_bands.csv`, `confusion_matrix_counts.json`, and `interpretability_summary.json`.
- Saved figures and test record: `../evidence_deliverable_5/figures/` and `../evidence_deliverable_5/pytest_results.txt`.
- Application and workflow implementation: `../app/streamlit_app.py` and `../src/fraud_detection/`.

## Practice-draft provenance

This file was prepared with AI assistance from the project documents and repository artifacts listed above. It is a practice synthesis, not an independently verified account of group experience or a substitute for the group checking the source files. No undocumented stakeholder feedback, production deployment, team reflection, or legal conclusion is asserted as fact.
