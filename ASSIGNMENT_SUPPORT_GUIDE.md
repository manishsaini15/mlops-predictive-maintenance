# MLOps  Guide

## Predictive Maintenance Classification with Local MLOps

This document is a learner support guide for the MLOps assignment.



It is designed to help you understand:
- the business problem
- the goals of each section
- the exact tasks you need to complete
- the rubric expectations
- what is already provided to you
- what you must submit

It does **not** give away the full notebook solution.

---

## Quick Start

### What Has Already Been Done For You

To keep the assignment manageable, the following parts are already provided:

- the dataset has already been prepared and split
- the stable and stress post-deployment batches are already available
- the starter notebook already contains the full section structure
- prompts and TODO placeholders are already written
- the grading rubric is already defined
- the dependency versions are already pinned for reproducibility
- MLflow tracking is preconfigured in the starter notebook using `mlflow.set_tracking_uri("sqlite:///mlflow.db")`

So your work should focus on:
- implementing the analysis correctly
- logging and monitoring properly
- interpreting the outputs clearly
- writing strong conclusions



Make sure it is:
- completed
- fully executed
- saved with outputs visible

---

## 1. Problem Statement

A heavy-equipment manufacturer wants to reduce unplanned machine downtime.
Each machine streams sensor readings, and the goal is to classify whether the
machine is operating normally or is heading toward one of four failure modes.

You have been given:
- historical labelled training data
- a stable post-deployment batch
- a shifted stress batch from a heavy-load production period

Your task is to build a **local MLOps workflow** that:
- validates the data before modeling
- compares and tracks candidate models
- tunes and registers the best model
- monitors incoming batches for drift
- explains model predictions
- recommends retraining when monitoring shows risk

---

## 2.  Goals

By the end of this assignment, you should be able to:
- validate tabular data with Pandera
- handle class imbalance correctly
- compare multiple models fairly using MLflow
- tune the strongest model with Optuna
- detect drift with Evidently
- interpret a multiclass tree model with SHAP
- connect technical outputs to engineering decisions

This assignment is not only about getting a good classifier.
It is about showing that you understand the **full MLOps loop**:

`validate -> train -> track -> tune -> monitor -> explain -> decide`

---

## 3. Files Already Provided

These are already done for you:

- [train.csv](data/train.csv)
  Historical labelled baseline data
- [current.csv](data/current.csv)
  Stable post-deployment batch
- [stress.csv](data/stress.csv)
  Shifted heavy-load batch
- [requirements.txt](requirements.txt)
  Pinned environment for deterministic execution
- [Starter Notebook](MLOps_Assignment_PredMaint_Starter.ipynb)
  The notebook you must complete and submit
- [Rubric CSV](MLOps_Assignment_rubrics.csv)
  Grading rubric used for evaluation

You do **not** need to:
- download the dataset
- create the split yourself
- build a full application or API

This is a **local notebook assignment**, designed to fit platform and autograder constraints.

---

## 4. What the Data Represents

The three datasets play different roles:

- `train.csv`
  Historical baseline used for training and validation

- `current.csv`
  A normal incoming production batch
  Expected story: mostly stable relative to training

- `stress.csv`
  A heavier-load post-deployment batch
  Expected story: operationally valid data, but distributionally shifted

Important:
- `stress.csv` is intended to be **valid but drifted**
- this means Pandera may still pass it
- the drift is expected to show up in Evidently, not necessarily in schema validation

That distinction is one of the key lessons of the assignment.

---

## 5. What the Starter Notebook Looks Like

The starter notebook already contains the full structure you should follow.

Main sections:

1. Data Loading, Schema Validation & EDA
2. Experiment Tracking & Model Selection
3. Drift Detection & Monitoring
4. Explainability & Insights
5. Conclusions

Sub-sections in the starter notebook:

### Section 1
- `1.1` Load the datasets
- `1.2` Define and apply a Pandera schema
- `1.3` Exploratory Data Analysis
- `1.4` Feature Engineering

### Section 2
- `2.1` Setup: features, split, SMOTE
- `2.2` Train and log 4 models with MLflow
- `2.3` Optuna tuning + MLflow Model Registry

### Section 3
- `3.1` Evidently - current batch
- `3.2` Evidently - stress batch
- `3.3` Retraining decision

### Section 4
- `4.1` SHAP analysis per failure class
- `4.2` Engineering insight

### Section 5
- `5.1` Key findings
- Final conclusions block

The notebook already gives you:
- imports
- section prompts
- TODO scaffolding
- the required structure

So your job is to fill in the logic, not redesign the notebook.

---

## 6. Stage-Wise Tasks

## Stage 1: Data Loading, Schema Validation & EDA

### Goal
Understand the data and prove that it is safe to use.

### What you need to do
- load all three CSV files
- print shapes
- display the training sample
- define the Pandera schema for the 7 raw columns
- validate `train`, `current`, and `stress`
- inspect class imbalance
- plot key distributions
- engineer `Power_W` and `Temp_diff`

### What you should notice
- the target is highly imbalanced
- some failure types are very rare
- the engineered features add useful physical meaning

### Common mistakes
- validating before fixing integer dtypes
- forgetting to validate `stress`
- computing engineered features for only one dataset

---

## Stage 2: Experiment Tracking & Model Selection

### Goal
Compare multiple models fairly, handle imbalance, and tune the strongest candidate.

### What you need to do
- encode the `Type` column
- create `X` and `y`
- perform a stratified 80/20 train-validation split
- apply SMOTE only on the training split
- train 4 models
- log all runs to MLflow
- compare them using `macro_f1`
- tune the strongest tree model with Optuna
- register the tuned best model in MLflow

### What you should notice
- accuracy is not a good selection metric here
- `macro_f1` is much more meaningful
- the rarest class remains the hardest

### Common mistakes
- applying SMOTE before the split
- using default `k_neighbors` instead of `3`
- choosing the winner based on accuracy
- logging incomplete metrics

---

## Stage 3: Drift Detection & Monitoring

### Goal
Show how a model behaves after deployment when new data arrives.

### What you need to do
- compare a basic statistic first (for example mean `Rotational speed`) between `current.csv` and `stress.csv`
- run Evidently on `current.csv`
- save `drift_current.html`
- state whether dataset drift is present
- run Evidently on `stress.csv`
- save `drift_stress.html`
- print per-feature drift details
- decide whether retraining is needed

### What you should notice
- `current.csv` should look relatively stable
- `stress.csv` should show meaningful drift
- drift should be concentrated in operationally important features
- this stage is a diagnosis task: explain why the model is stale on `stress.csv`, not how to force better stress-batch accuracy

### Common mistakes
- only saving the HTML without interpreting it
- mixing up reference and current datasets
- making a retraining recommendation without connecting it to evidence

---

## Stage 4: Explainability & Insights

### Goal
Explain the final tuned model in engineering terms.

### What you need to do
- load the saved best model
- compute SHAP values
- create a 4-panel multiclass SHAP plot
- save `shap_per_class.png`
- identify the top driver for each failure class
- explain the physical meaning of important features

### What you should notice
- each failure type is driven by a different pattern
- derived features are important, not just raw sensor columns
- explainability should support the retraining decision
- multiclass SHAP should be interpreted per class; do not collapse all classes into one global ranking

### Common mistakes
- treating multiclass SHAP like binary SHAP
- giving generic explanations instead of class-specific ones
- forgetting to save the SHAP figure

---

## Stage 5: Conclusions

### Goal
Summarize the assignment like a production engineering case.

### What you need to do
Write a conclusion covering:
- which model won and why
- why accuracy is misleading
- why the weakest class remains difficult
- what drifted in the stress batch
- one actionable engineering recommendation

### What makes a strong conclusion
- references actual results from your notebook
- mentions the correct metric
- connects model behavior to operations
- uses evidence from SHAP and drift outputs
- frames TWF underperformance as a data scarcity issue (30 real samples), even if tuned TWF F1 remains 0.0

---

## 7. Rubric Summary

Total marks: **50**

### Section 1: Data Loading, Validation and EDA - 15 marks
- Load datasets correctly: `3`
- Define schema: `3`
- Validate datasets correctly: `2`
- Class distribution + chart: `2`
- Distribution plots: `2`
- Compute `Power_W`: `2`
- Compute `Temp_diff` + grouped means: `1`

### Section 2: Experiment Tracking and Model Selection - 15 marks
- Split + SMOTE setup: `2`
- Log all 4 models to MLflow: `3`
- Log metrics correctly: `3`
- Print comparison + identify best model: `2`
- Run Optuna study: `3`
- Register tuned model in MLflow: `2`

### Section 3: Drift Detection and Monitoring - 10 marks
- Current batch monitoring: `4`
- Stress batch monitoring: `4`
- Retraining decision: `2`

### Section 4: Explainability and Insights - 5 marks
- SHAP per class: `3`
- Engineering interpretation: `2`

### Section 5: Conclusions - 5 marks
- Winning model and why: `1`
- Why accuracy is misleading: `1`
- Weakest class insight: `1`
- Stress drift implication: `1`
- Actionable recommendation: `1`

---

## 8. What the Grader Will Look For

The grader is not only checking whether your notebook runs.
It is also checking whether your work is **conceptually correct**.

It will look for:
- correct notebook structure
- correct saved files
- correct model-comparison logic
- correct use of `macro_f1`
- correct drift interpretation
- correct SHAP interpretation
- a sensible retraining recommendation

---

## 9. Common Errors and How To Avoid Them

These are the most common ways learners lose marks.

### Error 1: Using accuracy as the main selection metric

Why it is a problem:
- the dataset is highly imbalanced
- a model can look strong on accuracy while still missing rare failures

How to avoid it:
- use `macro_f1` for model comparison
- discuss accuracy only as a secondary metric

### Error 2: Applying SMOTE before the split

Why it is a problem:
- it leaks synthetic information into validation
- it makes evaluation unreliable

How to avoid it:
- split first
- apply SMOTE only on the training split

### Error 3: Forgetting `k_neighbors=3`

Why it is a problem:
- the rarest class is very small
- default SMOTE settings can fail or behave poorly

How to avoid it:
- explicitly set `k_neighbors=3`

### Error 4: Treating `stress.csv` as invalid data

Why it is a problem:
- the stress batch is meant to be valid but drifted
- drift and schema violations are not the same thing

How to avoid it:
- explain that Pandera checks validity
- explain that Evidently checks distribution shift

### Error 5: Fitting encoders separately on each dataset

Why it is a problem:
- category mappings can become inconsistent

How to avoid it:
- fit on `train`
- transform `current` and `stress` with the same encoder

### Error 6: Logging incomplete model runs

Why it is a problem:
- the rubric checks experiment tracking, not just raw training

How to avoid it:
- log each model run
- log model name, aggregate metrics, and per-class F1

### Error 7: Writing generic conclusions

Why it is a problem:
- the assignment expects evidence-based interpretation

How to avoid it:
- reference your own drift results
- reference your own SHAP findings
- reference the actual winning model and metric values

---

## 10. What Artifacts You Need To Produce

These artifacts should be created during notebook execution:

- `eda_distributions.png`
- `drift_current.html`
- `drift_stress.html`
- `best_model.pkl`
- `label_encoder.pkl`
- `shap_per_class.png`

These are generated outputs, not separate submission files unless your evaluator specifically asks for them.

---

## 11. What You Need To Submit

Primary submission:
- [Starter Notebook](MLOps_Assignment_PredMaint_Starter.ipynb)

Your notebook should be:
- fully completed
- fully executed
- saved with outputs visible

If the platform or evaluator also checks local generated files, make sure the notebook execution created:
- `drift_current.html`
- `drift_stress.html`
- `best_model.pkl`
- `label_encoder.pkl`
- `shap_per_class.png`

But the main learner-facing deliverable is still the completed notebook.

---

## 12. Key Lessons To Keep In Mind

If you feel lost, come back to these:

- valid data can still be drifted data
- accuracy can be misleading in rare-failure classification
- SMOTE helps, but it cannot fully solve real data scarcity
- drift monitoring becomes useful when connected to business action
- explainability is not decoration; it supports the retraining decision

In this assignment, the clearest example of that scarcity issue is `TWF`:
- it has very few real training examples
- SMOTE helps rebalance the training split
- but SMOTE cannot create the full real-world diversity of a rare failure mode
- so it is normal for `TWF` to remain the weakest class

---

## 13. Final Checklist Before Submission

Before submitting, make sure:

- all cells executed successfully
- all required plots/files were saved
- MLflow runs were created
- your winner model is chosen using `macro_f1`
- your stress-batch discussion names the important drifted features
- your retraining recommendation uses both drift evidence and SHAP evidence
- your conclusion is specific, not generic

---

## 14. If You Are Studying This Assignment Design

This assignment is intentionally kept:
- local
- deterministic
- notebook-friendly
- autograder-compatible

It still covers real MLOps concepts:
- validation
- experiment tracking
- model selection
- tuning
- registration
- monitoring
- explainability
- retraining decision-making

That is the core educational story of the assignment.
