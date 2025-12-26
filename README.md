# DATA ENGINEERING PROJECT
## Fraud Detection Pipeline for Transactionl Tabular Data

This is a reproducible, Colab-friendly framework for Fraud Detection across two dataseets: Credit Card (PCA Feasures) and PaySim (transaction logs).
It delivers tables and figures to five requestions (RQ1-RQ5), handles class imbalance, compares algorithms, designs an end-to-end pipeline, evaluate metrics that matter to rare events, uses feature engineering strategies with automatic saving of all oitputs

## Dataset Links
* https://www.kaggle.com/datasets/ealaxi/paysim1
* https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Primary Features
* **Two Datasets**: Credit Crad (Class) and PaySim(isFraud)
* End-toEnd experiments (RQ1-RQ5) with stratified 70/15/15 splits
* **Imbalance Handling**: Class weights, random undersampling, SMOTE
* **Algorithms**: Logistic Regression, Random Forest, XGBoost, Isolation Forest.
* **Metrics**: PR-AUC, MCC, F1, ROC-AUC, Recall@1%, plus Accuracy/Precicion/Recall/TNR
* Leak aeware Feature Engineering (Paysim balance residuals, ratios, cyclic time)
* Auto save tables (CSV/XLSX) and figures (PNG/SVG) for Research Questions
* **Colab-ready**: safe memory defaults, float32, optional subsampling

## Repository/Folder Structure

   | Function | Output
|----------|----------
| creditcard.csv/   | Credit Card dataset (Kaggle PCA features) 
| frauddetection.csv  | PaySim transactions dataset
| Fraud Detection Pipeline for Transactionl Tabular Data.ipynb  | Colab IDE
| Results/ |Auto-saved figures & tables
| README.md  |This file

## Prerequisites

1. **Environment**: Python 3.9+ which works wel in google Colab
2. **Libraries**
   * Numpy, matplotlib, scikit-learn
   * IMbalnced-learn (for SMOTE, undersamplin)
   * xgboost (for XGBOOST model)
  
## Process of code execution
1. Create a file on google Colab
   * Fraud Detection Pipeline for Transactionl Tabular Data.ipynb
  
2. Upload datasets to Colab file selection
   * creditcard.csv (colums include: Time, V1-V28, Amount, Class)
   * frauddetection.csv (columns: step, type, amount, balances, isFraud, isFlaggedFraud)
  
3. Import necessary Libraries or import for every single line of code as we progressed. Subsequest codelines will:
   * Reuse engineered arrays if already created (X_cc_tr, y_cc, X_ps_tr, y_ps), else builds minimal features safely
   * Data Cleaning ad Preprocessing
   * Feature engineering
   * computation of Tables and Figures as CSVXLSX or PNG/SVG respectively.
   * Find outputs and download necessary files to local machine
  
## Research Questions and OUtputs

**RQ1**: How do different imbalance handling techniques (under sampling, SMOTE oversampling, and class weighting) affect the performance of machine learning models in detecting fraudulent transactions?

**RQ2**: Which machine learning algorithms (Logistic Regression, Random Forest, XGBoost, Isolation Forest) demonstrate superior performance for fraud detection in highly imbalanced transactional datasets?

**RQ3**: How can an end-to-end fraud detection pipeline be designed to systematically address data preprocessing, imbalance handling, model training, and evaluation in a reproducible and scalable manner?

**RQ4**: Which evaluation metrics most effectively capture fraud detection performance in highly imbalanced scenarios, and how do they provide different insights compared to traditional accuracy measures?

**RQ5**: What preprocessing and feature engineering strategies most significantly improve fraud detection performance across different algorithms and datasets?

#### RQ1: Imbalance Handling vs Performance
**Compares**: Baseline, ClassWeight, Undersample, SMOTE (Logistic Regression base)
**Metrics**: PR‑AUC, MCC, F1, ROC‑AUC, Recall@1%.
Saved:
* RQ1_imbalance_results.csv/.xlsx
* RQ1_imbalance_PR_AUC_MCC.png/.svg

#### RQ2: Algorithm Comparison
**Models**: Logistic Regression, Random Forest, XGBoost (with scale_pos_weight), Isolation Forest (unsupervised
Saved:
* RQ2_algorithm_performance.csv/.xlsx
* RQ2_algorithm_PR_AUC_MCC.png/.svg

#### RQ3: End‑to‑End Pipeline Design
**Table of Ingestion Stages**: Ingestion → Cleaning → FE → Imbalance Handling → Training → Evaluation → Persistence

**Schematic Block Diagram**

![image](https://github.com/Freddyx94/DE-Fraud-Detection-Pipeline-for-Transactional-Tabular-Data/blob/main/RQ3_pipeline_schematic.png)
Saved:
* RQ3_pipeline_design.csv/.xlsx
* RQ3_pipeline_schematic.png/.svg

#### RQ4: Metrics Effectiveness vs Accuracy
Compares Accuracy against PR-AUC, MCC, F1, Recall@1%, Precision, Recall, Specificity (TNR).
Saved:
* RQ4_metrics_effectiveness.csv/.xlsx
* RQ4_metrics_vs_accuracy.png/.svg

#### RQ5: Preprocessing and Feature Engineering Strategies
Credit Card: Baseline: Time + Amount (log, cycles, night). Time + Amount + Bins (manual quantile one-hot)
PaySim: Baseline: Residuals (Leak aware deltas, ratios, zero flags, cycles time, robust z by type).
Saved:
* RQ5_fe_strategies_performance.csv/.xlsx
* RQ5_fe_strategies_PR_AUC_MCC.png/.svg

## Airflow DAG
