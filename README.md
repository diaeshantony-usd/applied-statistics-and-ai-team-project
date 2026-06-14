# Predictive Modeling for Credit Risk

This repository is the collaborative workspace for the Team project for Applied statistics & AI. We utilize the Credit Risk Modeling dataset to evaluate default risk and interest rate structures.

---

## 📝 Introduction: Project Context & Goals

### Project Context
In retail banking and consumer finance, credit risk modeling is one of the most critical applications of Applied Statistics & AI. When a borrower applies for a loan, the financial institution must evaluate the credit risk—namely, the likelihood that the borrower will fail to make required payments (default). 
- **Under-pricing risk (Type I Error)**: Approving a high-risk borrower who ultimately defaults leads to direct loss of principal.
- **Over-pricing risk (Type II Error)**: Rejecting a creditworthy borrower or charging an excessively high interest rate causes the bank to lose profitable business to competitors (opportunity cost).

This project uses the **Credit Risk Modeling Dataset** to analyze borrower profiles, optimize interest rate allocation, and predict borrower default probability.

### Project Goals
1. **Robust Data Preprocessing**: Establish a standard pipeline to handle missing observations using median imputation and filter unrepresentative outliers using Interquartile Range (IQR) filtering.
2. **Exploratory Analytics**: Evaluate data distributions, correct feature skewness using logarithmic transformations, and perform formal Shapiro-Wilk Gaussianity testing.
3. **Inferential Statistics & Causal Mapping**: Run Welch's $t$-tests and One-way ANOVA to evaluate mean differences across credit groups, and fit a Bayesian Network structure to map causal Directed Acyclic Graphs (DAGs) using conditional probabilities.
4. **Predictive Modeling & Diagnostics**: Fit statsmodels Multiple Linear Regression (OLS) and Logistic Regression (GLM), validating assumptions via Variance Inflation Factor (VIF) multicollinearity checks, residual Q-Q distribution plots, and Breusch-Pagan homoscedasticity testing.

---

## 📁 Repository Directory Structure

To keep the collaborative workspace organized and ensure seamless data sharing without Git conflicts, we use the following directory layout:

1. **`data/`**: The central repository for project datasets.
   * **`data/raw/`**: Contains the raw, un-preprocessed Credit Risk Modeling dataset.
   * **`data/processed/`**: Contains the cleaned and processed dataset (`credit_risk_cleaned.csv`) after missing-value median imputation and IQR outlier filtering.
2. **`01_Data_Prep_EDA/`**: Data Preparation & Exploratory Data Analysis.
   * **`A_Data_Prep_and_EDA.ipynb`**: Jupyter Notebook containing the full implementation of **Introduction (Context, Goals, & Hypotheses)**, **Data Preparation (Median Imputation, Outlier Filtering)**, and **Exploratory Data Analysis (EDA)** (Log-transforms, gridspec subplots, Shapiro-Wilk testing, and Central Limit Theorem simulations).
   * **`A_Data_Prep_and_EDA.pdf`**: Compiled PDF report of the complete Data Preparation & EDA subsystem, including all printed statistical outputs and visual figures.
   * **`Credit_Risk_Data_Prep_And_EDA_Report.docx`**: Detailed Word report for data preparation and exploratory data analysis.
3. **`02_Model_Selection/`**: Feature engineering, statistics, machine learning model selection, saved model artifacts, and inference.
   * **`B1_Data_Audit_Feature_Engineering.ipynb`**: Jupyter Notebook that audits the Phase 1 cleaned dataset, creates engineered/grouped features, performs focused EDA, and saves the Phase 2 modeling dataset at `data/processed/phase2/credit_risk_cleaned.csv`.
   * **`B1_Data_Audit_Feature_Engineering.pdf`**: Compiled PDF report for the data audit and feature engineering workflow.
   * **`B2_Statistics_Modeling_and_Inference.ipynb`**: Jupyter Notebook that loads the Phase 2 CSV and runs inferential statistics, Bayesian Network analysis, classification, regression, model saving, model comparison, and applicant-level inference.
   * **`B2_Statistics_Modeling_and_Inference.pdf`**: Compiled PDF report for the statistics, modeling, and inference workflow.
   * **`Credit_Risk_Model_Selection_Report.docx`**: Detailed Word report covering feature engineering, multicollinearity, train/test split, Bayesian Network reasoning, classification, regression, saved models, and applicant inference.
   * **Model artifacts**: Notebook B2 saves trained models under `data/models/classification/`, `data/models/regression/`, and `data/models/bayesian_network/`.
4. **`03_Model_Analysis_Diagnostics/`**: Predictive Modeling & Diagnostic Validation.
5. **`04_Master_Pipeline/`**: Final Joint Consolidation.

---

## 👥 Team Roles & Responsibilities

| Folder Subsystem | Owner | Milestone Deliverable |
| :--- | :--- | :--- |
| **`01_Data_Prep_EDA/`** | **Diaesh Antony** | Data Cleaning & Preparation, Exploratory Data Analysis (EDA) (Due Jun 01) |
| **`02_Model_Selection/`** | **N L N Sai Krishna Akula** | Team & Topic Selection, Hypothesis Testing & Model Selection (Due Jun 08) |
| **`03_Model_Analysis_Diagnostics/`** | **Ashok Bhairwal** | Predictive Modeling & Diagnostic Analysis (Due Jun 08) |
| **`04_Master_Pipeline/`** | **Joint Collaboration** | Draft Report (Due Jun 15) & Final Technical Report and Presentation Video (Due Jun 22) |
