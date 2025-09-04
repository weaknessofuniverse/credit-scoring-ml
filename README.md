# Credit Risk Scoring ML Project

This repository contains a machine learning pipeline for the ["Give Me Some Credit"](https://www.kaggle.com/competitions/GiveMeSomeCredit) Kaggle competition. The goal is to predict the probability of a consumer credit default.

## 🚀 Quick Overview

*   **Task:** Binary classification (imbalanced classes)
*   **Top Models:** Gradient Boosting & HistGradientBoosting
*   **Best CV AUC:** ~0.872
*   **Best Kaggle Score:** 0.85964 (Private)

## 🏗️ Project Architecture

The project is structured as a sequential data science pipeline:

credit-scoring-ml/
├─ notebooks/                          # interactive EDA and experiments
│  ├─ 01_EDA.ipynb/                    # Exploratory Data Analysis
│  ├─ 02_Preprocessing.ipynb           # Data cleaning, feature engineering, imputation
│  └─ 03_Modeling.ipynb                # Model training, selection, and prediction
├─ data/
│  ├─ raw/                    # raw files
│  └─ processed/              # outputs from preprocessing
├─ src/
│  ├─ model_pipeline.py/
│  └─ results_viewer.py/
├─ Article.pdf
├─ requirements.txt
└─ README.md


## 📋 Key Steps

1.  **EDA:** Handled class imbalance, missing data, and anomalous "magic values" (96, 98) in delinquency features.
2.  **Preprocessing:** Imputation, outlier clipping, log-transformation of income, and feature engineering (e.g., debt-to-income ratio, delinquency flags).
3.  **Modeling:** A three-stage approach:
    *   **Quick Scan:** Evaluate 7 models on a sample.
    *   **Refinement:** Retrain top-4 models on full data with/without SMOTE.
    *   **Final Tuning:** Exhaustive tuning of the two best models (Gradient Boosting variants).

## 📊 Results

Gradient Boosting models performed best, with **SMOTE surprisingly reducing performance** on this dataset.

| Model | CV AUC (no SMOTE) | Kaggle AUC (Private) |
| :--- | :--- | :--- |
| HistGradientBoosting | **0.8722** | **0.85964** |
| GradientBoosting | 0.8715 | 0.85944 |

## 🚀 How to Run

1.  Clone the repo and install requirements.
2.  Download competition data to `data/raw/`.
3.  Run Jupyter notebooks in sequential order (`01_EDA.ipynb`, `02_Preprocessing.ipynb`, `03_Modeling.ipynb`).

## 📄 License

This project is for educational purposes.
