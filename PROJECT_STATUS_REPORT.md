# 📊 Project Status Report — TMDB Movies (INSY 674)

**Report Date:** February 27, 2026  
**Scope:** Current status based on files and folders present in this repository workspace.

---

## 1. Project Overview

This project builds an end-to-end movie analytics workflow to support pre-release decision making with TMDB-based data.

Current implemented flow:
- Data extraction and enrichment
- Feature engineering and EDA
- Supervised popularity modeling
- Semi-supervised revenue-tier modeling
- Model export and Streamlit mockup app

---

## 2. Current Repository Structure

```
TMDB-Movies-INSY-674/
├── README.md
├── PROJECT_STATUS_REPORT.md
├── app/
│   ├── app_mockup.py
│   └── app_mockup2.py
├── data/
│   ├── movies_2010_2025.csv
│   ├── data_cleaned_engineered.csv
│   ├── data_features_master.csv
│   ├── data_supervised_popularity.csv
│   ├── data_supervised_revenue.csv
│   ├── data_ssl_revenue.csv
│   ├── best_ssl_model.joblib
│   ├── ssl_scaler.joblib
│   ├── ssl_model_comparison.csv
│   ├── best_ssl_confusion_matrix.png
│   └── best_ssl_confusion_matrix_pct.png
├── EDA/
│   └── EDA.ipynb
├── models/
│   ├── PopularityModelComparison.ipynb
│   ├── SemiSupervisedModels.ipynb
│   ├── SemiSupervisedModels_V2.ipynb
│   ├── export_best_models.py
│   ├── popularity_best_model.pkl
│   ├── ssl_best_model.pkl
│   ├── ssl_scaler.pkl
│   └── model_metadata.pkl
└── notebooks/
        ├── DataExtraction.ipynb
        └── FeatureEngineering.ipynb
```

---

## 3. Notebook and Model Status

### 3.1 Data and Feature Pipeline
- `notebooks/DataExtraction.ipynb`: extraction and enrichment pipeline.
- `notebooks/FeatureEngineering.ipynb`: cleaning and pre-release feature construction.
- `EDA/EDA.ipynb`: exploratory analysis.

### 3.2 Popularity Modeling
- `models/PopularityModelComparison.ipynb` is present and used for supervised popularity benchmarking.
- Packaged popularity model exists in `models/popularity_best_model.pkl`.

### 3.3 Revenue Tier Semi-Supervised Modeling
- `models/SemiSupervisedModels.ipynb`: baseline SSL workflow.
- `models/SemiSupervisedModels_V2.ipynb`: updated version with fine-tuning and expanded evaluation.

V2 status highlights:
- Compares base + tuned variants for supervised and SSL models.
- Comparison table includes: `accuracy`, `macro_f1`, `macro_precision`, `macro_recall`.
- Includes percentage confusion matrix artifact: `data/best_ssl_confusion_matrix_pct.png`.

---

## 4. Latest SSL Comparison Results (from `data/ssl_model_comparison.csv`)

| Rank | Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|------|-------|----------|----------|------------------|--------------|
| 1 | SelfTraining (SSL, tuned) | 0.6180 | 0.6247 | 0.6399 | 0.6180 |
| 2 | RandomForest (supervised, base) | 0.6161 | 0.6172 | 0.6228 | 0.6161 |
| 3 | RandomForest (supervised, tuned) | 0.6065 | 0.6064 | 0.6143 | 0.6065 |
| 4 | GradientBoosting (supervised, base) | 0.6027 | 0.6032 | 0.6061 | 0.6028 |
| 5 | GradientBoosting (supervised, tuned) | 0.6027 | 0.6024 | 0.6035 | 0.6027 |
| 6 | SelfTraining (SSL, base) | 0.5931 | 0.5909 | 0.5976 | 0.5931 |
| 7 | LabelSpreading (SSL, tuned) | 0.5470 | 0.5420 | 0.5415 | 0.5472 |
| 8 | LabelPropagation (SSL, tuned) | 0.5278 | 0.5277 | 0.5283 | 0.5280 |
| 9 | LabelPropagation (SSL, base) | 0.5029 | 0.5011 | 0.5000 | 0.5031 |
| 10 | LabelSpreading (SSL, base) | 0.4952 | 0.4943 | 0.4937 | 0.4954 |

Current best model by macro F1: **SelfTraining (SSL, tuned)**.

---

## 5. Export and App Integration Status

- `models/export_best_models.py` exists and packages models/metadata.
- Exported artifacts available:
    - `models/ssl_best_model.pkl`
    - `models/ssl_scaler.pkl`
    - `models/model_metadata.pkl`
    - `models/popularity_best_model.pkl`
- App files available:
    - `app/app_mockup.py`
    - `app/app_mockup2.py`

---

## 6. Current Completion Snapshot

- ✅ Data extraction notebook available
- ✅ Feature engineering notebook available
- ✅ EDA notebook available
- ✅ Popularity modeling notebook available
- ✅ Semi-supervised modeling notebook available
- ✅ V2 SSL notebook with tuning + full metrics available
- ✅ Model export script and packaged model files available
- ✅ Streamlit mockup app files available

---

## 7. Recommended Next Actions

1. Standardize one canonical SSL notebook (`SemiSupervisedModels_V2.ipynb`) for final reporting.
2. Re-run full notebook pipeline once before final delivery to refresh all artifacts consistently.
3. Confirm app consumes the latest exported metadata and feature schema.
4. Freeze dependency versions for reproducibility (`requirements.txt` / environment spec).
