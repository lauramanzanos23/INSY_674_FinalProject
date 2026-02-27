# TMDB Movie Success Prediction Project

## Executive Summary
This project delivers an end-to-end machine learning workflow to estimate movie success before release using TMDB-derived metadata.

It supports two prediction tasks:
1. Popularity prediction (regression).
2. Revenue tier prediction (semi-supervised classification: Low/Medium/High/Blockbuster).

The final pipeline includes:
1. Data extraction and enrichment from TMDB.
2. Pre-release-safe feature engineering.
3. Model comparison, cross-validation, and fine-tuning.
4. Model export for app serving.
5. Streamlit app (`app/app_final.py`) for scenario testing.

## Business Questions
1. How popular is a movie likely to be before release?
2. Which revenue tier is it likely to reach?
3. Which pre-release factors (cast, director, timing, genre, budget proxy) drive outcomes?
4. How do casting and release decisions change predicted results in what-if scenarios?

## Repository Structure
```text
INSY 674 FinalProject/
├── README.md
├── PROJECT_STATUS_REPORT.md
├── EDA/
│   └── EDA.ipynb
├── notebooks/
│   ├── DataExtraction.ipynb
│   └── FeatureEngineering.ipynb
├── data/
│   └── movies_2010_2025.csv
├── models/
│   ├── PopularityModelComparison.ipynb
│   ├── SemiSupervisedModels.ipynb
│   ├── SemiSupervisedModels_V2.ipynb
│   └── export_best_models.py
├── app/
│   └── app_final.py
└── src/
```

## End-to-End Pipeline
1. **Data extraction** (`notebooks/DataExtraction.ipynb`)
   - Collects 2010-2025 movie records from TMDB-style endpoints.
   - Enriches with cast, director, keywords, and metadata.
   - Produces base dataset (`movies_2010_2025.csv`).

2. **Feature engineering** (`notebooks/FeatureEngineering.ipynb`)
   - Creates a target-agnostic master table from pre-release features.
   - Enforces leakage controls (post-release variables excluded from model feature sets).
   - Handles zero-as-missing corrections for budget/revenue.
   - Builds derived supervised and SSL datasets.

3. **Popularity modeling** (`models/PopularityModelComparison.ipynb`)
   - Compares baseline, linear, tree ensemble, and boosting regressors.
   - Uses holdout metrics plus cross-validation for robustness.
   - Includes explainability (feature importance + SHAP) and tuning blocks.

4. **Revenue-tier SSL modeling** (`models/SemiSupervisedModels_V2.ipynb`)
   - Builds labeled + unlabeled setup for semi-supervised learning.
   - Compares supervised and SSL classifiers.
   - Selects best model by Macro F1 then Accuracy.

5. **Model packaging** (`models/export_best_models.py`)
   - Trains/selects final popularity model using notebook-consistent policy.
   - Loads best SSL artifacts and creates metadata used by app inference.

6. **Interactive app** (`app/app_final.py`)
   - Loads exported models and metadata.
   - Lets users lock director/cast and tune scenario inputs.
   - Returns popularity estimate, revenue tier, confidence, and composite fit score.

## Datasets and Artifacts
### Primary input
1. `data/movies_2010_2025.csv`: original extracted movie dataset.

### Generated datasets (pipeline outputs)
1. `data/data_features_master.csv`
2. `data/data_supervised_popularity.csv`
3. `data/data_supervised_revenue.csv`
4. `data/data_ssl_revenue.csv`

### Model artifacts
1. `models/popularity_best_model.pkl`
2. `models/ssl_best_model.pkl`
3. `models/ssl_scaler.pkl`
4. `models/model_metadata.pkl`

## Exploratory Data Analysis
### 1) Data Overview
1. Total movies: 9290
2. Labeled rows (revenue known):2604
3. Total feature: 52

### 2) Raw features
Groups | Categorical variables​ | Numerous variable​s 
--- | --- | --- 
Feature name | Title, ​Release date, Original language,​ Status, overview,​Genres, Keywords,​ Director name,​ Director department,​ Actor(1~5) name,​ Actor(1~5) character​ | Runtime, ​Popularity, ​Vote average, ​Vote count, Budget, ​Revenue, ​Director id, ​Director gender, Director Popularity,​ Actor(1~5) id, Actor(1~5) gender, Cast pop mean, Cast pop max​

### 3) Missing Value
Features name | Number of missing values
--- | --- 
runtime | 457
--- | --- 
Vote average | 1642
--- | --- 
Vote Count | 1640
--- | --- 
budget | 6527
--- | --- 
revenue | 6686
--- | --- 
cast_pop_mean | 222
--- | --- 
cast_pop_max | 222
--- | --- 
director_gender | 2285
--- | --- 
Missing_actor1 | 225
--- | --- 
Missing_actor2 | 384
--- | --- 
Missing_actor3 | 681
--- | --- 
Missing_actor4 | 1036
--- | --- 
Missing_actor5 | 1479
--- | --- 
genres | 489
--- | --- 
keywords | 3238

### 4) The distribution of log target variable​
#### Revenue


#### Popularity


### 5) Correlation map


### 6) Talent ranking
#### Ranking Director popularity​

## Modeling Summary
## 1) Popularity Regression
Notebook: `models/PopularityModelComparison.ipynb`

### Models compared
1. Dummy Mean
2. Linear Regression
3. RidgeCV
4. Random Forest
5. Extra Trees
6. Gradient Boosting
7. Hist Gradient Boosting
8. XGBoost
9. LightGBM

### Evaluation setup
1. 80/20 holdout split.
2. Metrics: RMSE, MAE, R2 (+ MedAE, MAPE, ExplainedVariance, MaxError, NRMSE).
3. K-Fold CV (3-fold) for `CV_RMSE`.
4. Repeated CV (5x2) for top-model stability.
5. Target ablation: raw popularity vs `log1p(popularity)`.
6. XGBoost RandomizedSearchCV fine-tuning block.

### Key reported results (current notebook outputs)
1. Best raw-target holdout model: **XGBoost**
   - RMSE `4.185`, MAE `1.626`, R2 `0.314`.
2. Best log-target setting: **Gradient Boosting + log1p(popularity)**
   - RMSE `3.507`, MAE `1.420`, R2 `0.519`.
3. Fine-tuned XGBoost improved over baseline XGBoost on holdout:
   - RMSE `4.149` vs `4.185`, MAE `1.570` vs `1.626`, R2 `0.326` vs `0.314`.

### Explainability
1. Global importances and SHAP summaries are included.
2. Top SHAP features (XGBoost view) include `release_year`, `log_budget`, `keyword_count`, and talent/timing variables.

## 2) Revenue Tier Semi-Supervised Classification
Notebook: `models/SemiSupervisedModels_V2.ipynb`

### Target
`y_ssl` classes:
1. `0` = Low
2. `1` = Medium
3. `2` = High
4. `3` = Blockbuster
5. `-1` = unlabeled rows (used in SSL training workflow)

### Model comparison
1. Supervised baselines (e.g., GradientBoosting, RandomForest).
2. SSL approaches (e.g., SelfTraining; graph-based methods depending on notebook section/version).
3. Selection based on Macro F1 then Accuracy.

### Reported comparison file
`data/ssl_model_comparison.csv` stores final metric table for SSL run.

## App Overview (`app/app_final.py`)
The final app is a casting sandbox for pre-release scenario analysis.

### What users can do
1. Select and lock a director.
2. Select cast (up to 5 actors).
3. Adjust genre, language, runtime, release year/month, keyword count, budget, and overview length.
4. View model outputs:
   - Predicted popularity (with percentile context vs training data).
   - Predicted revenue tier + confidence.
   - Composite casting fit score.
5. Compare scenario score versus director baseline.
6. Inspect top known-for movies from TMDB (with local fallback).
7. View global feature-importance chart for popularity model.

### Inference notes
1. Popularity model may be trained on `log1p` target; app applies `expm1` back-transform when metadata indicates it.
2. SSL path uses exported scaler and feature list from metadata.
3. Composite score in app is explicitly weighted:
   - 65% popularity score
   - 25% revenue tier score
   - 10% confidence

## How to Run
## 1) Export models
```bash
python models/export_best_models.py
```

## 2) Launch app
```bash
streamlit run app/app_final.py
```

## 3) Reproduce notebooks
1. Run `notebooks/DataExtraction.ipynb`
2. Run `notebooks/FeatureEngineering.ipynb`
3. Run `models/PopularityModelComparison.ipynb`
4. Run `models/SemiSupervisedModels_V2.ipynb`

## Environment
Core libraries used:
1. Python 3.x
2. pandas, numpy
3. scikit-learn
4. xgboost, lightgbm 
5. shap (optional explainability)
6. streamlit, altair
7. joblib, pickle

Optional env var:
1. `TMDB_API_KEY` for live TMDB enrichment in app profile lookups.

## Limitations
1. Model quality depends on TMDB coverage and data quality.
2. Targets are heavy-tailed and noisy; performance varies by split strategy.
3. Some app lookups rely on TMDB availability/network.
4. Causal-style analyses in notebooks are observational and not causal proof.

## Next Improvements
1. Add strict temporal CV for all final model selection decisions.
2. Add uncertainty intervals/calibration in app output.
3. Version artifact schema to prevent model-metadata mismatch.
4. Add automated tests for feature-row construction parity between training and app inference.
