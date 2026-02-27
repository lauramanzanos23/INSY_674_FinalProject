# TMDB Movie Success Prediction — One-Page Presentation Summary

## Project Goal
Build a pre-release decision-support system that predicts:
1. **Popularity** (regression).
2. **Revenue tier** (Low / Medium / High / Blockbuster, semi-supervised).

## Business Value
1. Supports casting and release strategy decisions before launch.
2. Quantifies likely performance using only pre-release information.
3. Enables rapid what-if scenario testing in an interactive app.

## Data and Scope
1. Source dataset: `data/movies_2010_2025.csv` (TMDB-derived, 2010-2025).
2. Feature pipeline generates:
   - `data_features_master.csv`
   - `data_supervised_popularity.csv`
   - `data_supervised_revenue.csv`
   - `data_ssl_revenue.csv`

## End-to-End Pipeline
1. **Data extraction** (`notebooks/DataExtraction.ipynb`).
2. **Feature engineering** (`notebooks/FeatureEngineering.ipynb`) with leakage controls.
3. **Popularity model comparison** (`models/PopularityModelComparison.ipynb`).
4. **Semi-supervised revenue modeling** (`models/SemiSupervisedModels_V2.ipynb`).
5. **Artifact export** (`models/export_best_models.py`).
6. **Production app** (`app/app_final.py`).

## Modeling Approach
### Popularity Regression
1. Compared baseline + linear + tree + boosting models.
2. Validation included holdout, K-Fold CV, repeated CV, ablation, and fine-tuning.

### Revenue Tier (SSL)
1. Built labeled/unlabeled framework (`y_ssl`, with unlabeled = `-1`).
2. Compared supervised and semi-supervised classifiers.
3. Selected best model by Macro F1 then Accuracy.

## Key Results
### Popularity
1. Best **raw-target** holdout model: **XGBoost**
   - RMSE `4.185`, MAE `1.626`, R2 `0.314`.
2. Best **log-target** setting: **Gradient Boosting + log1p(popularity)**
   - RMSE `3.507`, MAE `1.420`, R2 `0.519`.
3. Fine-tuned XGBoost improved over XGBoost baseline
   - RMSE `4.149` vs `4.185`.

### Explainability
1. SHAP/importance consistently highlighted:
   - `release_year`
   - `log_budget`
   - `keyword_count`
   - talent and release-timing features.

## Final App (Delivered)
`app/app_final.py` provides:
1. Director + cast selection and locking.
2. Scenario controls (genre, language, runtime, release timing, budget, text proxies).
3. Outputs:
   - predicted popularity,
   - predicted revenue tier,
   - confidence,
   - composite casting fit score.
4. TMDB known-for enrichment + local fallback context.

## Deliverables
1. Documentation: `README.md`, `PROJECT_STATUS_REPORT.md`.
2. Exported models: `popularity_best_model.pkl`, `ssl_best_model.pkl`, `ssl_scaler.pkl`, `model_metadata.pkl`.
3. Reproducible notebooks and deployable Streamlit app.

## Limitations and Next Steps
1. Heavy-tailed targets create split sensitivity; enforce stronger temporal validation.
2. Add prediction intervals/calibration to app outputs.
3. Add automated tests for training/inference feature parity and artifact versioning.
