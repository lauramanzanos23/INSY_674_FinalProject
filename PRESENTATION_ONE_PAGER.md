# TMDB Movie Success Prediction - One-Page Presentation Summary

## Project Goal
Build a pre-release decision-support system that predicts:
1. Popularity (regression).
2. Revenue tier (Low / Medium / High / Blockbuster, semi-supervised classification).

## Business Value
1. Supports greenlighting, casting, budget, and release strategy decisions before launch.
2. Quantifies expected performance using pre-release signals only.
3. Enables what-if scenario testing in an interactive Streamlit app.

## Data and Scope
1. Source dataset: `data/movies_2010_2025.csv` (TMDB-derived, 2010-2025).
2. Feature pipeline outputs:
   - `data/data_features_master.csv`
   - `data/data_supervised_popularity.csv`
   - `data/data_supervised_revenue.csv`
   - `data/data_ssl_revenue.csv`
3. Labeled rows are used for supervised training; unlabeled revenue rows are retained for SSL (`y_ssl = -1`).

## End-to-End Pipeline
1. Data extraction and enrichment: `notebooks/DataExtraction.ipynb`
2. Leakage-aware feature engineering: `notebooks/FeatureEngineering.ipynb`
3. Popularity model benchmarking: `models/PopularityModelComparison.ipynb`
4. Semi-supervised revenue-tier modeling: `models/SemiSupervisedModels_Final.ipynb`
5. Export and packaging: `models/export_best_models.py`
6. Final app: `app/app_final.py`

## Modeling Approach
### Popularity Regression
1. Compared baseline, linear, tree, and boosting models.
2. Validation included holdout, K-Fold CV, repeated CV, and target-transform ablation (`raw` vs `log1p`).
3. Final exported policy uses Gradient Boosting trained on `log1p(popularity)`.

### Revenue Tier (SSL)
1. Built mixed labeled/unlabeled training setup with leakage prevention.
2. Used labeled-only 60/20/20 train/validation/test split.
3. Compared supervised and SSL variants; selected by Macro F1 (accuracy as tiebreaker).

## Key Results
### Popularity
1. Best raw-target holdout model: XGBoost
   - RMSE `4.185`, MAE `1.626`, R2 `0.314`
2. Best log-target setting: Gradient Boosting + `log1p(popularity)`
   - RMSE `3.507`, MAE `1.420`, R2 `0.519`
3. XGBoost fine-tuning improved baseline slightly (RMSE `4.149` vs `4.185`).

### Revenue Tier (Validation Snapshot)
1. Best model: SelfTraining (SSL, tuned)
   - Accuracy `0.6238`, Macro F1 `0.6296`
2. Baseline tuned supervised alternatives were lower:
   - RandomForest: Accuracy `0.6046`, Macro F1 `0.6048`
   - GradientBoosting: Accuracy `0.6027`, Macro F1 `0.6024`

## Final App (Delivered)
`app/app_final.py` provides:
1. Director and cast scenario builder with lock controls.
2. Inputs for genre, language, runtime, release timing, budget, and text proxies.
3. Outputs:
   - predicted popularity,
   - predicted revenue tier,
   - class confidence,
   - composite casting fit score.
4. TMDB known-for enrichment with resilient local fallback behavior.

## Final Artifacts
1. Popularity model: `models/popularity_best_model.pkl`
2. Revenue-tier model: `models/ssl_best_model.pkl`
3. Scaler: `models/ssl_scaler.pkl`
4. Metadata: `models/model_metadata.pkl`

## Risks and Next Steps
1. Add mandatory temporal CV to reduce split and drift sensitivity.
2. Add uncertainty outputs (intervals/calibration) in the app.
3. Add automated feature-parity and artifact compatibility tests.
4. Expanding the dataset by collecting more movies with complete revenue information, or integrating additional databases to fill missing revenue values and reduce bias from incomplete records.

