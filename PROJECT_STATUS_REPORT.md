# Project Status Report — TMDB Movie Success Prediction

**Report Date:** March 1, 2026  
**Repository:** INSY 674 FinalProject  
**Course:** INSY 674

## 1. Current Project State
The project is in a complete end-to-end state with:
1. Data extraction and enrichment pipeline.
2. Leakage-aware feature engineering.
3. Popularity regression model comparison and selection.
4. Semi-supervised revenue-tier modeling with train/validation/test discipline.
5. Exported production artifacts in `models/`.
6. Final Streamlit app at `app/app_final.py` using exported models.

## 2. Delivered Scope
### 2.1 Data Assets
1. Source dataset: `data/movies_2010_2025.csv`
2. Engineered datasets:
   - `data/data_cleaned_engineered.csv`
   - `data/data_features_master.csv`
   - `data/data_supervised_popularity.csv`
   - `data/data_supervised_revenue.csv`
   - `data/data_ssl_revenue.csv`
3. SSL experiment outputs:
   - `data/ssl_model_comparison.csv`
   - `data/best_ssl_model.joblib`
   - `data/ssl_scaler.joblib`

### 2.2 Modeling Assets
1. Popularity modeling notebook: `models/PopularityModelComparison.ipynb`
2. Final SSL notebook: `models/SemiSupervisedModels_Final.ipynb`
3. Export script: `models/export_best_models.py`
4. Exported app-ready artifacts:
   - `models/popularity_best_model.pkl`
   - `models/ssl_best_model.pkl`
   - `models/ssl_scaler.pkl`
   - `models/model_metadata.pkl`

### 2.3 Application
1. Final application entrypoint: `app/app_final.py`
2. Loads model artifacts from `models/`.
3. Provides scenario inputs (director/cast + genre/language + runtime/release/budget/text proxies).
4. Returns popularity prediction, revenue tier, confidence, and casting fit score.
5. Includes TMDB known-for enrichment with fallback handling.

## 3. Repository Structure (Current)
```text
INSY 674 FinalProject/
├── README.md
├── PROJECT_STATUS_REPORT.md
├── PRESENTATION_ONE_PAGER.md
├── EDA/
│   └── EDA.ipynb
├── notebooks/
│   ├── DataExtraction.ipynb
│   └── FeatureEngineering.ipynb
├── data/
│   ├── movies_2010_2025.csv
│   ├── data_cleaned_engineered.csv
│   ├── data_features_master.csv
│   ├── data_supervised_popularity.csv
│   ├── data_supervised_revenue.csv
│   ├── data_ssl_revenue.csv
│   ├── ssl_model_comparison.csv
│   ├── best_ssl_model.joblib
│   └── ssl_scaler.joblib
├── models/
│   ├── PopularityModelComparison.ipynb
│   ├── SemiSupervisedModels_Final.ipynb
│   ├── export_best_models.py
│   ├── popularity_best_model.pkl
│   ├── ssl_best_model.pkl
│   ├── ssl_scaler.pkl
│   └── model_metadata.pkl
└── app/
    └── app_final.py
```

## 4. Stage-by-Stage Completion
### 4.1 Data Extraction (`notebooks/DataExtraction.ipynb`)
1. Pulled movie data for 2010-2025.
2. Enriched records with cast, director, and keyword metadata.
3. Output: `data/movies_2010_2025.csv`.

### 4.2 Feature Engineering (`notebooks/FeatureEngineering.ipynb`)
1. Created pre-release feature set across talent, content, temporal, and production signals.
2. Treated zero budget/revenue values as missing with indicator handling.
3. Preserved unlabeled revenue rows for semi-supervised learning.
4. Produced supervised and SSL-specific datasets.
5. Enforced leakage control in model-ready feature sets.

### 4.3 Popularity Modeling (`models/PopularityModelComparison.ipynb`)
1. Compared baseline, linear, tree, and boosting regressors.
2. Evaluated with holdout and cross-validation.
3. Performed target ablation (`raw` vs `log1p(popularity)`).
4. Added fine-tuning and explainability analyses.

Key results:
1. Best raw-target holdout: XGBoost (`RMSE 4.185`, `MAE 1.626`, `R2 0.314`)
2. Best transformed setting: Gradient Boosting + `log1p(popularity)` (`RMSE 3.507`, `MAE 1.420`, `R2 0.519`)
3. XGBoost tuning improved RMSE from `4.185` to `4.149`

### 4.4 Revenue-Tier SSL Modeling (`models/SemiSupervisedModels_Final.ipynb`)
1. Constructed `y_ssl` target with unlabeled samples as `-1`.
2. Removed post-release leakage features for SSL training.
3. Used labeled-only 60% train / 20% validation / 20% test split.
4. Fit preprocessing on train only; applied to validation/test/unlabeled.
5. Compared supervised and SSL model families; selected using validation Macro F1.
6. Exported comparison results to `data/ssl_model_comparison.csv`.

Validation snapshot from exported comparison table:
1. SelfTraining (SSL, tuned): Accuracy `0.6238`, Macro F1 `0.6296`
2. RandomForest (supervised, tuned): Accuracy `0.6046`, Macro F1 `0.6048`
3. GradientBoosting (supervised, tuned): Accuracy `0.6027`, Macro F1 `0.6024`

### 4.5 Export and Packaging (`models/export_best_models.py`)
1. Trains final popularity model on full supervised popularity data using `log1p`.
2. Loads best SSL model/scaler from `data/` joblib artifacts.
3. Saves app-ready pickle artifacts and metadata in `models/`.
4. Stores feature lists, target-transform policy, and revenue-tier label mapping.

### 4.6 Application (`app/app_final.py`)
1. Loads `popularity_best_model.pkl`, `ssl_best_model.pkl`, `ssl_scaler.pkl`, and `model_metadata.pkl`.
2. Builds inference rows aligned to training feature schema.
3. Performs popularity back-transform (`expm1`) when target policy is `log1p`.
4. Produces revenue tier and confidence outputs for UI display.

## 5. Validation and Robustness Coverage
1. Holdout evaluation on labeled data for popularity.
2. Cross-validation and repeated CV checks in popularity modeling.
3. Hyperparameter search in both supervised and SSL workflows.
4. Target-transform ablation for heavy-tailed popularity target.
5. Leakage filtering for pre-release feature policy.
6. Separation of train/validation/test roles for SSL model selection.

## 6. Risks and Gaps
1. Temporal drift risk remains under random split evaluation.
2. Heavy-tailed outcomes still affect RMSE stability.
3. Inference quality depends on strict feature parity with metadata.
4. TMDB network/API failures can reduce enrichment quality (fallbacks mitigate impact).

## 7. Recommended Next Steps
1. Add mandatory temporal cross-validation in final selection criteria.
2. Add calibrated uncertainty outputs to app predictions.
3. Add automated tests for feature schema parity and artifact compatibility.
4. Add model/version registry conventions for reproducible deployment snapshots.
