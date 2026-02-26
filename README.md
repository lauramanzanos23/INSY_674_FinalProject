# TMDB Movie Success Prediction Project

## Executive Summary
1. Built an end-to-end TMDB-based pipeline to predict pre-release movie popularity (regression) and revenue tier (classification).
2. Popularity modeling found strong signal vs baseline: best holdout model was XGBoost (`RMSE 4.185` vs baseline `5.061`, about 17% lower error).
3. Stability checks showed split sensitivity: repeated CV favored Random Forest/Gradient Boosting over XGBoost, so reliability depends on validation strategy.
4. Revenue-tier modeling showed semi-supervised learning value: SelfTraining (SSL) was best (`Macro F1 0.5486`, `Accuracy 0.5547`), outperforming supervised baselines.
5. Explainability highlighted release timing, budget-related features, and talent popularity as major drivers; causal estimates suggest a positive `has_budget` association but are observational, not causal proof.

## Overview
This repository is an end-to-end movie analytics project built on TMDB-style data to support pre-release decision making.

The project does five core things:
1. Builds cleaned, engineered datasets from movie metadata.
2. Predicts numeric movie popularity with supervised regression.
3. Predicts revenue tiers with semi-supervised classification.
4. Explains model behavior (feature importance + SHAP).
5. Provides an interactive Streamlit app for scenario testing (cast/director/release setup).

## Business Questions
1. How popular is a movie likely to be before release?
2. Which revenue tier is it likely to reach (Low/Medium/High/Blockbuster)?
3. Which controllable factors (cast, timing, budget proxy variables) move those predictions?
4. Does budget availability (`has_budget`) appear to be associated with higher popularity under observational assumptions?

## End-to-End Pipeline
1. Data extraction and enrichment from TMDB-oriented sources.
2. Data cleaning and feature engineering with pre-release constraints.
3. Supervised popularity modeling and comparison.
4. Semi-supervised revenue-tier modeling and comparison.
5. Explainability and causal sensitivity analysis.
6. Packaging best models for app inference.

Primary notebooks:
1. `notebooks/DataExtraction.ipynb`
2. `notebooks/FeatureEngineering.ipynb`
3. `models/PopularityModelComparison.ipynb`
4. `models/SemiSupervisedModels.ipynb`

## Data Artifacts
Main artifacts in `data/`:
1. `movies_2010_2025.csv`: enriched movie-level source table.
2. `data_cleaned_engineered.csv`: cleaned/engineered base table.
3. `data_features_master.csv`: feature master table.
4. `data_supervised_popularity.csv`: supervised regression dataset.
5. `data_ssl_revenue.csv`: semi-supervised revenue-tier dataset (`y_ssl` includes unlabeled rows as `-1`).
6. `data_supervised_revenue.csv`: supervised revenue table.

SSL output artifacts:
1. `best_ssl_model.joblib`
2. `ssl_scaler.joblib`
3. `ssl_model_comparison.csv`
4. `best_ssl_confusion_matrix.png`

## Features Used
Features are designed to be available pre-release:
1. Talent: director and top-cast popularity, cast aggregates, star count.
2. Content/context: genres, language flags, keywords, overview length.
3. Timing: release month/year/quarter, seasonality indicators.
4. Production: runtime and budget-derived indicators.

Leakage controls are explicitly applied, especially in SSL where columns containing `vote`, `review`, `rating`, `popularity`, `revenue`, or `budget` are removed from candidate features.

## Modeling Track 1: Popularity Regression
Notebook: `models/PopularityModelComparison.ipynb`

### Models Compared
1. Dummy Mean (baseline)
2. Linear Regression
3. RidgeCV
4. Random Forest
5. Extra Trees
6. Gradient Boosting
7. Hist Gradient Boosting
8. XGBoost (if installed)
9. LightGBM (if installed)

### Evaluation Setup
1. Train/test split: 80/20 holdout.
2. Metrics: MAE, RMSE, R2, MedAE, MAPE, Explained Variance, Max Error, NRMSE.
3. Cross-validation: 3-fold CV RMSE for comparison.
4. Additional robustness: repeated CV (5 folds x 2 repeats) on top holdout models.

### Key Results (from current notebook run)
Holdout best model by RMSE: `XGBoost`.

Top holdout RMSE:
1. XGBoost: `RMSE 4.185`, `MAE 1.626`, `R2 0.314`
2. Gradient Boosting: `RMSE 4.431`, `R2 0.231`
3. Random Forest: `RMSE 4.439`, `R2 0.228`
4. Dummy Mean baseline: `RMSE 5.061`, `R2 -0.003`

Interpretation:
1. The model clearly beats baseline error, so pre-release features contain predictive signal.
2. Absolute fit remains moderate (`R2 ~0.31`), meaning much variance is still unexplained.
3. Target distribution is right-skewed with extreme outliers (max popularity `378`), which inflates RMSE and creates split sensitivity.

Repeated CV on top holdout candidates (stability check):
1. Random Forest: `CV_RMSE_Mean 7.165`
2. Gradient Boosting: `CV_RMSE_Mean 7.312`
3. XGBoost: `CV_RMSE_Mean 7.630`, `CV_R2_Mean -0.020`

Interpretation:
1. XGBoost wins on the single holdout split but is less stable under repeated CV.
2. Random Forest / Gradient Boosting appear more robust across folds.
3. For production-like reliability, model selection should include stability criteria, not only one split.

## Explainability (Popularity Model)
Notebook section includes:
1. Feature importance/coefficients for best model type.
2. SHAP analysis (with XGBoost-safe native contribution fallback).

Top SHAP features from current run:
1. `release_year`
2. `log_budget`
3. `keyword_count`
4. `revenue_missing_flag`
5. `release_month`
6. Actor/director popularity aggregates (`actor1_popularity`, `director_popularity`, `cast_pop_max`, etc.)

Interpretation:
1. Temporal effects and budget-related information are major drivers.
2. Talent quality and metadata richness also contribute materially.

## Observational Causal Analysis (Popularity Notebook)
Treatment: `has_budget` (1 vs 0), using training split only.

Estimators:
1. IPW ATE
2. Doubly Robust (AIPW-style) ATE with CI

Current run:
1. IPW ATE: `+0.403` popularity points
2. Doubly Robust ATE: `+0.721` popularity points
3. DR 95% CI: `[0.282, 1.161]`
4. Propensity range: `[0.004, 0.999]`

Interpretation and caveat:
1. Estimated association is positive, but this is observational, not experimental causality.
2. Logistic propensity model showed a convergence warning, so estimates should be treated as sensitivity evidence, not causal proof.

## Modeling Track 2: Semi-Supervised Revenue Tier Classification
Notebook: `models/SemiSupervisedModels.ipynb`

### Target Definition
`y_ssl` classes:
1. `0` = Low
2. `1` = Medium
3. `2` = High
4. `3` = Blockbuster
5. `-1` = unlabeled rows used for SSL training only

### Models Compared
1. GradientBoosting (supervised baseline)
2. RandomForest (supervised baseline)
3. SelfTraining (SSL)
4. LabelSpreading (SSL, graph-based)
5. LabelPropagation (SSL, graph-based)

### Evaluation Rule
1. Primary metric: Macro F1.
2. Tiebreaker: Accuracy.
3. Held-out labeled test set is excluded from pseudo-labeling.

### Key Results (from `data/ssl_model_comparison.csv`)
1. SelfTraining (SSL): `Accuracy 0.5547`, `Macro F1 0.5486` (best)
2. RandomForest (supervised): `Accuracy 0.5240`, `Macro F1 0.5197`
3. GradientBoosting (supervised): `Accuracy 0.5067`, `Macro F1 0.5054`
4. LabelSpreading (SSL): `Accuracy 0.4280`, `Macro F1 0.4271`
5. LabelPropagation (SSL): `Accuracy 0.4261`, `Macro F1 0.4237`

Interpretation:
1. SSL helped when using SelfTraining, outperforming both supervised baselines.
2. Graph-based SSL underperformed in this feature space/configuration.
3. Pseudo-labeling with a tree base learner appears to transfer unlabeled information better than KNN graph propagation here.

## Model Packaging and Serving
Export script: `models/export_best_models.py`

Script outputs in `models/`:
1. `popularity_best_model.pkl` (XGBoost pipeline retrained on full popularity dataset)
2. `ssl_best_model.pkl`
3. `ssl_scaler.pkl`
4. `model_metadata.pkl` (feature lists + tier labels)

## Streamlit App
App: `app/app_mockup2.py`

What it does:
1. Loads packaged popularity and SSL models.
2. Recreates feature vectors from user-entered movie setup.
3. Scores popularity and revenue-tier outcomes.
4. Supports what-if testing for cast/director and release context.
5. Displays baseline vs scenario deltas and class-probability diagnostics.
6. Retrieves top movies for selected talent using TMDB first, local fallback second.

TMDB settings:
1. `TMDB_API_KEY` read from environment (fallback key exists in code).
2. Base URL: `https://api.themoviedb.org/3`.

## How to Run
1. Refresh packaged models:
```bash
python models/export_best_models.py
```

2. Launch app:
```bash
streamlit run app/app_mockup2.py
```

3. Reproduce notebook analyses:
1. Run `models/PopularityModelComparison.ipynb`
2. Run `models/SemiSupervisedModels.ipynb`

## Repository Layout
1. `app/`: Streamlit application code.
2. `data/`: datasets and saved model outputs.
3. `models/`: modeling notebooks, export script, and packaged artifacts.
4. `notebooks/`: extraction and feature engineering notebooks.
5. `EDA/`: exploratory analysis notebooks.
6. `src/`: support scripts (if present).

## Limitations
1. Results depend on TMDB data quality and coverage.
2. Popularity target is heavy-tailed, increasing variance and split sensitivity.
3. Causal section is observational and assumes no unobserved confounding.
4. App predictions are decision-support signals, not guarantees.

## Recommended Next Steps
1. Add temporal cross-validation to reduce optimism from random splits.
2. Tune and calibrate the SSL classifier probabilities.
3. Add regression prediction intervals and uncertainty flags in the app.
4. Improve causal robustness checks (alternative propensity/outcome models, trimming/sensitivity analysis).
5. Move TMDB API key fully to environment-only configuration.
