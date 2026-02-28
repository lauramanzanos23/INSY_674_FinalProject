# TMDB Movie Success Prediction Project

This repository implements an end-to-end data science and ML workflow to estimate movie outcomes before release using TMDB-derived metadata.

Primary decision support goals:
1. Estimate expected movie popularity.
2. Estimate likely revenue tier (Low, Medium, High, Blockbuster).
3. Support pre-release what-if decisions in an interactive Streamlit app.

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
│   ├── movies_2010_2025.csv
│   ├── data_features_master.csv
│   ├── data_supervised_popularity.csv
│   ├── data_supervised_revenue.csv
│   └── data_ssl_revenue.csv
├── models/
│   ├── PopularityModelComparison.ipynb
│   ├── SemiSupervisedModels.ipynb
│   ├── SemiSupervisedModels_V2.ipynb
│   ├── export_best_models.py
│   ├── popularity_best_model.pkl
│   ├── ssl_best_model.pkl
│   ├── ssl_scaler.pkl
│   └── model_metadata.pkl
├── app/
│   ├── app_final.py
│   └── app_mockup.py
├── docs/
│   └── figures/
└── requirements.txt
```

---

## 5. Data Science Lifecycle

### 5.1 Framing the Problem
Business problem:
1. Studios need pre-release guidance on likely movie performance.
2. Casting and release choices are expensive and high-risk.

ML framing:
1. Regression task: predict `popularity`.
2. Semi-supervised classification task: predict revenue tier (`y_ssl`).

Decision framing:
1. If predicted popularity is high but revenue confidence is low, prioritize risk mitigation (budget control, release strategy).
2. If both popularity and revenue outlook are strong, prioritize marketing scale-up.

### 5.2 Data Acquisition
Main source:
1. TMDB API (movies, cast, crew, keywords, metadata).

Acquisition notebook:
1. `notebooks/DataExtraction.ipynb`

Output:
1. `data/movies_2010_2025.csv`

Scope:
1. Movies from 2010 to 2025.
2. Talent attributes (director + top cast), language, genres, release metadata, budget/revenue/popularity fields.

### 5.3 Data Exploration
Restored EDA summary (as originally documented):

#### 1) Data Overview
1. Total movies: 9290
2. Labeled rows (revenue known): 2604
3. Total features (modeling set): 52

#### 2) Raw features
| Groups | Categorical variables | Numerical variables |
|---|---|---|
| Feature name | Title, Release date, Original language, Status, overview, Genres, Keywords, Director name, Director department, Actor(1-5) name, Actor(1-5) character | Runtime, Popularity, Vote average, Vote count, Budget, Revenue, Director id, Director gender, Director popularity, Actor(1-5) id, Actor(1-5) gender, Cast pop mean, Cast pop max |

#### 3) Missing Values
| Feature name | Missing values |
|---|---:|
| runtime | 457 |
| vote_average | 1642 |
| vote_count | 1640 |
| budget | 6527 |
| revenue | 6686 |
| cast_pop_mean | 222 |
| cast_pop_max | 222 |
| director_gender | 2285 |
| actor1 missing | 225 |
| actor2 missing | 384 |
| actor3 missing | 681 |
| actor4 missing | 1036 |
| actor5 missing | 1479 |
| genres | 489 |
| keywords | 3238 |

#### 4) Distribution of log target variables
Revenue
<img width="515" height="402" alt="Image" src="https://github.com/user-attachments/assets/df0371d8-7b35-47ff-83a6-e1487734efff" />

Popularity
<img width="544" height="406" alt="Image" src="https://github.com/user-attachments/assets/8a3f3ece-52d7-4197-9ae8-f454e87d9f26" />

#### 5) Correlation map
<img width="1004" height="703" alt="Image" src="https://github.com/user-attachments/assets/90f7900e-9ca6-4d44-a882-16c38b4270d5" />

#### 6) Genres and talent ranking
Genres
<img width="592" height="401" alt="Image" src="https://github.com/user-attachments/assets/73a2e91b-f1eb-45cf-ac4c-e7b8cad0a9eb" />

Top directors by popularity (sample from EDA):
| Rank | Director | Popularity |
|---|---|---:|
| 1 | Jackie Chan | 20.8778 |
| 2 | Tom Hanks | 15.8630 |
| 3 | Ben Affleck | 12.3544 |
| 4 | Angelina Jolie | 10.9173 |
| 5 | Sylvester Stallone | 10.5039 |

Top actors by popularity (sample from EDA):
| Rank | Actor | Popularity |
|---|---|---:|
| 1 | Kayden Kross | 224.1500 |
| 2 | Evelyn Claire | 145.2190 |
| 3 | Chanel Preston | 88.1540 |
| 4 | Akiho Yoshizawa | 85.4980 |
| 5 | Rosa Caracciolo | 75.3990 |

### 5.4 Data Preparation
Feature engineering notebook:
1. `notebooks/FeatureEngineering.ipynb`

Preparation strategy:
1. Parse and standardize genres/languages.
2. Build cast and director aggregate features.
3. Add release timing features (`release_month`, `release_quarter`, seasonal flags).
4. Add missingness indicators for budget/revenue and safe budget transforms (`has_budget`, `log_budget`).
5. Build text-derived proxies (`keyword_count`, `has_overview`, `overview_length`).
6. Enforce leakage-safe feature sets for pre-release prediction.

Generated datasets:
1. `data/data_features_master.csv`
2. `data/data_supervised_popularity.csv`
3. `data/data_supervised_revenue.csv`
4. `data/data_ssl_revenue.csv`

### 5.5 Modeling
Popularity notebook:
1. `models/PopularityModelComparison.ipynb`

Revenue-tier notebook:
1. `models/SemiSupervisedModels_V2.ipynb`

Model families used:
1. Regression: Dummy, linear, ridge, random forest, extra trees, gradient boosting, hist gradient boosting, XGBoost, LightGBM (availability-dependent).
2. Semi-supervised classification: supervised baselines + SSL approaches on partially labeled target.

### 5.6 Model Evaluation
Evaluation setup (popularity):
1. Train/holdout split (80/20).
2. Cross-validation + repeated CV stability checks.
3. Metrics: RMSE, MAE, R2 (plus additional diagnostics in notebook).
4. Ablation: raw target vs `log1p(popularity)` with back-transform (`expm1`) for comparable scale.

Evaluation setup (revenue tier):
1. Class metrics with focus on macro quality (e.g., Macro F1).
2. Confusion matrix diagnostics.
3. Selection among SSL candidates by performance and robustness.

### 5.7 Model Selection
Selection logic implemented in exporter:
1. `models/export_best_models.py`

Current exported models used by app:
1. Popularity model: `Gradient Boosting` with `log1p` target transform.
2. Revenue-tier model: best available SSL model from `SemiSupervisedModels_V2` artifacts.

Exported artifacts:
1. `models/popularity_best_model.pkl`
2. `models/ssl_best_model.pkl`
3. `models/ssl_scaler.pkl`
4. `models/model_metadata.pkl`

### 5.8 Model Fine-Tuning
In `PopularityModelComparison.ipynb`:
1. Hyperparameter tuning blocks exist (including advanced searches in prior sections).
2. Final model section includes dedicated repeated CV and targeted fine-tuning for the selected best log-target model.
3. SHAP explainability block added for selected final model.

---

## 5.9 Solution Presentation

### 5.9.1 Context
The solution is a decision-support pipeline for pre-release movie planning.

### 5.9.2 Hypothesis
Predictive hypotheses:
1. H1: Pre-release talent, content, timing, and production signals are predictive of popularity.
2. H2: Semi-supervised learning improves revenue-tier prediction when labeled revenue is limited.

Causal hypothesis (budget treatment):
1. H3: Budget-related treatment has positive effect on popularity under overlap/ignorability assumptions.

Null hypotheses:
1. H0 (predictive): models do not outperform baseline predictors materially.
2. H0 (causal): average treatment effect (ATE) is zero.

Outcomes and error types:
1. Regression errors: under/overprediction of popularity.
2. Classification errors: false optimism (predict high tier when low) and false pessimism.

Statistical communication:
1. Causal section reports uncertainty via confidence intervals (DR estimate).
2. P-values are not the primary model-selection criterion; predictive metrics and robust CV are primary.
3. Multiple testing / p-hacking mitigation: holdout discipline, repeated CV, and explicit ablations.

### 5.9.3 Data
Data sources and transformations are documented in:
1. `notebooks/DataExtraction.ipynb`
2. `notebooks/FeatureEngineering.ipynb`
3. EDA section above (before/after quality profile)

### 5.9.4 Model
Type:
1. Supervised regression for popularity.
2. Semi-supervised classification for revenue tier.

Modelling approach:
1. Multi-model benchmarking.
2. Leakage-safe features.
3. Target transform ablation.
4. Final model retraining and export for serving.

Model evaluation:
1. Train/validation/holdout strategy with CV checks.
2. Metrics aligned to task and business interpretability.

### 5.9.5 Results
Notebook figure highlights:

Popularity distribution
![Popularity distribution](docs/figures/PopularityModelComparison_files/PopularityModelComparison_7_0.png)

Model comparison snapshot
![Model comparison](docs/figures/PopularityModelComparison_files/PopularityModelComparison_15_2.png)

Residual diagnostics
![Residual diagnostics](docs/figures/PopularityModelComparison_files/PopularityModelComparison_18_0.png)

Target transform ablation (raw vs log)
![Target transform ablation](docs/figures/PopularityModelComparison_files/PopularityModelComparison_32_4.png)

Semi-supervised confusion matrix
![SSL confusion matrix](docs/figures/SemiSupervisedModels_V2_files/SemiSupervisedModels_V2_41_1.png)

### 5.9.6 Explainability of Results
Explainability included in popularity notebook:
1. Global feature importance plots.
2. SHAP summary plots (including final selected model block).

SHAP figure (final section):
![Final model SHAP](docs/figures/PopularityModelComparison_files/PopularityModelComparison_40_1.png)

### 5.9.7 Threats to Validity
Main risks:
1. Data quality and missingness in TMDB-derived fields.
2. Non-stationarity over time (market changes).
3. Semi-supervised label propagation bias.
4. Causal assumptions may be violated in observational data.

### 5.9.8 Conclusion
Project demonstrates practical predictive signal in pre-release features and deploys it in a usable app.

Current production-facing setup uses:
1. Popularity regression with log-target handling.
2. Semi-supervised revenue-tier classification.
3. Interpretable UI outputs with confidence context.

### 5.9.9 Lessons Learned and Next Steps
Lessons learned:
1. Validation strategy can change model ranking materially.
2. Target transforms (`log1p`) improve stability on heavy-tailed outcomes.
3. Leakage control is critical for realistic pre-release inference.

Next steps:
1. Add strict temporal cross-validation for final selection.
2. Add calibration and uncertainty intervals in UI.
3. Add automated parity tests between training features and app feature builder.
4. Track artifact/model versions with reproducible model registry metadata.

---

### 5.10 Launching, Monitoring and Maintenance
Launch:
1. Export models via `python models/export_best_models.py`.
2. Run app via `streamlit run app/app_final.py`.
3. Deploy on Streamlit Cloud with `requirements.txt` and `runtime.txt`.

Monitoring recommendations:
1. Data drift: monitor feature distributions vs training baseline.
2. Prediction drift: monitor percentile shifts in predicted popularity and revenue-tier frequencies.
3. Performance drift: periodically backtest on newly released movies.
4. Reliability: log model load/version metadata and prediction errors.

Maintenance plan:
1. Scheduled retraining cadence (e.g., quarterly or semi-annually).
2. Trigger retraining on drift thresholds.
3. Keep dependency versions pinned for artifact compatibility.
4. Keep notebook-to-export parity checks in CI.

---

## App Overview
Main app file:
1. `app/app_final.py`

Current app outputs:
1. Predicted popularity with percentile context.
2. Revenue outlook with confidence.
3. Actor/director popularity chart.
4. TMDB known-for movie panels (with fallback to dataset summaries).

## How to Run
1. Export/update model artifacts:
```bash
python models/export_best_models.py
```

2. Start Streamlit app:
```bash
streamlit run app/app_final.py
```

3. Reproduce notebooks:
1. `notebooks/DataExtraction.ipynb`
2. `notebooks/FeatureEngineering.ipynb`
3. `models/PopularityModelComparison.ipynb`
4. `models/SemiSupervisedModels_V2.ipynb`
