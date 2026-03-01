# The Next Blockbuster
Predicting Movie Success Using Pre-Release Data

Group 1: Laura Manzanos Zuriarrain, Maria Jose Beletanga, Emmanuel Okerein, Hank Shao, Ibukun Adeleye

This repository implements an end-to-end data science and ML workflow to estimate movie outcomes before release using TMDB-derived metadata.

Primary decision support goals:
1. Predict expected movie popularity.
2. Predict revenue tier (Low, Medium, High, Blockbuster) with semi-supervised learning.
3. Support pre-release what-if decisions in a Streamlit app.

---

**Context and Motivation**
- 80% of films do not have strong ROI despite high production costs (Forbes, 2019).
- This project helps production teams identify pre-release signals that drive popularity and revenue.

**Who Can Use This**
1. Studio Executive: budget and greenlight decisions.
2. Casting Director: cast shortlisting with ROI signals.
3. Independent Producer: reduce high-cost casting risk.
4. Production Data Analyst: scenario testing and decision support.

**Data Source**
- The Movie Database (TMDB) official API, 2010–2025.
- Entities: Movies, top-billed cast, directors.
- TMDB popularity as a proxy for audience interest.

**End-to-End Process**
1. Build a clean and enriched dataset with pre-release features.
2. Train and evaluate supervised models for popularity prediction.
3. Train and evaluate semi-supervised models for revenue-tier prediction.
4. Compare model performance with robust metrics.
5. Export best-performing models for integration into a decision-support app.

---

**Repository Structure**
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
│   ├── data_ssl_revenue.csv
│   └── ssl_model_comparison.csv
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

**Data Science Lifecycle**

**5.1 Framing the Problem**
Business problem:
1. Studios need pre-release guidance on likely movie performance.
2. Casting, timing, and marketing choices are expensive and high-risk.

ML framing:
1. Regression task: predict `popularity`.
2. Semi-supervised classification: predict revenue tier (`y_ssl`).

Decision framing:
1. If predicted popularity is high but revenue confidence is low, prioritize risk mitigation (budget control, release strategy).
2. If both popularity and revenue outlook are strong, prioritize marketing scale-up.

---

**5.2 Data Acquisition**
Main source:
1. TMDB API (movies, cast, crew, keywords, metadata).

Notebook:
1. `notebooks/DataExtraction.ipynb`

Output:
1. `data/movies_2010_2025.csv`

Scope:
1. Movies from 2010 to 2025.
2. Talent attributes (director + top cast), language, genres, release metadata, budget/revenue/popularity fields.

---

**5.3 Data Exploration**
Notebook:
1. `EDA/EDA.ipynb`

EDA summary:

1) Data overview
1. Total movies: 9290
2. Labeled rows (revenue known): 2604
3. Total features (modeling set): 52

2) Raw features
| Groups | Categorical variables | Numerical variables |
|---|---|---|
| Feature name | Title, Release date, Original language, Status, overview, Genres, Keywords, Director name, Director department, Actor(1-5) name, Actor(1-5) character | Runtime, Popularity, Vote average, Vote count, Budget, Revenue, Director id, Director gender, Director popularity, Actor(1-5) id, Actor(1-5) gender, Cast pop mean, Cast pop max |

3) Missing values
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

4) Distribution of log target variables
Revenue
<img width="515" height="402" alt="Image" src="https://github.com/user-attachments/assets/df0371d8-7b35-47ff-83a6-e1487734efff" />

Popularity
<img width="544" height="406" alt="Image" src="https://github.com/user-attachments/assets/8a3f3ece-52d7-4197-9ae8-f454e87d9f26" />

5) Correlation map
<img width="1004" height="703" alt="Image" src="https://github.com/user-attachments/assets/90f7900e-9ca6-4d44-a882-16c38b4270d5" />

6) Genres and talent ranking
Genres
<img width="592" height="401" alt="Image" src="https://github.com/user-attachments/assets/73a2e91b-f1eb-45cf-ac4c-e7b8cad0a9eb" />

Talent ranking rationale:
- Directors ranked by TMDB popularity validate `director_popularity` as a real pre-release signal.
- Actor popularity and appearance frequency capture distinct signals (star power vs prolificacy).

---

**5.4 Data Preparation**
Notebook:
1. `notebooks/FeatureEngineering.ipynb`

Guiding principles:
1. Target-agnostic master dataset: all features + raw targets kept together.
2. Zero correction: budget and revenue = 0 treated as missing, not true $0.
3. No data leakage: post-release metrics excluded from feature matrices.
4. Semi-supervised ready: unlabeled rows kept as `y = -1`.
5. Revenue tiers: quantile-based bins from labeled data only.

Feature categories:
1. Content: genre multi-hot encoding, keyword count, language flags.
2. Talent: cast & director popularity, star power, gender ratios.
3. Production: budget (log-transformed), runtime, overview signals.
4. Temporal: release month, year, quarter, summer/holiday flags.

Key engineered features:
1. `star_count`: number of actors above 75th-percentile popularity.
2. `cast_popularity_std`: spread of top-5 actor popularities.
3. `cast_gender_ratio`: percent female in top-5 billed cast.
4. `genre_*` (19 binary): multi-hot genre encoding.
5. `is_summer_release`, `is_holiday_release`: seasonal blockbuster flags.
6. `log_budget`: log1p-transformed budget (missing treated as unknown, not $0).

Generated datasets:
1. `data/data_features_master.csv`
2. `data/data_supervised_popularity.csv`
3. `data/data_supervised_revenue.csv`
4. `data/data_ssl_revenue.csv`

---

**5.5 Modeling**
Popularity notebook:
1. `models/PopularityModelComparison.ipynb`

Revenue-tier notebook:
1. `models/SemiSupervisedModels_Final.ipynb`

Regression models:
1. Dummy, Linear Regression, RidgeCV.
2. Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting.
3. XGBoost, LightGBM (availability-dependent).

Semi-supervised models:
1. Self Training with tuned Random Forest base estimator.
2. Label Spreading (KNN graph).
3. Label Propagation (KNN graph).

---

**5.6 Model Evaluation**
Evaluation setup (popularity):
1. Train/holdout split (80/20 on labeled data only).
2. Cross-validation + repeated CV stability checks.
3. Metrics: RMSE, MAE, R2, plus diagnostics in notebook.
4. Ablation: raw target vs `log1p(popularity)` with back-transform (`expm1`) for comparable scale.

Evaluation setup (revenue tier):
1. **60% train / 20% validation / 20% test** split on labeled data only (stratified).
2. Scaler and PCA fitted on train only — applied to val, test, and unlabeled.
3. Model selection based on **validation** Macro F1; test set evaluated once at the end.
4. Metrics: macro F1 (primary), accuracy, precision, recall.
5. Confusion matrix diagnostics.

---

**5.7 Model Selection**
Selection logic implemented in exporter:
1. `models/export_best_models.py`

Current exported models used by app:
1. Popularity model: Gradient Boosting with `log1p` target transform.
2. Revenue-tier model: best available SSL model from `SemiSupervisedModels_Final`.

Exported artifacts:
1. `models/popularity_best_model.pkl`
2. `models/ssl_best_model.pkl`
3. `models/ssl_scaler.pkl`
4. `models/model_metadata.pkl`

---

**5.8 Model Fine-Tuning**
In `models/PopularityModelComparison.ipynb`:
1. Hyperparameter tuning blocks exist.
2. Final model section includes repeated CV and targeted fine-tuning for the selected best log-target model.
3. SHAP explainability block added for selected final model.

Cross-validation and fine tuning:
1. CV used in model comparison (KFold, 3 folds) with CV_RMSE.
2. Repeated CV (5x2) stability check on the selected final log-target model.
3. RandomizedSearchCV used for XGBoost.

---

**5.9 Solution Presentation**

**Context**
The solution is a decision-support pipeline for pre-release movie planning.

**Hypothesis**
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

**Data**
Data sources and transformations are documented in:
1. `notebooks/DataExtraction.ipynb`
2. `notebooks/FeatureEngineering.ipynb`
3. EDA section above (before/after quality profile)

**Model**
Type:
1. Supervised regression for popularity.
2. Semi-supervised classification for revenue tier.

Modelling Approach:
1. Multi-model benchmarking.
2. Leakage-safe features.
3. Target transform ablation.
4. Final model retraining and export for serving.

Model Evaluation:
1. Train/validation/holdout strategy with CV checks.
2. Metrics aligned to task and business interpretability.

**Results (from notebooks in `models/`)**
Popularity prediction (regression):
- Best raw-target model: XGBoost.
- Best final model: Gradient Boosting with `log1p(popularity)`.

Holdout metrics (original popularity scale):
| Setting | Model | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| Best raw target | XGBoost | 4.1853 | 1.6265 | 0.3142 |
| Best log1p target (final) | Gradient Boosting | 3.5067 | 1.4196 | 0.5186 |

Revenue tier prediction (semi-supervised):
- Best model: SelfTraining (SSL) with tuned Random Forest base estimator.
- Primary metric: Macro F1 on held-out labeled test set.

Comparison (held-out labeled test set):
| Model | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| SelfTraining (SSL, tuned) | 0.6238 | 0.6296 | Pseudo-labeled: 5974 samples; threshold=0.7 |
| RandomForest (supervised, tuned) | 0.6046 | 0.6048 | Supervised baseline |
| GradientBoosting (supervised, tuned) | 0.6027 | 0.6024 | Supervised baseline |
| LabelSpreading (SSL, tuned) | 0.5470 | 0.5420 | Graph SSL |
| LabelPropagation (SSL, tuned) | 0.5278 | 0.5277 | Graph SSL |

**Explainability of Results**
Explainability included in popularity notebook:
1. Global feature importance plots.
2. SHAP summary plots (final selected model).

**Important Visualizations**
Popularity distribution
![Popularity distribution](docs/figures/PopularityModelComparison_files/PopularityModelComparison_7_0.png)

Model comparison snapshot
![Model comparison](docs/figures/PopularityModelComparison_files/PopularityModelComparison_15_2.png)

Residual diagnostics
![Residual diagnostics](docs/figures/PopularityModelComparison_files/PopularityModelComparison_18_0.png)

Target transform ablation (raw vs log1p)
![Target transform ablation](docs/figures/PopularityModelComparison_files/PopularityModelComparison_32_4.png)

Final model SHAP summary
![Final model SHAP](docs/figures/PopularityModelComparison_files/PopularityModelComparison_40_1.png)

Best SSL confusion matrix
![SSL confusion matrix](docs/figures/SemiSupervisedModels_Final_files/best_ssl_confusion_matrix.png)

**Threats to Validity**

Data & Distribution
- TMDB may not represent all markets or distribution channels.
- Revenue missingness may be systematic, affecting SSL assumptions.
- Temporal shifts may reduce future generalization.

Modeling & Evaluation
- Popularity and revenue distributions are skewed.
- Model ranking depends on validation strategy.

Leakage & Feature Risk
- Latent post-release proxies may remain despite filtering.

Interpretation & Deployment
- Results reflect associations, not causal effects.
- Production deployment requires monitoring and drift control.

**Conclusion**
- Pre-release metadata contains meaningful predictive signal for both popularity and revenue tier.
- Supervised models significantly outperform baseline error for popularity prediction.
- Semi-supervised Self Training achieved the best F1 for revenue tiers, improving over the best supervised model.
- Leveraging unlabeled data added measurable value when pseudo-labeling was carefully controlled.

Business Insight
- Release timing, budget-related proxies, and talent popularity are consistently strong predictors.
- Results provide interpretable signals for greenlighting, marketing, and release strategy decisions.

Next Steps
- Improve temporal robustness.
- Incorporate more pre-release variables such as marketing intensity, distribution scope, and social signals.
- Add temporal CV as required model-selection criterion.
- Add uncertainty intervals and drift monitoring in production.

**Lessons Learned**
- Semi-supervised learning adds value when labels are scarce.
- Feature engineering matters more than model complexity alone.
- AI is a tool, not a replacement for reasoning.

---

**5.10 Launching, Monitoring and Maintenance**
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

**App Overview**
Main app file:
1. `app/app_final.py`

Current app outputs:
1. Predicted popularity with percentile context.
2. Revenue outlook with confidence.
3. Actor/director popularity chart.
4. TMDB known-for movie panels (with fallback to dataset summaries).

---

**Key Files**
1. `notebooks/DataExtraction.ipynb`: TMDB ingestion.\n2. `notebooks/FeatureEngineering.ipynb`: feature pipeline.\n3. `models/PopularityModelComparison.ipynb`: supervised modeling.\n4. `models/SemiSupervisedModels_Final.ipynb`: SSL modeling (60/20/20 train/val/test).\n5. `models/export_best_models.py`: model export for app usage.
