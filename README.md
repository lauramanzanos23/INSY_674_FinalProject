# TMDB Movie Success Prediction Project

This repository implements an end-to-end data science and ML workflow to estimate movie outcomes before release using TMDB-derived metadata.

Primary decision support goals:
1. Predict expected movie popularity.
2. Predict revenue tier (Low, Medium, High, Blockbuster) with semi-supervised learning.
3. Support pre-release what-if decisions in a Streamlit app.

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

**Project Steps (End-to-End)**
1. **Data extraction**: Pull TMDB metadata. Notebook: `notebooks/DataExtraction.ipynb`.
2. **Feature engineering**: Build leakage-safe, pre-release features. Notebook: `notebooks/FeatureEngineering.ipynb`.
3. **Popularity modeling (supervised regression)**: Benchmark models and select best log-target model. Notebook: `models/PopularityModelComparison.ipynb`.
4. **Revenue tier modeling (semi-supervised)**: Compare supervised baselines vs SSL, select best macro F1. Notebook: `models/SemiSupervisedModels_V2.ipynb`.
5. **Export artifacts**: Generate app-ready models and metadata via `models/export_best_models.py`.
6. **App usage**: Serve results in Streamlit using `app/app_final.py`.

---

**How to Run**
1. Run extraction and feature engineering:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/DataExtraction.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/FeatureEngineering.ipynb
```

2. Run modeling notebooks:
```bash
jupyter nbconvert --to notebook --execute --inplace models/PopularityModelComparison.ipynb
jupyter nbconvert --to notebook --execute --inplace models/SemiSupervisedModels_V2.ipynb
```

3. Export best models:
```bash
python models/export_best_models.py
```

4. Launch the app:
```bash
streamlit run app/app_final.py
```

---

**Modeling Results (Notebook Outputs)**

**Popularity Prediction (Regression)**
Source: `models/PopularityModelComparison.ipynb`
- **Best raw-target model**: XGBoost.
- **Best final model**: Gradient Boosting with `log1p(popularity)` target transform.

**Holdout Metrics (Original Popularity Scale)**
| Setting | Model | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| Best raw target | XGBoost | 4.1853 | 1.6265 | 0.3142 |
| Best log1p target (final) | Gradient Boosting | 3.5067 | 1.4196 | 0.5186 |

**Revenue Tier Prediction (Semi-Supervised Classification)**
Source: `models/SemiSupervisedModels_V2.ipynb`
- **Best model**: SelfTraining (SSL) with tuned RandomForest base estimator.
- **Primary metric**: Macro F1 on held-out labeled test set.

**Comparison (Held-Out Labeled Test Set)**
| Model | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| SelfTraining (SSL, tuned) | 0.6238 | 0.6296 | Pseudo-labeled: 5974 samples; threshold=0.7 |
| RandomForest (supervised, tuned) | 0.6046 | 0.6048 | Supervised baseline |
| GradientBoosting (supervised, tuned) | 0.6027 | 0.6024 | Supervised baseline |
| LabelSpreading (SSL, tuned) | 0.5470 | 0.5420 | Graph SSL |
| LabelPropagation (SSL, tuned) | 0.5278 | 0.5277 | Graph SSL |

---

**Important Graphs**

**Popularity Notebook**
Popularity distribution
![Popularity distribution](docs/figures/PopularityModelComparison_files/PopularityModelComparison_7_0.png)

Model comparison snapshot (metrics bar chart)
![Model comparison](docs/figures/PopularityModelComparison_files/PopularityModelComparison_15_2.png)

Residual diagnostics
![Residual diagnostics](docs/figures/PopularityModelComparison_files/PopularityModelComparison_18_0.png)

Target transform ablation (raw vs log1p)
![Target transform ablation](docs/figures/PopularityModelComparison_files/PopularityModelComparison_32_4.png)

Final model SHAP summary
![Final model SHAP](docs/figures/PopularityModelComparison_files/PopularityModelComparison_40_1.png)

**Semi-Supervised Notebook**
Best SSL confusion matrix (Self-Training)
![SSL confusion matrix](docs/figures/SemiSupervisedModels_V2_files/SemiSupervisedModels_V2_41_1.png)

---

**Threats to Validity (Slide 37)**

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

---

**Conclusions (Slide 38)**
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

---

**Lessons Learned (Slide 39)**
- Semi-supervised learning adds value when labels are scarce.
- Feature engineering matters more than model complexity alone.
- AI is a tool, not a replacement for reasoning.

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

**Monitoring and Maintenance**
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
