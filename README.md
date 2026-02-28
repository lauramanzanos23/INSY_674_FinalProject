# TMDB Movie Success Prediction

End-to-end machine learning project to estimate movie outcomes before release using TMDB-derived metadata.

## What This Project Does

Two prediction tasks:
1. Popularity prediction (regression)
2. Revenue tier prediction (semi-supervised classification: Low / Medium / High / Blockbuster)

It includes:
1. Data extraction and feature engineering
2. Model comparison, validation, and fine-tuning
3. Explainability (feature importance + SHAP)
4. Causal inference analysis
5. Streamlit app for what-if casting scenarios

---

## Repository Structure

```text
app/
  app_final.py
data/
  movies_2010_2025.csv
  data_supervised_popularity.csv
  data_ssl_revenue.csv
models/
  PopularityModelComparison.ipynb
  SemiSupervisedModels_V2.ipynb
  export_best_models.py
  popularity_best_model.pkl
  ssl_best_model.pkl
  ssl_scaler.pkl
  model_metadata.pkl
docs/figures/
  PopularityModelComparison_files/*.png
  SemiSupervisedModels_V2_files/*.png
```

---

## Data and Features

Primary source:
1. TMDB API (movie metadata, cast/director, keywords, release info)

Core feature groups:
1. Talent: director and top-cast popularity
2. Content: genres, language
3. Timing: release month/year, seasonal flags
4. Production: runtime, budget indicators (`has_budget`, `log_budget`)
5. Text proxies: keyword count, overview length/presence

Leakage control:
1. Post-release signals are excluded from training features used in app inference.

---

## Modeling

### Popularity Regression

Notebook:
1. [`models/PopularityModelComparison.ipynb`](models/PopularityModelComparison.ipynb)

Highlights:
1. Baselines + multiple regressors
2. Holdout evaluation and cross-validation
3. Raw target vs `log1p(popularity)` ablation
4. Final-model validation + targeted fine-tuning
5. SHAP explainability for the selected final model

Current exported app model:
1. Gradient Boosting with `log1p` target transform (back-transformed with `expm1` at inference)

### Revenue Tier Semi-Supervised Classification

Notebook:
1. [`models/SemiSupervisedModels_V2.ipynb`](models/SemiSupervisedModels_V2.ipynb)

Highlights:
1. Semi-supervised setup with labeled + unlabeled examples
2. Feature policy aligned to pre-release setting
3. Selection by classification performance
4. Confusion matrix and class-level diagnostics

---

## Notebook Figures

### Popularity distribution
![Popularity distribution](docs/figures/PopularityModelComparison_files/PopularityModelComparison_7_0.png)

### Model comparison snapshot
![Model comparison](docs/figures/PopularityModelComparison_files/PopularityModelComparison_15_2.png)

### Best-model diagnostics
![Residual diagnostics](docs/figures/PopularityModelComparison_files/PopularityModelComparison_18_0.png)

### Feature importance
![Feature importance](docs/figures/PopularityModelComparison_files/PopularityModelComparison_20_1.png)

### SHAP summary (best/final model views)
![SHAP summary](docs/figures/PopularityModelComparison_files/PopularityModelComparison_40_1.png)

### Causal inference chart
![Causal inference](docs/figures/PopularityModelComparison_files/PopularityModelComparison_26_5.png)

### Semi-supervised confusion matrix
![SSL confusion matrix](docs/figures/SemiSupervisedModels_V2_files/SemiSupervisedModels_V2_41_1.png)

---

## Causal Inference

In the popularity notebook, budget-treatment analysis includes:
1. IPW estimate
2. Doubly robust estimate
3. Overlap/robustness checks (including placebo workflow)

Interpretation:
1. Treat as observational causal evidence, not randomized proof.
2. Use overlap and confidence intervals to judge reliability.

---

## Streamlit App

App file:
1. [`app/app_final.py`](app/app_final.py)

What the app shows:
1. Predicted popularity + percentile context
2. Revenue tier outlook + confidence
3. Actor/director popularity chart
4. TMDB “known-for” movies (with dataset fallback)

Run locally:
```bash
streamlit run app/app_final.py
```

---

## Model Export

Script:
1. [`models/export_best_models.py`](models/export_best_models.py)

Run:
```bash
python models/export_best_models.py
```

Generated artifacts:
1. `models/popularity_best_model.pkl`
2. `models/ssl_best_model.pkl`
3. `models/ssl_scaler.pkl`
4. `models/model_metadata.pkl`

---

## Deployment (Streamlit Cloud)

Required files:
1. [`requirements.txt`](requirements.txt)
2. [`runtime.txt`](runtime.txt)

If deployment fails on model load:
1. Check logs for missing module during PKL unpickling
2. Ensure dependency versions match training/export environment

---

## Limitations

1. TMDB metadata coverage varies by movie/person.
2. Popularity and revenue proxies are noisy and non-stationary.
3. Semi-supervised labels can propagate bias.
4. Causal estimates rely on assumptions and overlap quality.

## Next Steps

1. Add temporal validation for out-of-time robustness.
2. Add model registry/version tracking for artifacts.
3. Add calibration diagnostics in UI.
4. Add automated CI checks for artifact compatibility.
