import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parent

POPULARITY_DATA = DATA_DIR / "data_supervised_popularity.csv"
SSL_DATA = DATA_DIR / "data_ssl_revenue.csv"
SSL_MODEL_JOBLIB = DATA_DIR / "best_ssl_model.joblib"
SSL_SCALER_JOBLIB = DATA_DIR / "ssl_scaler.joblib"

POPULARITY_MODEL_PKL = MODEL_DIR / "popularity_best_model.pkl"
SSL_MODEL_PKL = MODEL_DIR / "ssl_best_model.pkl"
SSL_SCALER_PKL = MODEL_DIR / "ssl_scaler.pkl"
MODEL_METADATA_PKL = MODEL_DIR / "model_metadata.pkl"


def build_popularity_model() -> tuple[Pipeline, list[str], str, str]:
    df_pop = pd.read_csv(POPULARITY_DATA)
    pop_feature_cols = [c for c in df_pop.columns if c != "popularity"]
    X_pop = df_pop[pop_feature_cols]
    y_pop = df_pop["popularity"]
    y_pop_log_all = np.log1p(y_pop)

    # Match notebook final selection logic:
    # choose best log-target model by holdout RMSE on original scale.
    X_train, X_test, y_train, y_test = train_test_split(
        X_pop, y_pop, test_size=0.2, random_state=RANDOM_STATE
    )
    y_train_log = np.log1p(y_train)

    candidates = {
        "Gradient Boosting": Pipeline(
            [("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]
        ),
        "Hist Gradient Boosting": Pipeline(
            [("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))]
        ),
        "Extra Trees": Pipeline(
            [
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                )
            ]
        ),
        "XGBoost": Pipeline(
            [
                (
                    "model",
                    XGBRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                )
            ]
        ),
    }

    scores = []
    for model_name, pipe in candidates.items():
        m = clone(pipe)
        m.fit(X_train, y_train_log)
        pred = np.expm1(m.predict(X_test))
        pred = np.clip(pred, 0, None)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        scores.append((model_name, rmse))

    scores.sort(key=lambda x: x[1])
    best_name, best_rmse = scores[0]
    print("Popularity log-target holdout RMSE ranking:")
    for name, rmse in scores:
        print(f"  {name}: {rmse:.4f}")
    print(f"Selected popularity model: {best_name} (target transform: log1p)")
    print(f"Selected popularity holdout RMSE: {best_rmse:.4f}")

    pop_model = clone(candidates[best_name])
    pop_model.fit(X_pop, y_pop_log_all)
    return pop_model, pop_feature_cols, "log1p", best_name


def build_ssl_metadata() -> tuple[object, object, list[str]]:
    # Mirror SemiSupervisedModels_V2 feature policy.
    ssl_df = pd.read_csv(SSL_DATA)
    base_leakage_keywords = ["vote", "review", "rating", "revenue"]
    allowed_budget_features = {"has_budget", "log_budget", "budget_effective"}

    if {"has_budget", "log_budget"}.issubset(ssl_df.columns):
        ssl_df["budget_effective"] = ssl_df["has_budget"] * ssl_df["log_budget"]

    candidate_features = [c for c in ssl_df.columns if c != "y_ssl"]

    def is_leaked_feature(col_name: str) -> bool:
        name = col_name.lower()
        if name in allowed_budget_features:
            return False
        if any(kw in name for kw in base_leakage_keywords):
            return True
        if "budget" in name and name not in allowed_budget_features:
            return True
        if "popularity" in name:
            allowed_talent_popularity = (
                name == "director_popularity"
                or name == "cast_popularity_std"
                or name in {
                    "actor1_popularity",
                    "actor2_popularity",
                    "actor3_popularity",
                    "actor4_popularity",
                    "actor5_popularity",
                }
            )
            if not allowed_talent_popularity:
                return True
        return False

    leaked_cols = [c for c in candidate_features if is_leaked_feature(c)]
    ssl_feature_cols = [c for c in candidate_features if c not in leaked_cols]

    ssl_model = joblib.load(SSL_MODEL_JOBLIB)
    ssl_scaler = joblib.load(SSL_SCALER_JOBLIB)
    return ssl_model, ssl_scaler, ssl_feature_cols


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pop_model, pop_feature_cols, pop_target_transform, pop_model_name = build_popularity_model()
    with open(POPULARITY_MODEL_PKL, "wb") as f:
        pickle.dump(pop_model, f)

    ssl_model, ssl_scaler, ssl_feature_cols = build_ssl_metadata()
    with open(SSL_MODEL_PKL, "wb") as f:
        pickle.dump(ssl_model, f)
    with open(SSL_SCALER_PKL, "wb") as f:
        pickle.dump(ssl_scaler, f)

    metadata = {
        "popularity_feature_cols": pop_feature_cols,
        "popularity_target_transform": pop_target_transform,
        "popularity_model_name": pop_model_name,
        "ssl_feature_cols": ssl_feature_cols,
        "ssl_model_source_notebook": "SemiSupervisedModels_V2.ipynb",
        "ssl_tier_labels": {
            0: "Low",
            1: "Medium",
            2: "High",
            3: "Blockbuster",
        },
    }
    with open(MODEL_METADATA_PKL, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved popularity model: {POPULARITY_MODEL_PKL}")
    print(f"Popularity model target transform: {pop_target_transform}")
    print(f"Saved SSL model: {SSL_MODEL_PKL}")
    print(f"Saved SSL scaler: {SSL_SCALER_PKL}")
    print(f"Saved metadata: {MODEL_METADATA_PKL}")


if __name__ == "__main__":
    main()
