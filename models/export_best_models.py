import pickle
from pathlib import Path

import joblib
import pandas as pd
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


def build_popularity_model() -> tuple[Pipeline, list[str]]:
    df_pop = pd.read_csv(POPULARITY_DATA)
    pop_feature_cols = [c for c in df_pop.columns if c != "popularity"]
    X_pop = df_pop[pop_feature_cols]
    y_pop = df_pop["popularity"]

    pop_model = Pipeline(
        [
            (
                "model",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        ]
    )
    pop_model.fit(X_pop, y_pop)
    return pop_model, pop_feature_cols


def build_ssl_metadata() -> tuple[object, object, list[str]]:
    ssl_df = pd.read_csv(SSL_DATA)
    leakage_keywords = ["vote", "review", "rating", "popularity", "revenue", "budget"]
    candidate_features = [c for c in ssl_df.columns if c != "y_ssl"]
    leaked_cols = [
        c
        for c in candidate_features
        if any(keyword in c.lower() for keyword in leakage_keywords)
    ]
    ssl_feature_cols = [c for c in candidate_features if c not in leaked_cols]

    ssl_model = joblib.load(SSL_MODEL_JOBLIB)
    ssl_scaler = joblib.load(SSL_SCALER_JOBLIB)
    return ssl_model, ssl_scaler, ssl_feature_cols


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pop_model, pop_feature_cols = build_popularity_model()
    with open(POPULARITY_MODEL_PKL, "wb") as f:
        pickle.dump(pop_model, f)

    ssl_model, ssl_scaler, ssl_feature_cols = build_ssl_metadata()
    with open(SSL_MODEL_PKL, "wb") as f:
        pickle.dump(ssl_model, f)
    with open(SSL_SCALER_PKL, "wb") as f:
        pickle.dump(ssl_scaler, f)

    metadata = {
        "popularity_feature_cols": pop_feature_cols,
        "ssl_feature_cols": ssl_feature_cols,
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
    print(f"Saved SSL model: {SSL_MODEL_PKL}")
    print(f"Saved SSL scaler: {SSL_SCALER_PKL}")
    print(f"Saved metadata: {MODEL_METADATA_PKL}")


if __name__ == "__main__":
    main()
