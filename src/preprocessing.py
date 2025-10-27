"""
preprocessing.py
- Handles missing data
- Encodes categorical features
- Scales numeric features
- Exposes fit_transform and transform for use in train & API
"""
import os
from pathlib import Path
import yaml
import joblib
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "models"

def load_config():
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_preprocessor(X: pd.DataFrame):
    # detect numeric / categorical
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    # Remove target if present
    if "target" in numeric_cols:
        numeric_cols.remove("target")
    if "target" in categorical_cols:
        categorical_cols.remove("target")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder="drop")

    meta = {"numeric": numeric_cols, "categorical": categorical_cols}
    return preprocessor, meta

def fit_preprocessor_and_save(X: pd.DataFrame, artifact_name="preprocessor.pkl"):
    preprocessor, meta = build_preprocessor(X)
    preprocessor.fit(X)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "meta": meta}, ARTIFACTS_DIR/artifact_name)
    print(f"[preprocessing] Saved preprocessor to {ARTIFACTS_DIR/artifact_name}")
    return preprocessor, meta

def load_preprocessor(artifact_name="preprocessor.pkl"):
    path = ARTIFACTS_DIR / artifact_name
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {path}")
    obj = joblib.load(path)
    return obj["preprocessor"], obj["meta"]

def transform_df(X: pd.DataFrame, preprocessor):
    X_trans = preprocessor.transform(X)
    # Column names from transformer (approximate)
    # For simple purposes, return numpy array + metadata separately
    return X_trans

if __name__ == "__main__":
    cfg = load_config()
    data_dir = BASE_DIR / cfg["data"]["paths"]["data_dir"]
    csvs = sorted(list(Path(data_dir).glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    df = pd.read_csv(csvs[0])
    preprocessor, meta = fit_preprocessor_and_save(df.drop(columns=["target"], errors="ignore"))

