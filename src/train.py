"""
train.py
End-to-end training script:
 - generate data if none present
 - load latest CSV
 - validate
 - preprocess (fit preprocessor)
 - feature engineering
 - train multiple models with GridSearchCV
 - evaluate and persist best model
"""
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_generator import generate_csv, generate_sqlite_table
from data_validation import validate_dataframe, load_latest_csv
from preprocessing import fit_preprocessor_and_save, load_preprocessor, transform_df
from feature_engineering import create_derived_features
from persistence import save_model

BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def load_config():
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def basic_eda(df: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(8,6))
    sns.histplot(df['income'].dropna(), bins=30)
    plt.title("Income distribution")
    plt.tight_layout()
    plt.savefig(save_path / "income_dist.png")
    plt.close()

    plt.figure(figsize=(6,6))
    sns.countplot(x='target', data=df)
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(save_path / "target_count.png")
    plt.close()

def prepare_data(df: pd.DataFrame):
    # create derived features
    df = create_derived_features(df)
    # drop columns we won't use directly (like signup_date)
    df = df.drop(columns=["signup_date"], errors="ignore")
    # Separate target
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])
    return X, y

def train_models(X_train, y_train, preprocessor, cfg):
    # We'll construct pipelines that start from preprocessor
    results = {}
    # RandomForest
    pipe_rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=cfg["model"]["random_state"]))
    ])
    param_grid_rf = {
        "clf__n_estimators": cfg["model"]["rf"]["n_estimators"],
        "clf__max_depth": cfg["model"]["rf"]["max_depth"]
    }
    gs_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cfg["model"]["cv"], scoring=cfg["model"]["scoring"], n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    results["RandomForest"] = gs_rf

    # Logistic Regression
    pipe_lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=cfg["model"]["random_state"]))
    ])
    param_grid_lr = {"clf__C": cfg["model"]["lr"]["C"]}
    gs_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=cfg["model"]["cv"], scoring=cfg["model"]["scoring"], n_jobs=-1)
    gs_lr.fit(X_train, y_train)
    results["LogisticRegression"] = gs_lr

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier
        pipe_xgb = Pipeline([
            ("pre", preprocessor),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=cfg["model"]["random_state"]))
        ])
        param_grid_xgb = {
            "clf__n_estimators": cfg["model"]["xgb"]["n_estimators"],
            "clf__max_depth": cfg["model"]["xgb"]["max_depth"]
        }
        gs_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, cv=cfg["model"]["cv"], scoring=cfg["model"]["scoring"], n_jobs=-1)
        gs_xgb.fit(X_train, y_train)
        results["XGBoost"] = gs_xgb
    except Exception as e:
        print("[train] XGBoost not available or failed to import:", e)

    return results

def evaluate_and_select(results, X_test, y_test):
    eval_report = {}
    best_name = None
    best_score = -999
    for name, gs in results.items():
        best_est = gs.best_estimator_
        preds = best_est.predict_proba(X_test)[:,1] if hasattr(best_est, "predict_proba") else best_est.predict(X_test)
        # For classifiers that return hard predict, use predict
        try:
            auc = roc_auc_score(y_test, preds)
        except Exception:
            auc = None
        y_pred = best_est.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        eval_report[name] = {
            "best_params": gs.best_params_,
            "cv_best_score": float(gs.best_score_),
            "test_auc": float(auc) if auc is not None else None,
            "test_accuracy": float(acc)
        }
        if auc is not None and auc > best_score:
            best_score = auc
            best_name = name
    return best_name, eval_report

def main():
    cfg = load_config()
    data_dir = BASE_DIR / cfg["data"]["paths"]["data_dir"]
    # If no CSVs exist, generate data
    csvs = sorted(list(data_dir.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        print("[train] No CSV data found, generating synthetic data.")
        from data_generator import generate_csv
        generate_csv(cfg["data"]["n_rows"], cfg["data"]["random_state"])
        csvs = sorted(list(data_dir.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)

    df = pd.read_csv(csvs[0])
    # Validation
    from data_validation import validate_dataframe
    report = validate_dataframe(df)
    with open(LOGS_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[train] Validation report saved.")

    # Basic EDA
    basic_eda(df, LOGS_DIR)

    # Prepare data
    X, y = prepare_data(df)

    # Fit preprocessor and save
    preprocessor, meta = fit_preprocessor_and_save(X)
    # Load preprocessor to ensure saved object used in pipelines
    preprocessor, meta = load_preprocessor()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=cfg["data"]["random_state"], stratify=y)

    # Train models
    results = train_models(X_train, y_train, preprocessor, cfg)

    # Evaluate
    best_name, eval_report = evaluate_and_select(results, X_test, y_test)
    with open(LOGS_DIR / "evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"[train] Evaluation report saved. Best model: {best_name}")

    # Persist best model (the GridSearchCV object)
    best_gs = results[best_name]
    save_model(best_gs.best_estimator_, name="rf_model.pkl")
    print("[train] Training completed.")

if __name__ == "__main__":
    main()
