"""
evaluate.py
Load persisted model and run evaluation on a dataset.
"""
import joblib
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from feature_engineering import create_derived_features


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

def load_config():
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate_model(model_path=None, csv_path=None):
    cfg = load_config()
    if model_path is None:
        model_path = ARTIFACTS_DIR / "rf_model.pkl"
    model = joblib.load(model_path)
    if csv_path is None:
        # load latest data
        data_dir = BASE_DIR / cfg["data"]["paths"]["data_dir"]
        csvs = sorted(list(Path(data_dir).glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
        csv_path = csvs[0]
    df = pd.read_csv(csv_path)
    df = create_derived_features(df) 
    y = df["target"].astype(int)
    X = df.drop(columns=["target","signup_date"], errors="ignore")
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y, preds)
    print(classification_report(y, preds))
    if probs is not None:
        print("ROC AUC:", roc_auc_score(y, probs))
    # save confusion matrix
    LOGS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(LOGS_DIR / "confusion_matrix.png")
    print(f"Saved confusion matrix to {LOGS_DIR / 'confusion_matrix.png'}")

if __name__ == "__main__":
    evaluate_model()
