"""
persistence.py
Utilities for saving/loading models and artifacts (joblib)
"""
from pathlib import Path
import joblib
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def load_config():
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def save_model(model, name="rf_model.pkl"):
    path = ARTIFACTS_DIR / name
    joblib.dump(model, path)
    print(f"[persistence] Model saved to {path}")
    return path

def load_model(name="rf_model.pkl"):
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
