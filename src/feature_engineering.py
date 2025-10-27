"""
feature_engineering.py
- Create derived features
- Optionally run PCA and mutual information based selection
- Returns transformed X (numpy) and feature names
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import yaml
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "models"

def load_config():
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_derived_features(df: pd.DataFrame):
    df = df.copy()
    # example derived features:
    df["income_per_product"] = df["income"].fillna(df["income"].median()) / (df["num_products"].replace(0,1))
    df["age_tenure_ratio"] = df["age"] / (df["tenure"].replace(0,1) + 1)
    return df

def run_pca(X: np.ndarray, n_components=10):
    pca = PCA(n_components=min(n_components, X.shape[1]))
    X_p = pca.fit_transform(X)
    return X_p, pca

def select_by_mutual_info(X: np.ndarray, y, k=10):
    selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
    X_sel = selector.fit_transform(X, y)
    return X_sel, selector

def shap_feature_selection(model, X, feature_names, top_k=10):
    try:
        import shap
    except Exception as e:
        print(f"[feature_engineering] SHAP not available: {e}")
        return None, None
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    import numpy as np
    mean_abs = np.mean(np.abs(shap_values.values), axis=0)
    idx = np.argsort(mean_abs)[-top_k:]
    selected = [feature_names[i] for i in idx]
    return idx, selected

if __name__ == "__main__":
    # simple demonstration
    cfg = load_config()
    data_dir = BASE_DIR / cfg["data"]["paths"]["data_dir"]
    csvs = sorted(list(Path(data_dir).glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    import pandas as pd
    df = pd.read_csv(csvs[0])
    df = create_derived_features(df)
    print(df.head())
