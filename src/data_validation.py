"""
data_validation.py
Simple validation utilities: schema checks, missing value checks, duplicates, data types.
"""
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parents[1]

def load_config():
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def validate_dataframe(df: pd.DataFrame, expected_columns: Dict[str, Any]=None) -> Dict[str, Any]:
    """
    Runs quick checks and returns a report dictionary.
    expected_columns: optional mapping col -> dtype/type string (informational)
    """
    report = {}
    report["n_rows"] = int(df.shape[0])
    report["n_columns"] = int(df.shape[1])
    report["columns"] = df.columns.tolist()
    report["missing_perc"] = (df.isnull().sum() / len(df)).to_dict()
    report["duplicates"] = int(df.duplicated().sum())
    # ranges for numeric columns
    numeric = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    ranges = {}
    for c in numeric:
        try:
            ranges[c] = {"min": float(df[c].min()), "max": float(df[c].max())}
        except Exception:
            ranges[c] = {"min": None, "max": None}
    report["numeric_ranges"] = ranges
    if expected_columns:
        missing_cols = [c for c in expected_columns.keys() if c not in df.columns]
        report["missing_expected_columns"] = missing_cols
    return report

def load_latest_csv(data_dir: Path):
    # picks the latest CSV file in data dir
    csvs = sorted(list(data_dir.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    return pd.read_csv(csvs[0])

if __name__ == "__main__":
    cfg = load_config()
    data_dir = BASE_DIR / cfg["data"]["paths"]["data_dir"]
    df = load_latest_csv(data_dir)
    rep = validate_dataframe(df)
    import json
    print(json.dumps(rep, indent=2))
