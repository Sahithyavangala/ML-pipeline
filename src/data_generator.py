"""
data_generator.py
Generate synthetic multi-source data:
 - CSV file(s) (primary)
 - small SQLite DB table (simulated database source)
 - simulated API data returned as JSON (function)
Saves CSV(s) into data/ directory.
"""
import os
import json
import sqlite3
from faker import Faker
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

def load_config():
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def generate_csv(n_rows=5000, random_state=42):
    fake = Faker()
    np.random.seed(random_state)
    Faker.seed(random_state)

    # Example features: age, income, tenure, num_products, has_cr_card, is_active, country, signup_date
    ages = np.random.randint(18, 80, size=n_rows)
    income = np.round(np.random.normal(50000, 20000, size=n_rows)).astype(int)
    income = np.clip(income, 5000, None)
    tenure = np.random.randint(0, 240, size=n_rows)  # months
    num_products = np.random.randint(1, 6, size=n_rows)
    has_cr_card = np.random.choice([0,1], size=n_rows, p=[0.3, 0.7])
    is_active = np.random.choice([0,1], size=n_rows, p=[0.2, 0.8])
    country = np.random.choice(["India","USA","UK","Germany","France"], size=n_rows, p=[0.5,0.2,0.15,0.1,0.05])
    signup_date = [fake.date_between(start_date='-5y', end_date='today').isoformat() for _ in range(n_rows)]

    # Create synthetic target (binary classification) with some signal
    # Higher income and more products -> higher chance; low tenure -> lower chance.
    score = (income / income.max()) * 0.4 + (num_products / num_products.max()) * 0.3 + (tenure / tenure.max()) * 0.2 + (is_active * 0.1)
    prob = (score - score.min()) / (score.max() - score.min())
    y = (prob + np.random.normal(0, 0.05, size=n_rows) > 0.5).astype(int)

    df = pd.DataFrame({
        "age": ages,
        "income": income,
        "tenure": tenure,
        "num_products": num_products,
        "has_cr_card": has_cr_card,
        "is_active": is_active,
        "country": country,
        "signup_date": signup_date,
        "target": y
    })

    # Add some missingness and duplicates intentionally
    n_missing = max(1, int(0.01 * n_rows))
    for col in ["income", "signup_date"]:
        idx = np.random.choice(df.index, n_missing, replace=False)
        df.loc[idx, col] = None

    # duplicate some rows
    dup_idx = np.random.choice(df.index, size=int(0.005 * n_rows), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Save CSV
    cfg = load_config()
    data_dir = Path(BASE_DIR) / cfg["data"]["paths"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = data_dir / f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"[data_generator] Saved CSV to {filename}")
    return filename

def generate_sqlite_table(n_rows=5000, random_state=42):
    cfg = load_config()
    db_path = Path(BASE_DIR) / cfg["data"]["paths"]["data_dir"] / "source.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_csv(generate_csv(n_rows, random_state))
    # create a table for "customers" - subset of columns
    df2 = df[["age","income","num_products","is_active","country","target"]].copy()
    df2.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[data_generator] Wrote SQLite DB at {db_path}")
    return db_path

def simulate_api_response(n=10, random_state=42):
    # Returns a list of JSON records to simulate a simple API endpoint
    fake = Faker()
    np.random.seed(random_state)
    return [{"id": i, "event_time": fake.iso8601(), "metric": float(np.random.random())} for i in range(n)]

if __name__ == "__main__":
    cfg = load_config()
    generate_csv(cfg["data"]["n_rows"], cfg["data"]["random_state"])
    generate_sqlite_table(cfg["data"]["n_rows"], cfg["data"]["random_state"])

