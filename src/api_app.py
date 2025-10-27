"""
api_app.py
Simple Flask app to:
 - Provide a form-based UI to enter feature values
 - Provide /predict endpoint for JSON predictions
"""
import os
from pathlib import Path
from flask import Flask, request, render_template_string, jsonify
import joblib
import yaml
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "models"
CFG_PATH = BASE_DIR / "config.yaml"

app = Flask(__name__)

def load_config():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)

# load model and preprocessor
def load_artifacts():
    model_path = ARTIFACTS_DIR / "rf_model.pkl"
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.pkl"
    model = joblib.load(model_path)
    preobj = joblib.load(preprocessor_path)
    preprocessor = preobj["preprocessor"]
    meta = preobj["meta"]
    return model, preprocessor, meta

MODEL, PREPROCESSOR, META = None, None, None

def startup():
    """Load model and preprocessor once at startup."""
    global MODEL, PREPROCESSOR, META
    try:
        MODEL, PREPROCESSOR, META = load_artifacts()
        print("[api_app] ✅ Loaded model & preprocessor.")
    except Exception as e:
        print("[api_app] ❌ Could not load artifacts:", e)

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Pipeline Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }
        h1 { color: #343a40; }
        form { background: white; padding: 20px; border-radius: 8px; box-shadow: 0px 2px 10px rgba(0,0,0,0.1); width: 400px; }
        input, button { margin: 10px 0; padding: 8px; width: 100%; border-radius: 5px; border: 1px solid #ced4da; }
        button { background-color: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>ML Pipeline Prediction</h1>
    <form id="predictForm">
    <label>Age:</label>
    <input type="number" name="age" step="1" required>

    <label>Income:</label>
    <input type="number" name="income" step="0.01" required>

    <label>Tenure (months):</label>
    <input type="number" name="tenure" step="1" required>

    <label>Number of Products:</label>
    <input type="number" name="num_products" step="1" required>

    <label>Has Credit Card (1 = Yes, 0 = No):</label>
    <input type="number" name="has_cr_card" step="1" min="0" max="1" required>

    <label>Is Active (1 = Yes, 0 = No):</label>
    <input type="number" name="is_active" step="1" min="0" max="1" required>

    <label>Country:</label>
    <input type="text" name="country" required>

    <button type="submit">Predict</button>
</form>


    <div class="result" id="result"></div>

    <script>
        document.getElementById("predictForm").addEventListener("submit", async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            document.getElementById("result").innerHTML = 
                "<b>Prediction Result:</b> " + JSON.stringify(result);
        });
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    data = {
        "age": [int(request.form.get("age"))],
        "income": [float(request.form.get("income"))],
        "tenure": [int(request.form.get("tenure"))],
        "num_products": [int(request.form.get("num_products"))],
        "has_cr_card": [int(request.form.get("has_cr_card"))],
        "is_active": [int(request.form.get("is_active"))],
        "country": [request.form.get("country")]
    }
    df = pd.DataFrame(data)
    # derived features same way as training
    df["income_per_product"] = df["income"] / df["num_products"].replace(0,1)
    df["age_tenure_ratio"] = df["age"] / (df["tenure"].replace(0,1) + 1)
    try:
        probs = MODEL.predict_proba(df)[:,1] if MODEL is not None else [0.0]
        pred = MODEL.predict(df)[0] if MODEL is not None else 0
    except Exception as e:
        return f"Prediction failed: {e}", 500
    result = {"prob": float(probs[0]), "pred": int(pred)}
    return render_template_string(HTML_TEMPLATE, result=result)

@app.route("/predict", methods=["POST"])
def predict_json():
    req = request.get_json(force=True)
    df = pd.DataFrame([req])

    # Ensure all required columns exist (fill with defaults if missing)
    expected_cols = ["age", "income", "tenure", "num_products", "has_cr_card", "is_active", "country"]
    for col in expected_cols:
        if col not in df.columns:
            if col in ["tenure", "num_products", "has_cr_card", "is_active"]:
                df[col] = 1
            elif col == "country":
                df[col] = "India"
            else:
                df[col] = 0

    # Convert numeric columns
    numeric_cols = ["age", "income", "tenure", "num_products", "has_cr_card", "is_active"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived features
    df["income_per_product"] = df["income"].fillna(df["income"].median()) / df["num_products"].replace(0, 1)
    df["age_tenure_ratio"] = df["age"] / (df["tenure"].replace(0, 1) + 1)

    try:
        probs = MODEL.predict_proba(df)[:, 1].tolist() if MODEL is not None else [0.0]
        preds = MODEL.predict(df).tolist() if MODEL is not None else [0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"probabilities": probs, "predictions": preds})
if __name__ == "__main__":
    startup()  # 👈 Load artifacts before app starts
    cfg = load_config()
    app.run(host=cfg["api"]["host"], port=cfg["api"]["port"], debug=True)
