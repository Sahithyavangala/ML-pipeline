"""
run_pipeline.py
Main controller to automate the ML pipeline.
Executes all steps sequentially — from data generation to API startup.
"""
import subprocess
import time
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

# Define steps
steps = [
    "data_generator.py",
    "data_validation.py",
    "preprocessing.py",
    "feature_engineering.py",
    "train.py",
    "evaluate.py"
]

def run_step(script_name):
    print(f"\n🚀 Running: {script_name}")
    result = subprocess.run(["python", str(SRC_DIR / script_name)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed at step: {script_name}")
        print(result.stderr)
        exit(1)
    print(f"✅ Completed: {script_name}\n")

def main():
    print("===  Starting Full ML Pipeline ===\n")
    for step in steps:
        run_step(step)
        time.sleep(1)

    print("🎯 All stages completed successfully!\n")
    print("🔥 Launching Flask API...")
    subprocess.run(["python", str(SRC_DIR / "api_app.py")])

if __name__ == "__main__":
    main()
