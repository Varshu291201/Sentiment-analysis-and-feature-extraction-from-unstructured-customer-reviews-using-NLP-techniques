"""
Master runner: executes all 6 pipeline steps in sequence.
Run from the project directory: python run_all.py
"""

import subprocess
import sys
import time

steps = [
    ("01_data_loading.py",      "Data Loading & Exploration"),
    ("02_preprocessing.py",     "Text Preprocessing"),
    ("03_eda_visualization.py", "EDA & Visualization"),
    ("04_feature_extraction.py","Feature Extraction"),
    ("05_sentiment_models.py",  "Sentiment Classification Models"),
    ("06_insights_report.py",   "Final Insights & Dashboard"),
]

print("=" * 65)
print("  SENTIMENT ANALYSIS & FEATURE EXTRACTION – FULL PIPELINE")
print("=" * 65)

total_start = time.time()
for script, label in steps:
    print(f"\n>>> Running: {label}  ({script})")
    print("-" * 50)
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"\n[Done in {elapsed:.1f}s]")

total = time.time() - total_start
print("\n" + "=" * 65)
print(f"  ALL STEPS COMPLETE  |  Total time: {total:.1f}s")
print("  Outputs: data/  and  plots/")
print("=" * 65)
