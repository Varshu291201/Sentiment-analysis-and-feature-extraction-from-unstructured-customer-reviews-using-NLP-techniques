"""
Step 5: Train and compare three sentiment classification models.

Models:
  1. Logistic Regression  (TF-IDF features)
  2. Random Forest        (TF-IDF features)
  3. Linear SVM           (TF-IDF features)

Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import json
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.preprocessing import label_binarize

DATA_DIR = "data"
PLOTS_DIR = "plots"
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("STEP 5: SENTIMENT CLASSIFICATION MODELS")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "reviews_features.csv"))
df = df.dropna(subset=["tokens_str", "sentiment_label"])
print(f"Using {len(df)} reviews. Class distribution:")
print(df["sentiment_label"].value_counts())

LABEL_ORDER = ["negative", "neutral", "positive"]
le = LabelEncoder()
le.fit(LABEL_ORDER)

X = df["tokens_str"].fillna("")
y = df["sentiment_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ── Define pipelines ───────────────────────────────────────────
tfidf_params = dict(ngram_range=(1, 2), max_features=5000,
                    min_df=3, max_df=0.9, sublinear_tf=True)

pipelines = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)),
    ]),
}

results = {}
y_test_enc = le.transform(y_test)

for name, pipe in pipelines.items():
    print(f"\nTraining {name}...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

    # ROC-AUC (one-vs-rest)
    y_pred_enc = le.transform(y_pred)
    y_test_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
    y_pred_bin = label_binarize(y_pred_enc, classes=[0, 1, 2])
    try:
        roc = roc_auc_score(y_test_bin, y_pred_bin, multi_class="ovr", average="macro")
    except Exception:
        roc = None

    results[name] = {
        "accuracy": round(acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "roc_auc": round(roc, 4) if roc else "N/A",
        "confusion_matrix": cm.tolist(),
        "report": report,
    }

    roc_str = f"{roc:.4f}" if roc else "N/A"
    print(f"  Accuracy: {acc:.4f}  |  Macro F1: {report['macro avg']['f1-score']:.4f}  |  ROC-AUC: {roc_str}")
    print(classification_report(y_test, y_pred, target_names=LABEL_ORDER))

# ── Visualization 1: Model Comparison Bar Chart ───────────────
metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
model_names = list(results.keys())

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(model_names))
width = 0.2
colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

for i, metric in enumerate(metrics):
    vals = [results[m][metric] if isinstance(results[m][metric], float) else 0 for m in model_names]
    bars = ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title(), color=colors[i], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Model Comparison: Sentiment Classification", fontsize=13, fontweight="bold")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 08_model_comparison.png")

# ── Visualization 2: Confusion Matrices ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices – Test Set", fontsize=13, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    cm = np.array(res["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 09_confusion_matrices.png")

# ── Visualization 3: CV Learning Curves ──────────────────────
print("\nRunning 5-fold cross-validation for best model...")
best_model = "Logistic Regression"
best_pipe = pipelines[best_model]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
print(f"  {best_model} CV Macro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, 6), cv_scores, color="#3498db", alpha=0.8)
ax.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean={cv_scores.mean():.4f}")
ax.set_title(f"5-Fold Cross-Validation F1 – {best_model}", fontsize=12, fontweight="bold")
ax.set_xlabel("Fold")
ax.set_ylabel("Macro F1")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "10_cross_validation.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 10_cross_validation.png")

# Save results summary
summary = {name: {k: v for k, v in r.items() if k != "report" and k != "confusion_matrix"}
           for name, r in results.items()}
with open(os.path.join(DATA_DIR, "model_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved model results to {DATA_DIR}/model_results.json")
print("\nStep 5 complete.")
