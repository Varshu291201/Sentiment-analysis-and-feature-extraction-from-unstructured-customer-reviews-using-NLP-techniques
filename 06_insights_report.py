"""
Step 6: Generate final insights summary and combined report figure.
Aggregates all results into a single comprehensive dashboard plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
PLOTS_DIR = "plots"

print("=" * 60)
print("STEP 6: FINAL INSIGHTS & REPORT")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "reviews_features.csv"))
asp_df = pd.read_csv(os.path.join(DATA_DIR, "aspect_sentiment.csv"), index_col=0)
with open(os.path.join(DATA_DIR, "model_results.json")) as f:
    model_results = json.load(f)

# ── Summary Statistics ─────────────────────────────────────────
print("\n=== PROJECT SUMMARY ===")
print(f"Total reviews analysed   : {len(df):,}")
print(f"Avg polarity (all)       : {df['tb_polarity'].mean():.3f}")
print(f"Avg subjectivity (all)   : {df['tb_subjectivity'].mean():.3f}")

print("\nSentiment distribution:")
for lbl, cnt in df["sentiment_label"].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"  {lbl:<10}: {cnt:>4} ({pct:.1f}%)")

print("\nAvg polarity by label:")
print(df.groupby("sentiment_label")["tb_polarity"].mean().round(3).to_string())

print("\nMost discussed aspects (total mentions):")
asp_df["total"] = asp_df[["positive", "neutral", "negative"]].sum(axis=1)
print(asp_df[["positive", "negative", "total"]].sort_values("total", ascending=False).to_string())

print("\nModel Performance Summary:")
print(f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>10} {'ROC-AUC':>10}")
print("-" * 55)
for name, res in model_results.items():
    print(f"{name:<22} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f} {str(res['roc_auc']):>10}")

# ── Dashboard Figure ──────────────────────────────────────────
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor("#f8f9fa")
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}

# ── Panel 1: Sentiment Pie ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
sent_counts = df["sentiment_label"].value_counts()
ax1.pie(sent_counts.values, labels=sent_counts.index,
        autopct="%1.1f%%", colors=[COLORS[s] for s in sent_counts.index],
        startangle=90, textprops={"fontsize": 9})
ax1.set_title("Sentiment Distribution", fontweight="bold")

# ── Panel 2: Rating Distribution ──────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
rc = df["rating"].value_counts().sort_index()
bar_colors = ["#e74c3c", "#e74c3c", "#f39c12", "#2ecc71", "#2ecc71"]
ax2.bar(rc.index, rc.values, color=bar_colors)
ax2.set_title("Star Rating Distribution", fontweight="bold")
ax2.set_xlabel("Stars")
ax2.set_ylabel("Count")

# ── Panel 3: Polarity Distribution ────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
for lbl, clr in COLORS.items():
    ax3.hist(df[df["sentiment_label"] == lbl]["tb_polarity"],
             bins=30, alpha=0.6, color=clr, label=lbl, density=True)
ax3.axvline(0, color="black", linestyle="--", linewidth=1)
ax3.set_title("Polarity Distribution", fontweight="bold")
ax3.set_xlabel("Polarity")
ax3.legend(fontsize=8)

# ── Panel 4: Aspect Sentiment Heatmap ─────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
asp_heat = asp_df[["positive", "neutral", "negative"]].sort_values("positive", ascending=False)
sns.heatmap(asp_heat, annot=True, fmt="d", cmap="RdYlGn", ax=ax4,
            linewidths=0.5, cbar=False, annot_kws={"size": 9})
ax4.set_title("Aspect × Sentiment Heatmap", fontweight="bold")
ax4.set_xlabel("")

# ── Panel 5: Aspect Total Mentions ────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
asp_sorted = asp_df.sort_values("total")
ax5.barh(asp_sorted.index, asp_sorted["total"], color="#3498db", alpha=0.8)
ax5.set_title("Total Aspect Mentions", fontweight="bold")
ax5.set_xlabel("Count")

# ── Panel 6: Model Comparison ─────────────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
model_names = list(model_results.keys())
metrics = ["accuracy", "macro_f1"]
x = np.arange(len(model_names))
w = 0.3
for i, (metric, color) in enumerate(zip(metrics, ["#3498db", "#2ecc71"])):
    vals = [model_results[m][metric] for m in model_names]
    bars = ax6.bar(x + i*w, vals, w, label=metric.replace("_", " ").title(), color=color, alpha=0.85)
    for b in bars:
        ax6.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax6.set_xticks(x + w/2)
ax6.set_xticklabels(model_names)
ax6.set_ylim(0, 1.05)
ax6.set_title("Model Performance Comparison", fontweight="bold")
ax6.set_ylabel("Score")
ax6.legend()
ax6.grid(axis="y", alpha=0.3)

# ── Panel 7: Review Length Boxplot ────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
sns.boxplot(data=df, x="sentiment_label", y="word_count", ax=ax7,
            palette=COLORS, order=["positive", "neutral", "negative"])
ax7.set_ylim(0, 300)
ax7.set_title("Review Length by Sentiment", fontweight="bold")
ax7.set_xlabel("")
ax7.set_ylabel("Word Count")

# ── Panel 8: Polarity vs Subjectivity scatter ─────────────────
ax8 = fig.add_subplot(gs[3, :])
sample = df.sample(min(2000, len(df)), random_state=42)
scatter_colors = sample["sentiment_label"].map(COLORS)
ax8.scatter(sample["tb_polarity"], sample["tb_subjectivity"],
            c=scatter_colors, alpha=0.25, s=12)
ax8.set_title("Polarity vs Subjectivity (TextBlob)", fontweight="bold")
ax8.set_xlabel("Polarity")
ax8.set_ylabel("Subjectivity")
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=c, label=l) for l, c in COLORS.items()]
ax8.legend(handles=legend_els, loc="upper left")

# ── Main title ────────────────────────────────────────────────
fig.suptitle(
    "Sentiment Analysis & Feature Extraction from Customer Reviews\n"
    "Dataset: Amazon Electronics Reviews (McAuley-Lab/Amazon-Reviews-2023)",
    fontsize=14, fontweight="bold", y=0.98
)

plt.savefig(os.path.join(PLOTS_DIR, "00_DASHBOARD.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("\nSaved master dashboard: plots/00_DASHBOARD.png")
print("\nAll steps complete. See plots/ for all visualizations.")
