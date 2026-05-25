"""
Step 3: Exploratory Data Analysis and Visualization.
- Rating distribution, sentiment balance
- Review length distributions
- Word clouds per sentiment class
- Top N-grams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import ast
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}

print("=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "reviews_processed.csv"))
df["tokens"] = df["tokens_str"].apply(lambda x: str(x).split() if pd.notna(x) else [])
print(f"Loaded {len(df)} reviews.")

# ── Figure 1: Rating & Sentiment Distribution ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Amazon Electronics Reviews – Dataset Overview", fontsize=14, fontweight="bold")

rating_counts = df["rating"].value_counts().sort_index()
axes[0].bar(rating_counts.index, rating_counts.values,
            color=["#e74c3c", "#e74c3c", "#f39c12", "#2ecc71", "#2ecc71"])
axes[0].set_title("Rating Distribution (1–5 Stars)")
axes[0].set_xlabel("Star Rating")
axes[0].set_ylabel("Count")
for i, (x, y) in enumerate(zip(rating_counts.index, rating_counts.values)):
    axes[0].text(x, y + 10, str(y), ha="center", fontsize=9)

sent_counts = df["sentiment_label"].value_counts()
colors = [COLORS[s] for s in sent_counts.index]
wedges, texts, autotexts = axes[1].pie(
    sent_counts.values, labels=sent_counts.index,
    autopct="%1.1f%%", colors=colors, startangle=90,
    textprops={"fontsize": 11}
)
axes[1].set_title("Sentiment Label Distribution")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_rating_sentiment_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_rating_sentiment_distribution.png")

# ── Figure 2: Review Length by Sentiment ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Review Length Analysis by Sentiment", fontsize=14, fontweight="bold")

for label, color in COLORS.items():
    subset = df[df["sentiment_label"] == label]["word_count"]
    axes[0].hist(subset, bins=40, alpha=0.6, label=label, color=color, density=True)
axes[0].set_title("Word Count Distribution")
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Density")
axes[0].set_xlim(0, 300)
axes[0].legend()

sns.boxplot(data=df, x="sentiment_label", y="word_count", ax=axes[1],
            palette=COLORS, order=["positive", "neutral", "negative"])
axes[1].set_title("Word Count Boxplot by Sentiment")
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("Word Count")
axes[1].set_ylim(0, 300)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_review_length_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_review_length_analysis.png")

# ── Figure 3: Word Clouds ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Word Clouds by Sentiment Class", fontsize=14, fontweight="bold")

for ax, (label, color) in zip(axes, COLORS.items()):
    text = " ".join(df[df["sentiment_label"] == label]["tokens_str"].dropna())
    wc = WordCloud(
        width=500, height=350,
        background_color="white",
        colormap="RdYlGn" if label == "positive" else ("autumn_r" if label == "negative" else "copper"),
        max_words=100,
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(f"{label.capitalize()} Reviews", fontsize=12, color=color, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_wordclouds.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_wordclouds.png")

# ── Figure 4: Top 20 Unigrams per Sentiment ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Top 20 Words per Sentiment Class", fontsize=14, fontweight="bold")

for ax, (label, color) in zip(axes, COLORS.items()):
    tokens = [t for tlist in df[df["sentiment_label"] == label]["tokens"] for t in tlist]
    top = Counter(tokens).most_common(20)
    words, counts = zip(*top)
    ax.barh(words[::-1], counts[::-1], color=color, alpha=0.8)
    ax.set_title(f"{label.capitalize()}", fontsize=12)
    ax.set_xlabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_top_words_per_sentiment.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_top_words_per_sentiment.png")

print("\nEDA complete. All plots saved to /plots/")
