"""
Step 4: Feature Extraction from customer reviews.

Techniques implemented:
  A. TF-IDF (unigrams + bigrams)
  B. TextBlob Lexicon-based Sentiment Scores (polarity + subjectivity)
  C. spaCy Named Entity Recognition (product names, brands, etc.)
  D. Aspect-based feature extraction via noun-chunk + adjective dependency
  E. Topic Modelling via LDA (Latent Dirichlet Allocation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
import ast
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import spacy
from collections import Counter, defaultdict
from tqdm import tqdm

DATA_DIR = "data"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("STEP 4: FEATURE EXTRACTION")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "reviews_processed.csv"))
print(f"Loaded {len(df)} reviews.")

nlp = spacy.load("en_core_web_sm")

# ── A: TF-IDF Features ────────────────────────────────────────
print("\n[A] Computing TF-IDF features (unigrams + bigrams)...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=5,
    max_df=0.9,
    sublinear_tf=True,
)
tfidf_matrix = tfidf.fit_transform(df["tokens_str"].fillna(""))
feature_names = tfidf.get_feature_names_out()

print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

# Top TF-IDF terms per sentiment class
print("\n  Top 10 TF-IDF terms per sentiment class:")
for label in ["positive", "neutral", "negative"]:
    idx = df[df["sentiment_label"] == label].index
    mean_tfidf = tfidf_matrix[idx].mean(axis=0).A1
    top_idx = mean_tfidf.argsort()[-10:][::-1]
    top_terms = [(feature_names[i], round(mean_tfidf[i], 4)) for i in top_idx]
    print(f"  [{label}]: {top_terms}")

# ── B: Lexicon-based Sentiment Scores ─────────────────────────
print("\n[B] Computing TextBlob sentiment scores...")
polarities, subjectivities = [], []
for text in tqdm(df["text_clean"].fillna(""), desc="  TextBlob", ncols=70):
    blob = TextBlob(text)
    polarities.append(blob.sentiment.polarity)
    subjectivities.append(blob.sentiment.subjectivity)

df["tb_polarity"] = polarities
df["tb_subjectivity"] = subjectivities

print("\n  Avg polarity by sentiment label:")
print(df.groupby("sentiment_label")[["tb_polarity", "tb_subjectivity"]].mean().round(3))

# ── C: Named Entity Recognition ───────────────────────────────
print("\n[C] Running Named Entity Recognition (NER)...")
ent_types = defaultdict(list)
all_entities = []

sample_texts = df["text"].fillna("").tolist()[:2000]  # NER on 2k reviews for speed
for text in tqdm(sample_texts, desc="  NER", ncols=70):
    doc = nlp(text[:500])
    for ent in doc.ents:
        ent_types[ent.label_].append(ent.text.lower())
        all_entities.append((ent.text.lower(), ent.label_))

print("\n  Top entity types found:")
ent_counts = {k: len(v) for k, v in ent_types.items()}
for etype, count in sorted(ent_counts.items(), key=lambda x: -x[1])[:8]:
    print(f"    {etype}: {count} mentions")

print("\n  Most common PRODUCT entities:")
product_ents = [e[0] for e in all_entities if e[1] in ("PRODUCT", "ORG", "PERSON")]
print(f"  {Counter(product_ents).most_common(10)}")

# ── D: Aspect-Based Feature Extraction ────────────────────────
print("\n[D] Aspect-Based Sentiment Analysis (noun+adjective pairs)...")

ASPECT_KEYWORDS = {
    "battery":      ["battery", "charge", "charging", "power"],
    "sound":        ["sound", "audio", "bass", "treble", "volume", "noise"],
    "display":      ["screen", "display", "resolution", "brightness", "pixel"],
    "build_quality":["build", "quality", "durable", "material", "sturdy", "cheap", "flimsy"],
    "performance":  ["fast", "slow", "performance", "speed", "lag", "responsive"],
    "price_value":  ["price", "value", "worth", "expensive", "cheap", "cost", "money"],
    "ease_of_use":  ["easy", "simple", "intuitive", "difficult", "complicated", "setup"],
    "customer_service": ["service", "support", "warranty", "return", "refund"],
}

def detect_aspects(text: str) -> list[str]:
    text_lower = str(text).lower()
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(aspect)
    return found

df["aspects"] = df["text_clean"].apply(detect_aspects)

aspect_sentiment = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
for _, row in df.iterrows():
    for asp in row["aspects"]:
        aspect_sentiment[asp][row["sentiment_label"]] += 1

asp_df = pd.DataFrame(aspect_sentiment).T.fillna(0).astype(int)
asp_df["total"] = asp_df.sum(axis=1)
asp_df = asp_df.sort_values("total", ascending=False)

print("\n  Aspect mention counts by sentiment:")
print(asp_df)
asp_df.to_csv(os.path.join(DATA_DIR, "aspect_sentiment.csv"))

# ── E: LDA Topic Modeling ──────────────────────────────────────
print("\n[E] Topic Modelling with LDA (8 topics)...")
N_TOPICS = 8
lda_vectorizer = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.9)
lda_matrix = lda_vectorizer.fit_transform(df["tokens_str"].fillna(""))
lda_features = lda_vectorizer.get_feature_names_out()

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    learning_method="batch",
    max_iter=20,
)
lda.fit(lda_matrix)

print("\n  Top 10 words per topic:")
topic_labels = []
for i, comp in enumerate(lda.components_):
    top_words = [lda_features[j] for j in comp.argsort()[-10:][::-1]]
    print(f"  Topic {i+1}: {', '.join(top_words)}")
    topic_labels.append(f"T{i+1}: {top_words[0]}/{top_words[1]}")

doc_topics = lda.transform(lda_matrix)
df["dominant_topic"] = doc_topics.argmax(axis=1) + 1

# ── Visualization: Aspect Sentiment Heatmap ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Aspect-Based Sentiment Analysis", fontsize=14, fontweight="bold")

asp_plot = asp_df[["positive", "neutral", "negative"]].drop(columns=[], errors="ignore")
sns.heatmap(asp_plot, annot=True, fmt="d", cmap="RdYlGn", ax=axes[0],
            linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[0].set_title("Aspect × Sentiment Counts")
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Aspect")

asp_pct = asp_plot.div(asp_plot.sum(axis=1), axis=0) * 100
asp_pct.plot(kind="barh", stacked=True, ax=axes[1],
             color=["#2ecc71", "#f39c12", "#e74c3c"])
axes[1].set_title("Aspect Sentiment Share (%)")
axes[1].set_xlabel("Percentage")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_aspect_sentiment.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 05_aspect_sentiment.png")

# ── Visualization: Polarity Distribution ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("TextBlob Lexicon Sentiment Scores", fontsize=14, fontweight="bold")

for label, color in {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}.items():
    subset = df[df["sentiment_label"] == label]["tb_polarity"]
    axes[0].hist(subset, bins=40, alpha=0.6, label=label, color=color, density=True)
axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_title("Polarity Score Distribution")
axes[0].set_xlabel("Polarity (-1 to 1)")
axes[0].set_ylabel("Density")
axes[0].legend()

axes[1].scatter(
    df["tb_polarity"], df["tb_subjectivity"],
    c=df["sentiment_label"].map({"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}),
    alpha=0.3, s=10
)
axes[1].set_title("Polarity vs Subjectivity")
axes[1].set_xlabel("Polarity")
axes[1].set_ylabel("Subjectivity")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l)
                   for l, c in {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}.items()]
axes[1].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_polarity_subjectivity.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_polarity_subjectivity.png")

# ── Visualization: LDA Topic Distribution ─────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
topic_dist = df["dominant_topic"].value_counts().sort_index()
ax.bar([topic_labels[i-1] for i in topic_dist.index], topic_dist.values, color=sns.color_palette("tab10", N_TOPICS))
ax.set_title("LDA Topic Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Topic")
ax.set_ylabel("Number of Reviews")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_lda_topic_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_lda_topic_distribution.png")

# Save enriched dataframe
df.to_csv(os.path.join(DATA_DIR, "reviews_features.csv"), index=False)
print(f"\nSaved feature-enriched data to {DATA_DIR}/reviews_features.csv")
print("\nStep 4 complete.")
