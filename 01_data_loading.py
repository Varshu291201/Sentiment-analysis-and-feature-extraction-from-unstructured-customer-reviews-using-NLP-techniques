"""
Step 1: Load and explore the Yelp Reviews dataset.
Dataset: Yelp Review Full (yelp_review_full) via HuggingFace Datasets.
5-class star ratings mapped to positive / neutral / negative.
We use a 5000-sample stratified subset for manageable runtime.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SAMPLE_SIZE = 5000
RANDOM_SEED = 42

print("=" * 60)
print("STEP 1: DATA LOADING & EXPLORATION")
print("=" * 60)

print("\nLoading Yelp Review Full dataset (train split)...")
dataset = load_dataset("yelp_review_full", split="train")

# Dataset has 'label' (0-4, i.e. 1-5 stars) and 'text'
df_full = dataset.to_pandas()
df_full = df_full.rename(columns={"label": "rating_idx", "text": "text"})
df_full["rating"] = df_full["rating_idx"] + 1   # convert 0-4 → 1-5

# Stratified sample: 1000 per rating class → 5000 total
samples = []
for r in [1, 2, 3, 4, 5]:
    group = df_full[df_full["rating"] == r]
    n = min(len(group), SAMPLE_SIZE // 5)
    samples.append(group.sample(n, random_state=RANDOM_SEED))
df = pd.concat(samples, ignore_index=True)
df["review_id"] = df.index
df["title"] = ""        # Yelp dataset has no title column
df["helpful_vote"] = 0  # not in this dataset
df["verified_purchase"] = False

print(f"Loaded {len(df)} reviews (stratified).")

# Map rating → sentiment label
def rating_to_sentiment(r):
    if r >= 4:
        return "positive"
    elif r == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment_label"] = df["rating"].apply(rating_to_sentiment)

# Drop rows with empty text
df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
print(f"After removing empty reviews: {len(df)} records.")

print("\n--- Basic Statistics ---")
print(df[["rating", "helpful_vote"]].describe())

print("\n--- Sentiment Distribution ---")
print(df["sentiment_label"].value_counts())
print(df["sentiment_label"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

print("\n--- Sample Reviews ---")
for label in ["positive", "neutral", "negative"]:
    sample = df[df["sentiment_label"] == label]["text"].iloc[0][:200]
    print(f"\n[{label.upper()}]: {sample}...")

# Save
df.to_csv(os.path.join(DATA_DIR, "reviews_raw.csv"), index=False)
print(f"\nSaved raw data to {DATA_DIR}/reviews_raw.csv")
print("\nStep 1 complete.")
