"""
Step 2: Text preprocessing pipeline.
- Lowercasing, URL/HTML removal, punctuation cleanup
- Tokenization, stopword removal, lemmatization (spaCy)
- Saves cleaned data for downstream steps
"""

import pandas as pd
import numpy as np
import re
import os
import string
import nltk
import spacy
import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

DATA_DIR = "data"
STOP_WORDS = set(stopwords.words("english"))

print("=" * 60)
print("STEP 2: TEXT PREPROCESSING")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "reviews_raw.csv"))
print(f"Loaded {len(df)} reviews.")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"<[^>]+>", "", text)                  # HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)             # punctuation/special chars
    text = re.sub(r"\s+", " ", text).strip()             # extra whitespace
    return text


def lemmatize_tokens(text: str) -> list[str]:
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_space
        and token.is_alpha
        and len(token.lemma_) > 2
    ]
    return tokens


print("Cleaning text...")
df["text_clean"] = df["text"].apply(clean_text)

print("Lemmatizing tokens (this may take ~1-2 min)...")
df["tokens"] = df["text_clean"].apply(lemmatize_tokens)
df["tokens_str"] = df["tokens"].apply(lambda t: " ".join(t))

df["token_count"] = df["tokens"].apply(len)
df["char_count"] = df["text_clean"].apply(len)
df["word_count"] = df["text_clean"].apply(lambda x: len(x.split()))

print("\n--- Text Length Statistics (after cleaning) ---")
print(df[["word_count", "token_count", "char_count"]].describe().round(1))

df.to_csv(os.path.join(DATA_DIR, "reviews_processed.csv"), index=False)
print(f"\nSaved processed data to {DATA_DIR}/reviews_processed.csv")
print("Step 2 complete.")
