# Sentiment Analysis & Feature Extraction from Customer Reviews

> End-to-end NLP pipeline that classifies customer review sentiment and extracts actionable product/service insights using classical machine learning and linguistic feature engineering.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [NLP Pipeline](#nlp-pipeline)
- [Feature Extraction Techniques](#feature-extraction-techniques)
- [Models & Results](#models--results)
- [Visualizations](#visualizations)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)

---

## Overview

This project performs **sentiment classification** (Positive / Neutral / Negative) and **multi-faceted feature extraction** on unstructured customer reviews. The goal is to go beyond a simple star rating and extract *why* customers feel the way they do — which product aspects drive complaints, what topics appear across reviews, and what linguistic signals correlate with each sentiment class.

**What this project does:**

| Task | Approach |
|------|----------|
| Sentiment Classification | 3-class (Positive / Neutral / Negative) |
| Feature Representation | TF-IDF unigrams + bigrams |
| Lexicon Scoring | TextBlob polarity & subjectivity |
| Entity Extraction | spaCy Named Entity Recognition |
| Aspect Analysis | Keyword-based aspect–sentiment mapping |
| Topic Discovery | Latent Dirichlet Allocation (LDA) |
| Models Compared | Logistic Regression, Random Forest, Linear SVM |

---

## Dataset

**Yelp Review Full** — publicly available via [HuggingFace Datasets](https://huggingface.co/datasets/yelp_review_full)

| Property | Value |
|----------|-------|
| Full dataset size | 650,000 training reviews |
| Sample used | **5,000** (stratified, 1,000 per star class) |
| Fields | `text`, `label` (1–5 stars) |
| Domain | Restaurants, cafés, hotels, service businesses |
| Language | Primarily English |

**Star → Sentiment mapping:**

| Stars | Sentiment Label | Count |
|-------|----------------|-------|
| 4–5 ★ | Positive | 2,000 (40%) |
| 3 ★   | Neutral  | 1,000 (20%) |
| 1–2 ★ | Negative | 2,000 (40%) |

---

## Project Structure

```
├── 01_data_loading.py          # Load Yelp dataset, map ratings → sentiment labels
├── 02_preprocessing.py         # Text cleaning, tokenization, lemmatization
├── 03_eda_visualization.py     # EDA plots: distributions, word clouds, N-grams
├── 04_feature_extraction.py    # TF-IDF, TextBlob, NER, Aspect-based, LDA
├── 05_sentiment_models.py      # Train & evaluate LR / RF / SVM, confusion matrices
├── 06_insights_report.py       # Master dashboard figure + printed summary
├── run_all.py                  # Run the full pipeline end-to-end
├── requirements.txt            # Python dependencies
├── data/
│   ├── reviews_raw.csv         # Raw loaded reviews
│   ├── reviews_processed.csv   # After cleaning & lemmatization
│   ├── reviews_features.csv    # Enriched with all extracted features
│   ├── aspect_sentiment.csv    # Aspect × sentiment count matrix
│   └── model_results.json      # Model evaluation metrics
└── plots/
    ├── 00_DASHBOARD.png        # Master summary dashboard
    ├── 01_rating_sentiment_distribution.png
    ├── 02_review_length_analysis.png
    ├── 03_wordclouds.png
    ├── 04_top_words_per_sentiment.png
    ├── 05_aspect_sentiment.png
    ├── 06_polarity_subjectivity.png
    ├── 07_lda_topic_distribution.png
    ├── 08_model_comparison.png
    ├── 09_confusion_matrices.png
    └── 10_cross_validation.png
```

---

## NLP Pipeline

```
Raw Review Text
      │
      ▼
 ┌─────────────────────────────────────┐
 │  Step 1 · Data Loading              │  HuggingFace Datasets
 │  • Yelp Review Full (stratified)    │  5,000 reviews, 1,000/class
 └─────────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────────┐
 │  Step 2 · Preprocessing             │
 │  • Lowercase, strip URLs/HTML       │
 │  • Remove punctuation & digits      │
 │  • Tokenize (spaCy)                 │
 │  • Remove stopwords (NLTK)          │
 │  • Lemmatize (spaCy en_core_web_sm) │
 └─────────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────────┐
 │  Step 3 · EDA & Visualization       │
 │  • Rating/sentiment distributions   │
 │  • Review length analysis           │
 │  • Word clouds per class            │
 │  • Top-N unigram frequencies        │
 └─────────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────────┐
 │  Step 4 · Feature Extraction        │
 │  • TF-IDF (1+2-grams, 5K features) │
 │  • TextBlob polarity/subjectivity   │
 │  • spaCy NER (brands, places, $)   │
 │  • Aspect-based keyword mapping     │
 │  • LDA topic modeling (8 topics)    │
 └─────────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────────┐
 │  Step 5 · Sentiment Classification  │
 │  • Logistic Regression              │
 │  • Random Forest (200 trees)        │
 │  • Linear SVM                       │
 │  • 80/20 split + 5-fold CV          │
 └─────────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────────┐
 │  Step 6 · Insights & Reporting      │
 │  • Master dashboard (11 plots)      │
 │  • Aspect–sentiment heatmap         │
 │  • Model comparison summary         │
 └─────────────────────────────────────┘
```

---

## Feature Extraction Techniques

### A · TF-IDF (Term Frequency–Inverse Document Frequency)

- Unigrams + bigrams, `max_features=5,000`, `sublinear_tf=True`, `min_df=5`
- Captures discriminative n-grams per sentiment class
- **Top positive terms:** `great`, `good`, `love`, `friendly`, `delicious`
- **Top negative terms:** `bad`, `order`, `service`, `never`, `awful`

### B · Lexicon-Based Sentiment (TextBlob)

Assigns each review a **polarity** (−1 to +1) and **subjectivity** (0 to 1) score using the Pattern lexicon.

| Sentiment | Avg Polarity | Avg Subjectivity |
|-----------|-------------|-----------------|
| Positive  | +0.307      | 0.571           |
| Neutral   | +0.192      | 0.521           |
| Negative  | +0.023      | 0.518           |
| Overall   | +0.170      | 0.543           |

### C · Named Entity Recognition (spaCy NER)

Detected entity types across 2,000 sampled reviews:

| Entity Type | Mentions | Examples |
|-------------|----------|---------|
| CARDINAL    | 1,383    | quantities, counts |
| ORG         | 942      | Starbucks, Yelp |
| PERSON      | 885      | staff names |
| DATE        | 746      | "last Friday", "2 weeks ago" |
| TIME        | 694      | "30 minutes", "8pm" |
| GPE         | 484      | Las Vegas, Montreal |
| MONEY       | 379      | "$15", "twenty dollars" |

### D · Aspect-Based Sentiment Analysis

Eight product/service aspects detected via keyword matching:

| Aspect | Positive | Negative | Total |
|--------|----------|----------|-------|
| Price / Value     | 649 | 756 | 1,803 |
| Customer Service  | 589 | 833 | 1,781 |
| Performance       | 251 | 295 |   734 |
| Build Quality     | 200 | 274 |   612 |
| Ease of Use       | 128 | 135 |   336 |
| Battery           |  43 | 184 |   268 |
| Sound             |  74 |  68 |   172 |
| Display           |  32 |  20 |    75 |

### E · LDA Topic Modeling (8 Topics)

| Topic | Label | Top Words |
|-------|-------|-----------|
| T1 | Beauty & Health | hair, salon, stylist, haircut |
| T2 | Non-English Content | que, les, und, das |
| T3 | Food & Dining | food, order, chicken, place |
| T4 | Travel & Hotels | luxor, rio, bath, swim |
| T5 | Entertainment | theater, cirque, popcorn |
| T6 | General Positive | great, friendly, atmosphere |
| T7 | Cafés & Bakeries | starbucks, bagel, latte |
| T8 | General Reviews | place, time, service, room |

---

## Models & Results

All models use the same TF-IDF feature pipeline with an **80/20 stratified train-test split** (train=3,999 / test=1,000). Class weights are balanced to address the neutral-class minority.

### Test-Set Performance

| Model | Accuracy | Macro F1 | Macro Prec. | Macro Rec. | ROC-AUC |
|-------|----------|----------|-------------|------------|---------|
| **Logistic Regression** | **0.7040** | **0.6601** | **0.6603** | **0.6608** | **0.7569** |
| Linear SVM              | 0.6970 | 0.6399 | 0.6419 | 0.6400 | 0.7428 |
| Random Forest           | 0.6950 | 0.5584 | 0.6217 | 0.5925 | 0.7128 |

### Per-Class F1 (Best Model — Logistic Regression)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.78 | 0.80 | 0.79 | 400 |
| Neutral  | 0.41 | 0.45 | 0.43 | 200 |
| Positive | 0.78 | 0.74 | 0.76 | 400 |
| **Macro Avg** | **0.66** | **0.66** | **0.66** | **1000** |

### Cross-Validation (5-Fold, Logistic Regression)

```
Macro F1:  0.6717 ± 0.0062   (stable across folds)
```

---

## Visualizations

All plots are saved to the `plots/` directory. The master dashboard (`00_DASHBOARD.png`) combines all key figures.

| Plot | Description |
|------|-------------|
| `01_rating_sentiment_distribution.png` | Pie chart + bar chart of class distribution |
| `02_review_length_analysis.png` | Word count histogram & boxplot by sentiment |
| `03_wordclouds.png` | Word clouds for Positive / Neutral / Negative |
| `04_top_words_per_sentiment.png` | Top 20 unigrams per class |
| `05_aspect_sentiment.png` | Aspect × sentiment heatmap and stacked bars |
| `06_polarity_subjectivity.png` | TextBlob polarity distribution and scatter |
| `07_lda_topic_distribution.png` | LDA dominant-topic distribution |
| `08_model_comparison.png` | Accuracy / F1 / ROC-AUC grouped bar chart |
| `09_confusion_matrices.png` | Confusion matrices for all 3 models |
| `10_cross_validation.png` | 5-fold CV F1 scores per fold |

---

## Installation

**Prerequisites:** Python 3.11+

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy English model
python -m spacy download en_core_web_sm
```

### requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
nltk>=3.8.0
textblob>=0.17.0
wordcloud>=1.9.0
datasets>=2.14.0
spacy>=3.7.0
tqdm>=4.65.0
```

---

## Usage

### Run the full pipeline at once

```bash
python run_all.py
```

This executes all 6 steps in sequence and prints timing for each. Outputs land in `data/` and `plots/`.

**Estimated runtime:** ~5–8 minutes (dominated by spaCy lemmatisation and NER)

### Run individual steps

```bash
python 01_data_loading.py       # ~30s  — downloads & samples dataset
python 02_preprocessing.py      # ~2min — spaCy lemmatisation
python 03_eda_visualization.py  # ~20s  — generates 4 plots
python 04_feature_extraction.py # ~3min — TF-IDF, TextBlob, NER, LDA
python 05_sentiment_models.py   # ~1min — trains 3 models, 5-fold CV
python 06_insights_report.py    # ~10s  — master dashboard + summary
```

### Output files

```
data/reviews_raw.csv          — raw loaded reviews with sentiment labels
data/reviews_processed.csv    — after cleaning and lemmatisation
data/reviews_features.csv     — enriched with TF-IDF, polarity, aspects, topics
data/aspect_sentiment.csv     — aspect × sentiment count matrix
data/model_results.json       — accuracy, F1, ROC-AUC per model
plots/*.png                   — 11 visualisation files
```

---

## Key Findings

1. **Neutral class is the hardest to classify** — it scores F1 ≈ 0.43 across all models. Neutral reviews often contain balanced positive and negative language, making them lexically ambiguous.

2. **Customer service and price are the most discussed aspects** — together accounting for 3,584 mentions. Customer service generates significantly more negative mentions (833) than positive (589).

3. **Battery complaints skew heavily negative** — 184 negative vs. only 43 positive mentions, making it the most sentiment-polarised aspect.

4. **Logistic Regression beats Random Forest and SVM** on all metrics, demonstrating that a simple linear model with good feature engineering outperforms more complex approaches on sparse TF-IDF features.

5. **TextBlob polarity gaps are modest** — the difference between neutral (+0.19) and negative (+0.02) is small, confirming that lexicon scoring alone is insufficient for fine-grained 3-class classification.

6. **LDA topic T3 (Food & Dining) dominates** — consistent with Yelp's restaurant-heavy review base; 8 coherent topics emerge with minimal overlap.

---

## Technologies Used

| Category | Library / Tool | Version |
|----------|---------------|---------|
| Data loading | HuggingFace `datasets` | ≥2.14 |
| Data manipulation | `pandas`, `numpy` | ≥2.0, ≥1.24 |
| NLP preprocessing | `spaCy` (en_core_web_sm) | ≥3.7 |
| Stopwords | `NLTK` | ≥3.8 |
| Lexicon sentiment | `TextBlob` | ≥0.17 |
| Feature extraction | `scikit-learn` TF-IDF | ≥1.3 |
| Topic modeling | `scikit-learn` LDA | ≥1.3 |
| ML models | `scikit-learn` LR / RF / SVM | ≥1.3 |
| Visualisation | `matplotlib`, `seaborn` | ≥3.7, ≥0.12 |
| Word clouds | `wordcloud` | ≥1.9 |
| Language | Python | 3.11+ |

---

## License

This project is for educational purposes. The Yelp Review Full dataset is distributed under its own terms via HuggingFace Datasets.
