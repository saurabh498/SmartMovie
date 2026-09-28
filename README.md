# 🎬 SmartMovie — Intelligent Movie Recommendation System

Content-based movie recommender using **TF-IDF** + **Cosine Similarity** across five
movie features, with **explainable recommendations**, filters, diversity, and a
5-page **Streamlit** GUI. Built on the **TMDB 5000** dataset.

---

## 1. Project Overview

SmartMovie recommends movies a user is likely to enjoy by matching the **content
profile** of a reference movie (or of the user's stated preferences) against a
catalog of ~4,700 movies. It combines five features per movie into a single
weighted representation, vectorizes with TF-IDF, ranks by cosine similarity plus
a quality score, and explains every result.

## 2. Problem Statement

Users on streaming platforms face a discovery problem: too many movies, too
little time. Simple recommenders that match only on genre return technically
similar movies that are practically irrelevant. Given a catalog with structured
metadata, produce a ranked list of Top-N movies relevant to the user's input,
with reasons.

## 3. Existing System

Basic content-based recommender:

```
User selects movie → Similarity → Retrieve → Display
```

Typically uses a single feature (usually genre), no ranking layer, no
explanation, no filters.

## 4. Limitations of Existing System

| # | Limitation | Effect |
|---|---|---|
| 1 | Single-feature similarity | Same label ≠ same story |
| 2 | No user preferences | Cannot filter by rating, year, popularity |
| 3 | Cold-start | No recommendations for new users |
| 4 | No explanation | "Recommended" with no reason |
| 5 | No filtering | Low-rated or old movies can appear |
| 6 | No diversity | Near-duplicates (sequels, remakes) crowd the list |

## 5. Proposed System

Multi-feature weighted representation + layered ranking:

```
Movie Dataset → Preprocessing → Feature Engineering → TF-IDF
     → Cosine Similarity → Candidate Generation → Ranking & Filtering
     → Top-N Movies → Streamlit GUI → User
```

Two modes:

- **Content-based** — user picks a movie
- **Preference-based (cold start)** — user states preferences

## 6. Objectives

1. Multi-feature content similarity
2. Fast inference (sub-100 ms)
3. Explainable recommendations
4. Cold-start handling
5. Diversity-aware ranking
6. Interactive GUI
7. Robust error handling
8. Reproducible evaluation

## 7. Features

| Feature | Description |
|---|---|
| Search with partial match | `ince` → Inception |
| Multi-feature similarity | Genres + keywords + cast + director + overview |
| Weighted ranking | Similarity (0.60) + rating (0.20) + popularity (0.10) + preference (0.10) |
| Explainability | "Why recommended?" panel on every card |
| Filters | Genre, rating, year, popularity, Top-N |
| Diversity | MMR-lite penalises near-duplicates |
| Posters | Real TMDB posters with fallback placeholder |
| Cold-start | Preference tab for new users |
| Popular & Top Rated | Discovery sections on home |
| Model Evaluation page | Precision@K, Recall@K, coverage, timing |
| About page | Algorithm summary, weights, architecture |
| Error handling | Unknown movie, empty search, missing poster/director/cast/overview |

## 8. System Architecture

```
                 Movie Dataset
                      ↓
              Data Preprocessing
                      ↓
              Feature Engineering
                      ↓
                 TF-IDF
                      ↓
             Cosine Similarity
                      ↓
             Candidate Generation
                      ↓
              Ranking & Filtering
                      ↓
                Top-N Movies
                      ↓
              Streamlit Web GUI
                      ↓
                 User Results
```

## 9. Dataset

**Source:** TMDB 5000 Movie Dataset (Kaggle)

**Files:** `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`

**Merged, cleaned, and stored in `data/movies.csv`.**

**Columns:**

| Column | Purpose |
|---|---|
| id | Movie ID |
| title | Title |
| genres | Pipe-separated genres |
| keywords | Pipe-separated keywords |
| overview | Plot summary |
| cast | Top-5 cast members |
| director | Director |
| rating | Vote average |
| votes | Vote count |
| popularity | Popularity |
| year | Release year |
| poster_url | Full TMDB poster URL |

Duplicates removed, incomplete records dropped, movies before 1900 dropped.

## 10. Data Preprocessing

`prepare_tmdb.py` merges movies + credits, parses JSON-like list columns into
pipe-separated strings, extracts year from `release_date`, builds poster URLs,
drops empty/duplicate rows.

`src/preprocessing.py` coerces numeric columns and re-parses list fields at load time.

## 11. Feature Engineering

Each movie becomes a single weighted token string:

```
soup = genres*3 + keywords*2 + cast*2 + director*3 + overview*1
```

Weights live in `src/config.py` → `FEATURE_WEIGHTS`.

## 12. TF-IDF

```python
TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
```

Each movie → a sparse vector in a 50,000-term vocabulary.

## 13. Cosine Similarity

```
cos(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

Measures the angle between two TF-IDF vectors — independent of document length.

## 14. Recommendation Algorithm

1. Compute cosine similarity from source movie to all others
2. Zero out the source itself
3. Keep positive-similarity candidates
4. Apply user filters (genre, rating, year, popularity)
5. Normalize similarity, rating, popularity, preference-match
6. Blend with weights from `src/config.py`
7. Apply diversity (MMR-lite)
8. Return Top-N

## 15. Ranking Method

```
final_score = 0.60 × similarity
            + 0.20 × rating
            + 0.10 × popularity
            + 0.10 × preference_match
```

Weights are configurable in **one place**: `src/config.py`. This is a **ranking
score**, not accuracy.

## 16. GUI

**5-page Streamlit app:**

- 🏠 **Home** — search + Popular Movies + Top Rated
- 🎬 **Find Similar Movies** — pick a movie → recommendations with similarity + reasons
- 🎯 **Recommend by Preferences** — cold-start mode with genre/rating/popularity/year
- 📊 **Model Evaluation** — Precision@K, Recall@K, coverage, timing
- ℹ️ **About** — algorithm, weights, architecture, limitations

Sidebar contains global filters + diversity toggle.

## 17. Evaluation

`run_evaluation.py` computes:

- Precision@K (K = 5, 10)
- Recall@10 (pooled at top-50)
- Catalog coverage (200 random queries)
- Mean genre diversity
- Query latency (ms)

**Relevance proxy:** a recommendation is "relevant" if it shares at least one
genre with the source movie. This is a qualitative sanity check, not a
formal IR benchmark.

## 18. Results

| Metric | Value |
|---|---|
| Test set | 12 movies |
| Mean Precision@5 | ~0.867 |
| Mean Precision@10 | ~0.867 |
| Mean Recall@10 | ~0.196 |
| Catalog coverage (200 queries, top-10) | ~32.5% |
| Mean genre diversity per top-10 | ~8.6 |
| Cosine query time | ~5.5 ms |

> **This is a ranking score, not accuracy.** Precision@K measures content
> relevance, not user satisfaction. Recall@10 must be read against its
> theoretical ceiling of ~0.33 under top-50 pooling.

## 19. Limitations

- Content-based only (no collaborative filtering)
- No persistent user accounts
- Static dataset snapshot (~2017)
- Quality bounded by metadata completeness
- Similarity scores are content overlap, not probability of enjoyment

## 20. Future Scope

1. Collaborative filtering
2. Hybrid recommender
3. User accounts and watch history
4. Semantic embeddings (sentence-transformers)
5. Live TMDB API integration
6. Feedback-based learning (👍/👎)
7. Diversity-aware ranking with configurable weights

## 21. Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in `data/`.

## 22. How to Run

```bash
python prepare_tmdb.py         # merge + clean → data/movies.csv
python build_model.py          # train TF-IDF → models/*.pkl
python run_evaluation.py       # metrics → evaluation/*.csv + screenshots/*.png
streamlit run app.py           # launch GUI at http://localhost:8501
```

---

## Author

[Your Name] · Roll No: XXXXXXXX · [College Name]