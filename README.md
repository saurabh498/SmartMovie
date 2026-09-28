# 🎬 SmartMovie — Intelligent Movie Recommendation System

Content-based movie recommender using **TF-IDF** + **Cosine Similarity** across five
movie features, with **explainable recommendations**, filters, diversity, and a
5-page **Streamlit** GUI. Built on the **TMDB 5000** dataset.

---

## 1. Project Overview

SmartMovie recommends movies by matching the content profile of a reference movie
(or the user's stated preferences) against a catalog of ~4,700 movies. It combines
five features per movie into a weighted representation, vectorizes with TF-IDF,
ranks with cosine similarity plus a blended quality score, and explains every result.

## 2. Problem Statement

Users face a discovery problem: too many movies, too little time. Genre-only
recommenders return technically similar but practically irrelevant results. Given
a catalog with structured metadata, produce a ranked list of Top-N movies relevant
to the user's input, with reasons.

## 3. Existing System

```
User selects movie → Similarity → Retrieve → Display
```

Single-feature matching (usually genre), no ranking layer, no explanation, no filters.

## 4. Limitations of Existing System

| # | Limitation | Effect |
|---|---|---|
| 1 | Single-feature similarity | Same label ≠ same story |
| 2 | No user preferences | No filter by rating/year/popularity |
| 3 | Cold-start | No recommendations for new users |
| 4 | No explanation | "Recommended" with no reason |
| 5 | No filtering | Low-rated or old movies appear |
| 6 | No diversity | Sequels/remakes crowd the list |

## 5. Proposed System

Multi-feature weighted representation + layered ranking + diversity:

```
Movie Dataset → Preprocessing → Feature Engineering → TF-IDF
     → Cosine Similarity → Candidate Generation → Ranking & Filtering
     → Diversity → Top-N Movies → Streamlit GUI → User
```

Two modes: **content-based** (pick a movie) and **preference-based** (cold start).

## 6. Objectives

1. Multi-feature content similarity
2. Fast inference (sub-100 ms)
3. Explainable recommendations with score breakdown
4. Cold-start handling via preferences
5. Diversity-aware ranking
6. Interactive 5-page GUI
7. Robust error handling
8. Reproducible baseline-vs-proposed evaluation

## 7. Features

| Feature | Description |
|---|---|
| Partial-match search | `ince` → Inception |
| Multi-feature similarity | Genres + keywords + cast + director + overview |
| Weighted ranking | 0.60 sim + 0.20 rating + 0.10 popularity + 0.10 preference |
| Score breakdown | Every card shows all four normalized components |
| Dynamic explanations | Reasons reflect actual feature overlap, never invented |
| Low-similarity fallback | Warns when best match is below 15% similarity |
| Filters + Reset | Genre, rating, year, popularity, Top-N; Reset button |
| Diversity (MMR-lite) | Penalizes near-duplicates already picked |
| Posters | Real TMDB posters with gradient fallback |
| Cold-start mode | Preference tab for new users |
| Popular & Top Rated | Discovery sections on home |
| Model Evaluation page | Baseline vs Proposed comparison |
| About page | Algorithm, weights, architecture |
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

**Merged + cleaned → `data/movies.csv`.**

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
| poster_url | TMDB poster URL |

Duplicates removed, incomplete records dropped, movies before 1900 dropped.

## 10. Data Preprocessing

`prepare_tmdb.py` merges movies + credits, parses JSON-like list columns into
pipe-separated strings, extracts year, builds poster URLs, drops empty/duplicate rows.

`src/preprocessing.py` re-parses list fields at load time and coerces numeric columns.

## 11. Feature Engineering

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

Length-independent, so short and long overviews of the same concepts both score high.

## 14. Recommendation Algorithm

1. Compute cosine similarity from source to all movies
2. Zero out the source
3. Keep positive-similarity candidates
4. Apply filters (genre, rating, year, popularity)
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

Configurable in **one place**: `src/config.py`. This is a **ranking score**, not accuracy.

## 16. GUI

**5 pages:**

- 🏠 **Home** — search + How SmartMovie Works + Popular + Top Rated
- 🎬 **Find Similar Movies** — movie details + recommendations with similarity, score breakdown, dynamic reasons
- 🎯 **Recommend by Preferences** — cold-start mode
- 📊 **Model Evaluation** — Baseline vs Proposed comparison with charts
- ℹ️ **About** — algorithm, weights, architecture, limitations

Sidebar has filters + Reset Filters + diversity toggle.

## 17. Evaluation

`run_evaluation.py` compares **Baseline** (TF-IDF + cosine) vs **Proposed**
(cosine + weighted ranking + preference + diversity) on the same test set:

- Precision@5, Precision@10
- Recall@10 (pooled at top-50)
- Catalog coverage (200 random queries)
- Mean genre diversity
- Query latency (ms)

**Relevance proxy:** a recommendation is "relevant" if it shares at least one genre
with the source movie. This is a qualitative check, not a formal IR benchmark.

## 18. Results

See `evaluation/evaluation_summary.csv` for the current numbers on your dataset.
Typical values on TMDB 5000:

| Metric | Baseline | Proposed |
|---|---|---|
| Precision@5 | ~0.82 | ~0.87 |
| Precision@10 | ~0.80 | ~0.87 |
| Recall@10 | ~0.18 | ~0.20 |
| Coverage (200 queries) | ~28% | ~32% |
| Mean genre diversity | ~8.0 | ~8.6 |
| Query time (ms) | ~5.0 | ~5.5 |

> **These are ranking metrics, not accuracy.** Precision@K measures content
> relevance, not user satisfaction. Recall@10 is bounded above by ~0.33 under
> top-50 pooling.

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
python run_evaluation.py       # metrics → evaluation/*.csv + charts
streamlit run app.py           # launch GUI at http://localhost:8501
```