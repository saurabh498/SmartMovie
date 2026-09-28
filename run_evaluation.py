"""SmartMovie — evaluation.

Compares two models on the same test set:

    BASELINE  =  TF-IDF + Cosine Similarity (top-K by similarity only)
    PROPOSED  =  TF-IDF + Cosine Similarity + Weighted Ranking
                 + Preference Matching + Diversity

Note on preference matching: since the dataset has no real user history,
the source movie's own genres are used as the user preference vector.
This makes the preference component of the ranking formula actually active.
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath("."))

from src.config import RANKING_WEIGHTS, DIVERSITY_ENABLED
from src.ranking import score_candidates, diversify

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

MODELS = Path("models")
EVAL   = Path("evaluation")
SHOTS  = Path("screenshots")
EVAL.mkdir(exist_ok=True)
SHOTS.mkdir(exist_ok=True)

with open(MODELS / "movies.pkl", "rb") as f:
    movies = pickle.load(f)
with open(MODELS / "tfidf_matrix.pkl", "rb") as f:
    matrix = pickle.load(f)

print(f"Movies: {len(movies)}  |  Matrix: {matrix.shape}")


# ---------- Metrics ----------
def top_k_by_similarity(idx, k=10):
    sims = cosine_similarity(matrix[idx], matrix).flatten()
    sims[idx] = -1.0
    order = np.argsort(sims)[::-1][:k]
    return order, sims[order]


def precision_at_k(source_idx, rec_indices, k=10):
    src = set(movies.iloc[source_idx]["genres_list"])
    if not src:
        return np.nan
    top = rec_indices[:k]
    return sum(1 for i in top if set(movies.iloc[i]["genres_list"]) & src) / k


def recall_at_k(source_idx, rec_indices, k=10, pool_size=50):
    src = set(movies.iloc[source_idx]["genres_list"])
    if not src:
        return np.nan
    sims_all = cosine_similarity(matrix[source_idx], matrix).flatten()
    sims_all[source_idx] = -1.0
    pool = np.argsort(sims_all)[::-1][:pool_size]
    relevant = [i for i in pool if set(movies.iloc[i]["genres_list"]) & src]
    if not relevant:
        return 0.0
    top = rec_indices[:k]
    return sum(1 for i in top if i in set(relevant)) / len(relevant)


def genre_diversity(rec_indices):
    gs = set()
    for i in rec_indices:
        gs.update(movies.iloc[i]["genres_list"])
    return len(gs)


# ---------- Proposed model — preference now active ----------
def proposed_recommendations(source_idx, k=10):
    """Full pipeline: similarity + rating + popularity + preference + diversity.

    The source movie's own genres are used as the user-preference vector.
    This is the evaluation proxy for 'user wants more like this movie'.
    """
    source_genres = movies.iloc[source_idx]["genres_list"]

    sims = cosine_similarity(matrix[source_idx], matrix).flatten()
    sims[source_idx] = -1.0

    cand = movies.copy()
    cand["similarity"] = sims
    cand = cand[cand["similarity"] > 0]
    if cand.empty:
        return np.array([], dtype=int)

    # ← Fixed: user_genres is now passed so preference matching is active
    ranked = score_candidates(cand, cand["similarity"].values,
                              user_genres=source_genres)

    if DIVERSITY_ENABLED and len(ranked) > k:
        ranked = diversify(ranked, matrix, k)
    else:
        ranked = ranked.head(k)

    return ranked.index.to_numpy()


# ---------- Test set ----------
TEST_TITLES = [
    "Interstellar", "Inception", "The Dark Knight", "Pulp Fiction",
    "Avatar", "The Matrix", "Fight Club", "Forrest Gump",
    "The Shawshank Redemption", "The Godfather", "Whiplash", "Mad Max: Fury Road",
]

test_idx, test_names = [], []
for t in TEST_TITLES:
    m = movies[movies["title"].str.lower() == t.lower()]
    if not m.empty:
        test_idx.append(m.index[0])
        test_names.append(m.iloc[0]["title"])

print(f"Found {len(test_idx)} / {len(TEST_TITLES)} test movies\n")


# ---------- Run both models ----------
rows = []
for idx, name in zip(test_idx, test_names):
    base_idx, _ = top_k_by_similarity(idx, k=10)
    prop_idx = proposed_recommendations(idx, k=10)

    rows.append({
        "title": name,
        "base_P@5":  round(precision_at_k(idx, base_idx, 5), 3),
        "base_P@10": round(precision_at_k(idx, base_idx, 10), 3),
        "base_R@10": round(recall_at_k(idx, base_idx, 10, 50), 3),
        "base_div":  genre_diversity(base_idx),
        "prop_P@5":  round(precision_at_k(idx, prop_idx, 5), 3) if len(prop_idx) else np.nan,
        "prop_P@10": round(precision_at_k(idx, prop_idx, 10), 3) if len(prop_idx) else np.nan,
        "prop_R@10": round(recall_at_k(idx, prop_idx, 10, 50), 3) if len(prop_idx) else np.nan,
        "prop_div":  genre_diversity(prop_idx) if len(prop_idx) else 0,
    })

df_results = pd.DataFrame(rows)
print(df_results.to_string(index=False), "\n")


# ---------- Coverage ----------
rng = np.random.default_rng(42)
sample = rng.choice(len(movies), size=min(200, len(movies)), replace=False)

base_seen, prop_seen = set(), set()
for idx in sample:
    b_idx, _ = top_k_by_similarity(idx, k=10)
    base_seen.update(movies.iloc[b_idx]["id"].tolist())
    p_idx = proposed_recommendations(idx, k=10)
    if len(p_idx):
        prop_seen.update(movies.iloc[p_idx]["id"].tolist())

coverage_base = len(base_seen) / len(movies)
coverage_prop = len(prop_seen) / len(movies)


# ---------- Query time ----------
def timeit(fn, n=100):
    fn(0)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(0)
    return (time.perf_counter() - t0) / n * 1000


time_base = timeit(lambda i: top_k_by_similarity(i, k=10))
time_prop = timeit(lambda i: proposed_recommendations(i, k=10))


# ---------- Summary ----------
summary = pd.DataFrame({
    "Metric": [
        "Test set size",
        "Precision@5",
        "Precision@10",
        "Recall@10",
        "Coverage (200 queries)",
        "Mean genre diversity",
        "Query time (ms)",
    ],
    "Baseline": [
        len(test_idx),
        round(df_results["base_P@5"].mean(), 3),
        round(df_results["base_P@10"].mean(), 3),
        round(df_results["base_R@10"].mean(), 3),
        round(coverage_base, 4),
        round(df_results["base_div"].mean(), 2),
        round(time_base, 2),
    ],
    "Proposed": [
        len(test_idx),
        round(df_results["prop_P@5"].mean(), 3),
        round(df_results["prop_P@10"].mean(), 3),
        round(df_results["prop_R@10"].mean(), 3),
        round(coverage_prop, 4),
        round(df_results["prop_div"].mean(), 2),
        round(time_prop, 2),
    ],
})

print("=" * 60)
print("BASELINE vs PROPOSED")
print("=" * 60)
print(summary.to_string(index=False))


# ---------- Precision chart ----------
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(df_results))
w = 0.35
ax.bar(x - w/2, df_results["base_P@10"], width=w, label="Baseline P@10", color="#5cc8ff")
ax.bar(x + w/2, df_results["prop_P@10"], width=w, label="Proposed P@10", color="#7c5cff")
ax.set_xticks(x)
ax.set_xticklabels(df_results["title"], rotation=40, ha="right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_title("Precision@10 — Baseline vs Proposed", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(SHOTS / "eval_precision_at_k.png", bbox_inches="tight")
plt.close()


# ---------- Similarity distribution ----------
inter = movies[movies["title"] == "Interstellar"]
if not inter.empty:
    ii = inter.index[0]
    sims_all = cosine_similarity(matrix[ii], matrix).flatten()
    sims_all[ii] = 0
    top_sims = []
    for idx in test_idx:
        _, s = top_k_by_similarity(idx, 10)
        top_sims.extend(s)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(sims_all, bins=50, color="#7c5cff", alpha=0.85, edgecolor="white")
    axes[0].set_title("Similarity — Interstellar vs catalog")
    axes[0].set_xlabel("Cosine similarity")
    axes[1].hist(top_sims, bins=30, color="#ff5c8a", alpha=0.85, edgecolor="white")
    axes[1].set_title("Top-10 similarity across test set")
    axes[1].set_xlabel("Cosine similarity")
    plt.tight_layout()
    plt.savefig(SHOTS / "eval_similarity_distribution.png", bbox_inches="tight")
    plt.close()


# ---------- Save ----------
df_results.to_csv(EVAL / "evaluation_results.csv", index=False)
summary.to_csv(EVAL / "evaluation_summary.csv", index=False)

# Also dump the dataset size so README can stay in sync
with open(EVAL / "dataset_size.txt", "w") as f:
    f.write(str(len(movies)))

print(f"\n✓ Saved: {EVAL}/evaluation_results.csv")
print(f"✓ Saved: {EVAL}/evaluation_summary.csv")
print(f"✓ Saved: {EVAL}/dataset_size.txt")
print(f"✓ Saved: {SHOTS}/eval_precision_at_k.png")
print(f"✓ Saved: {SHOTS}/eval_similarity_distribution.png")