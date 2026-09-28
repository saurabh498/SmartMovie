"""SmartMovie — evaluation. Outputs to evaluation/."""
import os, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath("."))
from src.ranking import score_candidates, apply_filters

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
print(f"Movies: {len(movies)}  |  Matrix: {matrix.shape}\n")


def top_k_indices(idx, k=10):
    sims = cosine_similarity(matrix[idx], matrix).flatten()
    sims[idx] = -1.0
    return np.argsort(sims)[::-1][:k], sims


def precision_at_k(idx, k=10):
    src = set(movies.iloc[idx]["genres_list"])
    if not src: return np.nan
    top, _ = top_k_indices(idx, k)
    return sum(1 for i in top if set(movies.iloc[i]["genres_list"]) & src) / k


def recall_at_k(idx, k=10, pool_size=50):
    src = set(movies.iloc[idx]["genres_list"])
    if not src: return np.nan
    _, sims = top_k_indices(idx, pool_size)
    sims[idx] = -1.0
    pool = np.argsort(sims)[::-1][:pool_size]
    relevant = [i for i in pool if set(movies.iloc[i]["genres_list"]) & src]
    if not relevant: return 0.0
    top, _ = top_k_indices(idx, k)
    return sum(1 for i in top if i in set(relevant)) / len(relevant)


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

rows = []
for idx, name in zip(test_idx, test_names):
    rows.append({
        "title": name,
        "P@5":  round(precision_at_k(idx, 5), 3),
        "P@10": round(precision_at_k(idx, 10), 3),
        "R@10": round(recall_at_k(idx, 10, 50), 3),
    })
df_results = pd.DataFrame(rows)
print(df_results.to_string(index=False), "\n")

# Precision chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_results)); w = 0.35
ax.bar(x - w/2, df_results["P@5"],  width=w, label="P@5",  color="#7c5cff")
ax.bar(x + w/2, df_results["P@10"], width=w, label="P@10", color="#ff5c8a")
ax.set_xticks(x)
ax.set_xticklabels(df_results["title"], rotation=40, ha="right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_title("Precision@K per Test Movie", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(SHOTS / "eval_precision_at_k.png", bbox_inches="tight")
plt.close()

# Similarity distribution
inter = movies[movies["title"] == "Interstellar"]
if not inter.empty:
    ii = inter.index[0]
    sims_all = cosine_similarity(matrix[ii], matrix).flatten()
    sims_all[ii] = 0
    top_sims = []
    for idx in test_idx:
        _, s = top_k_indices(idx, 10)
        top_sims.extend(s)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(sims_all, bins=50, color="#7c5cff", alpha=0.85, edgecolor="white")
    axes[0].set_title("Similarity — Interstellar vs all movies")
    axes[0].set_xlabel("Cosine similarity")
    axes[1].hist(top_sims, bins=30, color="#ff5c8a", alpha=0.85, edgecolor="white")
    axes[1].set_title("Top-10 similarity across test set")
    axes[1].set_xlabel("Cosine similarity")
    plt.tight_layout()
    plt.savefig(SHOTS / "eval_similarity_distribution.png", bbox_inches="tight")
    plt.close()

# Coverage + diversity + speed
rng = np.random.default_rng(42)
sample = rng.choice(len(movies), size=min(200, len(movies)), replace=False)
big_ids = set()
for idx in sample:
    top, _ = top_k_indices(idx, 10)
    big_ids.update(movies.iloc[top]["id"].tolist())

divs = []
for idx in test_idx:
    top, _ = top_k_indices(idx, 10)
    gs = set()
    for i in top:
        gs.update(movies.iloc[i]["genres_list"])
    divs.append(len(gs))

_ = cosine_similarity(matrix[0], matrix)
N = 100
t0 = time.perf_counter()
for _ in range(N):
    cosine_similarity(matrix[0], matrix)
sim_ms = (time.perf_counter() - t0) / N * 1000

coverage = len(big_ids) / len(movies)
print(f"Coverage (200 queries): {coverage:.4f}  ({len(big_ids)} / {len(movies)})")
print(f"Mean genre diversity  : {np.mean(divs):.2f}")
print(f"Cosine query time     : {sim_ms:.2f} ms\n")

summary = pd.DataFrame({
    "Metric": ["Test size", "Mean P@5", "Mean P@10", "Mean R@10",
               "Coverage (200q)", "Genre diversity", "Query time (ms)"],
    "Value": [len(test_idx),
              round(df_results["P@5"].mean(), 3),
              round(df_results["P@10"].mean(), 3),
              round(df_results["R@10"].mean(), 3),
              round(coverage, 4),
              round(np.mean(divs), 2),
              round(sim_ms, 2)],
})
print("=" * 45)
print("SUMMARY")
print("=" * 45)
print(summary.to_string(index=False))

df_results.to_csv(EVAL / "evaluation_results.csv", index=False)
summary.to_csv(EVAL / "evaluation_summary.csv", index=False)
print(f"\n✓ Saved to {EVAL}/")
print(f"✓ Charts in {SHOTS}/")