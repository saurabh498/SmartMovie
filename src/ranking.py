"""Ranking: normalize + blend + filter + diversity."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import RANKING_WEIGHTS, COLD_START_WEIGHTS, DIVERSITY_PENALTY


def _minmax(values):
    v = np.asarray(values, dtype=float)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def _log_norm(values):
    v = np.log1p(np.clip(np.asarray(values, dtype=float), 0, None))
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def score_candidates(candidates_df, sims, user_genres=None, weights=None):
    weights = weights or RANKING_WEIGHTS
    df = candidates_df.copy()
    df["similarity"] = sims

    df["_sim_norm"]    = _minmax(df["similarity"])
    df["_rating_norm"] = _minmax(df["rating"])
    df["_pop_norm"]    = _log_norm(df["popularity"])

    user_genres = set(user_genres or [])
    if user_genres:
        df["_pref_norm"] = df["genres_list"].apply(
            lambda gs: len(user_genres & set(gs)) / len(user_genres)
        )
    else:
        df["_pref_norm"] = 0.0

    df["final_score"] = (
        weights["similarity"] * df["_sim_norm"]
        + weights["rating"]    * df["_rating_norm"]
        + weights["popularity"] * df["_pop_norm"]
        + weights["preference"] * df["_pref_norm"]
    )
    return df.sort_values("final_score", ascending=False)


def apply_filters(df, genres=None, min_rating=None, year_from=None, year_to=None,
                  min_popularity=None):
    out = df.copy()
    if genres:
        wanted = set(genres)
        out = out[out["genres_list"].apply(lambda gs: len(wanted & set(gs)) > 0)]
    if min_rating is not None:
        out = out[out["rating"] >= min_rating]
    if year_from is not None:
        out = out[out["year"] >= year_from]
    if year_to is not None:
        out = out[out["year"] <= year_to]
    if min_popularity is not None:
        out = out[out["popularity"] >= min_popularity]
    return out


def diversify(candidates_df, matrix, top_n, penalty=None):
    """MMR-lite: iteratively pick the top-scored candidate, then penalize
    remaining candidates by their max cosine similarity to already-picked ones.
    Reduces near-duplicate recommendations (sequels, remakes, spin-offs).
    """
    penalty = DIVERSITY_PENALTY if penalty is None else penalty
    if len(candidates_df) <= top_n or penalty <= 0:
        return candidates_df.head(top_n)

    local_idx = list(range(len(candidates_df)))
    global_idx = candidates_df.index.tolist()
    scores = candidates_df["final_score"].values

    picked_local = []
    remaining = list(local_idx)

    for _ in range(min(top_n, len(local_idx))):
        if not picked_local:
            best_local = int(np.argmax(scores))
        else:
            picked_global = [global_idx[i] for i in picked_local]
            rem_global    = [global_idx[i] for i in remaining]
            pairwise = cosine_similarity(matrix[rem_global], matrix[picked_global]).max(axis=1)
            adjusted = np.array([scores[i] for i in remaining]) - penalty * pairwise
            best_local = remaining[int(np.argmax(adjusted))]

        picked_local.append(best_local)
        remaining.remove(best_local)

    return candidates_df.iloc[picked_local]