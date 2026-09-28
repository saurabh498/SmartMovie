"""Ranking logic — combine similarity, rating, popularity, preference match."""
import numpy as np


# Default weights (configurable)
DEFAULT_WEIGHTS = {
    "similarity": 0.60,
    "rating": 0.20,
    "popularity": 0.10,
    "preference": 0.10,
}


def _minmax(values):
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _log_norm(values):
    values = np.asarray(values, dtype=float)
    values = np.log1p(np.clip(values, 0, None))
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def score_candidates(candidates_df, sims, user_genres=None, weights=None):
    """Add a final_score column combining similarity + rating + popularity + preference."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    df = candidates_df.copy()
    df["similarity"] = sims

    # Normalize components
    df["_sim_norm"] = _minmax(df["similarity"])
    df["_rating_norm"] = _minmax(df["rating"])
    df["_pop_norm"] = _log_norm(df["popularity"])

    # Preference match: fraction of user's chosen genres present in the movie
    user_genres = set(user_genres or [])
    if user_genres:
        df["_pref_norm"] = df["genres_list"].apply(
            lambda gs: len(user_genres & set(gs)) / max(len(user_genres), 1)
        )
    else:
        df["_pref_norm"] = 0.0

    df["final_score"] = (
        weights["similarity"] * df["_sim_norm"]
        + weights["rating"] * df["_rating_norm"]
        + weights["popularity"] * df["_pop_norm"]
        + weights["preference"] * df["_pref_norm"]
    )
    return df.sort_values("final_score", ascending=False)


def apply_filters(df, genres=None, min_rating=None, year_from=None, year_to=None):
    """Apply user-selected filters to the candidate pool."""
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
    return out