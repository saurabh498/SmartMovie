"""SmartMovie — Streamlit GUI. 5-page app with sidebar navigation."""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RANKING_WEIGHTS, COLD_START_WEIGHTS, DIVERSITY_ENABLED
from src.ranking import score_candidates, apply_filters, diversify

# ---------- Page config ----------
st.set_page_config(page_title="SmartMovie", page_icon="🎬", layout="wide")

# ---------- Global CSS ----------
st.markdown("""
<style>
  .stApp { background: #0f1117; }
  section[data-testid="stSidebar"] { background: #141824; }
  h1, h2, h3, h4 { color: #e8eaf0 !important; }
  .hero {
    background: linear-gradient(135deg, #7c5cff 0%, #ff5c8a 100%);
    padding: 40px 32px; border-radius: 16px; margin-bottom: 24px;
    color: white;
  }
  .hero h1 { color: white !important; margin: 0; font-size: 2.4rem; }
  .hero p { color: rgba(255,255,255,.85); margin: 8px 0 0 0; }
  .poster-fallback {
    background: linear-gradient(135deg, #7c5cff, #ff5c8a);
    height: 260px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 4rem; font-weight: 800; color: white;
  }
  .metric-card {
    background: #171b28; border: 1px solid #232838;
    border-radius: 12px; padding: 16px; text-align: center;
  }
  .metric-value { font-size: 1.8rem; font-weight: 800; color: #7c5cff; }
  .metric-label { font-size: .8rem; color: #8b93a7; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    required = [Path("models/movies.pkl"), Path("models/tfidf_matrix.pkl")]
    for p in required:
        if not p.exists() or p.stat().st_size == 0:
            return None, f"Missing or empty: {p}"
    try:
        with open(required[0], "rb") as f:
            movies = pickle.load(f)
        with open(required[1], "rb") as f:
            matrix = pickle.load(f)
        # Ensure poster_url column exists
        if "poster_url" not in movies.columns:
            movies["poster_url"] = ""
        return (movies, matrix), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


artifacts, load_error = load_artifacts()

if artifacts is None:
    st.error(f"⚠️ Model not ready — {load_error}")
    st.info("Run `python prepare_tmdb.py && python build_model.py`, then refresh.")
    st.stop()

movies_df, tfidf_matrix = artifacts


# ---------- Helpers ----------
def safe_poster(url):
    """Return valid URL or None."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url or url.lower() in ("nan", "none"):
        return None
    return url


def render_card(col, movie, similarity=None, reasons=None, show_score=False):
    """Render a recommendation card inside a column."""
    with col:
        with st.container(border=True):
            poster = safe_poster(movie.get("poster_url"))
            if poster:
                try:
                    st.image(poster, width='stretch')
                except Exception:
                    st.markdown(
                        f"<div class='poster-fallback'>{str(movie['title'])[0]}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"<div class='poster-fallback'>{str(movie['title'])[0]}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(f"**{movie['title']}**")
            st.caption(f"⭐ {movie['rating']}  ·  📅 {movie['year']}")

            genres = " • ".join(list(movie.get("genres_list", []))[:3])
            if genres:
                st.caption(genres)

            if similarity is not None:
                pct = int(round(float(similarity) * 100))
                st.progress(min(float(similarity), 1.0), text=f"Similarity: {pct}%")

            if show_score and "final_score" in movie:
                st.caption(f"Ranking score: {float(movie['final_score']):.3f}")

            if reasons:
                with st.expander("Why recommended?"):
                    for r in reasons:
                        st.markdown(f"• {r}")


def explain(source_row, target_row, user_genres=None, similarity=None):
    reasons = []
    sg = set(source_row["genres_list"]) & set(target_row["genres_list"])
    if sg:
        reasons.append(f"Same genre: **{', '.join(sorted(sg))}**")

    sk = set(source_row["keywords_list"]) & set(target_row["keywords_list"])
    if sk:
        reasons.append(f"Similar keywords: {', '.join(sorted(sk)[:4])}")

    if (source_row.get("director") and
            source_row["director"] == target_row["director"]):
        reasons.append(f"Same director: **{source_row['director']}**")

    sc = set(source_row["cast_list"]) & set(target_row["cast_list"])
    if sc:
        reasons.append(f"Common cast: {', '.join(sorted(sc)[:3])}")

    if target_row["rating"] >= 8.0:
        reasons.append(f"Highly rated: ⭐ {target_row['rating']}")

    if user_genres:
        matched = set(user_genres) & set(target_row["genres_list"])
        if matched:
            reasons.append(f"Matches your preferences: {', '.join(sorted(matched))}")

    if similarity is not None and similarity > 0:
        reasons.append(f"High content similarity: {int(similarity*100)}%")

    if not reasons:
        reasons.append("Overall content profile is similar")
    return reasons


def get_recommendations(source_idx, top_n, filters, user_genres=None, use_diversity=True):
    sims = cosine_similarity(tfidf_matrix[source_idx], tfidf_matrix).flatten()
    sims[source_idx] = -1.0

    cand = movies_df.copy()
    cand["similarity"] = sims
    cand = cand[cand["similarity"] > 0]

    cand = apply_filters(
        cand,
        genres=filters.get("genres"),
        min_rating=filters.get("min_rating"),
        year_from=filters.get("year_from"),
        year_to=filters.get("year_to"),
        min_popularity=filters.get("min_popularity"),
    )
    if cand.empty:
        return cand

    ranked = score_candidates(cand, cand["similarity"].values,
                              user_genres=user_genres or filters.get("genres"))

    if use_diversity and DIVERSITY_ENABLED and len(ranked) > top_n:
        ranked = diversify(ranked, tfidf_matrix, top_n)
    else:
        ranked = ranked.head(top_n)

    return ranked


def search_movie(query):
    """Partial, case-insensitive title search. Returns (row, match_type)."""
    if not query or not query.strip():
        return None, "empty"
    q = query.strip().lower()
    titles = movies_df["title"].str.lower()

    exact = movies_df[titles == q]
    if not exact.empty:
        return exact.iloc[0], "exact"

    starts = movies_df[titles.str.startswith(q)]
    if not starts.empty:
        return starts.iloc[0], "starts"

    contains = movies_df[titles.str.contains(q, regex=False)]
    if not contains.empty:
        return contains.iloc[0], "contains"

    return None, "not_found"


def render_movie_details(row):
    st.markdown(f"## {row['title']}")
    c1, c2 = st.columns([1, 3])
    with c1:
        poster = safe_poster(row.get("poster_url"))
        if poster:
            try:
                st.image(poster, width='stretch')
            except Exception:
                st.markdown(
                    f"<div class='poster-fallback'>{row['title'][0]}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div class='poster-fallback'>{row['title'][0]}</div>",
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown(f"⭐ **Rating:** {row['rating']}  |  📅 **Year:** {row['year']}")
        st.markdown(f"🔥 **Popularity:** {row['popularity']:.1f}")
        st.markdown(f"🎭 **Genres:** {', '.join(row['genres_list']) or '—'}")
        st.markdown(f"🎬 **Director:** {row.get('director') or '—'}")
        cast = ", ".join(list(row.get("cast_list", []))[:5]) or "—"
        st.markdown(f"👥 **Cast:** {cast}")
        st.markdown("**Overview:**")
        st.write(row.get("overview", "") or "No overview available.")


# ---------- Sidebar navigation ----------
with st.sidebar:
    st.markdown("## 🎬 SmartMovie")
    st.caption("Intelligent Movie Recommender")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🎬 Find Similar Movies", "🎯 Recommend by Preferences",
         "📊 Model Evaluation", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("### ⚙️ Filters")
    all_genres = sorted({g for gs in movies_df["genres_list"] for g in gs})
    sel_genres = st.multiselect("Genres", all_genres, default=[])
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 6.0, 0.1)
    yr_min = int(movies_df["year"].min())
    yr_max = int(movies_df["year"].max())
    year_range = st.slider("Release year range", yr_min, yr_max, (yr_min, yr_max))
    min_pop = st.slider("Minimum popularity", 0.0,
                        float(movies_df["popularity"].max()), 0.0, 1.0)
    top_n = st.selectbox("Number of recommendations", [5, 8, 10, 15], index=1)
    use_div = st.checkbox("Diversity-aware recommendations", value=True)

filters = {
    "genres": sel_genres,
    "min_rating": min_rating,
    "year_from": year_range[0],
    "year_to": year_range[1],
    "min_popularity": min_pop,
}


# ==================================================
# PAGE: Home
# ==================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
      <h1>SmartMovie</h1>
      <p>Content-based movie recommendations using TF-IDF and cosine similarity</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Search a movie")
    q = st.text_input("Search movie", placeholder="e.g. Inception, dark knight, ince…",
                      label_visibility="collapsed")

    if q:
        row, match = search_movie(q)
        if row is None:
            st.warning("No matching movie found. Try another title.")
        else:
            st.success(f"Found: **{row['title']}** ({row['year']})")
            st.session_state["preselect_idx"] = row.name
            st.info("Go to 🎬 **Find Similar Movies** in the sidebar to see recommendations.")

    # Popular movies
    st.markdown("### 🔥 Popular Movies")
    popular = movies_df.nlargest(10, "popularity")
    cols = st.columns(5)
    for i, (_, row) in enumerate(popular.iterrows()):
        with cols[i % 5]:
            with st.container(border=True):
                poster = safe_poster(row.get("poster_url"))
                if poster:
                    try:
                        st.image(poster, width='stretch')
                    except Exception:
                        st.markdown(
                            f"<div class='poster-fallback'>{row['title'][0]}</div>",
                            unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='poster-fallback'>{row['title'][0]}</div>",
                        unsafe_allow_html=True)
                st.markdown(f"**{row['title']}**")
                st.caption(f"⭐ {row['rating']}  ·  {row['year']}")

    # Top rated
    st.markdown("### ⭐ Top Rated Movies")
    st.caption("Discovery feature — not a recommendation from the ML model.")
    top_rated = movies_df[movies_df["votes"] >= 100].nlargest(10, "rating")
    cols = st.columns(5)
    for i, (_, row) in enumerate(top_rated.iterrows()):
        with cols[i % 5]:
            with st.container(border=True):
                poster = safe_poster(row.get("poster_url"))
                if poster:
                    try:
                        st.image(poster, width='stretch')
                    except Exception:
                        st.markdown(
                            f"<div class='poster-fallback'>{row['title'][0]}</div>",
                            unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='poster-fallback'>{row['title'][0]}</div>",
                        unsafe_allow_html=True)
                st.markdown(f"**{row['title']}**")
                st.caption(f"⭐ {row['rating']}  ·  {row['year']}")


# ==================================================
# PAGE: Find Similar Movies
# ==================================================
elif page == "🎬 Find Similar Movies":
    st.markdown("# 🎬 Find Similar Movies")
    st.caption("Pick a movie you like — get the most similar titles from the catalog.")

    pre = st.session_state.get("preselect_idx", None)
    all_titles = sorted(movies_df["title"].tolist())
    default_idx = 0
    if pre is not None and pre in movies_df.index:
        pre_title = movies_df.loc[pre, "title"]
        if pre_title in all_titles:
            default_idx = all_titles.index(pre_title) + 1

    selected = st.selectbox("Search a movie", options=[""] + all_titles, index=default_idx)

    if selected:
        match = movies_df[movies_df["title"] == selected]
        source_idx = match.index[0]
        source_row = movies_df.iloc[source_idx]

        st.markdown("---")
        render_movie_details(source_row)
        st.markdown("---")
        st.markdown(f"### ✨ Recommended Movies Based on *{source_row['title']}*")

        t0 = time.perf_counter()
        recs = get_recommendations(
            source_idx, top_n, filters,
            user_genres=sel_genres or None,
            use_diversity=use_div,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        if recs.empty:
            st.warning("No movies match your current filters. Try relaxing them.")
        else:
            st.caption(f"{len(recs)} recommendations · generated in {elapsed:.1f} ms")
            cols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                reasons = explain(source_row, row, sel_genres or None,
                                  similarity=row["similarity"])
                render_card(cols[i % 4], row,
                            similarity=row["similarity"],
                            reasons=reasons)


# ==================================================
# PAGE: Recommend by Preferences (Cold Start)
# ==================================================
elif page == "🎯 Recommend by Preferences":
    st.markdown("# 🎯 Recommend by Preferences")
    st.caption("New here? Tell us what you like and we'll rank the catalog for you. "
               "This is a preference-based mode of the same content-based engine.")

    all_genres = sorted({g for gs in movies_df["genres_list"] for g in gs})
    pref_candidates = ["Science Fiction", "Sci-Fi", "Drama", "Action", "Thriller"]
    default_genres = [g for g in pref_candidates if g in all_genres][:2]

    pref_genres = st.multiselect("Preferred genres", all_genres, default=default_genres)
    c1, c2 = st.columns(2)
    with c1:
        pref_min_rating = st.slider("Minimum rating", 0.0, 10.0, 7.0, 0.1, key="p_r")
    with c2:
        pref_min_pop = st.slider("Minimum popularity", 0.0,
                                 float(movies_df["popularity"].max()),
                                 0.0, 1.0, key="p_p")

    yr_min = int(movies_df["year"].min())
    yr_max = int(movies_df["year"].max())
    pref_years = st.slider("Preferred release year range",
                           yr_min, yr_max, (max(1990, yr_min), yr_max), key="p_y")
    pref_top_n = st.selectbox("Number of recommendations", [5, 8, 10, 15],
                              index=1, key="p_n")

    if st.button("🎯 Generate Recommendations", type="primary"):
        if not pref_genres:
            st.warning("Please pick at least one genre.")
        else:
            pool = apply_filters(
                movies_df,
                genres=pref_genres,
                min_rating=pref_min_rating,
                year_from=pref_years[0],
                year_to=pref_years[1],
                min_popularity=pref_min_pop,
            )
            if pool.empty:
                st.warning("No movies match your preferences. Try relaxing the filters.")
            else:
                pool = pool.copy()
                pool["similarity"] = 0.0
                ranked = score_candidates(
                    pool,
                    np.zeros(len(pool)),
                    user_genres=pref_genres,
                    weights=COLD_START_WEIGHTS,
                ).head(pref_top_n)

                st.markdown("---")
                st.markdown(f"### ✨ Top {len(ranked)} for your preferences")
                cols = st.columns(4)
                for i, (_, row) in enumerate(ranked.iterrows()):
                    matched = set(pref_genres) & set(row["genres_list"])
                    reasons = [
                        f"Matches your preference: **{', '.join(sorted(matched))}**",
                        f"Rating ⭐ {row['rating']} (≥ {pref_min_rating})",
                        f"Popularity {row['popularity']:.1f}",
                        f"Released in {row['year']}",
                    ]
                    render_card(cols[i % 4], row, similarity=None, reasons=reasons,
                                show_score=True)


# ==================================================
# PAGE: Model Evaluation
# ==================================================
elif page == "📊 Model Evaluation":
    st.markdown("# 📊 Model Evaluation")
    st.caption("Metrics are computed by `run_evaluation.py` on the current model "
               "and dataset. Relevance is defined as **genre overlap** with the source movie.")

    eval_summary = Path("evaluation/evaluation_summary.csv")
    eval_results = Path("evaluation/evaluation_results.csv")

    if not eval_summary.exists() or not eval_results.exists():
        st.warning("Evaluation files not found. Run `python run_evaluation.py` first.")
    else:
        summary = pd.read_csv(eval_summary)
        results = pd.read_csv(eval_results)

        # Top metric cards
        metrics = dict(zip(summary["Metric"], summary["Value"]))
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-value'>"
                    f"{metrics.get('Mean P@5', 0):.3f}</div>"
                    f"<div class='metric-label'>Precision@5</div></div>",
                    unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-value'>"
                    f"{metrics.get('Mean P@10', 0):.3f}</div>"
                    f"<div class='metric-label'>Precision@10</div></div>",
                    unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-value'>"
                    f"{metrics.get('Coverage (200q)', 0)*100:.1f}%</div>"
                    f"<div class='metric-label'>Catalog coverage</div></div>",
                    unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-value'>"
                    f"{metrics.get('Query time (ms)', 0):.1f} ms</div>"
                    f"<div class='metric-label'>Avg. query time</div></div>",
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Precision & Recall")
        chart_df = results.set_index("title")[["P@5", "P@10", "R@10"]]
        st.bar_chart(chart_df, height=360)

        st.markdown("### Full Summary")
        st.dataframe(summary, width='stretch', hide_index=True)

        st.markdown("### Per-Movie Results")
        st.dataframe(results, width='stretch', hide_index=True)

        # Eval charts (from screenshots)
        for fname, caption in [
            ("eval_precision_at_k.png", "Precision@K per test movie"),
            ("eval_similarity_distribution.png",
             "Similarity distribution across catalog"),
        ]:
            p = Path("screenshots") / fname
            if p.exists():
                st.markdown(f"### {caption}")
                st.image(str(p), width='stretch')

        st.info(
            "**Interpreting these numbers.** These are ranking metrics — "
            "**not accuracy**. Precision@K measures the fraction of top-K "
            "recommendations that share at least one genre with the source movie. "
            "Genre overlap is a proxy for relevance; it is not equivalent to real "
            "user satisfaction. The observed Recall@10 (~0.20) must be read "
            "against its definition, where the theoretical ceiling under "
            "top-50 pooling is roughly 0.33."
        )


# ==================================================
# PAGE: About
# ==================================================
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About the Model")

    st.markdown
