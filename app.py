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

st.set_page_config(page_title="SmartMovie", page_icon="🎬", layout="wide")

st.markdown("""
<style>
  .stApp { background: #0f1117; }
  section[data-testid="stSidebar"] { background: #141824; }
  h1, h2, h3, h4 { color: #e8eaf0 !important; }
  .hero {
    background: linear-gradient(135deg, #7c5cff 0%, #ff5c8a 100%);
    padding: 40px 32px; border-radius: 16px; margin-bottom: 24px; color: white;
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


# ---------- Load ----------
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

LOW_SIM_THRESHOLD = 0.15  # below this, we warn the user


# ---------- Helpers ----------
def safe_poster(url):
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url or url.lower() in ("nan", "none"):
        return None
    return url


def show_poster(movie):
    poster = safe_poster(movie.get("poster_url"))
    if poster:
        try:
            st.image(poster, width="stretch")
            return
        except Exception:
            pass
    st.markdown(
        f"<div class='poster-fallback'>{str(movie['title'])[0]}</div>",
        unsafe_allow_html=True,
    )


def dynamic_reasons(source_row, target_row, user_genres=None, similarity=None):
    """Generate reasons based on actual feature overlap. Never invents matches."""
    reasons = []

    sg = sorted(set(source_row["genres_list"]) & set(target_row["genres_list"]))
    if sg:
        reasons.append(f"Same genre: **{', '.join(sg)}**")

    if (source_row.get("director") and
            source_row["director"] == target_row["director"]):
        reasons.append(f"Same director: **{source_row['director']}**")

    sc = sorted(set(source_row["cast_list"]) & set(target_row["cast_list"]))
    if sc:
        reasons.append(f"Common cast: {', '.join(sc[:3])}")

    sk = sorted(set(source_row["keywords_list"]) & set(target_row["keywords_list"]))
    if sk:
        reasons.append(f"Similar themes: {', '.join(sk[:4])}")

    if target_row["rating"] >= 8.0:
        reasons.append(f"Highly rated: ⭐ {target_row['rating']}")

    if user_genres:
        matched = sorted(set(user_genres) & set(target_row["genres_list"]))
        if matched:
            reasons.append(f"Matches your preferences: {', '.join(matched)}")

    if similarity is not None and similarity > 0:
        pct = int(round(float(similarity) * 100))
        reasons.append(f"Content similarity: **{pct}%**")

    if not reasons:
        reasons.append("Best available match in the current catalog")
    return reasons


def score_breakdown_html(row):
    """Render the four normalized components + final score as a compact table."""
    sim = row.get("_sim_norm", 0.0)
    rat = row.get("_rating_norm", 0.0)
    pop = row.get("_pop_norm", 0.0)
    prf = row.get("_pref_norm", 0.0)
    fin = row.get("final_score", 0.0)
    return (
        f"<div style='font-size:.78rem;color:#9aa3ba;line-height:1.7'>"
        f"<b>Score breakdown</b><br>"
        f"Content similarity &nbsp; <b>{sim:.2f}</b><br>"
        f"Rating &nbsp; <b>{rat:.2f}</b><br>"
        f"Popularity &nbsp; <b>{pop:.2f}</b><br>"
        f"Preference match &nbsp; <b>{prf:.2f}</b><br>"
        f"<span style='color:#7c5cff'><b>Final score &nbsp; {fin:.3f}</b></span>"
        f"</div>"
    )


def render_card(col, movie, similarity=None, reasons=None, show_breakdown=False):
    with col:
        with st.container(border=True):
            show_poster(movie)
            st.markdown(f"**{movie['title']}**")
            st.caption(f"⭐ {movie['rating']}  ·  📅 {movie['year']}")
            genres = " • ".join(list(movie.get("genres_list", []))[:3])
            if genres:
                st.caption(genres)

            if similarity is not None:
                pct = int(round(float(similarity) * 100))
                st.progress(min(float(similarity), 1.0), text=f"Similarity: {pct}%")

            if show_breakdown:
                st.markdown(score_breakdown_html(movie), unsafe_allow_html=True)

            if reasons:
                with st.expander("Why recommended?"):
                    for r in reasons:
                        st.markdown(f"• {r}")


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
        show_poster(row)
    with c2:
        st.markdown(f"⭐ **Rating:** {row['rating']}  |  📅 **Year:** {row['year']}")
        st.markdown(f"🔥 **Popularity:** {row['popularity']:.1f}")
        st.markdown(f"🎭 **Genres:** {', '.join(row['genres_list']) or '—'}")
        st.markdown(f"🎬 **Director:** {row.get('director') or '—'}")
        cast = ", ".join(list(row.get("cast_list", []))[:5]) or "—"
        st.markdown(f"👥 **Cast:** {cast}")
        st.markdown("**Overview:**")
        st.write(row.get("overview", "") or "No overview available.")


# ---------- Sidebar ----------
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

    # Reset Filters button
    if st.button("🔄 Reset Filters", use_container_width=True):
        for k in ["f_genres", "f_rating", "f_years", "f_pop", "f_topn"]:
            st.session_state.pop(k, None)
        st.rerun()

    sel_genres = st.multiselect("Genres", all_genres, default=[], key="f_genres")
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 6.0, 0.1, key="f_rating")
    yr_min = int(movies_df["year"].min())
    yr_max = int(movies_df["year"].max())
    year_range = st.slider("Release year range", yr_min, yr_max, (yr_min, yr_max), key="f_years")
    min_pop = st.slider("Minimum popularity", 0.0,
                        float(movies_df["popularity"].max()), 0.0, 1.0, key="f_pop")
    top_n = st.selectbox("Number of recommendations", [5, 8, 10, 15], index=1, key="f_topn")
    use_div = st.checkbox("Diversity-aware recommendations", value=True)

filters = {
    "genres": sel_genres,
    "min_rating": min_rating,
    "year_from": year_range[0],
    "year_to": year_range[1],
    "min_popularity": min_pop,
}


# ==================================================
# HOME
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
            st.info("Open **🎬 Find Similar Movies** from the sidebar to see recommendations.")

    # ---- How SmartMovie Works ----
    st.markdown("---")
    st.markdown("### 🧠 How SmartMovie Works")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
**1. Feature Engineering**

Each movie is turned into a weighted combination of:
- Genres (weight 3)
- Director (weight 3)
- Keywords (weight 2)
- Cast (weight 2)
- Overview (weight 1)
""")
    c2.markdown("""
**2. Similarity**

- TF-IDF converts the weighted text into a sparse vector
- Cosine similarity measures how close two movies are
- Candidate movies = highest similarity matches
""")
    c3.markdown("""
**3. Ranking & Explanation**

- Final score = 0.60 similarity + 0.20 rating
  + 0.10 popularity + 0.10 preference
- Each result shows *why* it was recommended
""")

    st.markdown("---")
    st.markdown("### 🔥 Popular Movies")
    st.caption("Discovery feature — ranked by TMDB popularity, not the recommender.")
    popular = movies_df.nlargest(10, "popularity")
    cols = st.columns(5)
    for i, (_, row) in enumerate(popular.iterrows()):
        with cols[i % 5]:
            with st.container(border=True):
                show_poster(row)
                st.markdown(f"**{row['title']}**")
                st.caption(f"⭐ {row['rating']}  ·  {row['year']}")

    st.markdown("### ⭐ Top Rated Movies")
    st.caption("Discovery feature — ranked by TMDB rating, not the recommender.")
    top_rated = movies_df[movies_df["votes"] >= 100].nlargest(10, "rating")
    cols = st.columns(5)
    for i, (_, row) in enumerate(top_rated.iterrows()):
        with cols[i % 5]:
            with st.container(border=True):
                show_poster(row)
                st.markdown(f"**{row['title']}**")
                st.caption(f"⭐ {row['rating']}  ·  {row['year']}")


# ==================================================
# FIND SIMILAR MOVIES
# ==================================================
elif page == "🎬 Find Similar Movies":
    st.markdown("# 🎬 Find Similar Movies")

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
        recs = get_recommendations(source_idx, top_n, filters,
                                    user_genres=sel_genres or None,
                                    use_diversity=use_div)
        elapsed = (time.perf_counter() - t0) * 1000

        if recs.empty:
            st.warning("No movies match your current filters. Try relaxing them.")
        else:
            max_sim = float(recs["similarity"].max())
            if max_sim < LOW_SIM_THRESHOLD:
                st.warning(
                    f"⚠️ No highly similar movies found for **{source_row['title']}**. "
                    f"Best match is only **{int(max_sim*100)}%** similar. "
                    "Consider choosing a different reference movie or relaxing filters."
                )

            st.caption(f"{len(recs)} recommendations · generated in {elapsed:.1f} ms")
            cols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                reasons = dynamic_reasons(source_row, row, sel_genres or None,
                                           similarity=row["similarity"])
                render_card(cols[i % 4], row,
                            similarity=row["similarity"],
                            reasons=reasons,
                            show_breakdown=True)


# ==================================================
# PREFERENCES (COLD START)
# ==================================================
elif page == "🎯 Recommend by Preferences":
    st.markdown("# 🎯 Recommend by Preferences")
    st.caption("New here? Tell us what you like. This is a preference-based mode of the "
               "same content-based engine — no reference movie needed.")

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
                ranked = score_candidates(pool, np.zeros(len(pool)),
                                          user_genres=pref_genres,
                                          weights=COLD_START_WEIGHTS).head(pref_top_n)

                st.markdown("---")
                st.markdown(f"### ✨ Top {len(ranked)} for your preferences")
                cols = st.columns(4)
                for i, (_, row) in enumerate(ranked.iterrows()):
                    matched = sorted(set(pref_genres) & set(row["genres_list"]))
                    reasons = [
                        f"Matches your preference: **{', '.join(matched)}**",
                        f"Rating ⭐ {row['rating']} (≥ {pref_min_rating})",
                        f"Popularity {row['popularity']:.1f}",
                        f"Released in {row['year']}",
                    ]
                    render_card(cols[i % 4], row, similarity=None,
                                reasons=reasons, show_breakdown=True)


# ==================================================
# MODEL EVALUATION
# ==================================================
elif page == "📊 Model Evaluation":
    st.markdown("# 📊 Model Evaluation")
    st.caption("Metrics computed by `run_evaluation.py`. Relevance = sharing ≥1 genre "
               "with the source movie. These are ranking metrics, not accuracy.")

    summary_path = Path("evaluation/evaluation_summary.csv")
    results_path = Path("evaluation/evaluation_results.csv")

    if not summary_path.exists() or not results_path.exists():
        st.warning("Evaluation files not found. Run `python run_evaluation.py` first.")
    else:
        summary = pd.read_csv(summary_path)
        results = pd.read_csv(results_path)

        # --- Top cards (Proposed) ---
        prop = summary[["Metric", "Proposed"]].set_index("Metric")["Proposed"].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-value'>"
                    f"{prop.get('Precision@5', 0):.3f}</div>"
                    f"<div class='metric-label'>Precision@5</div></div>",
                    unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-value'>"
                    f"{prop.get('Precision@10', 0):.3f}</div>"
                    f"<div class='metric-label'>Precision@10</div></div>",
                    unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-value'>"
                    f"{prop.get('Coverage (200 queries)', 0)*100:.1f}%</div>"
                    f"<div class='metric-label'>Catalog coverage</div></div>",
                    unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-value'>"
                    f"{prop.get('Query time (ms)', 0):.1f} ms</div>"
                    f"<div class='metric-label'>Avg. query time</div></div>",
                    unsafe_allow_html=True)

        # --- Baseline vs Proposed ---
        st.markdown("---")
        st.markdown("### Baseline vs Proposed")
        st.caption(
            "**Baseline** = TF-IDF + Cosine Similarity (top-K by similarity only). "
            "**Proposed** = same similarity + weighted ranking + preference match + diversity."
        )
        st.dataframe(summary, width="stretch", hide_index=True)

        # --- Per-movie P@10 comparison chart ---
        st.markdown("### Per-movie Precision@10 — Baseline vs Proposed")
        chart_df = results[["title", "base_P@10", "prop_P@10"]].set_index("title")
        chart_df.columns = ["Baseline", "Proposed"]
        st.bar_chart(chart_df, height=380)

        # --- Per-movie table ---
        st.markdown("### Per-movie Results")
        st.dataframe(results, width="stretch", hide_index=True)

        # --- Eval images ---
        for fname, caption in [
            ("eval_precision_at_k.png", "Precision@K — Baseline vs Proposed"),
            ("eval_similarity_distribution.png", "Similarity distribution across catalog"),
        ]:
            p = Path("screenshots") / fname
            if p.exists():
                st.markdown(f"### {caption}")
                st.image(str(p), width="stretch")

        st.info(
            "**Reading these numbers.** Precision@K is a ranking metric — not accuracy. "
            "It measures the fraction of top-K recommendations that share at least one "
            "genre with the source movie. The relevance proxy is genre overlap, which "
            "is a sanity check, not real user satisfaction. Recall@10 is bounded above "
            "by ~0.33 under top-50 pooling."
        )


# ==================================================
# ABOUT
# ==================================================
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About the Model")

    st.markdown
