"""SmartMovie — Streamlit GUI."""
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.ranking import score_candidates, apply_filters


# -------------------- Page config --------------------
st.set_page_config(page_title="SmartMovie", page_icon="🎬", layout="wide")

st.markdown("""
<style>
  .stApp { background: #0f1117; }
  h1, h2, h3 { color: #e8eaf0 !important; }
  .movie-card {
    background: #171b28; border: 1px solid #232838; border-radius: 14px;
    padding: 16px; margin-bottom: 8px; transition: border-color .2s;
  }
  .movie-card:hover { border-color: #7c5cff; }
  .poster {
    height: 120px; border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 2.6rem; font-weight: 800; color: white;
    margin-bottom: 10px;
  }
  .badge {
    background: #222838; border-radius: 6px; padding: 2px 8px;
    font-size: .74rem; color: #a8b0c5; margin-right: 4px;
  }
  .sim-bar { height: 6px; border-radius: 4px; background: #232838; overflow: hidden; margin-top: 8px; }
  .sim-fill { height: 100%; background: linear-gradient(90deg, #7c5cff, #ff5c8a); }
  .why { font-size: .8rem; color: #9aa3ba; line-height: 1.7; margin-top: 8px; }
  .why b { color: #7c5cff; }
</style>
""", unsafe_allow_html=True)


# -------------------- Load artifacts (safe) --------------------
@st.cache_resource
def load_artifacts():
    """Return ((movies_df, tfidf_matrix), error_message)."""
    required = ["models/movies.pkl", "models/tfidf_matrix.pkl"]
    for p in required:
        if not Path(p).exists():
            return None, f"Missing file: {p}"
        if Path(p).stat().st_size == 0:
            return None, f"Empty file: {p}"

    try:
        with open("models/movies.pkl", "rb") as f:
            movies = pickle.load(f)
        with open("models/tfidf_matrix.pkl", "rb") as f:
            matrix = pickle.load(f)
        return (movies, matrix), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


artifacts, load_error = load_artifacts()

if artifacts is None:
    st.error(f"⚠️ Model not ready — {load_error}")
    st.info("Run `python build_model.py` in the project root, then refresh this page.")
    st.stop()

# This unpack is what was missing / misnamed in your version:
movies_df, tfidf_matrix = artifacts


# -------------------- Helpers --------------------
def gradient(seed):
    palettes = [
        ("#7c5cff", "#ff5c8a"), ("#5cc8ff", "#7c5cff"),
        ("#ffb35c", "#ff5c8a"), ("#5cffb4", "#5cc8ff"),
        ("#ff5c5c", "#ffb35c"), ("#b45cff", "#5cc8ff"),
    ]
    a, b = palettes[int(seed) % len(palettes)]
    return f"linear-gradient(135deg, {a}, {b})"


def explain(source_row, target_row, user_genres=None):
    reasons = []

    sg = set(source_row["genres_list"]) & set(target_row["genres_list"])
    if sg:
        reasons.append(f"Similar genres: <b>{', '.join(sorted(sg))}</b>")

    sk = set(source_row["keywords_list"]) & set(target_row["keywords_list"])
    if sk:
        reasons.append(f"Similar themes: <b>{', '.join(sorted(sk)[:4])}</b>")

    if source_row["director"] and source_row["director"] == target_row["director"]:
        reasons.append(f"Same director: <b>{source_row['director']}</b>")

    sc = set(source_row["cast_list"]) & set(target_row["cast_list"])
    if sc:
        reasons.append(f"Common cast: <b>{', '.join(sorted(sc)[:3])}</b>")

    if target_row["rating"] >= 8.0:
        reasons.append(f"Highly rated: <b>⭐ {target_row['rating']}</b>")

    if user_genres:
        matched = set(user_genres) & set(target_row["genres_list"])
        if matched:
            reasons.append(f"Matches your preferences: <b>{', '.join(sorted(matched))}</b>")

    if not reasons:
        reasons.append("Overall content profile is similar")

    return reasons


def render_card(row, sim=None, reasons=None):
    sim_html = ""
    if sim is not None:
        pct = int(round(float(sim) * 100))
        sim_html = (
            f'<div class="sim-bar"><div class="sim-fill" style="width:{pct}%"></div></div>'
            f'<div style="font-size:.72rem;color:#6f7891;margin-top:4px;'
            f'display:flex;justify-content:space-between">'
            f'<span>Similarity</span><span>{pct}%</span></div>'
        )

    why_html = ""
    if reasons:
        why_html = '<div class="why">' + "<br>".join(f"✓ {r}" for r in reasons) + "</div>"

    genres = ", ".join(list(row["genres_list"])[:3])

    html = (
        '<div class="movie-card">'
        f'<div class="poster" style="background:{gradient(row["id"])}">'
        f'{str(row["title"])[0]}</div>'
        f'<div style="font-weight:700;font-size:1rem;color:#e8eaf0">{row["title"]}</div>'
        f'<div style="margin:6px 0">'
        f'<span class="badge">⭐ {row["rating"]}</span>'
        f'<span class="badge">{row["year"]}</span>'
        f'</div>'
        f'<div style="font-size:.78rem;color:#8b93a7">{genres}</div>'
        f'{sim_html}'
        f'{why_html}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def get_recommendations(source_idx, top_n, filters, user_genres=None):
    sims = cosine_similarity(tfidf_matrix[source_idx], tfidf_matrix).flatten()
    sims[source_idx] = -1.0

    candidates = movies_df.copy()
    candidates["similarity"] = sims
    candidates = candidates[candidates["similarity"] > 0]

    candidates = apply_filters(
        candidates,
        genres=filters.get("genres"),
        min_rating=filters.get("min_rating"),
        year_from=filters.get("year_from"),
        year_to=filters.get("year_to"),
    )

    if candidates.empty:
        return candidates

    ranked = score_candidates(
        candidates,
        candidates["similarity"].values,
        user_genres=user_genres or filters.get("genres"),
    )
    return ranked.head(top_n)


# -------------------- Header --------------------
st.markdown("<h1 style='text-align:center'>🎬 SmartMovie</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#8b93a7'>Intelligent Movie Recommendation System · "
    "TF-IDF + Cosine Similarity</p>",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🔍 Find Similar Movies", "✨ Recommend by Preferences"])


# ============================================================
# TAB 1 — Content-based search
# ============================================================
with tab1:
    col_search, _ = st.columns([2, 1])
    with col_search:
        selected_title = st.selectbox(
            "Search a movie you like",
            options=[""] + sorted(movies_df["title"].tolist()),
            index=0,
        )

    with st.sidebar:
        st.header("⚙️ Filters")
        all_genres = sorted({g for gs in movies_df["genres_list"] for g in gs})
        sel_genres = st.multiselect("Genres", all_genres, default=[])
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 6.0, 0.1)
        year_min = int(movies_df["year"].min())
        year_max = int(movies_df["year"].max())
        year_range = st.slider("Release year range", year_min, year_max, (year_min, year_max))
        top_n = st.selectbox("Number of recommendations", [5, 8, 10, 15], index=1)

    filters = {
        "genres": sel_genres,
        "min_rating": min_rating,
        "year_from": year_range[0],
        "year_to": year_range[1],
    }

    if selected_title:
        match = movies_df[movies_df["title"] == selected_title]
        source_idx = match.index[0]
        source_row = movies_df.iloc[source_idx]

        st.markdown("---")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(
                f"<div class='poster' style='background:{gradient(source_row['id'])};"
                f"height:220px;font-size:4rem'>{source_row['title'][0]}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.subheader(source_row["title"])
            st.markdown(
                f"⭐ **{source_row['rating']}** &nbsp;·&nbsp; "
                f"📅 {source_row['year']} &nbsp;·&nbsp; "
                f"🎭 {', '.join(list(source_row['genres_list']))}"
            )
            if source_row["director"]:
                st.markdown(f"🎬 **Director:** {source_row['director']}")
            if source_row["cast_list"]:
                st.markdown(f"👥 **Cast:** {', '.join(list(source_row['cast_list'])[:4])}")
            st.write(source_row["overview"])

        st.markdown("---")
        st.subheader(f"✨ Top {top_n} recommendations")

        recs = get_recommendations(source_idx, top_n, filters, user_genres=sel_genres or None)

        if recs.empty:
            st.warning("No movies match your current filters. Try relaxing them.")
        else:
            cols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 4]:
                    reasons = explain(source_row, row, sel_genres or None)
                    render_card(row, sim=row["similarity"], reasons=reasons)


# ============================================================
# TAB 2 — Preference-based (cold start)
# ============================================================
with tab2:
    st.subheader("New here? Tell us what you like.")
    st.caption("Get recommendations without picking a movie first.")

    all_genres = sorted({g for gs in movies_df["genres_list"] for g in gs})
    pref_genres = st.multiselect("Preferred genres", all_genres, default=[])
    pref_min_rating = st.slider("Minimum rating", 0.0, 10.0, 7.0, 0.1, key="pref_rating")

    yr_min = int(movies_df["year"].min())
    yr_max = int(movies_df["year"].max())
    pref_years = st.slider(
        "Preferred release year range",
        yr_min, yr_max, (max(1990, yr_min), yr_max),
        key="pref_years",
    )
    pref_top_n = st.selectbox("How many recommendations?", [5, 8, 10, 15], index=1, key="pref_n")

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
                    weights={"similarity": 0.0, "rating": 0.45,
                             "popularity": 0.25, "preference": 0.30},
                ).head(pref_top_n)

                st.markdown("---")
                cols = st.columns(4)
                for i, (_, row) in enumerate(ranked.iterrows()):
                    with cols[i % 4]:
                        matched = set(pref_genres) & set(row["genres_list"])
                        reasons = [
                            f"Matches your preference: <b>{', '.join(sorted(matched))}</b>",
                            f"Rating <b>⭐ {row['rating']}</b> (≥ your threshold {pref_min_rating})",
                            f"Released in <b>{row['year']}</b>",
                        ]
                        render_card(row, sim=None, reasons=reasons)