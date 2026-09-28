"""Merge TMDB 5000 movies + credits into the SmartMovie schema.

Output:
    data/movies.csv  — columns: id, title, genres, keywords, overview,
                      cast, director, rating, votes, popularity, year, poster_url
"""
import ast
import os
from pathlib import Path

import pandas as pd

from src.config import TMDB_POSTER_BASE

DATA_DIR = "data"
MOVIES_FILE  = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_FILE = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
OUTPUT_FILE  = os.path.join(DATA_DIR, "movies.csv")


def parse_json_list(value, key="name", limit=None):
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ""
    names = [str(i[key]) for i in items if isinstance(i, dict) and key in i]
    if limit:
        names = names[:limit]
    return "|".join(names)


def extract_director(crew_json):
    if not isinstance(crew_json, str) or not crew_json.strip():
        return ""
    try:
        crew = ast.literal_eval(crew_json)
    except (ValueError, SyntaxError):
        return ""
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def build_poster_url(path):
    """Convert '/abc.jpg' → full TMDB URL. Empty string if missing."""
    if not isinstance(path, str) or not path.strip():
        return ""
    path = path.strip()
    if path.startswith("/"):
        path = path[1:]
    return f"{TMDB_POSTER_BASE}/{path}"


def main():
    if not os.path.exists(MOVIES_FILE) or not os.path.exists(CREDITS_FILE):
        raise FileNotFoundError(
            f"Place {MOVIES_FILE} and {CREDITS_FILE} in data/ first."
        )

    movies = pd.read_csv(MOVIES_FILE)
    credits = pd.read_csv(CREDITS_FILE)
    print(f"Loaded: movies={movies.shape}, credits={credits.shape}")

    # Credits
    credits["cast_names"] = credits["cast"].apply(lambda s: parse_json_list(s, "name", 5))
    credits["director"]   = credits["crew"].apply(extract_director)
    credits = credits.rename(columns={"movie_id": "id"})[["id", "cast_names", "director"]]

    # Movies
    movies["genres"]   = movies["genres"].apply(lambda s: parse_json_list(s, "name"))
    movies["keywords"] = movies["keywords"].apply(lambda s: parse_json_list(s, "name"))
    movies["poster_url"] = movies.get("poster_path", pd.Series([""] * len(movies))
                                     ).apply(build_poster_url)

    merged = movies.merge(credits, on="id", how="left")
    merged["rating"]     = pd.to_numeric(merged["vote_average"], errors="coerce").fillna(0.0)
    merged["votes"]      = pd.to_numeric(merged["vote_count"],  errors="coerce").fillna(0).astype(int)
    merged["popularity"] = pd.to_numeric(merged["popularity"],  errors="coerce").fillna(0.0)
    merged["year"]       = (pd.to_datetime(merged["release_date"], errors="coerce")
                              .dt.year.fillna(0).astype(int))

    final = merged[[
        "id", "title", "genres", "keywords", "overview", "cast_names", "director",
        "rating", "votes", "popularity", "year", "poster_url",
    ]].rename(columns={"cast_names": "cast"})

    # Clean
    final = final.dropna(subset=["title", "overview"])
    final = final[final["overview"].astype(str).str.strip() != ""]
    final = final[final["genres"].astype(str).str.strip() != ""]
    final = final[final["year"] > 1900]
    final = final.drop_duplicates(subset=["title"]).reset_index(drop=True)

    Path(DATA_DIR).mkdir(exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)

    with_posters = (final["poster_url"].astype(str).str.len() > 0).sum()
    print(f"\n✓ Wrote {len(final)} movies → {OUTPUT_FILE}")
    print(f"  With poster URL: {with_posters} ({with_posters/len(final)*100:.1f}%)")
    print(f"  Year range: {final['year'].min()}–{final['year'].max()}")


if __name__ == "__main__":
    main()