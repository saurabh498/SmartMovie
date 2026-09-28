"""Merge TMDB 5000 movies + credits into the SmartMovie schema.

Usage:
    python prepare_tmdb.py

Output:
    data/movies.csv  — columns: id, title, genres, keywords, overview,
                      cast, director, rating, votes, popularity, year
"""
import ast
import os
from pathlib import Path

import pandas as pd


DATA_DIR = "data"
MOVIES_FILE = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_FILE = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "movies.csv")


def parse_json_list(value, key="name", limit=None):
    """Extract names from a JSON-like list of dicts.

    Example input:  '[{"id": 18, "name": "Drama"}, {"id": 35, "name": "Comedy"}]'
    Example output: 'Drama|Comedy'
    """
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
    """Return the director's name from the crew JSON."""
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


def main():
    if not os.path.exists(MOVIES_FILE) or not os.path.exists(CREDITS_FILE):
        raise FileNotFoundError(
            f"Place {MOVIES_FILE} and {CREDITS_FILE} in the data/ folder first."
        )

    print("Loading CSV files...")
    movies = pd.read_csv(MOVIES_FILE)
    credits = pd.read_csv(CREDITS_FILE)

    print(f"  movies:  {movies.shape}")
    print(f"  credits: {credits.shape}")

    # --- Process credits ---
    credits["cast_names"] = credits["cast"].apply(lambda s: parse_json_list(s, "name", limit=5))
    credits["director"] = credits["crew"].apply(extract_director)
    credits = credits.rename(columns={"movie_id": "id"})
    credits = credits[["id", "cast_names", "director"]]

    # --- Process movies ---
    movies["genres"] = movies["genres"].apply(lambda s: parse_json_list(s, "name"))
    movies["keywords"] = movies["keywords"].apply(lambda s: parse_json_list(s, "name"))

    # --- Merge ---
    merged = movies.merge(credits, on="id", how="left")

    # --- Rename numeric fields to SmartMovie schema ---
    merged["rating"] = pd.to_numeric(merged["vote_average"], errors="coerce").fillna(0.0)
    merged["votes"] = pd.to_numeric(merged["vote_count"], errors="coerce").fillna(0).astype(int)
    merged["popularity"] = pd.to_numeric(merged["popularity"], errors="coerce").fillna(0.0)

    # --- Extract year from release_date ---
    merged["year"] = (
        pd.to_datetime(merged["release_date"], errors="coerce")
        .dt.year.fillna(0).astype(int)
    )

    # --- Select final columns ---
    final = merged[[
        "id", "title", "genres", "keywords", "overview",
        "cast_names", "director", "rating", "votes", "popularity", "year",
    ]].rename(columns={"cast_names": "cast"})

    # --- Clean ---
    final = final.dropna(subset=["title", "overview"])
    final = final[final["overview"].astype(str).str.strip() != ""]
    final = final.drop_duplicates(subset=["title"]).reset_index(drop=True)

    # Drop movies with no genres (can't be recommended meaningfully)
    final = final[final["genres"].astype(str).str.strip() != ""].reset_index(drop=True)

    Path(DATA_DIR).mkdir(exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✓ Wrote {len(final)} movies → {OUTPUT_FILE}")
    print(f"  Columns: {list(final.columns)}")
    print(f"  Year range: {final['year'].min()}–{final['year'].max()}")
    print(f"  Rating range: {final['rating'].min():.1f}–{final['rating'].max():.1f}")


if __name__ == "__main__":
    main()