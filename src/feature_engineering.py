"""Feature engineering — build the combined 'soup' for TF-IDF."""
import re
from .preprocessing import parse_genres, parse_list_field


def clean_text(text):
    """Lowercase and strip non-alphanumeric characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_soup(row, genre_weight=3, keyword_weight=2, cast_weight=2,
               director_weight=3, overview_weight=1):
    """Combine multiple movie features into one weighted token string."""
    parts = []

    # Genres (repeat by weight to emphasize)
    genres = parse_genres(row.get("genres", ""))
    parts.extend([clean_text(g) for g in genres] * genre_weight)

    # Keywords
    keywords = parse_list_field(row.get("keywords", ""))
    parts.extend([clean_text(k) for k in keywords] * keyword_weight)

    # Cast (top few)
    cast = parse_list_field(row.get("cast", ""))
    parts.extend([clean_text(c) for c in cast[:5]] * cast_weight)

    # Director
    director = row.get("director", "")
    if isinstance(director, str) and director.strip():
        parts.extend([clean_text(director)] * director_weight)

    # Overview
    overview = clean_text(row.get("overview", ""))
    if overview:
        parts.extend([overview] * overview_weight)

    return " ".join(p for p in parts if p)


def attach_lists(df):
    """Attach parsed list columns for later use in explanations."""
    df = df.copy()
    df["genres_list"] = df["genres"].apply(parse_genres)
    df["keywords_list"] = df["keywords"].apply(parse_list_field)
    df["cast_list"] = df["cast"].apply(parse_list_field)
    return df