"""Data loading and cleaning for SmartMovie."""
import ast
import pandas as pd


def parse_list_field(value, sep="|"):
    """Parse a field that may be JSON-like ('[{...}]') or pipe-separated ('A|B')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    value = str(value).strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [p.strip() for p in value.split(sep) if p.strip()]


def parse_genres(value):
    """Extract genre names from a JSON-like list of dicts or pipe string."""
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if parsed and isinstance(parsed[0], dict):
                return [d.get("name", "") for d in parsed if d.get("name")]
        except (ValueError, SyntaxError):
            pass
    return parse_list_field(value)


def load_dataset(path):
    """Load the movies CSV and return a clean DataFrame."""
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop rows missing essential data
    df = df.dropna(subset=["title", "overview"])
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    # Normalize text columns
    df["title"] = df["title"].astype(str).str.strip()
    df["overview"] = df["overview"].astype(str).str.strip()

    # Numeric coercion
    for col, default in [("rating", 0.0), ("votes", 0), ("popularity", 0.0), ("year", 0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # If release_date is present, extract year
    if "release_date" in df.columns and "year" not in df.columns:
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)

    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)