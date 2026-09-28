"""Central configuration for SmartMovie. Edit weights here, not in code."""

# Feature weights for the TF-IDF "soup"
FEATURE_WEIGHTS = {
    "genres":   3,
    "director": 3,
    "keywords": 2,
    "cast":     2,
    "overview": 1,
}

# Ranking weights (must sum to 1.0)
RANKING_WEIGHTS = {
    "similarity": 0.60,
    "rating":     0.20,
    "popularity": 0.10,
    "preference": 0.10,
}

# Cold-start weights (no similarity signal available)
COLD_START_WEIGHTS = {
    "similarity": 0.00,
    "rating":     0.45,
    "popularity": 0.25,
    "preference": 0.30,
}

# TF-IDF vectorizer parameters
TFIDF_PARAMS = {
    "max_features": 50000,
    "ngram_range":  (1, 2),
    "stop_words":   "english",
}

# Diversity (MMR-lite)
DIVERSITY_PENALTY = 0.30   # 0 = off, 1 = aggressive
DIVERSITY_ENABLED = True

# Poster
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"