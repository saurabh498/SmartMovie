"""Content-based recommendation engine using TF-IDF and cosine similarity."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    def __init__(self, max_features=50000, ngram_range=(1, 2),
                 stop_words="english", min_df=1):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            min_df=min_df,
        )
        self.tfidf_matrix = None
        self.movies = None

    def fit(self, movies_df, soup_column="soup"):
        self.movies = movies_df.reset_index(drop=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies[soup_column])
        return self

    def similar_to_index(self, idx, top_n=10):
        if self.tfidf_matrix is None:
            raise RuntimeError("Recommender not fitted yet.")
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sims[idx] = -1.0
        top_idx = np.argsort(sims)[::-1][:top_n]
        return top_idx, sims[top_idx]

    def similar_to_title(self, title, top_n=10):
        matches = self.movies[self.movies["title"].str.lower() == title.lower()]
        if matches.empty:
            return None, None
        idx = matches.index[0]
        return self.similar_to_index(idx, top_n)