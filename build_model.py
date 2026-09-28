"""Build the SmartMovie recommendation model.

Usage:
    python build_model.py

If data/movies.csv is missing, an embedded sample dataset is used.
"""
import os
import pickle
from pathlib import Path

import pandas as pd

from src.preprocessing import load_dataset
from src.feature_engineering import build_soup, attach_lists
from src.recommender import Recommender


# ---------- Embedded sample dataset (20 diverse movies) ----------
SAMPLE_DATA = [
    dict(id=1, title="Interstellar", genres="Sci-Fi|Drama", director="Christopher Nolan",
         cast="Matthew McConaughey|Anne Hathaway|Jessica Chastain|Michael Caine",
         keywords="space|time travel|astronaut|future|black hole|wormhole|sacrifice",
         overview="A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
         rating=8.7, votes=32000, popularity=150.5, year=2014),
    dict(id=2, title="Inception", genres="Sci-Fi|Thriller", director="Christopher Nolan",
         cast="Leonardo DiCaprio|Joseph Gordon-Levitt|Elliot Page|Tom Hardy",
         keywords="dream|mind|reality|subconscious|heist|memory",
         overview="A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
         rating=8.8, votes=34000, popularity=140.0, year=2010),
    dict(id=3, title="The Dark Knight", genres="Action|Crime|Drama", director="Christopher Nolan",
         cast="Christian Bale|Heath Ledger|Aaron Eckhart|Michael Caine",
         keywords="batman|joker|vigilante|gotham|chaos|moral dilemma",
         overview="When the menace known as the Joker wreaks havoc on Gotham, Batman must accept one of the greatest psychological tests.",
         rating=9.0, votes=36000, popularity=160.0, year=2008),
    dict(id=4, title="Pulp Fiction", genres="Crime|Drama", director="Quentin Tarantino",
         cast="John Travolta|Samuel L. Jackson|Uma Thurman|Bruce Willis",
         keywords="gangster|nonlinear|hitman|briefcase|los angeles|dark humor",
         overview="The lives of two mob hitmen, a boxer, and a pair of diner bandits intertwine in four tales of violence and redemption.",
         rating=8.9, votes=35000, popularity=130.0, year=1994),
    dict(id=5, title="Avatar", genres="Sci-Fi|Adventure|Fantasy", director="James Cameron",
         cast="Sam Worthington|Zoe Saldana|Sigourney Weaver|Stephen Lang",
         keywords="alien planet|native|marine|avatar|colonization|nature",
         overview="A paraplegic marine dispatched to the moon Pandora on a unique mission becomes torn between following orders and protecting the world.",
         rating=7.9, votes=30000, popularity=120.0, year=2009),
    dict(id=6, title="The Matrix", genres="Sci-Fi|Action", director="The Wachowskis",
         cast="Keanu Reeves|Laurence Fishburne|Carrie-Anne Moss|Hugo Weaving",
         keywords="virtual reality|hacker|simulation|dystopia|chosen one|artificial intelligence",
         overview="A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
         rating=8.7, votes=33000, popularity=145.0, year=1999),
    dict(id=7, title="Fight Club", genres="Drama|Thriller", director="David Fincher",
         cast="Brad Pitt|Edward Norton|Helena Bonham Carter|Meat Loaf",
         keywords="identity|split personality|underground club|consumerism|anarchy|twist",
         overview="An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.",
         rating=8.8, votes=34000, popularity=135.0, year=1999),
    dict(id=8, title="Forrest Gump", genres="Drama|Romance", director="Robert Zemeckis",
         cast="Tom Hanks|Robin Wright|Gary Sinise|Sally Field",
         keywords="history|love|vietnam|disability|destiny|american dream",
         overview="The presidencies of Kennedy and Johnson, Vietnam, Watergate, and other history unfold through the perspective of an Alabama man with an IQ of 75.",
         rating=8.8, votes=35000, popularity=125.0, year=1994),
    dict(id=9, title="The Shawshank Redemption", genres="Drama|Crime", director="Frank Darabont",
         cast="Tim Robbins|Morgan Freeman|Bob Gunton|William Sadler",
         keywords="prison|hope|friendship|escape|injustice|redemption",
         overview="Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
         rating=9.3, votes=38000, popularity=155.0, year=1994),
    dict(id=10, title="The Godfather", genres="Crime|Drama", director="Francis Ford Coppola",
         cast="Marlon Brando|Al Pacino|James Caan|Robert Duvall",
         keywords="mafia|family|power|loyalty|italian american|betrayal",
         overview="The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
         rating=9.2, votes=37000, popularity=148.0, year=1972),
    dict(id=11, title="Parasite", genres="Thriller|Drama|Comedy", director="Bong Joon-ho",
         cast="Song Kang-ho|Lee Sun-kyun|Cho Yeo-jeong|Choi Woo-shik",
         keywords="class|family|deception|poverty|social inequality|twist",
         overview="Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
         rating=8.5, votes=31000, popularity=138.0, year=2019),
    dict(id=12, title="Blade Runner 2049", genres="Sci-Fi|Drama|Mystery", director="Denis Villeneuve",
         cast="Ryan Gosling|Harrison Ford|Ana de Armas|Jared Leto",
         keywords="replicant|dystopia|identity|future|artificial intelligence|memory",
         overview="Young Blade Runner K's discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard.",
         rating=8.0, votes=27000, popularity=118.0, year=2017),
    dict(id=13, title="Whiplash", genres="Drama|Music", director="Damien Chazelle",
         cast="Miles Teller|J.K. Simmons|Melissa Benoist|Paul Reiser",
         keywords="jazz|obsession|mentor|perfectionism|ambition|abuse",
         overview="A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing.",
         rating=8.5, votes=29000, popularity=115.0, year=2014),
    dict(id=14, title="Mad Max: Fury Road", genres="Action|Adventure|Sci-Fi", director="George Miller",
         cast="Tom Hardy|Charlize Theron|Nicholas Hoult|Hugh Keays-Byrne",
         keywords="post-apocalyptic|desert|chase|survival|rebellion|practical effects",
         overview="In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners.",
         rating=8.1, votes=28000, popularity=128.0, year=2015),
    dict(id=15, title="Arrival", genres="Sci-Fi|Drama|Mystery", director="Denis Villeneuve",
         cast="Amy Adams|Jeremy Renner|Forest Whitaker|Michael Stuhlbarg",
         keywords="alien|language|time|communication|first contact|linguistics",
         overview="A linguist is recruited by the military to assist in translating alien communications after twelve mysterious spacecraft appear around the world.",
         rating=7.9, votes=26000, popularity=112.0, year=2016),
    dict(id=16, title="Django Unchained", genres="Western|Drama", director="Quentin Tarantino",
         cast="Jamie Foxx|Christoph Waltz|Leonardo DiCaprio|Samuel L. Jackson",
         keywords="slavery|revenge|bounty hunter|freedom|south|dark humor",
         overview="With the help of a German bounty hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner.",
         rating=8.4, votes=29000, popularity=122.0, year=2012),
    dict(id=17, title="The Prestige", genres="Drama|Mystery|Thriller", director="Christopher Nolan",
         cast="Christian Bale|Hugh Jackman|Scarlett Johansson|Michael Caine",
         keywords="magic|obsession|rivalry|illusion|sacrifice|twist",
         overview="After a tragic accident, two stage magicians in 1890s London engage in a battle to create the ultimate illusion while sacrificing everything.",
         rating=8.5, votes=28000, popularity=116.0, year=2006),
    dict(id=18, title="Get Out", genres="Horror|Thriller|Mystery", director="Jordan Peele",
         cast="Daniel Kaluuya|Allison Williams|Bradley Whitford|Catherine Keener",
         keywords="racism|hypnosis|social commentary|suburbia|mind control|twist",
         overview="A young African-American visits his white girlfriend's parents for the weekend, where his simmering uneasiness about their reception eventually reaches a boiling point.",
         rating=7.7, votes=24000, popularity=108.0, year=2017),
    dict(id=19, title="Spirited Away", genres="Animation|Fantasy|Adventure", director="Hayao Miyazaki",
         cast="Rumi Hiiragi|Miyu Irino|Mari Natsuki|Takashi Naito",
         keywords="spirit world|coming of age|witch|bathhouse|japanese folklore|magic",
         overview="During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits.",
         rating=8.6, votes=30000, popularity=126.0, year=2001),
    dict(id=20, title="La La Land", genres="Romance|Drama|Music", director="Damien Chazelle",
         cast="Ryan Gosling|Emma Stone|John Legend|Rosemarie DeWitt",
         keywords="jazz|ambition|los angeles|romance|musical|bittersweet",
         overview="While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.",
         rating=8.0, votes=27000, popularity=114.0, year=2016),
]


def ensure_dataset(data_dir="data"):
    """Ensure data/movies.csv exists AND is a valid non-empty CSV."""
    Path(data_dir).mkdir(exist_ok=True)
    path = os.path.join(data_dir, "movies.csv")

    valid = False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            probe = pd.read_csv(path)
            if not probe.empty and "title" in probe.columns:
                valid = True
        except Exception:
            valid = False

    if not valid:
        print(f"⚠ {path} missing/empty/invalid — regenerating from sample data")
        pd.DataFrame(SAMPLE_DATA).to_csv(path, index=False)
        print(f"✓ Created sample dataset → {path}")

    return path

def main():
    # 0. Clean stale artifacts so we never load a half-written file later
    for p in ["models/tfidf_vectorizer.pkl",
              "models/tfidf_matrix.pkl",
              "models/movies.pkl"]:
        if os.path.exists(p) and os.path.getsize(p) == 0:
            os.remove(p)

    # 1. Load
    path = ensure_dataset()
    df = load_dataset(path)
    print(f"✓ Loaded {len(df)} movies")
    # ... rest unchanged

def main():
    # 1. Load
    path = ensure_dataset()
    df = load_dataset(path)
    print(f"✓ Loaded {len(df)} movies")

    # 2. Feature engineering
    df = attach_lists(df)
    df["soup"] = df.apply(build_soup, axis=1)
    print("✓ Built feature soup")

    # 3. Fit TF-IDF + similarity
    rec = Recommender().fit(df, soup_column="soup")
    print(f"✓ TF-IDF matrix: {rec.tfidf_matrix.shape}")

    # 4. Save artifacts
    Path("models").mkdir(exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(rec.vectorizer, f)
    with open("models/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(rec.tfidf_matrix, f)
    with open("models/movies.pkl", "wb") as f:
        pickle.dump(df, f)
    df.to_csv("data/movies_processed.csv", index=False)
    print("✓ Saved models and processed data")

    # 5. Sanity check
    idx, sims = rec.similar_to_title("Interstellar", top_n=5)
    print("\nSanity check — Movies similar to Interstellar:")
    for i, s in zip(idx, sims):
        print(f"  {df.iloc[i]['title']:35s} sim={s:.3f}  score={df.iloc[i]['rating']}")


if __name__ == "__main__":
    main()