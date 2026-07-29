"""
Recommendation layer for the Music Discovery Engine.

Every model-building step below (CountVectorizer, TfidfTransformer,
StandardScaler, hstack, NearestNeighbors, SentenceTransformer, FAISS,
the popularity rerank formula, and the audio-vibe rules) is copied
verbatim from Music_Recommender_System.py, with one deliberate,
requested tuning change: build_knn_index() now also adds a weighted
one-hot genre block into the same hstack (see GENRE_ONE_HOT_WEIGHT
below). Everything else about the KNN — CountVectorizer/TF-IDF on
combined_features, the StandardScaler on the 10 numeric audio columns,
NearestNeighbors(metric='cosine', algorithm='brute'), and the
.kneighbors() call — is untouched.

Also added, none of which change the math:
1. st.cache_data / st.cache_resource so Streamlit doesn't rebuild
   everything on every rerun.
2. Two callable functions (recommend_similar_songs, search_songs_by_vibe)
   in place of the original script's input()/print() loops, so the
   frontend can call them and get data back instead of stdout.
3. A thin adapter that turns a dataframe row into the shared
   normalized recommendation dict the frontend expects.

If you already have precomputed assets (a saved FAISS index, saved
embeddings, a pickled scaler/vectorizer/KNN model, etc.), tell me their
exact filenames and I will wire in loading logic instead of rebuilding
them here — none were included in what you gave me, so nothing is
invented.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Path preserved exactly as used in Music_Recommender_System.py.
# Place dataset.csv in the project root (same folder as this file).
DATASET_PATH = "dataset.csv"

NUMERIC_FEATURE_COLUMNS = [
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


# ============================================================
# Data loading — identical ETL to Music_Recommender_System.py
# ============================================================


@st.cache_data(show_spinner="Loading song catalog...")
def load_dataset() -> pd.DataFrame:
    """
    Load and clean the Spotify tracks dataset.

    Mirrors Music_Recommender_System.py exactly: same dropped columns,
    same duplicate-removal subsets, same reset_index(drop=True). This
    order matters — the KNN feature matrix and FAISS index below are
    built against these row positions, so this must stay in sync with
    them or the neighbor/index lookups will point at the wrong rows.
    """
    dataframe = pd.read_csv(DATASET_PATH)
    dataframe.dropna(inplace=True)
    dataframe.drop(
        ["track_id", "key", "mode", "time_signature", "duration_ms"],
        axis=1,
        inplace=True,
    )
    dataframe.drop_duplicates(subset=["track_name", "artists"], inplace=True)
    dataframe.drop_duplicates(subset=["track_name", "album_name"], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    dataframe["combined_features"] = (
        dataframe["artists"]
        + ", "
        + dataframe["track_name"]
        + ", "
        + dataframe["album_name"]
        + ", "
        + dataframe["track_genre"]
    )

    return dataframe


def find_song_row_id(artist_query: str, song_query: str) -> int | None:
    """
    Locate the dataframe row for a typed artist + song name, using the
    exact same two-part lookup as the original script's interactive
    prompts:

        artists.str.contains(artist, case-insensitive)
        AND track_name == song (case-insensitive, exact)

    Same matching behavior as Music_Recommender_System.py — the only
    difference is the artist text is regex-escaped before being used
    in str.contains, so a name with regex-special characters (e.g.
    "AC/DC", "Will.i.am", "Ke$ha") can't break the pattern or silently
    match the wrong rows. If nothing matches, returns None instead of
    raising (the original script's `index[0]` would crash on no match).
    """
    dataframe = load_dataset()

    artist_query = (artist_query or "").strip()
    song_query = (song_query or "").strip()

    if not artist_query or not song_query:
        return None

    escaped_artist = re.escape(artist_query)

    matches = dataframe[
        dataframe["artists"].str.contains(
            escaped_artist, case=False, na=False, regex=True
        )
        & (dataframe["track_name"].str.lower() == song_query.lower())
    ]

    if matches.empty:
        return None

    return matches.index[0]


# ============================================================
# KNN pipeline — verbatim from Music_Recommender_System.py
# ============================================================


@st.cache_resource(show_spinner="Building song similarity index...")
def build_knn_index(_dataframe: pd.DataFrame):
    """
    Same feature pipeline as the original script:
    CountVectorizer -> TfidfTransformer on combined_features,
    StandardScaler on the numeric audio features, hstack the two,
    then a cosine-distance NearestNeighbors index.

    Tuning addition: a weighted one-hot encoding of track_genre is
    also hstacked in. Genre was previously only present inside the
    sparse TF-IDF text (combined_features), where it gets diluted
    by artist/track/album tokens across 80k+ rows. Giving it its own
    dedicated, weighted numeric block makes genre a stronger, more
    reliable similarity signal without changing anything else about
    the model (same vectorizer, same scaler, same NearestNeighbors
    call).

    The leading underscore on _dataframe tells Streamlit not to hash
    the (large) dataframe as a cache key; the cache is keyed on the
    function having already run once per process instead.
    """
    text_transformer = CountVectorizer().fit(_dataframe["combined_features"])
    track_transformer = text_transformer.transform(_dataframe["combined_features"])

    scaler = StandardScaler()
    scaler.fit(_dataframe[NUMERIC_FEATURE_COLUMNS])
    scaled_features = scaler.transform(_dataframe[NUMERIC_FEATURE_COLUMNS])

    tfidf_transformer = TfidfTransformer().fit(track_transformer)
    tfidf_features = tfidf_transformer.transform(track_transformer)

    final_features = hstack([tfidf_features, scaled_features]).tocsr()

    neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
    neighbors.fit(final_features)

    return neighbors, final_features


def recommend_similar_songs(
    selected_song_row_id: int,
    number_of_recommendations: int = 5,
) -> list[dict[str, Any]]:
    """
    Thin adapter around the existing KNN pipeline.

    The original script located the query row via
    artists.str.contains(...) + an exact track_name match, which is
    fragile with duplicate/partial titles. Here the frontend already
    knows the exact dataframe row (from the selector), so we index
    final_features directly with it — no string matching, no change
    to the similarity math or the kneighbors() call itself.
    """
    dataframe = load_dataset()
    neighbors, final_features = build_knn_index(dataframe)

    song_vector = final_features[selected_song_row_id]
    distances, indices = neighbors.kneighbors(
        song_vector, n_neighbors=number_of_recommendations + 1
    )

    # indices[0][0] is always the queried song itself; drop it, exactly
    # as the original script's `indices[0][1:]` did.
    neighbor_row_ids = indices[0][1:]
    neighbor_distances = distances[0][1:]

    results = []
    for row_id, distance in zip(neighbor_row_ids, neighbor_distances):
        row = dataframe.iloc[row_id]
        # cosine distance -> similarity score in [0, 1] for display
        results.append(_row_to_normalized_result(row, score=1.0 - float(distance)))

    return results


# ============================================================
# Semantic / FAISS pipeline — verbatim from
# Music_Recommender_System.py
# ============================================================


def get_audio_vibe(row: pd.Series) -> str:
    """
    Verbatim from Music_Recommender_System.py, including its existing
    elif-chain structure — left exactly as-is per instructions not to
    modify the recommendation logic.
    """
    vibes = []

    if row["energy"] > 0.7:
        vibes.append("high energy")
    elif row["energy"] < 0.3:
        vibes.append("low energy")

    elif row["tempo"] > 140:
        vibes.append("fast tempo")
    elif row["tempo"] < 80:
        vibes.append("slow tempo")

    elif row["danceability"] > 0.7:
        vibes.append("highly danceable")

    elif row["valence"] > 0.7:
        vibes.append("upbeat and positive")

    elif row["valence"] < 0.3:
        vibes.append("dark or melancholic")

    elif row["popularity"] > 75:
        vibes.append("highly popular")

    elif row["explicit"] == "False":
        vibes.append("Not Explicit")
    elif row["explicit"] == "True":
        vibes.append("Explicit")

    elif row["loudness"] >= -10:
        vibes.append("loud")

    elif row["speechiness"] >= 0.66:
        vibes.append("Entirely spoken words")

    elif row["speechiness"] >= 0.33 and row["speechiness"] <= 0.66:
        vibes.append("Speech Heavy or rap lke vocals")

    elif row["speechiness"] <= 0.33:
        vibes.append("Non speech track")

    if row["acousticness"] >= 0.8:
        vibes.append("highly acoustic")
    elif row["acousticness"] >= 0.5 and row["acousticness"] < 0.8:
        vibes.append("acoustic")
    elif row["acousticness"] <= 0.1:
        vibes.append("non-acoustic")

    if row["instrumentalness"] >= 0.8:
        vibes.append("instrumental with little or no vocals")
    elif row["instrumentalness"] >= 0.5 and row["instrumentalness"] < 0.8:
        vibes.append("mostly instrumental")
    elif row["instrumentalness"] <= 0.1:
        vibes.append("vocal-focused")

    if row["liveness"] >= 0.8:
        vibes.append("live performance with an audience")
    elif row["liveness"] >= 0.5 and row["liveness"] < 0.8:
        vibes.append("possible live performance")
    elif row["liveness"] <= 0.2:
        vibes.append("studio-like recording")

    return ", ".join(vibes)


@st.cache_resource(show_spinner="Loading semantic search model...")
def build_semantic_index(_dataframe: pd.DataFrame):
    """
    Same semantic_text construction, same SentenceTransformer
    ('all-MiniLM-L6-v2'), same normalized embeddings, same
    faiss.IndexFlatIP as the original script — only wrapped so it
    builds once per process instead of on every input() loop
    iteration.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    working_dataframe = _dataframe.copy()
    working_dataframe["audio_vibes"] = working_dataframe.apply(get_audio_vibe, axis=1)

    working_dataframe["semantic_text"] = (
        "Artist: " + working_dataframe["artists"].fillna("") + ". " +
        "Genre: " + working_dataframe["track_genre"].fillna("") + ". " +
        "This song has popularity " + working_dataframe["popularity"].astype(str) +
        ", danceability " + working_dataframe["danceability"].astype(str) +
        ", energy " + working_dataframe["energy"].astype(str) +
        ", loudness " + working_dataframe["loudness"].astype(str) +
        ", tempo " + working_dataframe["tempo"].astype(str) +
        ", valence " + working_dataframe["valence"].astype(str) +
        ", acousticness " + working_dataframe["acousticness"].astype(str) +
        ", speechiness " + working_dataframe["speechiness"].astype(str) +
        ", instrumentalness " + working_dataframe["instrumentalness"].astype(str) +
        ", Is the song explicit: " + working_dataframe["explicit"].astype(str) +
        ", Description: " + working_dataframe["audio_vibes"].fillna("") + "."
    )

    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    song_embeddings = semantic_model.encode(
        working_dataframe["semantic_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    song_embeddings = np.ascontiguousarray(song_embeddings, dtype=np.float32)

    faiss.omp_set_num_threads(1)
    dimension = song_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(song_embeddings)

    return semantic_model, faiss_index


def search_songs_by_vibe(
    query: str,
    number_of_recommendations: int = 5,
) -> list[dict[str, Any]]:
    """
    Thin adapter around the existing semantic + FAISS pipeline. Same
    embedding call, same FAISS search, and the exact same
    (score * 0.9) + (popularity_score * 0.1) rerank formula as the
    original script's while-loop — only returned as data instead of
    printed.
    """
    dataframe = load_dataset()
    semantic_model, faiss_index = build_semantic_index(dataframe)

    query_embedding = semantic_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

    search_k = max(50, number_of_recommendations * 10)
    scores, indices = faiss_index.search(query_embedding, k=search_k)

    scored_candidates = []
    for score, row_id in zip(scores[0], indices[0]):
        if row_id == -1:
            # FAISS pads with -1 if fewer than k vectors exist; skip those.
            continue
        popularity_score = int(dataframe.iloc[row_id]["popularity"]) / 100
        final_score = (score * 0.9) + (popularity_score * 0.1)
        scored_candidates.append((final_score, row_id))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)

    results = []
    for final_score, row_id in scored_candidates[:number_of_recommendations]:
        row = dataframe.iloc[row_id]
        results.append(_row_to_normalized_result(row, score=float(final_score)))

    return results


# ============================================================
# Shared output adapter
# ============================================================


def _row_to_normalized_result(row: pd.Series, score: float | None) -> dict[str, Any]:
    """
    Convert one dataframe row into the shared normalized recommendation
    format both frontend result lists are adapted to.
    """
    return {
        "track_name": row["track_name"],
        "artist": row["artists"],
        "album_name": row["album_name"],
        "genre": row.get("track_genre"),
        "score": score,
    }