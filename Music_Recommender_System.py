import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack

spotify_tracks = pd.read_csv('dataset.csv', )
pd.set_option('display.max_columns', None)
spotify_tracks.dropna(inplace=True)
spotify_tracks.drop(['track_id', 'key', 'mode', 'time_signature', 'duration_ms'], axis=1, inplace=True)
spotify_tracks.drop_duplicates(
    subset=['track_name', 'artists'],
    inplace=True)
spotify_tracks.drop_duplicates(
    subset=['track_name', 'album_name'],
    inplace=True
)

spotify_tracks.reset_index(drop=True, inplace=True)
# print(len(spotify_tracks))


from sklearn.feature_extraction.text import CountVectorizer
spotify_tracks['combined_features'] = spotify_tracks['artists'] + ', ' + spotify_tracks['track_name'] + ', ' + spotify_tracks['album_name'] + ', ' + spotify_tracks['track_genre']
print(spotify_tracks.head())
text_transformer = CountVectorizer().fit(spotify_tracks['combined_features'])
track_transformer = text_transformer.transform(spotify_tracks['combined_features'])
#print(track_transformer.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(spotify_tracks[['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])
scaler_features = scaler.transform(spotify_tracks[['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])
#new_df = pd.DataFrame(scaler_features, columns = [['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']], index = spotify_tracks['track_name'])
#new_df['combined_features'] = spotify_tracks['combined_features']
#print('')
#print('')
#print('')
#print('NEW DF:')
#print(new_df.head())
from sklearn.feature_extraction.text import TfidfTransformer
tfidftransformer1 = TfidfTransformer().fit(track_transformer)
tfidftransformer2 = tfidftransformer1.transform(track_transformer)


# ================
# COSINE SIMILARITY
# =================

# from sklearn.metrics.pairwise import cosine_similarity
# final_features = hstack([tfidftransformer2, scaler_features])
# final_similarity_matrix = cosine_similarity(final_features)
# final_df = pd.DataFrame(final_similarity_matrix, index = spotify_tracks['track_name'].str.strip() + ' - ' + spotify_tracks['artists'].str.strip(), columns = spotify_tracks['track_name'].str.strip() + ' - ' + spotify_tracks['artists'].str.strip())

# =================
# Nearest Neighbors
# =================


from sklearn.neighbors import NearestNeighbors
final_features = hstack([tfidftransformer2, scaler_features]).tocsr()
neighbors = NearestNeighbors(metric= 'cosine', algorithm= 'brute')
neighbors.fit(final_features)


artist = input("Enter Artist: ")
song = input("Enter Song Name: ")
num = int(input("Enter How Many Songs You Want Recommended: "))
song_vector_row = spotify_tracks[(spotify_tracks['artists'].str.lower().str.contains(artist.lower())) & (spotify_tracks['track_name'].str.lower() == song.lower())]
index = song_vector_row.index[0]

song_vector = final_features[index]
distances, indices = neighbors.kneighbors(song_vector, n_neighbors = num + 1)
recommendation_indices = indices[0][1:]
recommendations = spotify_tracks.iloc[recommendation_indices]
print(recommendations[['track_name', 'artists', 'album_name', 'track_genre']])



# =================
# =================
# =================
# =================


def get_audio_vibe(row):
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


spotify_tracks["audio_vibes"] = spotify_tracks.apply(get_audio_vibe, axis=1)

spotify_tracks["semantic_text"] = (
        "Artist: " + spotify_tracks["artists"].fillna("") + ". " +
        "Genre: " + spotify_tracks["track_genre"].fillna("") + ". " +
        "This song has popularity " + spotify_tracks["popularity"].astype(str) +
        ", danceability " + spotify_tracks["danceability"].astype(str) +
        ", energy " + spotify_tracks["energy"].astype(str) +
        ", loudness " + spotify_tracks["loudness"].astype(str) +
        ", tempo " + spotify_tracks["tempo"].astype(str) +
        ", valence " + spotify_tracks["valence"].astype(str) +
        ", acousticness " + spotify_tracks["acousticness"].astype(str) +
        ", speechiness " + spotify_tracks["speechiness"].astype(str) +
        ", instrumentalness " + spotify_tracks["instrumentalness"].astype(str) +
        ", Is the song explicit: " + spotify_tracks["explicit"].astype(str) +
        ", Description: " + spotify_tracks["audio_vibes"].fillna("") + "."
)

from sentence_transformers import SentenceTransformer

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

song_embeddings = semantic_model.encode(
    spotify_tracks["semantic_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

song_embeddings = np.ascontiguousarray(
    song_embeddings,
    dtype=np.float32
)

import faiss

faiss.omp_set_num_threads(1)

dimension = song_embeddings.shape[1]

faiss_index = faiss.IndexFlatIP(dimension)

faiss_index.add(song_embeddings)

while True:
    query = input("Describe the vibe you want: ")

    num_recs = int(input("Enter the amount of recommendations you want: "))

    query_embedding = semantic_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    scores, indices = faiss_index.search(query_embedding, k=max(50, num_recs * 10))

    result = zip(scores[0], indices[0])

    final_array = []

    for score, index in result:
        popularity_score = int(spotify_tracks.iloc[index]['popularity']) / 100
        final_score = (score * 0.9) + (popularity_score * 0.1)
        final_array.append(
            (final_score, spotify_tracks.iloc[index][["track_name", "artists", "album_name", "track_genre"]]))

    final_array = sorted(
        final_array,
        key=lambda x: x[0],
        reverse=True
    )

    for recommendation in final_array[:num_recs]:
        print(recommendation)


# =================
# =================
# =================
# =================


def recommend_songs(song_name, artist_name, n_rec):
    song_key = song_name + ' - ' + artist_name
    if song_key in final_df.index:
        recommendations = final_df.loc[song_key]
        recommendations = recommendations.sort_values(ascending=False)
        return recommendations[1:n_rec + 1]
    else:
        return "Song not found"
