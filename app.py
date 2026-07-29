from __future__ import annotations

from typing import Any

import streamlit as st

from Recommender import (
    find_song_row_id,
    load_dataset,
    recommend_similar_songs,
    search_songs_by_vibe,
)
from spotify_service import enrich_recommendations


st.set_page_config(
    page_title="Music Discovery Engine",
    page_icon="🎧",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-subtitle {
        color: #8b8b8b;
        margin-bottom: 1.5rem;
    }

    .song-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 16px;
        min-height: 100%;
    }

    .song-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 2px;
    }

    .song-artist {
        font-size: 0.95rem;
        margin-bottom: 3px;
    }

    .song-album {
        color: #8b8b8b;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }

    .song-meta {
        color: #8b8b8b;
        font-size: 0.8rem;
        margin-bottom: 10px;
    }

    div[data-testid="stImage"] img {
        border-radius: 12px;
        aspect-ratio: 1 / 1;
        object-fit: cover;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def display_song_card(song: dict[str, Any], key_suffix: str) -> None:
    """
    Render one recommendation. Prefers official Spotify metadata when
    a reliable match was found, and falls back to the local dataset
    fields (which are always present) otherwise — the card never
    disappears just because Spotify had no match.
    """

    with st.container(border=True):
        cover_url = song.get("cover_url")

        if cover_url:
            st.image(
                cover_url,
                use_container_width=True,
            )
        else:
            st.info("Album artwork unavailable")

        track_name = song.get("spotify_track_name") or song.get("track_name", "Unknown track")
        artist = song.get("spotify_artist") or song.get("artist", "Unknown artist")
        album_name = song.get("album_name")

        st.markdown(f"### {track_name}")
        st.write(artist)

        if album_name:
            st.caption(album_name)

        metadata_parts: list[str] = []

        if song.get("genre"):
            metadata_parts.append(str(song["genre"]))

        if song.get("score") is not None:
            metadata_parts.append(
                f"Match: {float(song['score']):.2f}"
            )

        if song.get("release_date"):
            metadata_parts.append(str(song["release_date"]))

        if metadata_parts:
            st.caption(" • ".join(metadata_parts))

        if song.get("explicit"):
            st.caption("🅴 Explicit")

        spotify_url = song.get("spotify_url")

        if spotify_url:
            st.link_button(
                "Open in Spotify ↗",
                spotify_url,
                use_container_width=True,
            )
        else:
            st.button(
                "Spotify match unavailable",
                disabled=True,
                use_container_width=True,
                key=f"missing_{key_suffix}",
            )


def display_recommendations(
    recommendations: list[dict[str, Any]],
) -> None:
    """
    Shared display for both recommendation modes — song-to-song and
    vibe search results are never given separate result UIs.
    """

    if not recommendations:
        st.warning("No recommendations were found.")
        return

    st.subheader("Your recommendations")

    for start_index in range(0, len(recommendations), 3):
        row = recommendations[start_index:start_index + 3]
        columns = st.columns(3)

        for column_index, (column, song) in enumerate(zip(columns, row)):
            with column:
                display_song_card(song, key_suffix=f"{start_index + column_index}")


st.title("🎧 Music Discovery Engine")

st.markdown(
    """
    <p class="main-subtitle">
        Discover music through song similarity or natural-language descriptions.
    </p>
    """,
    unsafe_allow_html=True,
)

mode = st.segmented_control(
    "Choose a recommendation method",
    options=[
        "Similar to a Song",
        "Search by Vibe",
        "Browse All Songs",
    ],
    default="Similar to a Song",
    selection_mode="single",
)


if mode == "Similar to a Song":
    st.subheader("Find songs similar to one you already like")

    artist_name = st.text_input(
        "Enter Artist",
        placeholder="e.g. The Weeknd",
    )

    song_name = st.text_input(
        "Enter Song Name",
        placeholder="e.g. Blinding Lights",
    )

    number_of_similar_songs = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="similar_song_count",
    )

    if st.button(
        "Find Similar Songs",
        type="primary",
        use_container_width=True,
    ):
        if not artist_name.strip() or not song_name.strip():
            st.warning("Enter both an artist and a song name.")
        else:
            selected_row_id = find_song_row_id(artist_name, song_name)

            if selected_row_id is None:
                st.warning(
                    "No matching song found for that artist and song name. "
                    "Check the spelling and try again."
                )
            else:
                with st.spinner("Finding similar songs..."):
                    recommendations = recommend_similar_songs(
                        selected_song_row_id=selected_row_id,
                        number_of_recommendations=number_of_similar_songs,
                    )

                with st.spinner("Retrieving Spotify metadata..."):
                    enriched_results = enrich_recommendations(recommendations)

                display_recommendations(enriched_results)


elif mode == "Search by Vibe":
    st.subheader("Describe the music you want")

    query = st.text_input(
        "Musical description",
        placeholder=(
            "e.g. dark high-energy hip-hop with a fast tempo "
            "and low valence"
        ),
    )

    number_of_results = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

    st.caption(
        "For the best results, mention genre, mood, energy, tempo, "
        "danceability, acousticness, instrumentalness, or vocals."
    )

    if st.button(
        "Search Music",
        type="primary",
        use_container_width=True,
    ):
        if not query.strip():
            st.warning("Describe the type of music you want.")
        else:
            with st.spinner("Searching the music library..."):
                recommendations = search_songs_by_vibe(
                    query=query,
                    number_of_recommendations=number_of_results,
                )

            with st.spinner("Retrieving Spotify metadata..."):
                enriched_results = enrich_recommendations(recommendations)

            display_recommendations(enriched_results)


elif mode == "Browse All Songs":
    st.subheader("Browse the song library")

    dataset = load_dataset()

    search_text = st.text_input(
        "Filter by track or artist",
        placeholder="Start typing a track or artist name...",
    )

    browse_columns = ["track_name", "artists", "album_name", "track_genre"]
    browse_view = dataset[browse_columns]

    if search_text.strip():
        match_mask = dataset["track_name"].str.contains(
            search_text, case=False, na=False, regex=False
        ) | dataset["artists"].str.contains(
            search_text, case=False, na=False, regex=False
        )
        browse_view = browse_view[match_mask]

    browse_view = browse_view.rename(
        columns={
            "track_name": "Track",
            "artists": "Artist",
            "album_name": "Album",
            "track_genre": "Genre",
        }
    )

    st.caption(f"Showing {len(browse_view):,} of {len(dataset):,} songs")

    st.dataframe(
        browse_view,
        use_container_width=True,
        hide_index=True,
        height=600,
    )


st.divider()

st.caption(
    "Built with Python, scikit-learn, Sentence Transformers, "
    "FAISS, Spotify Web API and Streamlit."
)