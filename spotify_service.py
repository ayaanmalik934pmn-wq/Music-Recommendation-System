"""
Spotify metadata enrichment for the Music Discovery Engine.

This module only looks up existing recommendations on Spotify for
display purposes (artwork, official name, link). It never generates
or influences recommendations.

Auth: Client Credentials flow only (app-level, no user login, no
redirect URI). Credentials are read exclusively from st.secrets and
are never logged, printed, or included in exception messages.
"""

from __future__ import annotations

import re
from typing import Any

import spotipy
import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials


# ============================================================
# Client
# ============================================================


@st.cache_resource(show_spinner=False)
def get_spotify_client() -> spotipy.Spotify:
    """
    Create and cache one Spotify API client for the app's lifetime.
    Never recreated per song lookup.
    """
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )

    return spotipy.Spotify(
        auth_manager=auth_manager,
        requests_timeout=10,
        retries=3,
    )


# ============================================================
# Text cleaning / matching helpers
#
# These exist only for frontend song selection, duplicate handling,
# and Spotify lookups/validation. They never touch the dataframe
# columns or text the ML models were trained/built on.
# ============================================================


_WHITESPACE_PATTERN = re.compile(r"\s+")

# Bracketed/parenthetical feature credits: (feat. X), [ft. X], etc.
_BRACKETED_FEATURE_PATTERN = re.compile(
    r"[\(\[]\s*(?:feat\.?|ft\.?|featuring)\s+[^\)\]]*[\)\]]",
    re.IGNORECASE,
)

# Trailing "feat./ft./featuring Artist" with no brackets at all.
_INLINE_FEATURE_PATTERN = re.compile(
    r"\s+(?:feat\.?|ft\.?|featuring)\s+.+$",
    re.IGNORECASE,
)

# Common version/edit tags, bracketed or trailing after a hyphen.
_VERSION_TAG_PATTERN = re.compile(
    r"""
    (?:
        [\(\[]\s*
        (?:re-?master(?:ed)?(?:\s*\d{2,4})?|remix(?:ed)?|radio\s*edit|
           live(?:\s*version)?|deluxe(?:\s*(?:version|edition))?|
           extended(?:\s*(?:version|mix))?|explicit|clean|
           single\s*version|mono|stereo)
        \s*[\)\]]
    )
    |
    (?:
        \s*-\s*
        (?:re-?master(?:ed)?(?:\s*\d{2,4})?|remix(?:ed)?|radio\s*edit|
           live(?:\s*version)?|deluxe(?:\s*(?:version|edition))?|
           extended(?:\s*(?:version|mix))?|single\s*version|mono|stereo)
        \s*$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Conservative artist separators: only split on clearly delimiting
# punctuation/words, never on punctuation that could be part of an
# intentional artist name.
_ARTIST_SEPARATOR_PATTERN = re.compile(
    r"\s*(?:,|;|&|\bfeat\.?\b|\bft\.?\b|\bfeaturing\b|\band\b)\s*",
    re.IGNORECASE,
)


def normalize_for_comparison(value: str | None) -> str:
    """Lowercase, strip, collapse whitespace for safe comparisons only."""
    if not value:
        return ""
    value = value.strip().lower()
    return _WHITESPACE_PATTERN.sub(" ", value)


def clean_track_title(track_name: str | None) -> str:
    """
    Build a simplified title for fallback Spotify searches by removing
    bracketed feature credits and common version/edit tags. The
    original track_name is always kept untouched for display.
    """
    if not track_name:
        return ""

    cleaned = _BRACKETED_FEATURE_PATTERN.sub("", track_name)
    cleaned = _INLINE_FEATURE_PATTERN.sub("", cleaned)
    cleaned = _VERSION_TAG_PATTERN.sub("", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip(" -")

    return cleaned or track_name


def primary_artist(artist_field: str | None) -> str:
    """
    Extract one primary-artist string for Spotify search from a field
    that may contain multiple artists (comma/semicolon/ampersand/
    "and"/"feat."/"ft."/"featuring" separated, matching this dataset's
    formatting, e.g. "Ingrid Michaelson;ZAYN"). The full original
    artist string is never altered for display — this is search-only.
    """
    if not artist_field:
        return ""

    segments = [
        segment
        for segment in _ARTIST_SEPARATOR_PATTERN.split(artist_field.strip())
        if segment
    ]

    return segments[0] if segments else artist_field


def _escape_for_query(value: str) -> str:
    """Strip characters that would break a quoted Spotify search field."""
    return value.replace('"', "").strip()


def _extract_candidate_artists(candidate: dict[str, Any]) -> str:
    return " ".join(
        artist.get("name", "") for artist in candidate.get("artists", [])
    )


def _tracks_reasonably_match(
    candidate: dict[str, Any],
    local_track_name: str,
    local_primary_artist: str,
) -> bool:
    """
    Validate a Spotify candidate against the local recommendation
    before accepting it, so we don't silently accept a cover, karaoke
    version, tribute, unrelated remix, or same-title/different-artist
    track from a broad query.
    """
    candidate_track = normalize_for_comparison(candidate.get("name", ""))
    candidate_artists = normalize_for_comparison(_extract_candidate_artists(candidate))

    local_track = normalize_for_comparison(local_track_name)
    local_artist = normalize_for_comparison(local_primary_artist)

    if not candidate_track or not local_track:
        return False

    track_matches = local_track in candidate_track or candidate_track in local_track
    artist_matches = bool(local_artist) and local_artist in candidate_artists

    return track_matches and artist_matches


def _normalize_spotify_track(candidate: dict[str, Any]) -> dict[str, Any]:
    album = candidate.get("album", {})
    images = album.get("images", [])
    cover_url = images[0]["url"] if images else None

    artist_names = [
        artist.get("name", "") for artist in candidate.get("artists", [])
    ]

    return {
        "spotify_id": candidate.get("id"),
        "spotify_track_name": candidate.get("name"),
        "spotify_artist": ", ".join(artist_names) if artist_names else None,
        "album_name": album.get("name"),
        "cover_url": cover_url,
        "spotify_url": candidate.get("external_urls", {}).get("spotify"),
        "release_date": album.get("release_date"),
        "explicit": candidate.get("explicit", False),
    }


# ============================================================
# Lookup
# ============================================================


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def search_spotify_track(
    track_name: str,
    artist_name: str,
) -> dict[str, Any] | None:
    """
    Find the Spotify track matching a local recommendation, using a
    staged search so odd formatting (feat. credits, remaster tags,
    multiple artists) doesn't cause a miss or a wrong match:

    1. exact original title + original artist
    2. cleaned title + primary artist
    3. cleaned title + primary artist, unqualified/broad query
    4. cleaned title only (final fallback)

    Every candidate is validated against the local track/artist before
    being accepted — a broad or fallback query never auto-accepts the
    first result. Returns None (no metadata) if nothing reliable is
    found; the caller falls back to local data.
    """
    spotify = get_spotify_client()

    lead_artist = primary_artist(artist_name)
    cleaned_title = clean_track_title(track_name)

    safe_original_track = _escape_for_query(track_name)
    safe_original_artist = _escape_for_query(artist_name)
    safe_cleaned_title = _escape_for_query(cleaned_title)
    safe_lead_artist = _escape_for_query(lead_artist)

    staged_queries = [
        f'track:"{safe_original_track}" artist:"{safe_original_artist}"',
        f'track:"{safe_cleaned_title}" artist:"{safe_lead_artist}"',
        f"{safe_cleaned_title} {safe_lead_artist}".strip(),
        safe_cleaned_title,
    ]

    seen_queries: set[str] = set()

    for query in staged_queries:
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)

        try:
            response = spotify.search(q=query, type="track", limit=5)
        except Exception:
            continue

        candidates = response.get("tracks", {}).get("items", [])

        for candidate in candidates:
            if _tracks_reasonably_match(candidate, track_name, lead_artist):
                return _normalize_spotify_track(candidate)

    return None


def enrich_recommendations(
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add Spotify metadata to recommendation results. Local fields
    (track_name, artist, album_name, genre, score) are always kept as
    a fallback; spotify_* fields are added on top when a reliable
    match is found, and left absent (not fabricated) otherwise.
    """
    enriched_results: list[dict[str, Any]] = []

    for recommendation in recommendations:
        spotify_data = search_spotify_track(
            track_name=recommendation.get("track_name", ""),
            artist_name=recommendation.get("artist", ""),
        )

        enriched_result = recommendation.copy()

        if spotify_data:
            enriched_result.update(spotify_data)

        enriched_results.append(enriched_result)

    return enriched_results