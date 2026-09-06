# 🎵 Music Recommendation System

A hybrid music recommendation system built using machine learning, semantic search, and audio features to generate personalized song recommendations from a dataset of **80,000+ tracks**.

The project combines traditional similarity-based recommendation with natural-language semantic search, allowing users to either select a song they like or describe the kind of music they want to hear.

---

## Overview

I built this project to explore how recommendation systems can combine multiple types of information instead of relying on a single similarity metric.

The system uses:

- **K-Nearest Neighbors (KNN)** for song-to-song similarity
- **TF-IDF** representations for text-based metadata
- **FAISS** for semantic search from free-form user queries
- Spotify audio features such as acousticness, instrumentalness, liveness, energy, danceability, and other musical characteristics
- **Popularity weighting** to improve recommendation quality
- **Spotify API** integration for album artwork and song links
- **Streamlit** for an interactive frontend

The goal was to build a recommender that considers both **what a song is and how it feels**.

---

## Features

### 🎧 Song-Based Recommendations

Users can select a song and receive similar tracks based on a combination of:

- Textual similarity
- Audio characteristics
- Feature scaling
- Nearest-neighbor search
- Popularity weighting

### 🔎 Natural Language Search

Users can also describe what they want to listen to using free-form queries such as:

> "chill acoustic songs for studying"

or

> "high-energy music for the gym"

The system processes the query and uses semantic similarity with **FAISS vector search** to retrieve relevant songs from the dataset.

### 🎼 Audio-Aware Recommendations

The recommender uses musical and audio characteristics to better understand the vibe of a track, including:

- Acousticness
- Instrumentalness
- Liveness
- Energy
- Danceability
- Valence
- Tempo
- Time signature

I also implemented additional rules around certain audio features to improve how closely recommendations match the user's intended mood and listening context.

---

## Tech Stack

**Language**
- Python

**Machine Learning & Retrieval**
- scikit-learn
- K-Nearest Neighbors (KNN)
- TF-IDF
- FAISS

**Data Processing**
- Pandas
- NumPy

**Frontend**
- Streamlit

**External Integration**
- Spotify Web API

---

## Dataset

The project uses a music dataset containing **more than 80,000 tracks**, including song metadata and numerical audio features.

Before being used by the recommendation system, the data is cleaned, transformed, and scaled so that features with very different numerical ranges can be compared effectively.

Working with a dataset of this size also gave me experience designing a recommendation pipeline where retrieval needed to remain efficient while still incorporating multiple types of song information.

---

## How It Works

The recommendation pipeline roughly follows this process:

1. **Load and clean the dataset** containing 80,000+ tracks.

2. **Process textual metadata** using TF-IDF to represent relevant textual information numerically.

3. **Scale numerical audio features** so that features with different ranges can contribute appropriately to similarity calculations.

4. **Combine relevant features** to create representations of each track.

5. **Use K-Nearest Neighbors** to identify tracks with similar characteristics.

6. **Apply popularity weighting and audio-based rules** to further refine and rank recommendations.

7. For free-form natural-language searches, **FAISS vector search** is used to efficiently retrieve semantically relevant tracks.

8. The recommendations are presented through an interactive **Streamlit frontend**.

9. The **Spotify API** is used to retrieve album artwork and Spotify links for the recommended tracks.

---

## Why I Built This

I wanted to understand recommendation systems beyond simply calling an existing recommendation API.

I was especially interested in the problem that musical similarity isn't necessarily defined by one thing. Two songs might share similar metadata but sound completely different, while two songs from different artists or genres might still fit the exact same mood.

That led me to experiment with combining **machine learning, textual similarity, numerical audio features, semantic vector search, and popularity signals** within the same system.

Building the project gave me hands-on experience with:

- Working with a dataset containing tens of thousands of records
- Feature engineering
- Data preprocessing and scaling
- Similarity-based machine learning
- Vector search and semantic retrieval
- Recommendation ranking
- API integration
- Building an interactive frontend
- Connecting multiple components into one complete application

One of my favorite parts of the project was learning **FAISS** and seeing how vector search could turn a natural-language request into useful retrieval from a large collection of songs.

More broadly, this project reinforced how much I enjoy taking an idea, learning whatever technologies are necessary to make it work, and turning it into a complete system that someone can actually interact with.

---

## Running the Application

### Main File

The main entry point for the application is:

`app.py`

This file launches the Streamlit interface and connects the recommendation pipeline, semantic search functionality, and Spotify API integration into the interactive application.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Configure Spotify API Credentials

If Spotify API credentials are required, add them to your environment:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

**Do not commit your actual Spotify API credentials to GitHub.**

### 4. Launch the Application

Run:

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local URL that can be opened in your browser.

---

## Project Structure

A simplified structure of the project looks like:

```text
music-recommendation-system/
│
├── app.py                  # Main Streamlit application
├── recommender.py          # Recommendation logic
├── requirements.txt        # Python dependencies
├── data/                   # Dataset / processed data
├── .env                    # API credentials (not committed)
└── README.md               # Project documentation
```

> Note: The exact structure may vary depending on how the project files are organized.

---

## Future Improvements

There are several directions I would like to explore further:

- **Personalized user profiles** based on individual listening preferences
- **Collaborative filtering** using user listening history
- More advanced **embedding models** for semantic search
- Automatic **playlist generation**
- Learning from user likes and dislikes
- More sophisticated recommendation ranking
- Direct Spotify account integration
- Recommendation explanations showing **why** a song was suggested
- Deployment as a publicly accessible web application
- Experimentation with hybrid recommendation architectures combining content-based and collaborative approaches

---

## Project Status

The core recommendation system and interactive prototype are complete.

I am continuing to experiment with recommendation ranking, semantic retrieval, and personalization techniques to improve the quality of the system.

---

## Author

**Ayaan Malik**

Built as part of my exploration of **machine learning, recommendation systems, semantic search, and applied AI**.
