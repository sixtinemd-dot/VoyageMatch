from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing import FEATURE_COLUMNS


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

INTEREST_COLUMNS = [
    "beach",
    "culture",
    "nature",
    "nightlife",
    "food",
    "adventure",
    "relaxation",
]


@dataclass
class UserPreferences:
    budget_per_day_usd: float
    preferred_temp_c: float
    beach: int
    culture: int
    nature: int
    nightlife: int
    food: int
    adventure: int
    relaxation: int
    safety: int
    popularity: int
    method: str = "Weighted scoring"


def _user_vector(preferences: UserPreferences) -> np.ndarray:
    return np.array(
        [
            preferences.budget_per_day_usd,
            preferences.preferred_temp_c,
            preferences.beach,
            preferences.culture,
            preferences.nature,
            preferences.nightlife,
            preferences.food,
            preferences.adventure,
            preferences.relaxation,
            preferences.popularity,
            preferences.safety,
        ],
        dtype=float,
    )


def weighted_score(df: pd.DataFrame, preferences: UserPreferences) -> pd.DataFrame:
    scored = df.copy()

    budget_fit = 1 - (
        abs(scored["cost_per_day_usd"] - preferences.budget_per_day_usd)
        / preferences.budget_per_day_usd
    )
    temp_fit = 1 - (abs(scored["avg_temp_c"] - preferences.preferred_temp_c) / 30)
    interest_fit = np.zeros(len(scored))

    for column in INTEREST_COLUMNS:
        user_value = getattr(preferences, column)
        interest_fit += 1 - (abs(scored[column] - user_value) / 10)

    interest_fit = interest_fit / len(INTEREST_COLUMNS)
    safety_fit = 1 - (abs(scored["safety"] - preferences.safety) / 10)
    popularity_fit = 1 - (abs(scored["popularity"] - preferences.popularity) / 10)

    scored["match_score"] = (
        0.30 * budget_fit.clip(0, 1)
        + 0.15 * temp_fit.clip(0, 1)
        + 0.35 * interest_fit.clip(0, 1)
        + 0.10 * safety_fit.clip(0, 1)
        + 0.10 * popularity_fit.clip(0, 1)
    )
    return scored.sort_values("match_score", ascending=False)


def cosine_recommend(df: pd.DataFrame, preferences: UserPreferences) -> pd.DataFrame:
    features = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    user = _user_vector(preferences).reshape(1, -1)

    scaler = MinMaxScaler()
    all_vectors = scaler.fit_transform(np.vstack([features, user]))
    destination_vectors = all_vectors[:-1]
    user_vector = all_vectors[-1].reshape(1, -1)

    scored = df.copy()
    scored["match_score"] = cosine_similarity(destination_vectors, user_vector).ravel()
    return scored.sort_values("match_score", ascending=False)


def knn_recommend(
    df: pd.DataFrame,
    preferences: UserPreferences,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    features = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    user = _user_vector(preferences).reshape(1, -1)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_user = scaler.transform(user)

    model = NearestNeighbors(n_neighbors=min(n_neighbors, len(df)), metric="euclidean")
    model.fit(scaled_features)
    distances, indices = model.kneighbors(scaled_user)

    result = df.iloc[indices.ravel()].copy()
    result["match_score"] = 1 / (1 + distances.ravel())
    return result.sort_values("match_score", ascending=False)


def add_travel_style_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    clustered = df.copy()
    features = clustered[INTEREST_COLUMNS].to_numpy(dtype=float)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=min(n_clusters, len(clustered)),
        random_state=42,
        n_init="auto",
    )
    clustered["travel_style_cluster"] = model.fit_predict(scaled)
    return clustered


def recommend_destinations(
    df: pd.DataFrame,
    preferences: UserPreferences,
    top_n: int = 5,
) -> pd.DataFrame:
    if preferences.method == "Cosine similarity":
        ranked = cosine_recommend(df, preferences)
    elif preferences.method == "K-nearest neighbors":
        ranked = knn_recommend(df, preferences, n_neighbors=top_n)
    else:
        ranked = weighted_score(df, preferences)

    return ranked.head(top_n)
