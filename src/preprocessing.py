import pandas as pd


MONTHLY_TEMPERATURE_COLUMNS = [f"temp_month_{month}" for month in range(1, 13)]

DESTINATION_COLUMNS = [
    "city",
    "country",
    "region",
    "latitude",
    "longitude",
    "cost_per_day_usd",
    "avg_temp_c",
    "beach",
    "culture",
    "nature",
    "nightlife",
    "food",
    "adventure",
    "relaxation",
    "popularity",
    "safety",
] + MONTHLY_TEMPERATURE_COLUMNS

FEATURE_COLUMNS = [
    "cost_per_day_usd",
    "avg_temp_c",
    "beach",
    "culture",
    "nature",
    "nightlife",
    "food",
    "adventure",
    "relaxation",
    "popularity",
    "safety",
]


def clean_destinations(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup for destination records."""
    cleaned = df.copy()
    for column in DESTINATION_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = None

    cleaned["city"] = cleaned["city"].str.strip()
    cleaned["country"] = cleaned["country"].str.strip()
    cleaned["region"] = cleaned["region"].fillna("Unknown").astype(str).str.strip()

    numeric_columns = (
        FEATURE_COLUMNS
        + MONTHLY_TEMPERATURE_COLUMNS
        + ["latitude", "longitude"]
    )
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["city", "country", "latitude", "longitude"])
    cleaned["cost_per_day_usd"] = cleaned["cost_per_day_usd"].fillna(135)
    cleaned["avg_temp_c"] = cleaned["avg_temp_c"].fillna(cleaned["avg_temp_c"].median())
    for column in MONTHLY_TEMPERATURE_COLUMNS:
        cleaned[column] = cleaned[column].fillna(cleaned["avg_temp_c"])
    cleaned[FEATURE_COLUMNS] = cleaned[FEATURE_COLUMNS].fillna(
        cleaned[FEATURE_COLUMNS].median()
    )
    cleaned[FEATURE_COLUMNS] = cleaned[FEATURE_COLUMNS].clip(lower=0)
    rating_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in ["cost_per_day_usd", "avg_temp_c"]
    ]
    cleaned[rating_columns] = cleaned[rating_columns].clip(0, 10)
    return cleaned.drop_duplicates(subset=["city", "country"]).reset_index(drop=True)
