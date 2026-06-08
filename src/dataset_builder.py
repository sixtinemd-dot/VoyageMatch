import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.preprocessing import MONTHLY_TEMPERATURE_COLUMNS, clean_destinations


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
CUSTOM_DIR = ROOT_DIR / "data" / "custom"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "destinations.csv"

BUDGET_TO_DAILY_COST = {
    "budget": 70,
    "mid-range": 135,
    "mid range": 135,
    "midrange": 135,
    "luxury": 240,
}


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {column.lower().strip().replace(" ", "_"): column for column in df.columns}
    for candidate in candidates:
        key = candidate.lower().strip().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def _rating(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.max(skipna=True) <= 5:
        numeric = numeric * 2
    return numeric.clip(0, 10)


def _parse_monthly_average(value: object) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, dict):
        parsed = value
    else:
        text = str(value).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return pd.to_numeric(text, errors="coerce")

    if isinstance(parsed, dict):
        values = [
            month_data.get("avg") if isinstance(month_data, dict) else month_data
            for month_data in parsed.values()
        ]
    elif isinstance(parsed, list):
        values = parsed
    else:
        return np.nan

    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def _parse_monthly_temperatures(value: object) -> dict[str, float]:
    temperatures = {column: np.nan for column in MONTHLY_TEMPERATURE_COLUMNS}
    if pd.isna(value):
        return temperatures

    if isinstance(value, dict):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value).strip())
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(str(value).strip())
            except (ValueError, SyntaxError):
                return temperatures

    if not isinstance(parsed, dict):
        return temperatures

    for month in range(1, 13):
        month_data = parsed.get(str(month), parsed.get(month))
        value = month_data.get("avg") if isinstance(month_data, dict) else month_data
        temperatures[f"temp_month_{month}"] = pd.to_numeric(value, errors="coerce")
    return temperatures


def _budget_to_cost(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    return BUDGET_TO_DAILY_COST.get(str(value).strip().lower(), np.nan)


def normalize_travel_city_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a travel-city Kaggle-style CSV into VoyageMatch columns."""
    city_col = _find_column(raw_df, ["city", "city_name", "name"])
    country_col = _find_column(raw_df, ["country", "country_name"])
    region_col = _find_column(raw_df, ["region", "continent"])
    lat_col = _find_column(raw_df, ["latitude", "lat"])
    lon_col = _find_column(raw_df, ["longitude", "lng", "lon"])
    budget_col = _find_column(raw_df, ["budget_level", "budget", "cost_level", "cost"])
    temp_col = _find_column(
        raw_df,
        [
            "avg_temp_c",
            "average_temperature",
            "avg_temp_monthly",
            "monthly_avg_temp",
            "climate",
        ],
    )

    required = {
        "city": city_col,
        "country": country_col,
        "latitude": lat_col,
        "longitude": lon_col,
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    normalized = pd.DataFrame()
    normalized["city"] = raw_df[city_col]
    normalized["country"] = raw_df[country_col]
    normalized["region"] = raw_df[region_col] if region_col else "Unknown"
    normalized["latitude"] = raw_df[lat_col]
    normalized["longitude"] = raw_df[lon_col]

    if budget_col:
        normalized["cost_per_day_usd"] = raw_df[budget_col].apply(_budget_to_cost)
    else:
        normalized["cost_per_day_usd"] = np.nan

    if temp_col:
        normalized["avg_temp_c"] = raw_df[temp_col].apply(_parse_monthly_average)
        monthly_temperatures = raw_df[temp_col].apply(_parse_monthly_temperatures)
        normalized[MONTHLY_TEMPERATURE_COLUMNS] = pd.DataFrame(
            monthly_temperatures.tolist(),
            index=raw_df.index,
        )
    else:
        normalized["avg_temp_c"] = np.nan

    column_map = {
        "beach": ["beach", "beaches", "beach_rating", "beaches_rating"],
        "culture": ["culture", "culture_rating"],
        "nature": ["nature", "nature_rating"],
        "nightlife": ["nightlife", "nightlife_rating"],
        "food": ["food", "cuisine", "cuisine_rating", "food_rating"],
        "adventure": ["adventure", "adventure_rating"],
        "relaxation": ["relaxation", "wellness", "wellness_rating"],
        "popularity": ["popularity", "overall", "overall_rating", "urban", "urban_rating"],
        "safety": ["safety", "safety_rating"],
    }

    for output_column, candidates in column_map.items():
        source = _find_column(raw_df, candidates)
        if source:
            normalized[output_column] = _rating(raw_df[source])
        elif output_column == "safety":
            normalized[output_column] = 7
        elif output_column == "popularity":
            available = [
                column
                for column in ["culture", "nature", "food", "nightlife"]
                if column in normalized
            ]
            normalized[output_column] = (
                normalized[available].mean(axis=1) if available else 7
            )
        else:
            normalized[output_column] = 5

    normalized["source_dataset"] = "kaggle_worldwide_travel_cities"
    description_col = _find_column(
        raw_df,
        ["description", "short_description", "summary"],
    )
    normalized["description"] = raw_df[description_col] if description_col else ""
    return clean_destinations(normalized)


def build_processed_dataset(
    raw_csv_path: Path,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_csv_path)
    processed = normalize_travel_city_dataset(raw_df)
    custom_files = sorted(CUSTOM_DIR.glob("*.csv")) if CUSTOM_DIR.exists() else []
    if custom_files:
        custom_destinations = pd.concat(
            [clean_destinations(pd.read_csv(path)) for path in custom_files],
            ignore_index=True,
        )
        processed = pd.concat(
            [custom_destinations, processed],
            ignore_index=True,
        )
        processed = clean_destinations(processed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return processed


def discover_raw_csv(raw_dir: Path = RAW_DIR) -> Path | None:
    candidates = sorted(
        path for path in raw_dir.glob("*.csv") if path.name != "sample_destinations.csv"
    )
    return candidates[0] if candidates else None
