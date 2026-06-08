from calendar import month_name
from datetime import date

import pandas as pd


def months_in_range(start_date: date, end_date: date) -> list[int]:
    months = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        months.append(month)
        month = 1 if month == 12 else month + 1
        year = year + 1 if month == 1 else year
    return months


def format_travel_period(start_date: date, end_date: date) -> str:
    if start_date.year == end_date.year:
        if start_date.month == end_date.month:
            return (
                f"{month_name[start_date.month]} "
                f"{start_date.day}-{end_date.day}, {start_date.year}"
            )
        return (
            f"{month_name[start_date.month]} {start_date.day} - "
            f"{month_name[end_date.month]} {end_date.day}, {start_date.year}"
        )
    return f"{start_date:%B %d, %Y} - {end_date:%B %d, %Y}"


def season_name(latitude: float, months: list[int]) -> str:
    representative_month = months[len(months) // 2]
    if abs(latitude) < 15:
        return "Tropical season"

    northern_seasons = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    southern_swap = {
        "Winter": "Summer",
        "Spring": "Autumn",
        "Summer": "Winter",
        "Autumn": "Spring",
    }
    season = northern_seasons[representative_month]
    return season if latitude >= 0 else southern_swap[season]


def seasonal_cost_multiplier(latitude: float, months: list[int]) -> float:
    """Estimate destination cost changes by local travel season."""
    if abs(latitude) < 15:
        multipliers = [1.08 if month in {12, 1, 2, 3} else 1.0 for month in months]
    else:
        local_summer = {6, 7, 8} if latitude >= 0 else {12, 1, 2}
        local_winter = {12, 1, 2} if latitude >= 0 else {6, 7, 8}
        multipliers = []
        for month in months:
            if month in local_summer:
                multipliers.append(1.18)
            elif month in local_winter:
                multipliers.append(0.92)
            else:
                multipliers.append(1.05)
    return sum(multipliers) / len(multipliers)


def apply_travel_dates(
    destinations: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    dated = destinations.copy()
    months = months_in_range(start_date, end_date)
    temperature_columns = [f"temp_month_{month}" for month in months]
    available_columns = [
        column for column in temperature_columns if column in dated.columns
    ]

    dated["annual_avg_temp_c"] = dated["avg_temp_c"]
    if available_columns:
        dated["avg_temp_c"] = dated[available_columns].mean(axis=1)

    dated["base_cost_per_day_usd"] = dated["cost_per_day_usd"]
    dated["seasonal_multiplier"] = dated["latitude"].apply(
        lambda latitude: seasonal_cost_multiplier(float(latitude), months)
    )
    dated["cost_per_day_usd"] = (
        dated["base_cost_per_day_usd"] * dated["seasonal_multiplier"]
    ).round(0)
    dated["travel_season"] = dated["latitude"].apply(
        lambda latitude: season_name(float(latitude), months)
    )
    return dated
