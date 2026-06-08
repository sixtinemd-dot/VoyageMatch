import requests


def get_forecast_summary(latitude: float, longitude: float) -> dict:
    """Fetch a simple forecast summary from Open-Meteo."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
        "forecast_days": 7,
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()["daily"]

    return {
        "source": "Open-Meteo",
        "avg_high_c": round(sum(data["temperature_2m_max"]) / len(data["temperature_2m_max"]), 1),
        "avg_low_c": round(sum(data["temperature_2m_min"]) / len(data["temperature_2m_min"]), 1),
        "total_precipitation_mm": round(sum(data["precipitation_sum"]), 1),
    }
