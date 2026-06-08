from dataclasses import dataclass

import requests


CURRENCY_LABELS = {
    "USD": "USD - US Dollar",
    "EUR": "EUR - Euro",
    "GBP": "GBP - British Pound",
    "ILS": "ILS - Israeli New Shekel",
    "CAD": "CAD - Canadian Dollar",
    "AUD": "AUD - Australian Dollar",
    "JPY": "JPY - Japanese Yen",
    "CHF": "CHF - Swiss Franc",
}

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "ILS": "₪",
    "CAD": "C$",
    "AUD": "A$",
    "JPY": "¥",
    "CHF": "CHF ",
}

FALLBACK_USD_RATES = {
    "USD": 1.0,
    "EUR": 0.88,
    "GBP": 0.75,
    "ILS": 3.45,
    "CAD": 1.37,
    "AUD": 1.52,
    "JPY": 157.0,
    "CHF": 0.80,
}


@dataclass
class ExchangeRates:
    rates: dict[str, float]
    date: str
    source: str
    is_fallback: bool = False


def fetch_exchange_rates() -> ExchangeRates:
    quotes = ",".join(code for code in CURRENCY_LABELS if code != "USD")
    try:
        response = requests.get(
            "https://api.frankfurter.dev/v2/rates",
            params={"base": "USD", "quotes": quotes},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        rates = {"USD": 1.0}
        for item in payload:
            rates[item["quote"]] = float(item["rate"])

        if not all(code in rates for code in CURRENCY_LABELS):
            raise ValueError("Currency response was incomplete.")

        return ExchangeRates(
            rates=rates,
            date=payload[0]["date"],
            source="Frankfurter reference rates",
        )
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return ExchangeRates(
            rates=FALLBACK_USD_RATES,
            date="offline estimate",
            source="Built-in fallback rates",
            is_fallback=True,
        )


def from_usd(amount_usd: float, currency: str, rates: ExchangeRates) -> float:
    return amount_usd * rates.rates[currency]


def to_usd(amount: float, currency: str, rates: ExchangeRates) -> float:
    return amount / rates.rates[currency]


def format_money(amount: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS[currency]
    decimals = 0 if currency == "JPY" or abs(amount) >= 100 else 2
    return f"{symbol}{amount:,.{decimals}f}"
