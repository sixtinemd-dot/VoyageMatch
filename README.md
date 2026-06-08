# VoyageMatch

VoyageMatch is a GenAI and machine learning final project that recommends travel destinations from user preferences, weather expectations, budget, and travel style.

## Project Idea

The app asks the user for trip criteria and returns:

- destination recommendations
- match scores and explanation
- date- and season-aware climate estimates
- season-adjusted daily budget fit
- user-selectable display currencies with daily reference-rate conversion
- a generated itinerary with distinct daily themes and morning/afternoon/evening pacing
- dynamically loaded destination preview images with source attribution

## ML Components

The first version includes:

- weighted scoring
- cosine similarity
- K-nearest neighbors
- optional clustering by travel style

## Implemented Data Sources

- Kaggle: Worldwide Travel Cities (Ratings and Climate), 560 destinations
- Open-Meteo API: live 7-day weather snapshot
- Frankfurter API: daily reference exchange rates with an offline fallback
- Wikipedia / Wikimedia Commons: destination preview images through the MediaWiki API
- Optional Unsplash API: higher-quality destination photography when
  `UNSPLASH_ACCESS_KEY` is configured
- Optional OpenAI Responses API: GenAI itinerary generation when `OPENAI_API_KEY` is set

The Kaggle monthly climate JSON is expanded into 12 temperature features. A user's
travel dates select the relevant monthly temperature, local season, and a documented
seasonal cost multiplier before destinations are ranked.

The Kaggle dataset is normalized into:

```text
data/processed/destinations.csv
```

The app automatically uses the processed Kaggle dataset when it exists. If it does not exist, it falls back to the small sample dataset.

Custom destinations live in `data/custom/`. The build script cleans and merges every
CSV in that directory with the Kaggle data before writing the processed dataset.
This keeps project-specific rows safe when the Kaggle dataset is rebuilt.

The included Israeli destinations use monthly temperature means calculated from
Open-Meteo historical daily data for 2016-2025. Their interest ratings and budget
levels are explicitly curated for VoyageMatch and classified in the Middle East
travel region.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/build_dataset.py
streamlit run app.py
```

If `python` is not available on Windows, use the Python launcher:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/build_dataset.py
streamlit run app.py
```

## Optional API Setup

Create a `.env` file or set environment variables:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.2
UNSPLASH_ACCESS_KEY=your_unsplash_access_key_here
```

Without an OpenAI key, VoyageMatch uses the offline itinerary generator. Without
an Unsplash key, it falls back to attributed Wikimedia destination images.

## Current Status

VoyageMatch now runs with the full 560-destination Kaggle dataset, three recommendation approaches, travel-style clustering, weather integration, and optional GenAI itinerary generation.

The interface uses an electric-blue visual theme and supports USD, EUR, GBP, ILS,
CAD, AUD, JPY, and CHF. Recommendation calculations remain normalized in USD,
while inputs and results are converted to the user's selected currency.
