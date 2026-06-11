# VoyageMatch

VoyageMatch is a full-stack travel recommendation application built with
Python and Streamlit. It combines data engineering, recommendation algorithms,
external APIs, optional generative AI, and persistent user accounts.

Users can create an account, describe their ideal trip, receive ranked
destination recommendations, generate an itinerary, and save their
preferences, favorites, recommendation history, and itineraries.

## Main Features

- Account registration, login, and logout
- Persistent user data with Neon PostgreSQL
- Automatic local SQLite fallback for development
- Region and country filtering
- Date-aware destination climate
- Season-adjusted estimated costs
- Eight display currencies
- Weighted scoring, cosine similarity, and K-nearest neighbors
- K-means travel-style clustering
- Five ranked destination recommendations
- Interactive destination map
- Live seven-day weather when departure is close
- Unsplash or Wikimedia destination images with attribution
- OpenAI or offline rule-based itinerary generation
- RAG assistant grounded in retrieved destination records
- OpenAI vector embeddings stored with Neon pgvector
- Transparent retrieval sources and local TF-IDF fallback
- Saved preferences, favorites, recommendation history, and itineraries

## Application Flow

1. Destination data is loaded and cleaned with pandas.
2. The user's dates select the relevant monthly climate values.
3. Estimated destination costs are adjusted for the local season.
4. Destinations are filtered by the selected region and country.
5. The selected recommendation algorithm compares the user profile with each
   destination.
6. The five strongest matches are displayed.
7. External services add exchange rates, images, weather, and optional AI.
8. Signed-in users can save results and return to them later.

## Recommendation Methods

### Weighted scoring

Combines interpretable category scores:

- budget fit: 30%
- temperature fit: 15%
- interests: 35%
- safety: 10%
- popularity: 10%

### Cosine similarity

Scales the numerical features and measures the similarity between the user's
preference vector and each destination vector.

### K-nearest neighbors

Scales the same features and finds the destinations with the smallest
Euclidean distance from the user profile.

### K-means clustering

Groups destinations into four broad travel-style clusters. Clustering is an
additional unsupervised-learning component and does not determine the final
ranking.

## Destination Data

The main source is the Kaggle **Worldwide Travel Cities Ratings and Climate**
dataset, containing 560 destinations. VoyageMatch adds seven custom
destinations, producing a processed dataset of **567 destinations**.

The build process:

- normalizes source column names
- converts ratings to a common 0-10 scale
- expands monthly climate JSON into 12 temperature features
- converts budget categories into estimated daily USD costs
- fills required missing values
- merges CSV files from `data/custom/`
- writes `data/processed/destinations.csv`

If the processed dataset is unavailable, the app uses the bundled sample CSV.

## Database and Accounts

VoyageMatch supports two database backends:

- **Neon PostgreSQL:** hosted storage for deployed, multi-user use
- **SQLite:** automatic local fallback when `DATABASE_URL` is not configured

The application uses six tables. The five account tables are created during
database startup; the vector table is created when the RAG assistant is opened
or indexed.

| Table | Purpose |
|---|---|
| `users` | Email, display name, password hash, and account creation time |
| `user_preferences` | The user's latest saved trip preferences |
| `favorites` | Saved destinations |
| `recommendation_history` | Saved searches and their ranked results |
| `saved_itineraries` | Generated itineraries and travel dates |
| `destination_embeddings` | Destination documents and semantic vectors |

Every saved record is associated with a user ID. Foreign keys and cascade
deletion maintain relationships between account data.

Passwords are never stored directly. They are salted and hashed with
PBKDF2-HMAC-SHA256 using 600,000 iterations. Database queries use parameters
instead of inserting user input directly into SQL.

## External Services

| Service | Purpose | Required? | Fallback |
|---|---|---:|---|
| Kaggle | Original destination dataset | Data preparation | Bundled raw/sample data |
| Neon | Hosted PostgreSQL database | No | Local SQLite |
| Frankfurter | Reference exchange rates | No key | Built-in approximate rates |
| Open-Meteo | Live seven-day weather | No key | Historical monthly climate |
| Unsplash | Destination photography | Optional key | Wikipedia/Wikimedia |
| Wikipedia/Wikimedia | Images and attribution | No key | App continues without image |
| OpenAI | Personalized itineraries | Optional key | Rule-based itinerary |
| OpenAI Embeddings | Semantic destination retrieval | Optional key | Local TF-IDF |

## Project Structure

```text
VoyageMatch/
├── app.py                         # Streamlit interface and application flow
├── src/
│   ├── database.py                # Accounts and persistent user data
│   ├── data_loader.py             # Destination CSV loading
│   ├── dataset_builder.py         # Kaggle normalization and custom merge
│   ├── preprocessing.py           # Data cleaning and feature preparation
│   ├── recommender.py             # Scoring, cosine, KNN, and K-means
│   ├── seasonality.py             # Monthly climate and seasonal costs
│   ├── currency.py                # Exchange rates and conversions
│   ├── weather.py                 # Open-Meteo forecast
│   ├── destination_image.py       # Unsplash and Wikimedia images
│   ├── itinerary_generator.py     # OpenAI and offline itineraries
│   └── rag.py                     # Embeddings, retrieval, grounded answers
├── scripts/build_dataset.py       # Rebuilds the processed dataset
├── scripts/index_embeddings.py    # Builds the destination vector index
├── data/raw/                      # Original and sample CSV files
├── data/custom/                   # Project-specific destinations
├── data/processed/                # Final application dataset
├── .streamlit/config.toml         # Streamlit visual theme
├── .env.example                   # Environment-variable template
└── requirements.txt               # Python dependencies
```

## Local Setup

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_dataset.py
streamlit run app.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/build_dataset.py
streamlit run app.py
```

## Environment Variables

Copy `.env.example` to `.env` and replace only the values you use:

```env
# Hosted account storage. Without this, the app uses local SQLite.
DATABASE_URL=postgresql://user:password@host/neondb?sslmode=require

# Optional AI itinerary generation.
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5.2
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional higher-quality destination images.
UNSPLASH_ACCESS_KEY=your_unsplash_access_key
```

The `.env` file is ignored by Git. Never commit database credentials, API
keys, or `.streamlit/secrets.toml`.

## Neon Setup

1. Create a project at [Neon](https://neon.tech/).
2. Open **Connect** in the Neon dashboard.
3. Select the pooled connection option.
4. Copy the PostgreSQL connection string.
5. Put it in `.env` as `DATABASE_URL`.
6. Restart Streamlit.

The application creates its tables on the first successful connection. The
sidebar displays `Storage: Neon PostgreSQL` when Neon is active.

## RAG and Vector Index Setup

The AI travel assistant uses retrieval-augmented generation:

1. Each destination becomes a document containing its description, region,
   climate, cost, and strongest travel qualities.
2. OpenAI converts each document into a 1,536-dimensional embedding.
3. Neon stores the vectors with the PostgreSQL `pgvector` extension.
4. The user's question is embedded with the same model.
5. Cosine distance retrieves the most relevant destination records.
6. Only those retrieved records are sent to the OpenAI transformer.
7. The answer cites the retrieved destination sources.

After configuring `OPENAI_API_KEY` and `DATABASE_URL`, build the index from the
app's **AI travel assistant** tab or run:

```bash
python scripts/index_embeddings.py
```

The app enables the `vector` extension and creates the vector table
automatically. If OpenAI embedding quota is unavailable, indexing automatically
uses deterministic local 1,536-dimensional text vectors stored in the same
pgvector table. Before an index exists, retrieval falls back to local TF-IDF.
Transformer-generated answers still require available OpenAI API quota.

For Streamlit Community Cloud, add the connection string to the app's Secrets:

```toml
DATABASE_URL = "postgresql://user:password@host/neondb?sslmode=require"
```

## Reliability

VoyageMatch remains useful when optional services are unavailable:

- SQLite replaces Neon locally.
- Sample data replaces the processed dataset.
- Built-in rates replace Frankfurter.
- Monthly climate replaces unavailable or distant forecasts.
- Wikimedia replaces Unsplash.
- The rule-based planner replaces OpenAI.
- Local TF-IDF replaces vector and pgvector retrieval.

## Current Status

The application currently uses 567 destinations, three recommendation
approaches, K-means clustering, account authentication, persistent user data,
seasonality, currency conversion, weather integration, attributed images, and
optional generative-AI itineraries. Its RAG assistant supports OpenAI
embeddings, Neon pgvector retrieval, transformer inference, and source
citations.
