# VoyageMatch Presentation Guide

## 1. The Project in One Sentence

VoyageMatch is a Streamlit travel recommendation app that compares a user's
budget, preferred weather, travel dates, location choices, and interests with a
dataset of destinations, then recommends the closest matches and creates a
personalized itinerary.

## 2. What the User Can Do

The user chooses:

- a region and optionally a country
- travel dates
- a display currency
- a daily budget
- a preferred average temperature
- interest levels for beaches, culture, nature, nightlife, food, adventure,
  and relaxation
- safety and popularity preferences
- one of three recommendation methods

The user can also:

- create an account and log in securely
- save and automatically restore travel preferences
- save or remove favorite destinations
- save recommendation searches and their five results
- save, review, and delete generated itineraries
- ask natural-language questions about the destination knowledge base
- inspect the destination records retrieved for each answer

The app then displays:

- the best destination match
- four additional recommendations
- a match percentage
- a season-adjusted estimated budget
- climate information for the selected travel months
- a destination image and map
- a live seven-day forecast when the trip is close enough
- a generated day-by-day itinerary

## 3. How the App Works

The main flow is:

1. Load the processed destination dataset.
2. Clean missing or invalid values.
3. Group destinations into broad travel-style clusters.
4. Read the user's choices from the Streamlit sidebar.
5. Adjust each destination's temperature and estimated cost for the selected
   travel dates.
6. Filter destinations by region or country.
7. Compare the user profile with each remaining destination.
8. Rank the destinations and select the five best matches.
9. Fetch optional live information such as exchange rates, images, and weather.
10. Generate an itinerary with OpenAI or the offline rule-based generator.
11. Save account data, preferences, favorites, history, and itineraries in
    SQLite locally or Neon PostgreSQL when configured.
12. For RAG questions, retrieve semantically related destinations and give
    those records to the transformer as grounded context.

## 4. Files and Their Purposes

### `app.py`

This is the main application and user interface.

It:

- configures the Streamlit page and visual theme
- renders account registration, login, logout, and saved-travel controls
- creates all sidebar controls
- calls the data, recommendation, weather, currency, image, and itinerary code
- displays the best match, alternatives, map, metrics, and developer details
- sends account-specific save and load operations to the database layer
- caches data and API results to avoid unnecessary repeated work
- catches weather errors so an unavailable API does not crash the app

This is the file started by:

```bash
streamlit run app.py
```

### `src/data_loader.py`

Loads destination data.

It prefers `data/processed/destinations.csv`. If that file does not exist, it
uses the smaller sample dataset so the application can still start.

### `src/database.py`

Provides authentication and persistent user storage.

It:

- creates the required database tables automatically
- stores salted PBKDF2 password hashes rather than plain-text passwords
- supports account registration and login
- saves and reloads each user's preferences
- stores favorite destinations, recommendation history, and itineraries
- associates saved records with the authenticated user's ID
- uses parameterized SQL queries
- connects to Neon PostgreSQL through `DATABASE_URL`
- falls back to a local SQLite database during development

### `src/preprocessing.py`

Cleans and validates destination data.

It:

- creates any missing required columns
- removes unusable rows without a city, country, or coordinates
- converts model features to numbers
- fills missing costs, temperatures, and ratings
- limits ratings to the 0-10 scale
- removes duplicate city-country combinations

### `src/dataset_builder.py`

Converts the original Kaggle data into the format required by VoyageMatch.

It:

- identifies columns even when source column names vary
- converts ratings from a 0-5 scale to a 0-10 scale when necessary
- parses monthly climate JSON into 12 temperature columns
- converts budget labels such as `budget`, `mid-range`, and `luxury` into
  estimated daily costs
- adds default values for missing features
- merges custom destination CSV files with the Kaggle data
- saves the final processed dataset

### `scripts/build_dataset.py`

Runs the dataset-building process.

It uses the existing Kaggle CSV in `data/raw/`. If no raw CSV is available, it
uses `kagglehub` to download the Kaggle dataset first.

### `src/recommender.py`

Contains the machine-learning and recommendation logic.

It defines the user's preference profile and supports three methods:

#### Weighted scoring

Calculates separate scores for:

- budget fit: 30%
- temperature fit: 15%
- interests: 35%
- safety: 10%
- popularity: 10%

The scores are combined into one match score. This method is easy to explain
because every category has a visible importance.

#### Cosine similarity

Represents the user and every destination as vectors of numbers. It measures
how similar the direction of each destination vector is to the user's vector.

Before comparison, `MinMaxScaler` places different features on comparable
scales. This prevents a feature such as cost from dominating ratings that only
range from 0 to 10.

#### K-nearest neighbors

Also scales the features, then finds destinations with the smallest Euclidean
distance from the user profile. The closest destinations are the user's
"nearest neighbors."

#### K-means clustering

K-means is not used to produce the final ranking. It groups destinations into
four broad travel-style clusters based on their interest ratings. The cluster
number is shown in the developer details as an additional ML component.

### `src/seasonality.py`

Makes recommendations sensitive to the selected travel dates.

It:

- finds all months included in the trip
- uses the matching monthly temperature columns instead of only an annual
  average
- determines the local season using latitude and hemisphere
- applies a simple seasonal cost multiplier

The cost assumptions are:

- local summer: `1.18x`
- local winter: `0.92x`
- spring or autumn: `1.05x`
- tropical destinations: `1.08x` from December through March, otherwise `1.0x`

These are transparent project estimates, not live hotel prices.

### `src/currency.py`

Handles display currencies.

It:

- fetches current reference exchange rates
- converts the user's budget into US dollars for consistent recommendation
  calculations
- converts results back into the selected display currency
- formats currency symbols and decimals
- uses built-in approximate rates if the external service is unavailable

The underlying dataset and model always use US dollars.

### `src/weather.py`

Requests a live seven-day forecast from Open-Meteo.

It calculates:

- average forecast high
- average forecast low
- total forecast precipitation

The app only requests this forecast when the trip begins within seven days.
For later trips, it displays the historical monthly climate from the dataset.

### `src/destination_image.py`

Finds a preview image for the recommended destination.

It first tries Unsplash when an Unsplash access key is configured. Without a
key, or if Unsplash fails, it searches Wikipedia and Wikimedia Commons. Image
source and photographer attribution are displayed in the interface.

### `src/itinerary_generator.py`

Creates the day-by-day trip plan.

It first checks for an OpenAI API key. When one is present, it sends the
destination, dates, budget, climate, and user preferences to the OpenAI
Responses API.

When no key is available or the AI request fails, it uses a built-in
rule-based generator. That generator chooses themes from the user's highest
interests and produces different morning, afternoon, and evening ideas.

This means the app still works without generative AI or internet access.

### `src/rag.py`

Implements retrieval-augmented generation.

It:

- turns every destination into a structured text document
- creates OpenAI vector embeddings in batches
- falls back to local 1,536-dimensional text vectors when API quota is unavailable
- stores embeddings in Neon PostgreSQL with pgvector
- embeds user questions and retrieves the closest destination documents
- sends only retrieved context to the transformer
- asks the model to cite sources and avoid unsupported claims
- uses local TF-IDF retrieval and a grounded template without OpenAI

### `scripts/index_embeddings.py`

Builds or refreshes the destination vector index.

### `src/__init__.py`

Marks `src` as a Python package so its modules can be imported cleanly.

### Data files

#### `data/raw/Worldwide Travel Cities Dataset (Ratings and Climate).csv`

The original external Kaggle dataset.

#### `data/raw/sample_destinations.csv`

A small backup dataset used if the processed dataset is unavailable.

#### `data/custom/israel_destinations.csv`

Project-specific destinations added separately from the Kaggle source.

#### `data/processed/destinations.csv`

The final cleaned and merged dataset used by the app. It currently contains
567 destinations.

### Configuration files

#### `requirements.txt`

Lists the Python packages needed to run the project.

#### `.env.example`

Shows example variables for Neon, OpenAI, and Unsplash without containing real
credentials.

#### `.streamlit/config.toml`

Defines Streamlit theme colors.

#### `.gitignore`

Prevents local virtual environments, secret files, caches, and logs from being
committed to Git.

#### `README.md`

Provides the project overview and setup instructions.

## 5. Technologies and Python Libraries

### Streamlit

Builds the web interface entirely in Python. It provides sliders, date inputs,
tabs, metrics, maps, data tables, caching, and session state.

### pandas

Loads CSV files and handles table operations such as cleaning, filtering,
merging, adding columns, and sorting recommendations.

### NumPy

Provides numerical arrays and calculations used in recommendation scoring and
missing-value handling.

### scikit-learn

Provides:

- `MinMaxScaler` for feature normalization
- `cosine_similarity` for cosine recommendations
- `NearestNeighbors` for K-nearest neighbors
- `KMeans` for travel-style clustering

### Requests

Sends HTTP requests to currency, weather, Wikipedia, and Unsplash APIs.

### python-dotenv

Loads optional API keys from a local `.env` file.

### kagglehub

Downloads the Kaggle dataset when a local raw copy is not already available.

### OpenAI Python library

Calls the OpenAI Responses API for optional AI-generated itineraries.

### psycopg

Connects the Python application to Neon PostgreSQL.

### SQLite and Neon PostgreSQL

SQLite provides a zero-setup local development database. Neon provides hosted
PostgreSQL for deployed accounts and persistent multi-user data.

## 6. Database Design

The database is relational: a user is stored once, and saved travel data refers
back to that user with a foreign key.

| Table | Important fields | Purpose |
|---|---|---|
| `users` | ID, email, display name, password hash | Account identity and login |
| `user_preferences` | User ID, preferences JSON, update time | One current preference profile per user |
| `favorites` | User ID, city, country, destination JSON | Saved destinations |
| `recommendation_history` | User ID, preferences JSON, results JSON | Previous saved searches |
| `saved_itineraries` | User ID, destination, dates, itinerary | Generated plans |
| `destination_embeddings` | Destination, text, metadata, vector | Semantic RAG index |

The relationships are one-to-many: one user can have many favorites, saved
searches, and itineraries. Preferences are one-to-one because each user has one
latest saved preference profile.

JSON is used inside some database columns because a complete preference profile
or recommendation result contains several related values. PostgreSQL remains
the database system; JSON is only the format used for those structured values.

### Database security

- Passwords are never stored as readable text.
- Every password receives a random salt.
- Passwords are hashed with PBKDF2-HMAC-SHA256 and 600,000 iterations.
- Emails are normalized to lowercase and must be unique.
- SQL parameters reduce SQL-injection risk.
- `DATABASE_URL` and API keys are stored in `.env` or deployment secrets.
- The `.env` file and local SQLite files are excluded from Git.

For a larger production system, managed authentication, email verification,
password resets, rate limiting, and database migrations would be sensible next
steps.

## 7. External Data and Services

### Kaggle

- Dataset: Worldwide Travel Cities Ratings and Climate
- Purpose: main destination records, ratings, coordinates, climate, budget
  categories, and descriptions
- Used during dataset preparation, not queried every time the app runs

### Open-Meteo

- Purpose: live seven-day weather forecast
- Data used: minimum temperature, maximum temperature, and precipitation
- No API key is required
- Historical monthly climate in the Kaggle data remains the main source for
  future travel dates

### Frankfurter

- Purpose: daily reference exchange rates
- Base currency: USD
- The app includes offline fallback rates

### Wikipedia and Wikimedia Commons

- Purpose: destination preview images and attribution
- Used as the standard image source when Unsplash is not configured

### Unsplash

- Purpose: optional higher-quality destination photographs
- Requires `UNSPLASH_ACCESS_KEY`
- Photographer and Unsplash attribution are displayed

### OpenAI

- Purpose: optional personalized itinerary generation
- Requires `OPENAI_API_KEY`
- The model can be selected with `OPENAI_MODEL`
- The app defaults to a local rule-based itinerary when OpenAI is unavailable

### Neon

- Purpose: hosted PostgreSQL storage for account data
- Stores users, preferences, favorites, history, and itineraries
- Requires `DATABASE_URL`
- Local development falls back to SQLite

### pgvector

- PostgreSQL extension used for vector columns and similarity search
- Stores 1,536-dimensional destination embeddings
- Uses cosine distance to retrieve relevant destinations

### OpenAI Embeddings

- Converts destination documents and questions into semantic vectors
- Default model: `text-embedding-3-small`
- Enables meaning-based retrieval instead of only keyword matching

## 8. Model Features

Every destination and user profile is compared with these 11 features:

1. daily cost in USD
2. average temperature for the selected travel months
3. beach rating
4. culture rating
5. nature rating
6. nightlife rating
7. food rating
8. adventure rating
9. relaxation rating
10. popularity rating
11. safety rating

The interest, popularity, and safety values use a 0-10 scale.

## 9. What Is Machine Learning and What Is Rule-Based?

Machine-learning components:

- cosine similarity
- K-nearest neighbors
- K-means clustering
- feature scaling
- vector embeddings and semantic similarity retrieval

Rule-based or formula-based components:

- weighted scoring
- season names and cost multipliers
- data cleaning defaults
- offline itinerary generation

Generative-AI component:

- optional OpenAI itinerary generation
- retrieval-augmented travel-question answering

It is useful to explain this distinction honestly. The project combines ML,
traditional scoring, APIs, data engineering, and optional generative AI.

## 10. Reliability and Fallbacks

The app is designed not to depend completely on external services:

- no Neon configuration: use the local SQLite database
- no processed dataset: use the sample dataset
- no exchange-rate connection: use built-in rates
- no live weather: show a warning or use historical climate
- no Unsplash key: use Wikipedia/Wikimedia
- no image result: continue without an image
- no OpenAI key or failed AI request: use the rule-based itinerary
- no embeddings or OpenAI key: use local TF-IDF retrieval
- no OpenAI embedding quota: store local text vectors in pgvector

Streamlit also caches data and API results, which improves speed and reduces
repeated requests.

## 11. Limitations to Mention

- Destination ratings and costs are estimates, not real-time prices.
- Seasonal multipliers are project assumptions rather than booking data.
- The estimated trip budget excludes flights.
- The live forecast only covers seven days.
- Match percentages express model similarity, not a guarantee that a traveler
  will enjoy the destination.
- Some destination values are defaults when the original data did not provide
  a feature.
- The clusters are numbered groups and do not automatically receive human
  labels such as "beach lovers."
- The offline itinerary suggests types of activities, not verified businesses
  or reservations.
- The custom login system does not yet include email verification, password
  reset emails, or multi-factor authentication.
- Recommendation history is saved only when the signed-in user chooses to save
  the search.

## 12. Short Presentation Script

"VoyageMatch is a travel recommendation application built with Python and
Streamlit. Users can create an account, then enter a destination area, travel
dates, budget, preferred temperature, and ratings for different travel
interests.

The app loads a cleaned dataset based mainly on a Kaggle travel-cities dataset,
with 560 source destinations and seven custom destinations, for 567 in total.
It adjusts temperatures and estimated costs for the user's travel months, then
compares the user profile with the available destinations.

The user can choose weighted scoring, cosine similarity, or K-nearest
neighbors. The app ranks the destinations and displays the five best matches.
It also uses K-means to group destinations into travel-style clusters.

Several external services add live information. Frankfurter provides exchange
rates, Open-Meteo provides short-term weather, and Wikipedia or Unsplash
provides destination images. OpenAI can create the itinerary, but the app also
has an offline itinerary generator, so it does not depend on an API key.

For persistent user data, the application uses Neon PostgreSQL. It stores
accounts, preferences, favorites, recommendation history, and itineraries.
During local development it can use SQLite instead. Passwords are salted and
hashed before storage.

The AI assistant uses retrieval-augmented generation. Destination descriptions
and travel features are converted into OpenAI embeddings and stored in Neon
with pgvector. A question is embedded, the closest destinations are retrieved,
and only those records are sent to the transformer. The interface displays the
retrieved sources.

The main strength of the project is that it combines data preparation,
machine-learning methods, API integration, generative AI, a relational
database, vector search, RAG, authentication, and a complete user interface
while keeping fallbacks for unavailable services."

## 13. Likely Questions and Answers

### Why normalize the data?

Cost may be around 100 or 200 while ratings range from 0 to 10. Scaling puts
the features on comparable ranges so cost does not dominate similarity.

### Why offer three recommendation methods?

They demonstrate different approaches. Weighted scoring is transparent,
cosine similarity compares profile direction, and KNN finds the closest
profiles by distance.

### Is the match score an accuracy percentage?

No. It is a similarity or fit score produced by the selected recommendation
method.

### Is K-means used for the recommendations?

Not directly. It groups destinations by interest patterns and demonstrates
unsupervised learning. The selected recommendation algorithm creates the final
ranking.

### Why use monthly climate instead of the live forecast?

A live weather service cannot reliably forecast months in advance. Monthly
climate data is more appropriate for future trip planning, while the live API
is useful only near the departure date.

### Does the app require OpenAI?

No. OpenAI improves itinerary personalization, but the offline generator keeps
the feature available without a key.

### Why convert every budget to USD?

The dataset uses USD. Keeping one internal currency makes comparisons
consistent, while exchange rates let the user view and enter money in a
familiar currency.

### Is this a supervised model?

No. There is no labeled answer saying which destination is correct for each
user. The project uses similarity methods and unsupervised clustering.

### Why use Neon?

Neon provides hosted PostgreSQL without requiring a database server on the
user's computer. It makes account data persistent across devices and is more
suitable than a local CSV or SQLite file for a deployed multi-user app.

### Why keep SQLite?

SQLite makes development and demonstrations easy because it needs no account,
server, or network connection. The same database service switches to Neon when
`DATABASE_URL` is configured.

### What information is stored?

The app stores the user's email, display name, password hash, saved preference
profile, favorite destinations, saved recommendation searches, and saved
itineraries. It does not store a readable password.

### Why are some database fields JSON?

Preferences and recommendation results contain several values that are usually
read and written together. JSON preserves that structure while PostgreSQL
still manages users, ownership, uniqueness, and relationships.

### Is the login system production ready?

It demonstrates secure password hashing and user-specific storage, but a
large public product should also add email verification, password reset,
rate limiting, multi-factor authentication, and managed sessions.

### What makes this RAG?

The system retrieves destination records before generation. The transformer
receives the top semantically related documents from the vector database and
is instructed to answer only from that context.

### What is an embedding?

An embedding is a numerical vector representing the meaning of text. Texts
with similar meanings have vectors that are close together, even if they use
different words.

### Why pgvector instead of FAISS?

VoyageMatch already uses Neon PostgreSQL. pgvector keeps persistent embeddings
beside the application's other data and avoids adding a separate FAISS index.

### What would you improve next?

- use live accommodation and flight prices
- collect user feedback and evaluate recommendation quality
- label and explain the K-means clusters
- add transport time, visa, language, and accessibility preferences
- add automated tests and deployment monitoring
- use more detailed seasonal and climate data
- add email verification and password resets
- introduce database migrations and account deletion controls

## Features

### MVP features

- Account registration and login
- User preference form
- Destination filtering
- Ranked destination recommendations
- Match scores
- Budget and climate matching
- Destination details
- Basic itinerary generation
- Persistent storage of user preferences

### Additional features

- Neon PostgreSQL with SQLite fallback
- Favorites
- Recommendation history
- Saved itineraries
- Multiple currencies
- Three recommendation methods
- K-means clustering
- Date-aware climate and seasonal prices
- Live weather
- Unsplash/Wikimedia images
- Optional OpenAI itineraries
- RAG travel assistant
- OpenAI vector embeddings
- Neon pgvector semantic search
- Retrieved-source citations
- Interactive map
- Offline service fallbacks

