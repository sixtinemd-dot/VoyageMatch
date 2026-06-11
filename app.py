#  Imports 

from datetime import date, datetime, timedelta
import json
import os
import re

import pandas as pd
import streamlit as st

from src.database import Database, user_from_dict, user_to_dict
from src.data_loader import load_destinations
from src.destination_image import get_destination_image
from src.currency import (
    CURRENCY_LABELS,
    fetch_exchange_rates,
    format_money,
    from_usd,
    to_usd,
)
from src.itinerary_generator import generate_itinerary
from src.preprocessing import clean_destinations
from src.rag import (
    answer_travel_question,
    embedding_backend_label,
    embedding_count,
    index_destinations,
    openai_is_configured,
)
from src.recommender import (
    INTEREST_COLUMNS,
    UserPreferences,
    add_travel_style_clusters,
    recommend_destinations,
)
from src.seasonality import apply_travel_dates, format_travel_period
from src.weather import get_forecast_summary


#  Streamlit page configuration and visual design 

st.set_page_config(
    page_title="VoyageMatch",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f8ff;
    }
    .block-container {
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #082c5c;
        letter-spacing: 0;
        font-family: "Aptos Display", "Segoe UI", Arial, sans-serif;
    }
    p, label, div, span, input, button {
        font-family: "Aptos", "Segoe UI", Arial, sans-serif;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #cfe0ff;
        border-top: 3px solid #0066ff;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        box-shadow: 0 3px 14px rgba(0, 102, 255, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #486581;
    }
    div[data-testid="stSidebar"] {
        background: #eaf2ff;
        border-right: 1px solid #c9dcff;
    }
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        color: #082c5c;
    }
    .destination-kicker {
        color: #0066ff;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
    }
    .destination-name {
        color: #082c5c;
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }
    .match-reason {
        color: #334e68;
        font-size: 1rem;
        line-height: 1.6;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    button[data-baseweb="tab"] {
        color: #486581;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0066ff;
        border-bottom-color: #0066ff;
    }
    div[data-testid="stAlert"] {
        border-left-color: #0066ff;
        background: #eef5ff;
    }
    div.stButton > button[kind="primary"] {
        background: #0066ff;
        border-color: #0066ff;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #0052cc;
        border-color: #0052cc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


#  Display labels and recommendation options 

REGION_LABELS = {
    "africa": "Africa",
    "asia": "Asia",
    "europe": "Europe",
    "middle_east": "Middle East",
    "north_america": "North America",
    "south_america": "South America",
    "oceania": "Oceania",
}

METHOD_LABELS = {
    "Balanced match": "Weighted scoring",
    "Similar travel profile": "Cosine similarity",
    "Closest destination matches": "K-nearest neighbors",
}

INTEREST_LABELS = {
    "beach": "Beaches",
    "culture": "Culture and history",
    "nature": "Nature",
    "nightlife": "Nightlife",
    "food": "Food",
    "adventure": "Adventure",
    "relaxation": "Relaxation",
}


#  Cached data and external services 

@st.cache_resource(show_spinner=False)
def get_database():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        try:
            database_url = st.secrets.get("DATABASE_URL")
        except FileNotFoundError:
            database_url = None
    return Database(database_url)


@st.cache_data
def get_data():
    return add_travel_style_clusters(clean_destinations(load_destinations()))


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(latitude: float, longitude: float):
    return get_forecast_summary(latitude, longitude)


@st.cache_data(ttl=21600, show_spinner=False)
def get_exchange_rates():
    return fetch_exchange_rates()


@st.cache_data(ttl=3600, show_spinner=False)
def get_preview_image(city: str, country: str):
    return get_destination_image(city, country)


#  Formatting and display helper functions 

def region_name(region: str) -> str:
    return REGION_LABELS.get(region, region.replace("_", " ").title())


def strongest_interests(preferences: UserPreferences, count: int = 3) -> list[str]:
    ranked = sorted(
        INTEREST_COLUMNS,
        key=lambda column: getattr(preferences, column),
        reverse=True,
    )
    return [INTEREST_LABELS[column] for column in ranked[:count]]


def render_trip_plan(markdown_text: str) -> None:
    main_text, separator, practical_notes = markdown_text.partition(
        "### Practical notes"
    )
    sections = re.split(r"(?=^## Day \d+)", main_text, flags=re.MULTILINE)

    introduction = sections[0].strip()
    if introduction:
        st.markdown(introduction)

    for section in sections[1:]:
        title, _, body = section.partition("\n")
        with st.container(border=True):
            st.markdown(f"### {title.removeprefix('## ').strip()}")
            st.markdown(body.strip())

    if separator:
        st.markdown("### Practical notes")
        st.markdown(practical_notes.strip())


def serialize_record(record) -> dict:
    serialized = {}
    for key, value in record.items():
        if pd.isna(value):
            serialized[key] = None
        elif hasattr(value, "item"):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized


#  Saved preferences and account helper functions 

def preference_value(name: str, default):
    saved = st.session_state.get("saved_preferences") or {}
    return saved.get(name, default)


def set_authenticated_user(user, database: Database) -> None:
    for key in list(st.session_state):
        if key.startswith("trip_"):
            del st.session_state[key]
    st.session_state["authenticated_user"] = user_to_dict(user)
    st.session_state["saved_preferences"] = database.load_preferences(user.id) or {}


def render_account_panel(database: Database) -> None:
    user_data = st.session_state.get("authenticated_user")
    if user_data:
        user = user_from_dict(user_data)
        st.success(f"Signed in as {user.display_name}")
        st.caption(user.email)
        if st.button("Log out", width="stretch"):
            for key in list(st.session_state):
                if key.startswith("trip_") or key in {
                    "authenticated_user",
                    "saved_preferences",
                }:
                    del st.session_state[key]
            st.rerun()
        return

    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )
            submitted = st.form_submit_button("Log in", width="stretch")
        if submitted:
            user = database.authenticate(email, password)
            if user:
                set_authenticated_user(user, database)
                st.rerun()
            st.error("Incorrect email or password.")

    with register_tab:
        with st.form("register_form"):
            display_name = st.text_input("Name", key="register_name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input(
                "Password (8+ characters)",
                type="password",
                key="register_password",
            )
            submitted = st.form_submit_button(
                "Create account",
                width="stretch",
            )
        if submitted:
            try:
                user = database.register_user(email, password, display_name)
            except ValueError as exc:
                st.error(str(exc))
            else:
                set_authenticated_user(user, database)
                st.rerun()


#  Load application data and services 

destinations = get_data()
exchange_rates = get_exchange_rates()
try:
    database = get_database()
except Exception as exc:
    st.error(f"The account database could not be initialized: {exc}")
    st.stop()

current_user = (
    user_from_dict(st.session_state["authenticated_user"])
    if st.session_state.get("authenticated_user")
    else None
)
dataset_sources = (
    set(destinations["source_dataset"].dropna().astype(str))
    if "source_dataset" in destinations.columns
    else {"sample_destinations"}
)


#  Sidebar: account and travel preferences 

with st.sidebar:
    st.header("👤 Your account")
    render_account_panel(database)
    st.caption(f"Storage: {database.backend_name}")
    st.divider()

    st.header("🧭 Plan your vacation")
    st.caption("Tell VoyageMatch what your ideal trip feels like.")

    region_values = sorted(destinations["region"].dropna().unique().tolist())
    region_options = ["anywhere"] + region_values
    saved_region = preference_value("region", "anywhere")
    if saved_region not in region_options:
        saved_region = "anywhere"
    selected_region = st.selectbox(
        "🌍 Where would you like to go?",
        region_options,
        index=region_options.index(saved_region),
        format_func=lambda value: "Anywhere in the world"
        if value == "anywhere"
        else region_name(value),
        key="trip_region",
    )

    country_source = destinations
    if selected_region != "anywhere":
        country_source = destinations[
            destinations["region"] == selected_region
        ]
    country_options = ["anywhere"] + sorted(
        country_source["country"].dropna().unique().tolist()
    )
    saved_country = preference_value("country", "anywhere")
    if saved_country not in country_options:
        saved_country = "anywhere"
    selected_country = st.selectbox(
        "📍 Preferred country",
        country_options,
        index=country_options.index(saved_country),
        format_func=lambda value: "Any country"
        if value == "anywhere"
        else value,
        key="trip_country",
    )

    default_start = date.today() + timedelta(days=60)
    default_end = default_start + timedelta(days=6)
    try:
        default_start = date.fromisoformat(
            preference_value("start_date", default_start.isoformat())
        )
        default_end = date.fromisoformat(
            preference_value("end_date", default_end.isoformat())
        )
    except (TypeError, ValueError):
        pass
    if default_start < date.today():
        default_start = date.today() + timedelta(days=60)
    if default_start > date.today() + timedelta(days=730):
        default_start = date.today() + timedelta(days=60)
    if default_end < default_start:
        default_end = default_start + timedelta(days=6)
    if default_end > date.today() + timedelta(days=730):
        default_end = min(
            default_start + timedelta(days=6),
            date.today() + timedelta(days=730),
        )
    selected_dates = st.date_input(
        "📅 Travel dates",
        value=(default_start, default_end),
        min_value=date.today(),
        max_value=date.today() + timedelta(days=730),
        help="Your dates affect destination weather, season, trip length, and estimated cost.",
        key="trip_dates",
    )

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Select both a departure and return date.")
        st.stop()

    start_date, end_date = selected_dates
    if end_date < start_date:
        st.error("The return date must be after the departure date.")
        st.stop()

    days = (end_date - start_date).days + 1
    travel_period = format_travel_period(start_date, end_date)
    st.caption(f"{days} days · {travel_period}")

    saved_currency = preference_value("currency", "USD")
    if saved_currency not in CURRENCY_LABELS:
        saved_currency = "USD"
    selected_currency = st.selectbox(
        "💱 Currency",
        list(CURRENCY_LABELS),
        index=list(CURRENCY_LABELS).index(saved_currency),
        format_func=lambda code: CURRENCY_LABELS[code],
        key="trip_currency",
    )
    currency_rate = exchange_rates.rates[selected_currency]
    minimum_budget = max(1, round(40 * currency_rate))
    maximum_budget = max(minimum_budget + 1, round(300 * currency_rate))
    default_budget = int(
        preference_value("budget_display", round(120 * currency_rate))
    )
    default_budget = min(maximum_budget, max(minimum_budget, default_budget))
    budget_step = max(1, round(5 * currency_rate))
    budget_display = st.slider(
        "💳 Daily budget per traveler",
        minimum_budget,
        maximum_budget,
        default_budget,
        budget_step,
        key=f"trip_budget_{selected_currency}",
    )
    budget = to_usd(budget_display, selected_currency, exchange_rates)
    st.caption(
        f"Approximately {format_money(budget, 'USD')} for recommendation matching."
    )
    preferred_temp = st.slider(
        "☀️ Ideal average temperature",
        0,
        35,
        int(preference_value("preferred_temp", 24)),
        format="%d C",
        key="trip_preferred_temp",
    )

    st.subheader("✨ What matters to you?")
    beach = st.slider(
        "🏖️ Beaches", 0, 10, int(preference_value("beach", 6)), key="trip_beach"
    )
    culture = st.slider(
        "🏛️ Culture and history",
        0,
        10,
        int(preference_value("culture", 8)),
        key="trip_culture",
    )
    nature = st.slider(
        "🌿 Nature", 0, 10, int(preference_value("nature", 7)), key="trip_nature"
    )
    nightlife = st.slider(
        "🌙 Nightlife",
        0,
        10,
        int(preference_value("nightlife", 5)),
        key="trip_nightlife",
    )
    food = st.slider(
        "🍽️ Food", 0, 10, int(preference_value("food", 8)), key="trip_food"
    )
    adventure = st.slider(
        "🥾 Adventure",
        0,
        10,
        int(preference_value("adventure", 6)),
        key="trip_adventure",
    )
    relaxation = st.slider(
        "🌊 Relaxation",
        0,
        10,
        int(preference_value("relaxation", 7)),
        key="trip_relaxation",
    )

    with st.expander("Advanced recommendation settings"):
        saved_method = preference_value("method_label", "Balanced match")
        if saved_method not in METHOD_LABELS:
            saved_method = "Balanced match"
        method_label = st.selectbox(
            "Matching approach",
            list(METHOD_LABELS),
            index=list(METHOD_LABELS).index(saved_method),
            help="Choose how destination similarity is calculated.",
            key="trip_method",
        )
        safety = st.slider(
            "Safety priority",
            0,
            10,
            int(preference_value("safety", 8)),
            key="trip_safety",
        )
        popularity = st.slider(
            "Well-known destinations",
            0,
            10,
            int(preference_value("popularity", 7)),
            help="Lower values favor quieter, less prominent destinations.",
            key="trip_popularity",
        )


#  Prepare and run the recommendation system 

dated_destinations = apply_travel_dates(destinations, start_date, end_date)
filtered_destinations = dated_destinations
if selected_region != "anywhere":
    filtered_destinations = dated_destinations[
        dated_destinations["region"] == selected_region
    ]
if selected_country != "anywhere":
    filtered_destinations = filtered_destinations[
        filtered_destinations["country"] == selected_country
    ]

preferences = UserPreferences(
    budget_per_day_usd=budget,
    preferred_temp_c=preferred_temp,
    beach=beach,
    culture=culture,
    nature=nature,
    nightlife=nightlife,
    food=food,
    adventure=adventure,
    relaxation=relaxation,
    safety=safety,
    popularity=popularity,
    method=METHOD_LABELS[method_label],
)

recommendations = recommend_destinations(filtered_destinations, preferences, top_n=5)
top_destination = recommendations.iloc[0]
top_interests = strongest_interests(preferences)
estimated_trip_cost = top_destination["cost_per_day_usd"] * days
display_daily_cost = from_usd(
    top_destination["cost_per_day_usd"],
    selected_currency,
    exchange_rates,
)
display_trip_cost = from_usd(
    estimated_trip_cost,
    selected_currency,
    exchange_rates,
)
region_label = (
    selected_country
    if selected_country != "anywhere"
    else (
        "Worldwide"
        if selected_region == "anywhere"
        else region_name(selected_region)
    )
)


#  Main result: best destination match 

st.title("VoyageMatch")
st.caption(
    f"📅 {travel_period}  ·  🌍 {region_label}  ·  "
    f"💳 {format_money(budget_display, selected_currency)} daily target"
)

st.markdown('<div class="destination-kicker">Your best match</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="destination-name">{top_destination["city"]}, {top_destination["country"]}</div>',
    unsafe_allow_html=True,
)

description = top_destination.get("description", "")
if isinstance(description, str) and description:
    st.markdown(f'<div class="match-reason">{description}</div>', unsafe_allow_html=True)

preview_image = get_preview_image(
    str(top_destination["city"]),
    str(top_destination["country"]),
)

match_metric, cost_metric, climate_metric, season_metric = st.columns(4)
match_metric.metric("🎯 Match", f"{top_destination['match_score'] * 100:.0f}%")
cost_metric.metric(
    "💳 Estimated trip budget",
    format_money(display_trip_cost, selected_currency),
)
climate_metric.metric(
    "🌡️ Average for your dates",
    f"{top_destination['avg_temp_c']:.1f} C",
)
season_metric.metric("🍂 Local season", top_destination["travel_season"])
st.caption(
    "The budget uses the dataset's destination cost level plus a transparent seasonal adjustment. "
    "It covers typical spending and does not include flights."
)

st.info(
    f"Recommended for {', '.join(top_interests)}. "
    f"Estimated spending for this period is "
    f"{format_money(display_daily_cost, selected_currency)} "
    f"per traveler per day."
)


#  Save preferences, history, and favorites 

saved_preference_payload = {
    "region": selected_region,
    "country": selected_country,
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat(),
    "currency": selected_currency,
    "budget_display": budget_display,
    "budget_per_day_usd": round(budget, 2),
    "preferred_temp": preferred_temp,
    "beach": beach,
    "culture": culture,
    "nature": nature,
    "nightlife": nightlife,
    "food": food,
    "adventure": adventure,
    "relaxation": relaxation,
    "safety": safety,
    "popularity": popularity,
    "method_label": method_label,
}
recommendation_records = [
    serialize_record(row)
    for row in recommendations[
        ["city", "country", "region", "match_score", "cost_per_day_usd"]
    ].to_dict(orient="records")
]

save_search_column, favorite_column = st.columns(2)
with save_search_column:
    if current_user:
        if st.button("💾 Save preferences and results", width="stretch"):
            database.save_preferences(current_user.id, saved_preference_payload)
            database.save_recommendation_history(
                current_user.id,
                saved_preference_payload,
                recommendation_records,
            )
            st.session_state["saved_preferences"] = saved_preference_payload
            st.success("Your preferences and these recommendations were saved.")
    else:
        st.caption("Log in to save your preferences and recommendation history.")

with favorite_column:
    if current_user:
        top_city = str(top_destination["city"])
        top_country = str(top_destination["country"])
        top_is_favorite = database.is_favorite(
            current_user.id,
            top_city,
            top_country,
        )
        favorite_label = (
            "★ Remove from favorites" if top_is_favorite else "☆ Add to favorites"
        )
        if st.button(favorite_label, width="stretch"):
            if top_is_favorite:
                database.remove_favorite(current_user.id, top_city, top_country)
                st.success("Destination removed from favorites.")
            else:
                database.add_favorite(
                    current_user.id,
                    serialize_record(top_destination),
                )
                st.success("Destination added to favorites.")
            st.rerun()


#  Main application tabs 

overview_tab, alternatives_tab, plan_tab, assistant_tab, saved_tab, developer_tab = st.tabs(
    [
        "🧭 Your match",
        "🌍 Other destinations",
        "🗓️ Trip plan",
        "💬 AI travel assistant",
        "👤 My saved travel",
        "⚙️ Developer details",
    ]
)


#  Tab 1: destination overview and weather 

with overview_tab:
    details_column, weather_column = st.columns([1.35, 1])

    with details_column:
        st.subheader("🗺️ Destination overview")
        if preview_image:
            st.image(
                preview_image.url,
                caption=preview_image.page_title,
                width="stretch",
            )
            if preview_image.photographer_name and preview_image.photographer_url:
                st.caption(
                    f"Photo by [{preview_image.photographer_name}]"
                    f"({preview_image.photographer_url}) on "
                    f"[Unsplash]({preview_image.page_url})"
                )
            else:
                st.caption(
                    f"[{preview_image.source} image and attribution]"
                    f"({preview_image.page_url})"
                )
        st.write(
            f"VoyageMatch placed **{top_destination['city']}** first from "
            f"**{len(filtered_destinations):,} destinations** in your selected search area."
        )
        st.map(
            recommendations.rename(columns={"latitude": "lat", "longitude": "lon"}),
            latitude="lat",
            longitude="lon",
        )

    with weather_column:
        st.subheader("☀️ Weather for your dates")
        st.metric(
            "Historical monthly average",
            f"{top_destination['avg_temp_c']:.1f} C",
        )
        st.caption(
            f"Based on Kaggle monthly climate data for {travel_period}. "
            f"Annual average: {top_destination['annual_avg_temp_c']:.1f} C."
        )

        if start_date <= date.today() + timedelta(days=7):
            try:
                weather = get_weather(
                    float(top_destination["latitude"]),
                    float(top_destination["longitude"]),
                )
                st.write("**Live 7-day forecast**")
                high_column, low_column = st.columns(2)
                high_column.metric("High", f"{weather['avg_high_c']} C")
                low_column.metric("Low", f"{weather['avg_low_c']} C")
                st.metric(
                    "Precipitation",
                    f"{weather['total_precipitation_mm']} mm",
                )
                st.caption(f"Live source: {weather['source']}")
            except Exception:
                st.warning("The live forecast is temporarily unavailable.")
        else:
            st.info(
                "A live forecast is not available this far ahead. "
                "VoyageMatch is using historical monthly climate instead."
            )


#  Tab 2: alternative destination recommendations 

with alternatives_tab:
    st.subheader("🌍 Your next best matches")
    friendly_recommendations = recommendations[
        [
            "city",
            "country",
            "region",
            "cost_per_day_usd",
            "avg_temp_c",
            "match_score",
            "travel_season",
        ]
    ].copy()
    friendly_recommendations["cost_per_day_usd"] = friendly_recommendations[
        "cost_per_day_usd"
    ].apply(
        lambda amount: format_money(
            from_usd(amount, selected_currency, exchange_rates),
            selected_currency,
        )
    )
    friendly_recommendations["region"] = friendly_recommendations["region"].map(
        region_name
    )
    friendly_recommendations["match_score"] = (
        friendly_recommendations["match_score"] * 100
    ).round(0)
    friendly_recommendations["avg_temp_c"] = friendly_recommendations[
        "avg_temp_c"
    ].round(1)
    friendly_recommendations = friendly_recommendations.rename(
        columns={
            "city": "City",
            "country": "Country",
            "region": "Region",
            "cost_per_day_usd": f"Daily cost ({selected_currency})",
            "avg_temp_c": "Typical temperature (C)",
            "match_score": "Match (%)",
            "travel_season": "Season",
        }
    )
    st.dataframe(
        friendly_recommendations,
        width="stretch",
        hide_index=True,
    )


#  Tab 3: personalized itinerary 

with plan_tab:
    st.subheader(f"🗓️ Build a {days}-day plan for {top_destination['city']}")
    plan_key = (
        top_destination["city"],
        top_destination["country"],
        days,
        budget,
        preferred_temp,
        tuple(getattr(preferences, interest) for interest in INTEREST_COLUMNS),
        safety,
        popularity,
        preferences.method,
        start_date,
        end_date,
        selected_currency,
        exchange_rates.date,
    )

    if st.button("Create my trip plan", type="primary", width="stretch"):
        with st.spinner("Creating your personalized trip plan..."):
            st.session_state["trip_plan"] = generate_itinerary(
                top_destination,
                preferences,
                days,
                travel_period,
                selected_currency,
                display_daily_cost,
            )
            st.session_state["trip_plan_key"] = plan_key

    if st.session_state.get("trip_plan_key") == plan_key:
        st.markdown(
            """
            <div style="
                background:#eaf2ff;
                border-left:4px solid #0066ff;
                border-radius:8px;
                padding:0.8rem 1rem;
                margin:0.5rem 0 1.2rem 0;
                color:#102a43;">
                A balanced outline with a different focus each day. Treat it as a
                flexible starting point rather than a list of fixed bookings.
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_trip_plan(st.session_state["trip_plan"])
        if current_user and st.button(
            "Save this itinerary",
            width="stretch",
            key="save_current_itinerary",
        ):
            database.save_itinerary(
                current_user.id,
                str(top_destination["city"]),
                str(top_destination["country"]),
                start_date,
                end_date,
                st.session_state["trip_plan"],
            )
            st.success("The itinerary was saved to your account.")
        elif not current_user:
            st.caption("Log in to save this itinerary.")
    else:
        st.caption("Generate a day-by-day itinerary for your current match.")


#  Tab 4: RAG travel assistant and vector search 

with assistant_tab:
    st.subheader("💬 Ask the VoyageMatch knowledge base")
    st.write(
        "Ask a natural-language travel question. VoyageMatch retrieves the most "
        "relevant destination records first, then produces an answer grounded in "
        "those records."
    )

    indexed_destinations = st.session_state.get("indexed_destinations")

    assistant_a, assistant_b = st.columns(2)
    assistant_a.metric("Destination documents", f"{len(destinations):,}")
    assistant_b.metric(
        "Stored embeddings",
        f"{indexed_destinations:,}" if indexed_destinations is not None else "Not checked",
    )

    if indexed_destinations is None:
        st.caption(
            "Vector status is checked only when requested so a sleeping Neon "
            "database does not delay application startup."
        )
        if st.button(
            "Check vector index status",
            width="stretch",
            key="check_embedding_index",
        ):
            with st.spinner("Checking the vector index..."):
                try:
                    st.session_state["indexed_destinations"] = embedding_count(database)
                except Exception as exc:
                    st.error(f"The vector store is not available: {exc}")
                else:
                    st.rerun()
    elif indexed_destinations < len(destinations):
        st.info(
            "Build or refresh the vector index to activate semantic vector "
            "retrieval. OpenAI embeddings are used when quota is available; "
            "otherwise VoyageMatch creates local text vectors."
        )
        if st.button(
            "Build destination embedding index",
            type="primary",
            width="stretch",
            key="build_embedding_index",
        ):
            with st.spinner(
                "Creating destination vectors and storing them in the vector index..."
            ):
                try:
                    indexed = index_destinations(database, destinations)
                    vector_backend = embedding_backend_label(database)
                except Exception as exc:
                    st.error(f"The embedding index could not be built: {exc}")
                else:
                    st.session_state["indexed_destinations"] = indexed
                    st.success(
                        f"Indexed {indexed} destination documents using "
                        f"{vector_backend}."
                    )
                    st.rerun()
    else:
        vector_backend = embedding_backend_label(database)
        st.success(
            f"Vector retrieval is ready with {vector_backend} and "
            f"{'Neon pgvector' if database.is_postgres else 'the local vector store'}."
        )
        if not openai_is_configured():
            st.caption(
                "Transformer answers require `OPENAI_API_KEY`; retrieval itself "
                "works with the local vector index."
            )

    with st.form("rag_question_form"):
        rag_question = st.text_area(
            "Travel question",
            placeholder=(
                "Which destinations are warm, culturally rich, and suitable "
                "for a moderate budget?"
            ),
            height=100,
        )
        ask_rag = st.form_submit_button(
            "Search destinations and answer",
            width="stretch",
        )

    if ask_rag:
        with st.spinner("Retrieving relevant destinations and preparing an answer..."):
            try:
                st.session_state["rag_answer"] = answer_travel_question(
                    database,
                    destinations,
                    rag_question,
                )
                st.session_state["rag_question"] = rag_question
            except Exception as exc:
                st.error(f"The travel assistant could not answer: {exc}")

    rag_answer = st.session_state.get("rag_answer")
    if rag_answer:
        st.markdown(rag_answer.answer)
        st.caption(
            f"Retrieval: {rag_answer.retrieval_mode} · "
            f"Generation: {rag_answer.generation_mode}"
        )
        with st.expander("Retrieved destination sources"):
            for number, source in enumerate(rag_answer.sources, start=1):
                st.markdown(
                    f"**[{number}] {source.city}, {source.country}** "
                    f"· similarity {source.similarity:.3f}"
                )
                st.caption(source.content)


#  Tab 5: saved favorites, itineraries, and searches 

with saved_tab:
    st.subheader("👤 My saved travel")
    if not current_user:
        st.info("Create an account or log in to save and revisit your travel plans.")
    else:
        st.write(f"Welcome back, **{current_user.display_name}**.")
        favorites_section, itineraries_section, history_section = st.tabs(
            ["Favorites", "Itineraries", "Recommendation history"]
        )

        with favorites_section:
            favorites = database.list_favorites(current_user.id)
            if not favorites:
                st.caption("You have not saved any favorite destinations yet.")
            for favorite in favorites:
                favorite_columns = st.columns([3, 1])
                favorite_columns[0].write(
                    f"**{favorite['city']}, {favorite['country']}**"
                )
                if favorite_columns[1].button(
                    "Remove",
                    key=f"remove_favorite_{favorite['id']}",
                    width="stretch",
                ):
                    database.remove_favorite(
                        current_user.id,
                        favorite["city"],
                        favorite["country"],
                    )
                    st.rerun()

        with itineraries_section:
            saved_itineraries = database.list_itineraries(current_user.id)
            if not saved_itineraries:
                st.caption("You have not saved any itineraries yet.")
            for itinerary in saved_itineraries:
                title = (
                    f"{itinerary['city']}, {itinerary['country']} · "
                    f"{itinerary['start_date']} to {itinerary['end_date']}"
                )
                with st.expander(title):
                    render_trip_plan(itinerary["itinerary"])
                    if st.button(
                        "Delete itinerary",
                        key=f"delete_itinerary_{itinerary['id']}",
                    ):
                        database.delete_itinerary(
                            current_user.id,
                            int(itinerary["id"]),
                        )
                        st.rerun()

        with history_section:
            history = database.list_recommendation_history(current_user.id)
            if not history:
                st.caption("No recommendation searches have been saved yet.")
            for entry in history:
                created_at = datetime.fromisoformat(entry["created_at"]).strftime(
                    "%Y-%m-%d %H:%M"
                )
                results = json.loads(entry["recommendations_json"])
                with st.expander(f"Saved search · {created_at}"):
                    history_table = pd.DataFrame(results)
                    if not history_table.empty:
                        history_table["match_score"] = (
                            history_table["match_score"] * 100
                        ).round(0)
                        history_table = history_table.rename(
                            columns={
                                "city": "City",
                                "country": "Country",
                                "region": "Region",
                                "match_score": "Match (%)",
                                "cost_per_day_usd": "Daily cost (USD)",
                            }
                        )
                        st.dataframe(
                            history_table,
                            width="stretch",
                            hide_index=True,
                        )


#  Tab 6: technical and dataset details 

with developer_tab:
    st.subheader("⚙️ Recommendation details")
    developer_a, developer_b, developer_c = st.columns(3)
    developer_a.metric("Dataset rows", f"{len(destinations):,}")
    developer_b.metric("Countries", f"{destinations['country'].nunique():,}")
    developer_c.metric(
        "Travel-style cluster",
        int(top_destination["travel_style_cluster"]),
    )

    if len(dataset_sources) > 1:
        dataset_name = "Kaggle Worldwide Travel Cities + VoyageMatch custom data"
    else:
        dataset_source = next(iter(dataset_sources))
        dataset_name = {
            "kaggle_worldwide_travel_cities": "Kaggle Worldwide Travel Cities",
            "sample_destinations": "Bundled sample destinations",
        }.get(dataset_source, dataset_source.replace("_", " ").title())
    st.write(f"**Dataset:** {dataset_name}")
    st.write(f"**Active algorithm:** {preferences.method}")
    st.write(f"**Selected travel period:** {travel_period}")
    st.write(
        f"**Display currency:** {selected_currency} using {exchange_rates.source} "
        f"({exchange_rates.date})"
    )
    if exchange_rates.is_fallback:
        st.warning(
            "The currency service was unavailable, so approximate built-in rates are being used."
        )
    st.write(
        f"**Seasonal cost multiplier for the top match:** "
        f"{top_destination['seasonal_multiplier']:.2f}x"
    )
    st.write(f"**Destinations after region filter:** {len(filtered_destinations):,}")
    st.write(f"**Account storage:** {database.backend_name}")
    vector_rows = st.session_state.get("indexed_destinations")
    st.write(
        "**Vector documents indexed:** "
        + (f"{vector_rows:,}" if vector_rows is not None else "Not checked")
    )
    st.write(
        "**RAG retrieval:** OpenAI embeddings with pgvector when indexed; "
        "local TF-IDF fallback otherwise."
    )
    st.write(
        "**Model features:** season-adjusted budget, selected-month climate, beaches, "
        "culture, nature, nightlife, "
        "food, adventure, relaxation, popularity, and safety."
    )

    with st.expander("Processed dataset preview"):
        st.dataframe(
            filtered_destinations,
            width="stretch",
            hide_index=True,
        )
