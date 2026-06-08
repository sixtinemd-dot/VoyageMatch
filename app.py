from datetime import date, timedelta
import re

import streamlit as st

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
from src.recommender import (
    INTEREST_COLUMNS,
    UserPreferences,
    add_travel_style_clusters,
    recommend_destinations,
)
from src.seasonality import apply_travel_dates, format_travel_period
from src.weather import get_forecast_summary


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


destinations = get_data()
exchange_rates = get_exchange_rates()
dataset_sources = (
    set(destinations["source_dataset"].dropna().astype(str))
    if "source_dataset" in destinations.columns
    else {"sample_destinations"}
)

with st.sidebar:
    st.header("🧭 Plan your vacation")
    st.caption("Tell VoyageMatch what your ideal trip feels like.")

    region_values = sorted(destinations["region"].dropna().unique().tolist())
    region_options = ["anywhere"] + region_values
    selected_region = st.selectbox(
        "🌍 Where would you like to go?",
        region_options,
        format_func=lambda value: "Anywhere in the world"
        if value == "anywhere"
        else region_name(value),
    )

    country_source = destinations
    if selected_region != "anywhere":
        country_source = destinations[
            destinations["region"] == selected_region
        ]
    country_options = ["anywhere"] + sorted(
        country_source["country"].dropna().unique().tolist()
    )
    selected_country = st.selectbox(
        "📍 Preferred country",
        country_options,
        format_func=lambda value: "Any country"
        if value == "anywhere"
        else value,
    )

    default_start = date.today() + timedelta(days=60)
    default_end = default_start + timedelta(days=6)
    selected_dates = st.date_input(
        "📅 Travel dates",
        value=(default_start, default_end),
        min_value=date.today(),
        max_value=date.today() + timedelta(days=730),
        help="Your dates affect destination weather, season, trip length, and estimated cost.",
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

    selected_currency = st.selectbox(
        "💱 Currency",
        list(CURRENCY_LABELS),
        format_func=lambda code: CURRENCY_LABELS[code],
    )
    currency_rate = exchange_rates.rates[selected_currency]
    minimum_budget = max(1, round(40 * currency_rate))
    maximum_budget = max(minimum_budget + 1, round(300 * currency_rate))
    default_budget = round(120 * currency_rate)
    budget_step = max(1, round(5 * currency_rate))
    budget_display = st.slider(
        "💳 Daily budget per traveler",
        minimum_budget,
        maximum_budget,
        default_budget,
        budget_step,
        key=f"budget_{selected_currency}",
    )
    budget = to_usd(budget_display, selected_currency, exchange_rates)
    st.caption(
        f"Approximately {format_money(budget, 'USD')} for recommendation matching."
    )
    preferred_temp = st.slider(
        "☀️ Ideal average temperature",
        0,
        35,
        24,
        format="%d C",
    )

    st.subheader("✨ What matters to you?")
    beach = st.slider("🏖️ Beaches", 0, 10, 6)
    culture = st.slider("🏛️ Culture and history", 0, 10, 8)
    nature = st.slider("🌿 Nature", 0, 10, 7)
    nightlife = st.slider("🌙 Nightlife", 0, 10, 5)
    food = st.slider("🍽️ Food", 0, 10, 8)
    adventure = st.slider("🥾 Adventure", 0, 10, 6)
    relaxation = st.slider("🌊 Relaxation", 0, 10, 7)

    with st.expander("Advanced recommendation settings"):
        method_label = st.selectbox(
            "Matching approach",
            list(METHOD_LABELS),
            help="Choose how destination similarity is calculated.",
        )
        safety = st.slider("Safety priority", 0, 10, 8)
        popularity = st.slider(
            "Well-known destinations",
            0,
            10,
            7,
            help="Lower values favor quieter, less prominent destinations.",
        )

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

overview_tab, alternatives_tab, plan_tab, developer_tab = st.tabs(
    ["🧭 Your match", "🌍 Other destinations", "🗓️ Trip plan", "⚙️ Developer details"]
)

with overview_tab:
    details_column, weather_column = st.columns([1.35, 1])

    with details_column:
        st.subheader("🗺️ Destination overview")
        if preview_image:
            st.image(
                preview_image.url,
                caption=preview_image.page_title,
                use_column_width=True,
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
        use_container_width=True,
        hide_index=True,
    )

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

    if st.button("Create my trip plan", type="primary", use_container_width=True):
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
    else:
        st.caption("Generate a day-by-day itinerary for your current match.")

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
    st.write(
        "**Model features:** season-adjusted budget, selected-month climate, beaches, "
        "culture, nature, nightlife, "
        "food, adventure, relaxation, popularity, and safety."
    )

    with st.expander("Processed dataset preview"):
        st.dataframe(
            filtered_destinations,
            use_container_width=True,
            hide_index=True,
        )
