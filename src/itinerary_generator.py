import os

import pandas as pd
from dotenv import load_dotenv

from src.recommender import INTEREST_COLUMNS, UserPreferences


load_dotenv()


def _preference_summary(preferences: UserPreferences) -> str:
    values = {
        "budget_per_day_usd": preferences.budget_per_day_usd,
        "preferred_temp_c": preferences.preferred_temp_c,
        "beach": preferences.beach,
        "culture": preferences.culture,
        "nature": preferences.nature,
        "nightlife": preferences.nightlife,
        "food": preferences.food,
        "adventure": preferences.adventure,
        "relaxation": preferences.relaxation,
        "safety": preferences.safety,
        "popularity": preferences.popularity,
    }
    return ", ".join(f"{key}={value}" for key, value in values.items())


def generate_ai_itinerary(
    destination: pd.Series,
    preferences: UserPreferences,
    days: int,
    travel_period: str | None = None,
    display_currency: str = "USD",
    display_daily_cost: float | None = None,
) -> str | None:
    """Generate an itinerary with OpenAI when an API key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = f"""
Create a practical, weather-aware travel idea for a GenAI demo app.

Destination:
- City: {destination['city']}
- Country: {destination['country']}
- Region: {destination['region']}
- Estimated daily destination spending: {
    display_daily_cost
    if display_daily_cost is not None
    else destination['cost_per_day_usd']
:.0f} {display_currency}
- Average climate temperature: {destination['avg_temp_c']:.1f}C
- Destination description: {destination.get('description', '')}

User preferences:
{_preference_summary(preferences)}

Trip length: {days} days
Travel dates: {travel_period or "Not specified"}

Return:
1. A short trip overview.
2. A day-by-day itinerary using Markdown headings in this exact style:
   ## Day 1 - Theme
   **Morning:** ...
   **Afternoon:** ...
   **Evening:** ...
3. Give every day a different primary theme and avoid repeating the same main activity.
4. Progress naturally from arrival and orientation to deeper exploration, an excursion,
   a slower day, and a memorable final day.
5. Finish with concise budget and weather-aware planning notes.

Keep the answer concise and presentation-friendly.
"""
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
            input=prompt,
        )
        return response.output_text
    except Exception:
        return None


def generate_rule_based_itinerary(
    destination: pd.Series,
    preferences: UserPreferences,
    days: int,
    travel_period: str | None = None,
) -> str:
    """Generate a simple itinerary without an LLM, so the app works offline."""
    ranked_interests = sorted(
        INTEREST_COLUMNS,
        key=lambda column: getattr(preferences, column),
        reverse=True,
    )

    city = destination["city"]
    country = destination["country"]
    interest_text = ", ".join(ranked_interests[:3])

    lines = [
        f"### Your {city} trip",
        "",
        f"A {days}-day journey through {city}, {country}"
        f"{f' during {travel_period}' if travel_period else ''}, shaped around "
        f"{interest_text}. The pace starts gently, builds into deeper exploration, "
        "and leaves room for a relaxed finish.",
        "",
    ]

    interest_ideas = {
        "beach": (
            "Start near the waterfront and find a calm stretch of coast",
            "Try a beach activity or explore a nearby coastal district",
            "Stay for sunset and choose a casual seafood or local dinner",
        ),
        "culture": (
            "Visit a major historic area before the busiest hours",
            "Explore a museum, landmark, or architecture-focused route",
            "Walk through an atmospheric old quarter after dinner",
        ),
        "nature": (
            "Begin with a botanical garden, park, or scenic walking route",
            "Continue to a viewpoint or a nearby natural escape",
            "Return slowly through a quieter neighborhood or riverside area",
        ),
        "nightlife": (
            "Take an easy morning and explore a creative local district",
            "Visit independent shops, galleries, or a lively public square",
            "Choose live music, an evening market, or a well-known nightlife area",
        ),
        "food": (
            "Have breakfast at a neighborhood cafe and browse a local market",
            "Join a tasting route or sample regional specialties for lunch",
            "Book a signature dinner built around the city's best-known cuisine",
        ),
        "adventure": (
            "Set out early for an active guided experience",
            "Continue with a hike, water activity, cycling route, or outdoor challenge",
            "Recover over a relaxed meal and an easy scenic walk",
        ),
        "relaxation": (
            "Keep the morning open for a slow breakfast and wellness time",
            "Choose a gentle scenic walk, spa, or peaceful local escape",
            "Enjoy an unhurried dinner with no fixed schedule afterward",
        ),
    }

    special_days = [
        (
            "Arrival and first impressions",
            f"Settle in, learn the area around your accommodation, and get oriented in {city}",
            "Take a short walk through a central neighborhood without overplanning",
            "Choose an easy local dinner and rest after the journey",
        ),
        (
            "Local life and hidden corners",
            "Explore a residential neighborhood and stop at an independent cafe",
            "Browse small shops, local streets, and a less-visited landmark",
            "Try a neighborhood restaurant away from the main tourist route",
        ),
        (
            "A change of scenery",
            "Leave the city center for a nearby town, coast, landscape, or district",
            "Make the excursion the day's main experience and keep the route flexible",
            "Return for a simple dinner close to your accommodation",
        ),
        (
            "A slower day",
            "Sleep in and revisit a favorite cafe or neighborhood",
            "Keep one flexible activity and leave time for spontaneous discoveries",
            "Plan a relaxed evening with a scenic view or comfortable meal",
        ),
        (
            "Farewell favorites",
            "Return to one place you especially enjoyed or pick up local gifts",
            "Take a final walk and leave buffer time before departure",
            "End with a memorable meal that reflects the trip's strongest theme",
        ),
    ]

    for day in range(1, days + 1):
        if day == 1:
            theme, morning, afternoon, evening = special_days[0]
        elif day == days:
            theme, morning, afternoon, evening = special_days[-1]
        elif days >= 5 and day == days - 1:
            theme, morning, afternoon, evening = special_days[-2]
        elif days >= 4 and day == max(3, days // 2):
            theme, morning, afternoon, evening = special_days[2]
        elif day > 7 and day % 4 == 0:
            theme, morning, afternoon, evening = special_days[1]
        else:
            focus = ranked_interests[(day - 2) % len(ranked_interests)]
            theme = {
                "beach": "Coast and sunset",
                "culture": "History and character",
                "nature": "Green spaces and views",
                "nightlife": "Creative districts after dark",
                "food": "Markets and local flavors",
                "adventure": "Active exploration",
                "relaxation": "Slow travel and wellbeing",
            }[focus]
            morning, afternoon, evening = interest_ideas[focus]

        lines.extend(
            [
                f"## Day {day} - {theme}",
                f"**Morning:** {morning}.",
                "",
                f"**Afternoon:** {afternoon}.",
                "",
                f"**Evening:** {evening}.",
                "",
            ]
        )

    lines.extend(
        [
            "### Practical notes",
            f"- Plan around an average temperature of {destination['avg_temp_c']:.1f} C.",
            "- Keep one flexible block each day for weather changes or spontaneous finds.",
            "- Reserve major excursions and signature dinners ahead of time.",
        ]
    )
    return "\n".join(lines)


def generate_itinerary(
    destination: pd.Series,
    preferences: UserPreferences,
    days: int,
    travel_period: str | None = None,
    display_currency: str = "USD",
    display_daily_cost: float | None = None,
) -> str:
    ai_text = generate_ai_itinerary(
        destination,
        preferences,
        days,
        travel_period,
        display_currency,
        display_daily_cost,
    )
    if ai_text:
        return ai_text
    return generate_rule_based_itinerary(destination, preferences, days, travel_period)
