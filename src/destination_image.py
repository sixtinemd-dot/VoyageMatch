import os
from dataclasses import dataclass
from urllib.parse import quote

import requests
from dotenv import load_dotenv


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"

load_dotenv()


@dataclass
class DestinationImage:
    url: str
    page_title: str
    page_url: str
    source: str = "Wikipedia / Wikimedia Commons"
    photographer_name: str | None = None
    photographer_url: str | None = None


def _image_from_pages(payload: dict) -> DestinationImage | None:
    pages = payload.get("query", {}).get("pages", {})
    page_items = pages if isinstance(pages, list) else pages.values()
    for page in page_items:
        thumbnail = page.get("thumbnail", {})
        if thumbnail.get("source"):
            title = page.get("title", "Destination")
            return DestinationImage(
                url=thumbnail["source"],
                page_title=title,
                page_url=page.get(
                    "fullurl",
                    f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                ),
            )
    return None


def _get_unsplash_image(city: str, country: str) -> DestinationImage | None:
    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        return None

    headers = {
        "Authorization": f"Client-ID {access_key}",
        "Accept-Version": "v1",
    }
    search_queries = [
        f"{city} {country}",
        f"{city} travel",
        f"{country} travel landscape",
    ]
    results = []
    for query in search_queries:
        response = requests.get(
            UNSPLASH_API_URL,
            params={
                "query": query,
                "orientation": "landscape",
                "content_filter": "high",
                "per_page": 10,
            },
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            break

    if not results:
        return None

    photo = results[0]
    user = photo.get("user", {})
    photographer_url = user.get("links", {}).get("html")
    if photographer_url:
        photographer_url = f"{photographer_url}?utm_source=voyagematch&utm_medium=referral"

    return DestinationImage(
        url=f"{photo['urls']['regular']}&auto=format&fit=crop&w=1400&q=85",
        page_title=photo.get("alt_description") or f"{city}, {country}",
        page_url=(
            f"{photo['links']['html']}?utm_source=voyagematch&utm_medium=referral"
        ),
        source="Unsplash",
        photographer_name=user.get("name"),
        photographer_url=photographer_url,
    )


def _get_wikipedia_image(city: str, country: str) -> DestinationImage | None:
    """Fetch a representative destination image from Wikipedia."""
    common_params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "pageimages|info",
        "piprop": "thumbnail",
        "pithumbsize": 1200,
        "inprop": "url",
        "redirects": 1,
        "origin": "*",
    }
    headers = {"User-Agent": "VoyageMatch/1.0 educational travel recommender"}

    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            params={**common_params, "titles": f"{city}, {country}"},
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        image = _image_from_pages(response.json())
        if image:
            return image

        response = requests.get(
            WIKIPEDIA_API_URL,
            params={
                **common_params,
                "generator": "search",
                "gsrsearch": f"{city} {country}",
                "gsrnamespace": 0,
                "gsrlimit": 3,
            },
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        return _image_from_pages(response.json())
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None


def get_destination_image(city: str, country: str) -> DestinationImage | None:
    try:
        unsplash_image = _get_unsplash_image(city, country)
        if unsplash_image:
            return unsplash_image
    except (requests.RequestException, KeyError, TypeError, ValueError):
        pass
    return _get_wikipedia_image(city, country)
