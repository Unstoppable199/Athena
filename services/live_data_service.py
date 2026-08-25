"""
Live data service.

Structured lookups for facts that change: weather, share prices and
exchange rates.

These exist because scraping prose for numbers was the weakest part of
the pipeline. A weather page hands back forty-odd sentences of hourly
tables, FAQ text and air-quality notices, and the model then has to
pick a temperature out of it and cite a sentence that supports it -
which is where answers were failing verification and falling back to a
raw evidence dump. These endpoints return the number itself, so there
is nothing to extract and nothing to misread.

Every source here is free and needs no API key, so nothing has to be
configured before Athena will run.
"""

import requests


# Long enough for a slow link, short enough that a hung endpoint doesn't
# hold up the reply.
#
# Raised from 6s after a currency lookup timed out and fell through to
# a web search - which took longer than the wait it avoided, and gave a
# worse answer from scraped pages. When the structured source is the
# better answer, it is worth waiting a little longer for it.
_TIMEOUT = 10

# One retry, because these endpoints drop the occasional request: the
# same conversion failed and then succeeded seconds later.
_ATTEMPTS = 2

# Yahoo rejects requests without a browser-shaped agent.
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _get_json(url, params=None, headers=None):
    """GET and decode JSON, retrying once on a network failure."""

    last_error = None

    for attempt in range(_ATTEMPTS):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=_TIMEOUT,
            )
            return response.json()

        except Exception as error:
            last_error = error
            print(f"[LIVE DATA] attempt {attempt + 1} failed: {error}")

    raise last_error


# WMO weather interpretation codes, which is what Open-Meteo reports
# instead of a description. Only the buckets a person would recognise -
# an unmapped code degrades to the plain number rather than guessing.
_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _place_label(place: dict) -> str:
    """Name the place without repeating itself.

    The three fields overlap more often than not: Singapore's region
    and country are both "Singapore", and a city that shares its name
    with its state produced "Japan, Japan" - which reads like a bug
    even when the weather behind it is right.
    """

    label = []

    for part in (place.get("name"), place.get("admin1"), place.get("country")):

        part = (part or "").strip()

        if part and part.lower() not in {p.lower() for p in label}:
            label.append(part)

    return ", ".join(label)


class LiveDataService:

    # ------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------

    def weather(self, location: str) -> dict:
        """Current conditions for a place name, via Open-Meteo."""

        if not location or not str(location).strip():
            # Worded as the question it stands in for. The router
            # normally holds "what's the weather" back so it can be
            # asked properly, but the planner can still reach here on
            # its own - and when it did, the whole reply was the string
            # "No location was provided.", which tells the user what
            # went wrong internally rather than what to do about it.
            return {
                "success": False,
                "needs_clarification": True,
                "error": "Which city's weather would you like?",
            }

        location = str(location).strip()

        try:
            geo = _get_json(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
            )

            results = geo.get("results") or []

            if not results:
                return {
                    "success": False,
                    "error": f"I couldn't find a place called '{location}'.",
                }

            place = results[0]

            # A country name geocodes to a single point somewhere in
            # the middle of it, and the API answers for that point
            # without a word of complaint. "The weather in India" then
            # comes back as the weather in one unnamed field in Madhya
            # Pradesh, presented as the weather in India.
            #
            # The feature code is what separates them: countries are
            # PCLI, populated places are PPL and its variants. Asking
            # which city is the only honest answer here - the question
            # as asked does not have one.
            if str(place.get("feature_code", "")).startswith("PCL"):
                return {
                    "success": False,
                    "needs_clarification": True,
                    "error": (
                        f"'{location}' is a whole country, which doesn't have "
                        "one set of weather. Which city?"
                    ),
                }

            data = _get_json(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": (
                        "temperature_2m,relative_humidity_2m,"
                        "apparent_temperature,precipitation,"
                        "wind_speed_10m,weather_code"
                    ),
                    "timezone": "auto",
                },
            )

            current = data.get("current") or {}
            units = data.get("current_units") or {}

            if "temperature_2m" not in current:
                return {
                    "success": False,
                    "error": "The weather service didn't return conditions.",
                }

            code = current.get("weather_code")

            # Reported without units so the caller can label them from
            # the API's own unit fields rather than assuming Celsius.
            return {
                "success": True,
                "data": {
                    "place": _place_label(place),
                    "conditions": _WEATHER_CODES.get(code, f"code {code}"),
                    "temperature": current.get("temperature_2m"),
                    "temperature_unit": units.get("temperature_2m", "°C"),
                    "feels_like": current.get("apparent_temperature"),
                    "humidity": current.get("relative_humidity_2m"),
                    "humidity_unit": units.get("relative_humidity_2m", "%"),
                    "precipitation": current.get("precipitation"),
                    "precipitation_unit": units.get("precipitation", "mm"),
                    "wind_speed": current.get("wind_speed_10m"),
                    "wind_unit": units.get("wind_speed_10m", "km/h"),
                    "observed_at": current.get("time"),
                    "timezone": data.get("timezone"),
                },
            }

        except Exception as error:
            return {
                "success": False,
                "error": f"The weather lookup failed: {error}",
            }

    # ------------------------------------------------------------
    # Share prices
    # ------------------------------------------------------------

    def quote(self, symbol: str) -> dict:
        """Latest price for a ticker symbol, via Yahoo Finance."""

        if not symbol or not str(symbol).strip():
            return {"success": False, "error": "No ticker symbol was provided."}

        symbol = str(symbol).strip().upper()

        try:
            payload = _get_json(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"range": "1d", "interval": "1d"},
                headers=_HEADERS,
            )

            results = (payload.get("chart") or {}).get("result") or []

            if not results:
                return {
                    "success": False,
                    "error": f"I couldn't find a ticker called '{symbol}'.",
                }

            meta = results[0].get("meta") or {}
            price = meta.get("regularMarketPrice")
            previous = meta.get("chartPreviousClose") or meta.get("previousClose")

            if price is None:
                return {
                    "success": False,
                    "error": f"No price was returned for '{symbol}'.",
                }

            change = None
            change_percent = None

            if previous:
                change = round(price - previous, 4)
                change_percent = round((price - previous) / previous * 100, 2)

            return {
                "success": True,
                "data": {
                    "symbol": meta.get("symbol", symbol),
                    "name": meta.get("longName") or meta.get("shortName"),
                    "price": price,
                    "currency": meta.get("currency"),
                    "previous_close": previous,
                    "change": change,
                    "change_percent": change_percent,
                    "exchange": meta.get("fullExchangeName"),
                },
            }

        except Exception as error:
            return {
                "success": False,
                "error": f"The price lookup failed: {error}",
            }

    # ------------------------------------------------------------
    # Exchange rates
    # ------------------------------------------------------------

    def exchange(self, base: str, target: str, amount=1) -> dict:
        """Convert between currencies, via Frankfurter (ECB rates)."""

        if not base or not target:
            return {
                "success": False,
                "error": "Both a source and a target currency are needed.",
            }

        base = str(base).strip().upper()
        target = str(target).strip().upper()

        try:
            amount = float(amount) if amount is not None else 1.0
        except (TypeError, ValueError):
            amount = 1.0

        try:
            payload = _get_json(
                "https://api.frankfurter.dev/v1/latest",
                params={"base": base, "symbols": target},
            )

            rates = payload.get("rates") or {}

            if target not in rates:
                return {
                    "success": False,
                    "error": (
                        f"I couldn't convert {base} to {target}. "
                        "Only currencies published by the ECB are covered."
                    ),
                }

            rate = rates[target]

            return {
                "success": True,
                "data": {
                    "base": base,
                    "target": target,
                    "rate": rate,
                    "amount": amount,
                    "converted": round(amount * rate, 4),
                    "rates_published": payload.get("date"),
                },
            }

        except Exception as error:
            return {
                "success": False,
                "error": f"The exchange-rate lookup failed: {error}",
            }
