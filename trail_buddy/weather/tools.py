from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

from langchain_core.tools import tool

from trail_buddy.weather.service import (
    OPEN_METEO_MAX_FORECAST_DAYS,
    WeatherUnavailable,
    fetch_forecast,
    fetch_historical,
    geocode,
)


logger = logging.getLogger(__name__)

WEATHER_CODE_LABELS = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "heavy showers",
    82: "violent showers",
    95: "thunderstorm",
    96: "thunderstorm w/ hail",
    99: "severe thunderstorm w/ hail",
}


def _label(code: Any) -> str:
    try:
        return WEATHER_CODE_LABELS.get(int(code), f"code {code}")
    except (TypeError, ValueError):
        return "unknown"


def _forecast_days_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    rows: list[dict[str, Any]] = []
    for i, day in enumerate(times):
        try:
            row = {
                "date": day,
                "tmin_c": daily["temperature_2m_min"][i],
                "tmax_c": daily["temperature_2m_max"][i],
                "precip_mm": daily["precipitation_sum"][i],
                "precip_hours": daily["precipitation_hours"][i],
                "wind_kmh_max": daily["windspeed_10m_max"][i],
                "weather_code": daily["weathercode"][i],
            }
        except (KeyError, IndexError, TypeError):
            continue
        row["weather"] = _label(row["weather_code"])
        rows.append(row)
    return rows


def _aggregate_year(daily: dict[str, Any]) -> dict[str, Any] | None:
    times = daily.get("time") or []
    if not times:
        return None
    tmins = [v for v in (daily.get("temperature_2m_min") or []) if v is not None]
    tmaxs = [v for v in (daily.get("temperature_2m_max") or []) if v is not None]
    precs = [v for v in (daily.get("precipitation_sum") or []) if v is not None]
    winds = [v for v in (daily.get("windspeed_10m_max") or []) if v is not None]
    if not (tmins and tmaxs):
        return None
    return {
        "days": len(times),
        "tmin_avg_c": round(sum(tmins) / len(tmins), 1),
        "tmax_avg_c": round(sum(tmaxs) / len(tmaxs), 1),
        "precip_total_mm": round(sum(precs), 1) if precs else 0.0,
        "precip_days": sum(1 for v in precs if v and v > 0.5),
        "wind_max_kmh": round(max(winds), 1) if winds else 0.0,
    }


def _historical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    years_payload: list[dict[str, Any]] = []
    for year, year_payload in sorted((payload.get("years") or {}).items()):
        agg = _aggregate_year(year_payload.get("daily") or {})
        years_payload.append({"year": int(year), "stats": agg})
    return {
        "target_date": payload.get("target_date"),
        "window_days": payload.get("window_days"),
        "years": years_payload,
    }


def _today() -> dt.date:
    return dt.date.today()


def _parse_date(target_date: str | None) -> dt.date | None:
    if not target_date:
        return None
    try:
        return dt.date.fromisoformat(target_date)
    except ValueError as exc:
        raise WeatherUnavailable(
            f"target_date must be ISO YYYY-MM-DD, got {target_date!r}."
        ) from exc


def _derive_forecast_days(parsed_date: dt.date | None, today: dt.date) -> tuple[int | None, bool, str | None]:
    """Return (forecast_days_override, skip_forecast, note).

    - No date → use env default (override=None), no skip, no note.
    - 0..MAX days ahead → cover up through the target day.
    - >MAX days ahead → skip forecast, leave a note.
    - Past date → keep env default; the historical block covers the target.
    """
    if parsed_date is None:
        return None, False, None
    days_until = (parsed_date - today).days
    if days_until > OPEN_METEO_MAX_FORECAST_DAYS:
        return None, True, (
            f"Target date is {days_until} days out; short-range forecast unavailable "
            f"beyond {OPEN_METEO_MAX_FORECAST_DAYS} days. Showing historical climatology only."
        )
    if 0 <= days_until <= OPEN_METEO_MAX_FORECAST_DAYS:
        return days_until + 1, False, None
    return None, False, None


def _error_json(location: str, message: str) -> str:
    return json.dumps({"error": f"Weather lookup failed: {message}", "location": location})


@tool("trail_weather_search", parse_docstring=True)
def trail_weather_search(location: str, target_date: str | None = None) -> str:
    """Look up trail weather as JSON: forecast plus historical climatology for the same date in prior years.

    Returns a JSON string with keys: ``location``, ``forecast`` (omitted when the
    target date is beyond the short-range horizon), ``historical`` (only when
    ``target_date`` is given), and optional ``notes``.

    Use this when the user asks whether to run, race, or train on a specific trail or
    in a specific area — especially if they mention a date, weekend, or month. Elicit
    the date before calling whenever the answer depends on it.

    Args:
        location: Place name to geocode (e.g. "Boka Bay Montenegro", "Chamonix",
            "Cheget Elbrus"). Pass a city, region, or named trail area.
        target_date: Optional ISO date (YYYY-MM-DD). Drives both the forecast horizon
            (covers up through that day, up to 16 days out) and the historical
            climatology lookup. Beyond 16 days out the forecast is omitted and only
            historical norms are returned.
    """
    logger.info("[weather] tool_call location=%r target_date=%r", location, target_date)
    try:
        parsed_date = _parse_date(target_date)
        forecast_days, skip_forecast, note = _derive_forecast_days(parsed_date, _today())

        geo = geocode(location)
        result: dict[str, Any] = {
            "location": {
                "name": geo.name,
                "country": geo.country,
                "latitude": geo.latitude,
                "longitude": geo.longitude,
                "elevation_m": geo.elevation_m,
            }
        }
        notes: list[str] = []
        if note:
            notes.append(note)

        if not skip_forecast:
            forecast = fetch_forecast(
                geo.latitude,
                geo.longitude,
                forecast_days=forecast_days,
            )
            result["forecast"] = {"days": _forecast_days_payload(forecast)}

        if parsed_date is not None:
            historical = fetch_historical(geo.latitude, geo.longitude, parsed_date)
            result["historical"] = _historical_payload(historical)

        if notes:
            result["notes"] = notes
        return json.dumps(result)
    except WeatherUnavailable as exc:
        logger.warning("[weather] unavailable: %s", exc)
        return _error_json(location, str(exc))


WEATHER_TOOLS = [trail_weather_search]
