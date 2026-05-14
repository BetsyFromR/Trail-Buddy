from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from langchain_core.tools import tool

from trail_buddy.weather.service import (
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


def _format_forecast(payload: dict[str, Any]) -> str:
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return "Forecast: no data."

    rows: list[str] = []
    for i, day in enumerate(times):
        try:
            tmin = daily["temperature_2m_min"][i]
            tmax = daily["temperature_2m_max"][i]
            precip = daily["precipitation_sum"][i]
            precip_h = daily["precipitation_hours"][i]
            wind = daily["windspeed_10m_max"][i]
            code = daily["weathercode"][i]
        except (KeyError, IndexError, TypeError):
            continue
        rows.append(
            f"  {day}: {tmin}–{tmax}°C, "
            f"precip {precip} mm over {precip_h} h, "
            f"wind {wind} km/h, {_label(code)}"
        )
    return "Forecast (daily):\n" + "\n".join(rows)


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
        "tmin_avg": round(sum(tmins) / len(tmins), 1),
        "tmax_avg": round(sum(tmaxs) / len(tmaxs), 1),
        "precip_total": round(sum(precs), 1) if precs else 0.0,
        "precip_days": sum(1 for v in precs if v and v > 0.5),
        "wind_max": round(max(winds), 1) if winds else 0.0,
    }


def _format_historical(payload: dict[str, Any]) -> str:
    target = payload.get("target_date")
    window = payload.get("window_days")
    years = payload.get("years") or {}
    if not years:
        return "Historical: no data."

    rows: list[str] = []
    for year, year_payload in sorted(years.items()):
        agg = _aggregate_year(year_payload.get("daily") or {})
        if not agg:
            rows.append(f"  {year}: no data")
            continue
        rows.append(
            f"  {year}: "
            f"min {agg['tmin_avg']}°C / max {agg['tmax_avg']}°C avg, "
            f"{agg['precip_total']} mm over {agg['precip_days']} wet days, "
            f"peak wind {agg['wind_max']} km/h"
        )
    return (
        f"Historical climatology for {target} ±{window} days:\n" + "\n".join(rows)
    )


def _parse_date(target_date: str | None) -> dt.date | None:
    if not target_date:
        return None
    try:
        return dt.date.fromisoformat(target_date)
    except ValueError as exc:
        raise WeatherUnavailable(
            f"target_date must be ISO YYYY-MM-DD, got {target_date!r}."
        ) from exc


@tool("trail_weather_search", parse_docstring=True)
def trail_weather_search(location: str, target_date: str | None = None) -> str:
    """Look up trail weather: 7-day forecast plus historical climatology for the same date in prior years.

    Use this when the user asks whether to run, race, or train on a specific trail or
    in a specific area — especially if they mention a date, weekend, or month.

    Args:
        location: Place name to geocode (e.g. "Boka Bay Montenegro", "Chamonix",
            "Cheget Elbrus"). Pass a city, region, or named trail area.
        target_date: Optional ISO date (YYYY-MM-DD) for historical lookup. If omitted,
            only the forecast is returned. Pass the user's planned run date when known.
    """
    logger.info("[weather] tool_call location=%r target_date=%r", location, target_date)
    parsed_date = _parse_date(target_date)
    try:
        geo = geocode(location)
        forecast = fetch_forecast(geo.latitude, geo.longitude)
        header = (
            f"Location: {geo.name}"
            + (f", {geo.country}" if geo.country else "")
            + f" ({geo.latitude:.3f}, {geo.longitude:.3f}"
            + (f", elev {geo.elevation_m:.0f} m" if geo.elevation_m is not None else "")
            + ")"
        )
        sections = [header, _format_forecast(forecast)]
        if parsed_date is not None:
            historical = fetch_historical(geo.latitude, geo.longitude, parsed_date)
            sections.append(_format_historical(historical))
        return "\n\n".join(sections)
    except WeatherUnavailable as exc:
        logger.warning("[weather] unavailable: %s", exc)
        return f"Weather lookup failed: {exc}"


WEATHER_TOOLS = [trail_weather_search]
