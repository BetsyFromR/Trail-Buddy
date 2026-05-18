from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from trail_buddy.weather.config import WeatherSettings, get_weather_settings


logger = logging.getLogger(__name__)


class WeatherUnavailable(RuntimeError):
    """Raised when Open-Meteo cannot be reached or returns an unusable response.

    ``category`` lets callers distinguish:
      - ``transient``  — network/HTTP issues; safe to retry.
      - ``validation`` — caller-side bad input (empty location, bad date).
      - ``business``   — request was well-formed but the upstream has no data.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str = "transient",
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = bool(retryable) if retryable is not None else category == "transient"


@dataclass(frozen=True)
class GeocodeResult:
    name: str
    country: str | None
    latitude: float
    longitude: float
    elevation_m: float | None
    admin1: str | None = None


DEFAULT_GEOCODE_CANDIDATES = 5


def _get_json(url: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    try:
        response = httpx.get(url, params=params, timeout=timeout_s)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise WeatherUnavailable(
            f"Open-Meteo request failed: {exc}", category="transient", retryable=True
        ) from exc


def geocode(
    location: str,
    settings: WeatherSettings | None = None,
    max_candidates: int = DEFAULT_GEOCODE_CANDIDATES,
) -> list[GeocodeResult]:
    """Return ranked geocoding candidates for ``location`` (best first).

    The caller decides what to do when more than one result comes back — there's
    no silent pick, so disambiguation can happen up at the tool/LLM layer.
    """
    if not location.strip():
        raise WeatherUnavailable("Empty location.", category="validation", retryable=False)
    resolved = settings or get_weather_settings()
    count = max(1, int(max_candidates))
    payload = _get_json(
        resolved.geocode_url,
        {"name": location, "count": count, "language": "en", "format": "json"},
        resolved.request_timeout_s,
    )
    results = payload.get("results") or []
    if not results:
        raise WeatherUnavailable(
            f"No coordinates found for {location!r}.",
            category="business",
            retryable=False,
        )
    return [
        GeocodeResult(
            name=item.get("name") or location,
            country=item.get("country"),
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
            elevation_m=float(item["elevation"]) if item.get("elevation") is not None else None,
            admin1=item.get("admin1"),
        )
        for item in results
    ]


_FORECAST_DAILY = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_hours",
    "windspeed_10m_max",
    "weathercode",
]


OPEN_METEO_MAX_FORECAST_DAYS = 16


def fetch_forecast(
    latitude: float,
    longitude: float,
    settings: WeatherSettings | None = None,
    forecast_days: int | None = None,
) -> dict[str, Any]:
    resolved = settings or get_weather_settings()
    days = forecast_days if forecast_days is not None else resolved.forecast_days
    days = max(1, min(int(days), OPEN_METEO_MAX_FORECAST_DAYS))
    return _get_json(
        resolved.forecast_url,
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join(_FORECAST_DAILY),
            "forecast_days": days,
            "timezone": "auto",
            "windspeed_unit": "kmh",
        },
        resolved.request_timeout_s,
    )


def fetch_historical(
    latitude: float,
    longitude: float,
    target_date: dt.date,
    settings: WeatherSettings | None = None,
) -> dict[str, Any]:
    """Fetch a window of ±N days around ``target_date`` for the past N years."""
    resolved = settings or get_weather_settings()
    window = resolved.historical_window_days
    years = resolved.historical_years_back

    today = dt.date.today()
    end_year = today.year - 1
    start_year = end_year - years + 1

    years_payload: dict[str, dict[str, Any]] = {}
    for year in range(start_year, end_year + 1):
        try:
            start = target_date.replace(year=year) - dt.timedelta(days=window)
            end = target_date.replace(year=year) + dt.timedelta(days=window)
        except ValueError:
            start = dt.date(year, target_date.month, min(target_date.day, 28)) - dt.timedelta(days=window)
            end = start + dt.timedelta(days=2 * window)

        years_payload[str(year)] = _get_json(
            resolved.archive_url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": ",".join(_FORECAST_DAILY),
                "timezone": "auto",
                "windspeed_unit": "kmh",
            },
            resolved.request_timeout_s,
        )

    return {
        "target_date": target_date.isoformat(),
        "window_days": window,
        "years": years_payload,
    }
