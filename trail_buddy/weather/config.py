from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class WeatherSettings:
    """Open-Meteo endpoints + tuning knobs. No API key required."""

    forecast_url: str = field(
        default_factory=lambda: os.getenv("TRAIL_BUDDY_WEATHER_FORECAST_URL", DEFAULT_FORECAST_URL)
    )
    archive_url: str = field(
        default_factory=lambda: os.getenv("TRAIL_BUDDY_WEATHER_ARCHIVE_URL", DEFAULT_ARCHIVE_URL)
    )
    geocode_url: str = field(
        default_factory=lambda: os.getenv("TRAIL_BUDDY_WEATHER_GEOCODE_URL", DEFAULT_GEOCODE_URL)
    )
    forecast_days: int = field(
        default_factory=lambda: _int_env("TRAIL_BUDDY_WEATHER_FORECAST_DAYS", 7)
    )
    historical_years_back: int = field(
        default_factory=lambda: _int_env("TRAIL_BUDDY_WEATHER_HISTORICAL_YEARS", 3)
    )
    historical_window_days: int = field(
        default_factory=lambda: _int_env("TRAIL_BUDDY_WEATHER_HISTORICAL_WINDOW_DAYS", 7)
    )
    request_timeout_s: float = field(
        default_factory=lambda: _float_env("TRAIL_BUDDY_WEATHER_TIMEOUT_S", 10.0)
    )


def get_weather_settings() -> WeatherSettings:
    return WeatherSettings()
