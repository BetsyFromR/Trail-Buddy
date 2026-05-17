from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WebSearchSettings:
    """Tavily-backed web search tuning. ``api_key`` is required for live calls."""

    api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY") or None)
    max_results: int = field(
        default_factory=lambda: _int_env("TRAIL_BUDDY_WEB_SEARCH_MAX_RESULTS", 5)
    )
    search_depth: str = field(
        default_factory=lambda: os.getenv("TRAIL_BUDDY_WEB_SEARCH_DEPTH", "basic")
    )
    include_answer: bool = field(
        default_factory=lambda: _bool_env("TRAIL_BUDDY_WEB_SEARCH_INCLUDE_ANSWER", True)
    )


def get_web_search_settings() -> WebSearchSettings:
    return WebSearchSettings()
