from __future__ import annotations

import logging

from langchain_core.tools import BaseTool

from trail_buddy.web_search.config import WebSearchSettings, get_web_search_settings


logger = logging.getLogger(__name__)

TRAIL_WEB_SEARCH_DESCRIPTION = (
    "Search the public web for trail-running information that is current, niche, "
    "or not in the local knowledge base. Use for: race news, registration windows, "
    "course updates, recent results, gear reviews, regulations, or anything the user "
    "asks about that you do not already know reliably. Do NOT use for weather "
    "(use `trail_weather_search` instead) or for generic technique/nutrition advice "
    "you can answer directly. Input is a focused search query, not a question."
)


def build_web_search_tools(settings: WebSearchSettings | None = None) -> list[BaseTool]:
    """Return Tavily-backed search tools, or [] if no API key is configured."""
    resolved = settings or get_web_search_settings()
    if not resolved.api_key:
        logger.info("[web_search] TAVILY_API_KEY not set; web search tool disabled")
        return []

    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        logger.warning("[web_search] langchain-tavily not installed; web search disabled")
        return []

    tool = TavilySearch(
        max_results=resolved.max_results,
        search_depth=resolved.search_depth,
        include_answer=resolved.include_answer,
        description=TRAIL_WEB_SEARCH_DESCRIPTION,
    )
    return [tool]


WEB_SEARCH_TOOLS: list[BaseTool] = build_web_search_tools()
