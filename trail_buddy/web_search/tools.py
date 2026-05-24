from __future__ import annotations

import json
import logging

from langchain_core.tools import BaseTool, tool

from trail_buddy.web_search.config import WebSearchSettings, get_web_search_settings


logger = logging.getLogger(__name__)
SEARCH_RESULT_PREVIEW_CHARS = 220
SEARCH_RESULT_LOG_LIMIT = 5

TRAIL_WEB_SEARCH_DESCRIPTION = (
    "Search the public web for trail-running information that is current, niche, "
    "or not in the local knowledge base. Returns a JSON string with the search "
    "results (and optional answer). Use for: local trail or route recommendations "
    "in a specific country/region/city/park, race news, registration windows, course "
    "updates, recent results, gear reviews, regulations, or anything the user asks "
    "about that you do not already know reliably. Do NOT use for weather (use "
    "`trail_weather_search` instead) or for generic technique/nutrition advice you "
    "can answer directly. Input is a focused search query, not a question."
)


def _preview(value: object, *, limit: int = SEARCH_RESULT_PREVIEW_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _log_tavily_result(query: str, result: object) -> None:
    if not isinstance(result, dict):
        logger.info(
            "[web_search] tavily result query=%r type=%s",
            query,
            type(result).__name__,
        )
        return

    answer = _preview(result.get("answer"))
    results = result.get("results") or []
    logger.info(
        "[web_search] tavily answer query=%r results=%s answer=%s",
        query,
        len(results) if isinstance(results, list) else "unknown",
        answer or "<none>",
    )

    if not isinstance(results, list):
        return

    for index, item in enumerate(results[:SEARCH_RESULT_LOG_LIMIT], start=1):
        if not isinstance(item, dict):
            logger.info("[web_search] result %s: %s", index, _preview(item))
            continue
        logger.info(
            "[web_search] result %s: title=%s | url=%s | content=%s",
            index,
            _preview(item.get("title")),
            item.get("url") or "<none>",
            _preview(item.get("content") or item.get("raw_content")),
        )


def _make_tavily_tool(underlying: BaseTool) -> BaseTool:
    """Wrap a TavilySearch tool so it returns a JSON string instead of a dict.

    Keeps the tool name ``tavily_search`` for prompt/test stability.
    """

    @tool("tavily_search", description=TRAIL_WEB_SEARCH_DESCRIPTION)
    def tavily_search(query: str) -> str:
        try:
            result = underlying.invoke({"query": query})
        except Exception as exc:  # noqa: BLE001 — surface tool errors as JSON
            logger.warning("[web_search] tavily error: %s", exc)
            return json.dumps({"error": f"Web search failed: {exc}", "query": query})
        _log_tavily_result(query, result)
        return json.dumps(result, default=str)

    return tavily_search


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

    underlying = TavilySearch(
        max_results=resolved.max_results,
        search_depth=resolved.search_depth,
        include_answer=resolved.include_answer,
    )
    return [_make_tavily_tool(underlying)]


WEB_SEARCH_TOOLS: list[BaseTool] = build_web_search_tools()
