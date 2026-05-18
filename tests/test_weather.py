import datetime as dt
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from trail_buddy.graph import build_graph
from trail_buddy.weather import service as weather_service
from trail_buddy.weather import tools as weather_tools
from trail_buddy.weather.service import WeatherUnavailable, fetch_forecast, geocode
from trail_buddy.weather.tools import (
    OPEN_METEO_MAX_FORECAST_DAYS,
    _derive_forecast_days,
    trail_weather_search,
)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


@pytest.fixture
def patch_httpx(monkeypatch):
    """Replace httpx.get inside the weather service with a routing stub."""
    routes: dict[str, object] = {}

    def fake_get(url, params=None, timeout=None):
        for key, payload in routes.items():
            if key in url:
                if callable(payload):
                    return _FakeResponse(payload(params))
                return _FakeResponse(payload)
        raise AssertionError(f"unrouted url: {url}")

    monkeypatch.setattr(weather_service.httpx, "get", fake_get)
    return routes


def test_geocode_returns_top_candidate(patch_httpx):
    patch_httpx["geocoding-api"] = {
        "results": [
            {"name": "Chamonix", "country": "France", "latitude": 45.92, "longitude": 6.87, "elevation": 1035}
        ]
    }
    candidates = geocode("Chamonix")
    assert len(candidates) == 1
    top = candidates[0]
    assert top.name == "Chamonix"
    assert top.country == "France"
    assert top.latitude == pytest.approx(45.92)
    assert top.elevation_m == pytest.approx(1035)


def test_geocode_returns_multiple_candidates(patch_httpx):
    patch_httpx["geocoding-api"] = {
        "results": [
            {
                "name": "Boka",
                "country": "Montenegro",
                "admin1": "Kotor",
                "latitude": 42.45,
                "longitude": 18.7,
                "elevation": 0,
            },
            {
                "name": "Boka",
                "country": "India",
                "admin1": "West Bengal",
                "latitude": 25.0,
                "longitude": 88.0,
                "elevation": 30,
            },
        ]
    }
    candidates = geocode("Boka Bay")
    assert [c.country for c in candidates] == ["Montenegro", "India"]
    assert candidates[0].admin1 == "Kotor"


def test_geocode_no_results_raises(patch_httpx):
    patch_httpx["geocoding-api"] = {"results": []}
    with pytest.raises(WeatherUnavailable) as exc_info:
        geocode("Nowhere-12345")
    assert exc_info.value.category == "business"
    assert exc_info.value.retryable is False


def test_geocode_empty_location_is_validation_error():
    with pytest.raises(WeatherUnavailable) as exc_info:
        geocode("   ")
    assert exc_info.value.category == "validation"
    assert exc_info.value.retryable is False


def test_fetch_forecast_passes_coordinates(patch_httpx):
    captured: dict = {}

    def forecast_payload(params):
        captured.update(params)
        return {"daily": {"time": ["2026-05-12"]}}

    patch_httpx["api.open-meteo.com/v1/forecast"] = forecast_payload
    fetch_forecast(45.0, 6.0)
    assert captured["latitude"] == 45.0
    assert captured["longitude"] == 6.0
    assert "temperature_2m_max" in captured["daily"]


def test_trail_weather_search_returns_json(monkeypatch, patch_httpx):
    monkeypatch.setattr(weather_tools, "_today", lambda: dt.date(2026, 5, 10))
    patch_httpx["geocoding-api"] = {
        "results": [
            {"name": "Chamonix", "country": "France", "latitude": 45.92, "longitude": 6.87, "elevation": 1035}
        ]
    }
    patch_httpx["api.open-meteo.com/v1/forecast"] = {
        "daily": {
            "time": ["2026-05-10", "2026-05-11", "2026-05-12"],
            "temperature_2m_max": [11.0, 13.0, 14.0],
            "temperature_2m_min": [2.0, 3.0, 4.0],
            "precipitation_sum": [0.0, 1.0, 5.2],
            "precipitation_hours": [0, 1, 4],
            "windspeed_10m_max": [9.0, 10.0, 22.0],
            "weathercode": [1, 2, 63],
        }
    }
    patch_httpx["archive-api.open-meteo.com/v1/archive"] = {
        "daily": {
            "time": ["2024-05-05", "2024-05-12", "2024-05-19"],
            "temperature_2m_max": [11.0, 13.0, 15.0],
            "temperature_2m_min": [2.0, 3.0, 4.0],
            "precipitation_sum": [1.0, 0.0, 3.0],
            "precipitation_hours": [1, 0, 2],
            "windspeed_10m_max": [15.0, 18.0, 20.0],
            "weathercode": [1, 2, 61],
        }
    }
    raw = trail_weather_search.invoke({"location": "Chamonix", "target_date": "2026-05-12"})
    payload = json.loads(raw)

    assert payload["location"]["name"] == "Chamonix"
    assert payload["location"]["country"] == "France"
    days = payload["forecast"]["days"]
    assert [d["date"] for d in days] == ["2026-05-10", "2026-05-11", "2026-05-12"]
    assert days[-1]["weather"] == "rain"  # weathercode 63
    historical = payload["historical"]
    assert historical["target_date"] == "2026-05-12"
    assert 2024 in {y["year"] for y in historical["years"]}


def test_trail_weather_search_skips_history_when_no_date(monkeypatch, patch_httpx):
    monkeypatch.setattr(weather_tools, "_today", lambda: dt.date(2026, 5, 10))
    patch_httpx["geocoding-api"] = {
        "results": [{"name": "Boka Bay", "country": "ME", "latitude": 42.4, "longitude": 18.7, "elevation": 0}]
    }
    patch_httpx["api.open-meteo.com/v1/forecast"] = {
        "daily": {
            "time": ["2026-05-10"],
            "temperature_2m_max": [22.0],
            "temperature_2m_min": [14.0],
            "precipitation_sum": [0.0],
            "precipitation_hours": [0],
            "windspeed_10m_max": [12.0],
            "weathercode": [0],
        }
    }
    payload = json.loads(trail_weather_search.invoke({"location": "Boka Bay"}))
    assert "forecast" in payload
    assert "historical" not in payload


def test_trail_weather_search_far_future_returns_climatology_only(monkeypatch, patch_httpx):
    monkeypatch.setattr(weather_tools, "_today", lambda: dt.date(2026, 5, 10))
    patch_httpx["geocoding-api"] = {
        "results": [{"name": "Chamonix", "country": "France", "latitude": 45.92, "longitude": 6.87, "elevation": 1035}]
    }
    # No forecast route registered — far-future target must NOT hit the forecast URL.
    patch_httpx["archive-api.open-meteo.com/v1/archive"] = {
        "daily": {
            "time": ["2024-08-10"],
            "temperature_2m_max": [25.0],
            "temperature_2m_min": [12.0],
            "precipitation_sum": [0.0],
            "precipitation_hours": [0],
            "windspeed_10m_max": [10.0],
            "weathercode": [0],
        }
    }
    payload = json.loads(
        trail_weather_search.invoke({"location": "Chamonix", "target_date": "2026-08-15"})
    )
    assert "forecast" not in payload
    assert payload["historical"]["target_date"] == "2026-08-15"
    assert any("beyond" in note for note in payload.get("notes", []))


def _invoke_as_tool_call(args: dict, *, call_id: str = "tc-test"):
    """Invoke the tool with a ToolCall payload so we get back a ToolMessage
    carrying both ``content`` (for the LLM) and ``artifact`` (for observability).
    """
    return trail_weather_search.invoke(
        {
            "args": args,
            "id": call_id,
            "name": "trail_weather_search",
            "type": "tool_call",
        }
    )


def test_trail_weather_search_handles_failure(monkeypatch):
    """Transient network failure must surface as a categorized artifact."""

    def boom(*_a, **_k):
        raise WeatherUnavailable("Open-Meteo request failed: network down")

    monkeypatch.setattr(weather_tools, "geocode", boom)

    message = _invoke_as_tool_call({"location": "Chamonix"})

    # Structured artifact stays in state for observability.
    artifact = message.artifact
    assert artifact["status"] == "error"
    assert artifact["location"] == "Chamonix"
    assert artifact["error"]["category"] == "transient"
    assert artifact["error"]["retryable"] is True
    assert "network down" in artifact["error"]["message"]
    assert "alternatives" not in artifact["error"]

    # Content (what the LLM sees) mirrors the categorization.
    payload = json.loads(message.content)
    assert "Weather lookup failed" in payload["error"]
    assert payload["category"] == "transient"
    assert payload["retryable"] is True
    assert payload["location"] == "Chamonix"


def test_trail_weather_search_validation_error_is_not_retryable(monkeypatch):
    """Bad ISO date is a validation error — LLM should fix the call, not retry."""
    monkeypatch.setattr(weather_tools, "_today", lambda: dt.date(2026, 5, 10))
    message = _invoke_as_tool_call({"location": "Chamonix", "target_date": "not-a-date"})

    artifact = message.artifact
    assert artifact["error"]["category"] == "validation"
    assert artifact["error"]["retryable"] is False


def test_trail_weather_search_ambiguous_input_returns_candidates(monkeypatch, patch_httpx):
    """Multiple geocoding matches: surface them as ambiguous_input + candidates."""
    monkeypatch.setattr(weather_tools, "_today", lambda: dt.date(2026, 5, 10))
    patch_httpx["geocoding-api"] = {
        "results": [
            {
                "name": "Boka",
                "country": "Montenegro",
                "admin1": "Kotor",
                "latitude": 42.45,
                "longitude": 18.7,
                "elevation": 0,
            },
            {
                "name": "Boka",
                "country": "India",
                "admin1": "West Bengal",
                "latitude": 25.0,
                "longitude": 88.0,
                "elevation": 30,
            },
        ]
    }

    message = _invoke_as_tool_call({"location": "Boka Bay"})

    artifact = message.artifact
    assert artifact["status"] == "error"
    error = artifact["error"]
    assert error["category"] == "ambiguous_input"
    assert error["retryable"] is False
    alternatives = error["alternatives"]
    assert {alt["country"] for alt in alternatives} == {"Montenegro", "India"}
    assert any(alt["admin1"] == "Kotor" for alt in alternatives)

    # LLM-visible content carries the same candidate list under `candidates`.
    payload = json.loads(message.content)
    assert payload["category"] == "ambiguous_input"
    assert {c["country"] for c in payload["candidates"]} == {"Montenegro", "India"}


def test_trail_weather_search_success_has_ok_artifact(monkeypatch, patch_httpx):
    """Sanity check: success path attaches a status=ok artifact mirroring content."""
    monkeypatch.setattr(weather_tools, "_today", lambda: dt.date(2026, 5, 10))
    patch_httpx["geocoding-api"] = {
        "results": [
            {"name": "Chamonix", "country": "France", "latitude": 45.92, "longitude": 6.87, "elevation": 1035}
        ]
    }
    patch_httpx["api.open-meteo.com/v1/forecast"] = {
        "daily": {
            "time": ["2026-05-10"],
            "temperature_2m_max": [12.0],
            "temperature_2m_min": [3.0],
            "precipitation_sum": [0.0],
            "precipitation_hours": [0],
            "windspeed_10m_max": [9.0],
            "weathercode": [1],
        }
    }
    message = _invoke_as_tool_call({"location": "Chamonix"})
    assert message.artifact["status"] == "ok"
    assert message.artifact["location"]["name"] == "Chamonix"


def test_derive_forecast_days_horizons():
    today = dt.date(2026, 5, 10)
    # No date → defaults
    assert _derive_forecast_days(None, today) == (None, False, None)
    # Same day → 1
    override, skip, _ = _derive_forecast_days(dt.date(2026, 5, 10), today)
    assert override == 1 and skip is False
    # Inside horizon → days_until + 1
    override, skip, _ = _derive_forecast_days(dt.date(2026, 5, 15), today)
    assert override == 6 and skip is False
    # Edge of horizon
    override, skip, _ = _derive_forecast_days(
        today + dt.timedelta(days=OPEN_METEO_MAX_FORECAST_DAYS), today
    )
    assert override == OPEN_METEO_MAX_FORECAST_DAYS + 1 and skip is False
    # Beyond horizon → skip
    override, skip, note = _derive_forecast_days(
        today + dt.timedelta(days=OPEN_METEO_MAX_FORECAST_DAYS + 1), today
    )
    assert override is None and skip is True and note
    # Past date → use defaults
    assert _derive_forecast_days(dt.date(2026, 5, 1), today) == (None, False, None)


def test_graph_routes_tool_call_and_returns_final_answer():
    """Stub LLM emits a tool_call; graph should run the tool then ask LLM again."""

    @tool("trail_weather_search")
    def stub_tool(location: str, target_date: str | None = None) -> str:
        """Stub."""
        return f"WEATHER stub for {location} on {target_date or 'today'}"

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "trail_weather_search",
                "args": {"location": "Chamonix", "target_date": "2026-06-15"},
                "id": "call_1",
            }
        ],
    )
    final_msg = AIMessage(content="Forecast looks dry — go for it.")

    llm = FakeMessagesListChatModel(responses=[tool_call_msg, final_msg])
    graph = build_graph(llm=llm, retriever=lambda q: [], tools=[stub_tool])

    result = graph.invoke(
        {"messages": [HumanMessage(content="Should I run Chamonix on June 15?")]},
        config={"configurable": {"thread_id": "tool-test"}},
    )

    contents = [getattr(m, "content", "") for m in result["messages"]]
    assert any("WEATHER stub for Chamonix on 2026-06-15" in str(c) for c in contents)
    assert result["messages"][-1].content == "Forecast looks dry — go for it."


def test_graph_without_tools_keeps_linear_pipeline():
    """Passing tools=[] should skip the tools node entirely (back-compat)."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    llm = FakeListChatModel(responses=["Plain answer."])
    graph = build_graph(llm=llm, retriever=lambda q: [], tools=[])
    result = graph.invoke(
        {"messages": [HumanMessage(content="Hi")]},
        config={"configurable": {"thread_id": "no-tools"}},
    )
    assert result["messages"][-1].content == "Plain answer."
