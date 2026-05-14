import datetime as dt

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from trail_buddy.graph import build_graph
from trail_buddy.weather import service as weather_service
from trail_buddy.weather.service import WeatherUnavailable, fetch_forecast, geocode
from trail_buddy.weather.tools import trail_weather_search


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


def test_geocode_parses_first_result(patch_httpx):
    patch_httpx["geocoding-api"] = {
        "results": [
            {"name": "Chamonix", "country": "France", "latitude": 45.92, "longitude": 6.87, "elevation": 1035}
        ]
    }
    result = geocode("Chamonix")
    assert result.name == "Chamonix"
    assert result.country == "France"
    assert result.latitude == pytest.approx(45.92)
    assert result.elevation_m == pytest.approx(1035)


def test_geocode_no_results_raises(patch_httpx):
    patch_httpx["geocoding-api"] = {"results": []}
    with pytest.raises(WeatherUnavailable):
        geocode("Nowhere-12345")


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


def test_trail_weather_search_returns_summary(patch_httpx):
    patch_httpx["geocoding-api"] = {
        "results": [
            {"name": "Chamonix", "country": "France", "latitude": 45.92, "longitude": 6.87, "elevation": 1035}
        ]
    }
    patch_httpx["api.open-meteo.com/v1/forecast"] = {
        "daily": {
            "time": ["2026-05-12", "2026-05-13"],
            "temperature_2m_max": [12.0, 14.0],
            "temperature_2m_min": [3.0, 4.0],
            "precipitation_sum": [0.0, 5.2],
            "precipitation_hours": [0, 4],
            "windspeed_10m_max": [10.0, 22.0],
            "weathercode": [1, 63],
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
    summary = trail_weather_search.invoke({"location": "Chamonix", "target_date": "2026-05-12"})
    assert "Chamonix" in summary
    assert "Forecast" in summary
    assert "Historical" in summary
    assert "2026-05-12" in summary
    assert "rain" in summary  # weathercode 63 → "rain"


def test_trail_weather_search_skips_history_when_no_date(patch_httpx):
    patch_httpx["geocoding-api"] = {
        "results": [{"name": "Boka Bay", "country": "ME", "latitude": 42.4, "longitude": 18.7, "elevation": 0}]
    }
    patch_httpx["api.open-meteo.com/v1/forecast"] = {
        "daily": {
            "time": ["2026-05-12"],
            "temperature_2m_max": [22.0],
            "temperature_2m_min": [14.0],
            "precipitation_sum": [0.0],
            "precipitation_hours": [0],
            "windspeed_10m_max": [12.0],
            "weathercode": [0],
        }
    }
    summary = trail_weather_search.invoke({"location": "Boka Bay"})
    assert "Forecast" in summary
    assert "Historical" not in summary


def test_trail_weather_search_handles_failure(monkeypatch):
    def boom(*_a, **_k):
        raise WeatherUnavailable("network down")

    from trail_buddy.weather import tools as weather_tools

    monkeypatch.setattr(weather_tools, "geocode", boom)
    summary = trail_weather_search.invoke({"location": "Chamonix"})
    assert "Weather lookup failed" in summary


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
