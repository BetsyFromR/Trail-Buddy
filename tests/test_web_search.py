import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from trail_buddy.graph import build_graph
from trail_buddy.web_search import build_web_search_tools
from trail_buddy.web_search.config import WebSearchSettings, get_web_search_settings


def test_web_search_settings_resolve_from_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("TRAIL_BUDDY_WEB_SEARCH_MAX_RESULTS", "8")
    monkeypatch.setenv("TRAIL_BUDDY_WEB_SEARCH_DEPTH", "advanced")
    monkeypatch.setenv("TRAIL_BUDDY_WEB_SEARCH_INCLUDE_ANSWER", "false")

    settings = get_web_search_settings()
    assert settings.api_key == "tvly-test"
    assert settings.max_results == 8
    assert settings.search_depth == "advanced"
    assert settings.include_answer is False


def test_web_search_tools_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    settings = WebSearchSettings()
    assert build_web_search_tools(settings) == []


def test_web_search_tool_built_when_api_key_present(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    settings = WebSearchSettings()
    tools = build_web_search_tools(settings)
    assert len(tools) == 1
    assert tools[0].name == "tavily_search"
    assert "trail-running" in tools[0].description


def test_graph_builds_without_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    llm = FakeListChatModel(responses=["ok"])
    graph = build_graph(llm=llm, retriever=lambda q: [])
    result = graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "no-key"}},
    )
    assert result["messages"][-1].content == "ok"


def test_graph_routes_web_search_tool_call():
    @tool("tavily_search")
    def stub_tavily(query: str) -> str:
        """Stub."""
        return f"WEB results for: {query}"

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "tavily_search",
                "args": {"query": "UTMB 2026 registration"},
                "id": "call_web",
            }
        ],
    )
    final_msg = AIMessage(content="Registration opens in December.")

    llm = FakeMessagesListChatModel(responses=[tool_call_msg, final_msg])
    graph = build_graph(llm=llm, retriever=lambda q: [], tools=[stub_tavily])
    result = graph.invoke(
        {"messages": [HumanMessage(content="When does UTMB 2026 registration open?")]},
        config={"configurable": {"thread_id": "web-tool"}},
    )

    contents = [getattr(m, "content", "") for m in result["messages"]]
    assert any("WEB results for: UTMB 2026 registration" in str(c) for c in contents)
    assert result["messages"][-1].content == "Registration opens in December."
