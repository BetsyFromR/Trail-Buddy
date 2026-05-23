import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from trail_buddy.effort import EFFORT_TOOLS, flat_kilometer_equivalent
from trail_buddy.graph import build_graph
from trail_buddy.prompts import render_system_prompt


def test_flat_kilometer_equivalent_returns_json():
    payload = json.loads(
        flat_kilometer_equivalent.invoke(
            {"distance_km": 35, "elevation_gain_m": 2000}
        )
    )

    assert payload["distance_km"] == 35.0
    assert payload["elevation_gain_m"] == 2000.0
    assert payload["elevation_flat_km"] == 20.0
    assert payload["flat_km_equivalent"] == 55.0
    assert payload["ascent_m_per_flat_km"] == 100.0


def test_flat_kilometer_equivalent_validates_inputs():
    payload = json.loads(
        flat_kilometer_equivalent.invoke(
            {"distance_km": 35, "elevation_gain_m": -100}
        )
    )

    assert payload["category"] == "validation"
    assert payload["retryable"] is False
    assert "elevation_gain_m" in payload["error"]


def test_effort_tool_is_exported_and_prompted():
    assert [tool.name for tool in EFFORT_TOOLS] == ["flat_kilometer_equivalent"]

    rendered = render_system_prompt()
    assert "flat_kilometer_equivalent" in rendered
    assert "35 km with 2000 m D+ is 55 flat km" in rendered
    assert "MUST call this tool before answering" in rendered
    assert "Do not calculate this" in rendered
    assert "mentally or inline" in rendered


def test_graph_routes_flat_kilometer_tool_call(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "flat_kilometer_equivalent",
                "args": {"distance_km": 35, "elevation_gain_m": 2000},
                "id": "call_flat_km",
            }
        ],
    )
    final_msg = AIMessage(content="Roughly 55 flat km.")

    llm = FakeMessagesListChatModel(responses=[tool_call_msg, final_msg])
    graph = build_graph(llm=llm, retriever=lambda q: [])
    result = graph.invoke(
        {"messages": [HumanMessage(content="35 km with 2000 m elevation?")]},
        config={"configurable": {"thread_id": "flat-km-tool"}},
    )

    contents = [getattr(message, "content", "") for message in result["messages"]]
    assert any('"flat_km_equivalent": 55.0' in str(content) for content in contents)
    assert result["messages"][-1].content == "Roughly 55 flat km."
