from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from trail_buddy.graph import build_graph
from trail_buddy.prompts import render_system_prompt


def _invoke(graph, text: str, thread: str):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread}},
    )


def test_graph_builds_and_responds_with_fake_llm():
    llm = FakeListChatModel(responses=["Sure — what's the race name and your half marathon PR?"])
    graph = build_graph(llm=llm)

    result = _invoke(
        graph,
        "I want to run a trail. Can I cope with 35 km / 2000 D+?",
        thread="t1",
    )

    last = result["messages"][-1]
    assert "race name" in last.content.lower()


def test_multi_turn_uses_checkpointer():
    llm = FakeListChatModel(responses=["Which race?", "Roughly 6:00–7:00 hours."])
    graph = build_graph(llm=llm)

    first = _invoke(graph, "Can I run 35 km / 2000 D+?", thread="multi")
    assert len(first["messages"]) == 2  # human + ai

    second = _invoke(graph, "Boka Bay Trail blue, HM PR 1:30", thread="multi")
    # Checkpointer preserved prior turn → 4 messages now.
    assert len(second["messages"]) == 4
    assert "6:00" in second["messages"][-1].content


def test_system_prompt_includes_profile_and_retrieved():
    rendered = render_system_prompt(
        profile={"target_race": "Boka Bay Trail blue", "half_marathon_pr": "1:30"},
        retrieved=["Race profile: 35 km, 2000 D+, technical descents."],
    )
    assert "Boka Bay" in rendered
    assert "1:30" in rendered
    assert "Retrieved context" in rendered


def test_state_has_messages_after_invocation():
    llm = FakeListChatModel(responses=["Wash them."])
    graph = build_graph(llm=llm)
    result = _invoke(graph, "А кроссовки надо стирать?", thread="ru")
    assert result["messages"][-1].content == "Wash them."
    assert result["retrieved"] == []
