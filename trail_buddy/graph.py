from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from trail_buddy.effort import EFFORT_TOOLS
from trail_buddy.llm import build_llm
from trail_buddy.nodes import make_advisor_node, make_retrieve_node
from trail_buddy.state import State
from trail_buddy.weather import WEATHER_TOOLS
from trail_buddy.web_search import build_web_search_tools


def build_graph(
    llm: BaseChatModel | None = None,
    checkpointer=None,
    retriever=None,
    tools=None,
):
    """Build the Trail Buddy LangGraph app.

    Pass ``llm``, ``retriever``, or ``tools`` to inject fakes in tests.
    ``checkpointer`` defaults to an in-process MemorySaver so multi-turn
    clarification works. ``tools`` defaults to bundled tools; pass
    ``[]`` to disable tool calling entirely.
    """
    llm = llm or build_llm()
    checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    resolved_tools = (
        [*WEATHER_TOOLS, *EFFORT_TOOLS, *build_web_search_tools()]
        if tools is None
        else tools
    )

    graph = StateGraph(State)
    retrieve = make_retrieve_node(retriever) if retriever is not None else make_retrieve_node()
    graph.add_node("retrieve", retrieve)
    graph.add_node("advisor", make_advisor_node(llm, tools=resolved_tools))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "advisor")

    if resolved_tools:
        graph.add_node("tools", ToolNode(resolved_tools))
        graph.add_conditional_edges("advisor", tools_condition)
        graph.add_edge("tools", "advisor")
    else:
        graph.add_edge("advisor", END)

    return graph.compile(checkpointer=checkpointer)
