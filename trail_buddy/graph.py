from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from trail_buddy.llm import build_llm
from trail_buddy.nodes import make_advisor_node, make_retrieve_node
from trail_buddy.state import State


def build_graph(llm: BaseChatModel | None = None, checkpointer=None, retriever=None):
    """Build the Trail Buddy LangGraph app.

    Pass ``llm`` and ``retriever`` to inject fakes in tests. ``checkpointer``
    defaults to an in-process MemorySaver so multi-turn clarification works.
    """
    llm = llm or build_llm()
    checkpointer = checkpointer if checkpointer is not None else MemorySaver()

    graph = StateGraph(State)
    retrieve = make_retrieve_node(retriever) if retriever is not None else make_retrieve_node()
    graph.add_node("retrieve", retrieve)
    graph.add_node("advisor", make_advisor_node(llm))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "advisor")
    graph.add_edge("advisor", END)
    return graph.compile(checkpointer=checkpointer)
