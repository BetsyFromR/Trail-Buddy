from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from trail_buddy.prompts import render_system_prompt
from trail_buddy.state import State


def retrieve_node(state: State) -> dict:
    """RAG seam. Returns no context in the MVP; replaced when retrieval lands."""
    return {"retrieved": []}


def make_advisor_node(llm: BaseChatModel):
    def advisor_node(state: State) -> dict:
        system = SystemMessage(
            content=render_system_prompt(
                profile=state.get("profile") or {},
                retrieved=state.get("retrieved") or [],
            )
        )
        response = llm.invoke([system, *state["messages"]])
        return {"messages": [response]}

    return advisor_node
