import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from trail_buddy.prompts import render_system_prompt
from trail_buddy.retrieval import RetrievalUnavailable, retrieve_context
from trail_buddy.state import State


TEXT_PREVIEW_CHARS = 180
logger = logging.getLogger(__name__)


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content or "")


def _latest_user_text(state: State) -> str:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) != "human":
            continue
        return _message_text(getattr(message, "content", ""))
    return ""


def _doc_summary(context: str) -> str:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    title = next((line for line in lines if line.startswith("[") or line.startswith("Title:")), "Untitled")
    source = next((line for line in lines if line.startswith("Source:")), "Source: unknown")
    chunk_id = next((line.removeprefix("Chunk ID:").strip() for line in lines if line.startswith("Chunk ID:")), "unknown")
    body = next(
        (
            line
            for line in lines
            if not line.startswith("[")
            and not line.startswith("Title:")
            and not line.startswith("Source:")
            and not line.startswith("Chunk ID:")
        ),
        "",
    )
    preview = body[:TEXT_PREVIEW_CHARS]
    if len(body) > TEXT_PREVIEW_CHARS:
        preview += "..."
    return f"chunk_id={chunk_id} | {title} | {source} | text={preview}"


def _log_retrieval_trace(query: str, retrieved: list[str]) -> None:
    logger.info("[RAG] query: %s", query)
    logger.info("[RAG] retrieved_docs: %s", len(retrieved))
    for index, context in enumerate(retrieved, start=1):
        logger.info("[RAG] doc %s: %s", index, _doc_summary(context))


def _log_answer_trace(response) -> None:
    answer = _message_text(getattr(response, "content", response)).strip()
    logger.info("[RAG] answer: %s", answer)


def make_retrieve_node(retriever=retrieve_context):
    def _retrieve_node(state: State) -> dict:
        query = _latest_user_text(state)
        if not query.strip():
            _log_retrieval_trace(query, [])
            return {"retrieved": []}

        try:
            retrieved = retriever(query)
            _log_retrieval_trace(query, retrieved)
            return {"retrieved": retrieved}
        except RetrievalUnavailable as exc:
            logger.warning("[RAG] retrieval_unavailable: %s", exc)
            _log_retrieval_trace(query, [])
            return {"retrieved": []}
        except Exception:
            logger.exception("[RAG] retrieval_error")
            _log_retrieval_trace(query, [])
            return {"retrieved": []}

    return _retrieve_node


def retrieve_node(state: State) -> dict:
    return make_retrieve_node()(state)


def _non_system_messages(messages):
    return [
        message
        for message in messages
        if getattr(message, "type", None) != "system"
    ]


def make_advisor_node(llm: BaseChatModel, tools=None):
    """Build the advisor node. If ``tools`` is given they are bound to the LLM so
    the model can emit tool_calls; the graph routes those to a ToolNode.
    """
    if tools:
        try:
            bound_llm = llm.bind_tools(tools)
        except (NotImplementedError, AttributeError):
            logger.warning("[tools] llm.bind_tools unsupported; running without tools")
            bound_llm = llm
    else:
        bound_llm = llm

    def advisor_node(state: State) -> dict:
        system = SystemMessage(
            content=render_system_prompt(
                profile=state.get("profile") or {},
                retrieved=state.get("retrieved") or [],
            )
        )
        response = bound_llm.invoke([system, *_non_system_messages(state["messages"])])
        _log_answer_trace(response)
        return {"messages": [response]}

    return advisor_node
