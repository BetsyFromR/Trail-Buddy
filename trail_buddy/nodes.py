import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from trail_buddy.prompts import render_system_prompt
from trail_buddy.retrieval import (
    RetrievalTrace,
    RetrievalUnavailable,
    format_retrieved_docs,
    retrieve_with_trace,
)
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


def _doc_summary(doc: Document) -> str:
    title = doc.metadata.get("title") or "Untitled"
    source = doc.metadata.get("url") or doc.metadata.get("source") or "unknown"
    collection = doc.metadata.get("collection") or "unknown"
    chunk_id = (
        getattr(doc, "id", None)
        or doc.metadata.get("id")
        or doc.metadata.get("chunk_id")
        or "unknown"
    )
    body = " ".join(doc.page_content.split())
    preview = body[:TEXT_PREVIEW_CHARS]
    if len(body) > TEXT_PREVIEW_CHARS:
        preview += "..."
    return (
        f"chunk_id={chunk_id} | collection={collection} | title={title} | "
        f"source={source} | text={preview}"
    )


def _trace_summary(trace: RetrievalTrace | None) -> str:
    if trace is None:
        return "retrievers=unknown | fusion=unknown"

    parts = [
        f"retrievers={','.join(trace.retrievers)}",
        f"fusion={trace.fusion or 'none'}",
        f"vector_k={trace.vector_k}",
        f"collections={','.join(trace.collections) or 'none'}",
    ]
    if trace.bm25_k is not None:
        parts.insert(3, f"bm25_k={trace.bm25_k}")
    if trace.rrf_rank_constant is not None:
        parts.insert(3, f"rrf_rank_constant={trace.rrf_rank_constant}")
    return " | ".join(parts)


def _log_retrieval_trace(
    query: str,
    docs: list[Document],
    trace: RetrievalTrace | None = None,
) -> None:
    logger.info("[RAG] query: %s", query)
    logger.info("[RAG] %s", _trace_summary(trace))
    logger.info("[RAG] retrieved_docs: %s", len(docs))
    for index, doc in enumerate(docs, start=1):
        logger.info("[RAG] doc %s: %s", index, _doc_summary(doc))


def _log_legacy_retrieval_trace(query: str, retrieved: list[str]) -> None:
    logger.info("[RAG] query: %s", query)
    logger.info("[RAG] retrievers=custom | fusion=unknown")
    logger.info("[RAG] retrieved_docs: %s", len(retrieved))


def _log_failed_retrieval_trace(query: str, retriever) -> None:
    if retriever is not None:
        _log_legacy_retrieval_trace(query, [])
        return

    _log_retrieval_trace(query, [])


def _log_answer_trace(response) -> None:
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        requested = ", ".join(
            f"{call.get('name', 'unknown')}({call.get('args', {})})"
            for call in tool_calls
        )
        logger.info("[tools] requested: %s", requested)
        return

    answer = _message_text(getattr(response, "content", response)).strip()
    logger.info("[RAG] answer: %s", answer)


def make_retrieve_node(retriever=None):
    def _retrieve_node(state: State) -> dict:
        query = _latest_user_text(state)
        if not query.strip():
            _log_retrieval_trace(query, [])
            return {"retrieved": []}

        try:
            if retriever is not None:
                retrieved = retriever(query)
                _log_legacy_retrieval_trace(query, retrieved)
                return {"retrieved": retrieved}

            docs, trace = retrieve_with_trace(query)
            retrieved = format_retrieved_docs(docs)
            _log_retrieval_trace(query, docs, trace)
            return {"retrieved": retrieved}
        except RetrievalUnavailable as exc:
            logger.warning("[RAG] retrieval_unavailable: %s", exc)
            _log_failed_retrieval_trace(query, retriever)
            return {"retrieved": []}
        except Exception:
            logger.exception("[RAG] retrieval_error")
            _log_failed_retrieval_trace(query, retriever)
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
