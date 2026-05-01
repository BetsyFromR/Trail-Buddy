from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from trail_buddy.prompts import render_system_prompt
from trail_buddy.retrieval import RetrievalUnavailable, retrieve_context
from trail_buddy.state import State


TEXT_PREVIEW_CHARS = 180


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


def _print_retrieval_trace(query: str, retrieved: list[str]) -> None:
    print(f"[RAG] query: {query}", flush=True)
    print(f"[RAG] retrieved_docs: {len(retrieved)}", flush=True)
    for index, context in enumerate(retrieved, start=1):
        print(f"[RAG] doc {index}: {_doc_summary(context)}", flush=True)


def _print_answer_trace(response) -> None:
    answer = _message_text(getattr(response, "content", response)).strip()
    print(f"[RAG] answer: {answer}", flush=True)


def make_retrieve_node(retriever=retrieve_context):
    def _retrieve_node(state: State) -> dict:
        query = _latest_user_text(state)
        if not query.strip():
            _print_retrieval_trace(query, [])
            return {"retrieved": []}

        try:
            retrieved = retriever(query)
            _print_retrieval_trace(query, retrieved)
            return {"retrieved": retrieved}
        except RetrievalUnavailable as exc:
            print(f"[RAG] retrieval_unavailable: {exc}", flush=True)
            _print_retrieval_trace(query, [])
            return {"retrieved": []}
        except Exception as exc:
            print(f"[RAG] retrieval_error: {type(exc).__name__}: {exc}", flush=True)
            _print_retrieval_trace(query, [])
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


def make_advisor_node(llm: BaseChatModel):
    def advisor_node(state: State) -> dict:
        system = SystemMessage(
            content=render_system_prompt(
                profile=state.get("profile") or {},
                retrieved=state.get("retrieved") or [],
            )
        )
        response = llm.invoke([system, *_non_system_messages(state["messages"])])
        _print_answer_trace(response)
        return {"messages": [response]}

    return advisor_node
