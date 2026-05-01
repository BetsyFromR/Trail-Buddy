from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from trail_buddy.retrieval.config import (
    RetrievalSettings,
    available_collection_names,
    get_retrieval_settings,
)


class RetrievalUnavailable(RuntimeError):
    """Raised when the configured local RAG store cannot be queried."""


def _load_embedding_model(model_name: str) -> Any:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RetrievalUnavailable(
            "Install langchain-huggingface and sentence-transformers to use RAG."
        ) from exc

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=16)
def _cached_vectorstore(
    chroma_dir: str,
    collection_name: str,
    embedding_model: str,
) -> Any:
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RetrievalUnavailable("Install langchain-chroma and chromadb to use RAG.") from exc

    persist_directory = Path(chroma_dir)
    if not (persist_directory / "chroma.sqlite3").exists():
        raise RetrievalUnavailable(f"Chroma database not found at {persist_directory}.")

    embeddings = _load_embedding_model(embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    return vectorstore


def _format_doc(doc: Document, index: int) -> str:
    title = doc.metadata.get("title") or "Untitled"
    source = doc.metadata.get("url") or doc.metadata.get("source") or "unknown"
    collection = doc.metadata.get("collection")
    chunk_id = getattr(doc, "id", None) or doc.metadata.get("id") or doc.metadata.get("chunk_id")
    text = " ".join(doc.page_content.split())
    parts = [
        f"[{index}] Title: {title}",
        f"Source: {source}",
    ]
    if collection:
        parts.append(f"Collection: {collection}")
    if chunk_id:
        parts.append(f"Chunk ID: {chunk_id}")
    parts.append(text)
    return "\n".join(parts)


def format_retrieved_docs(docs: list[Document]) -> list[str]:
    return [_format_doc(doc, index) for index, doc in enumerate(docs, start=1)]


def retrieve_documents(
    query: str,
    settings: RetrievalSettings | None = None,
) -> list[Document]:
    resolved = settings or get_retrieval_settings()
    collection_names = (
        [resolved.collection_name]
        if resolved.collection_name
        else available_collection_names(resolved.rag_store_dir)
    )
    if not collection_names:
        raise RetrievalUnavailable("No Chroma collections found in the configured RAG store.")

    scored_docs: list[tuple[Document, float]] = []
    for collection_name in collection_names:
        vectorstore = _cached_vectorstore(
            str(resolved.chroma_dir),
            collection_name,
            resolved.embedding_model,
        )
        for doc, score in vectorstore.similarity_search_with_score(
            query,
            k=resolved.retriever_k,
        ):
            doc.metadata.setdefault("collection", collection_name)
            scored_docs.append((doc, score))

    scored_docs.sort(key=lambda item: item[1])
    return [doc for doc, _score in scored_docs[:resolved.retriever_k]]


def retrieve_context(
    query: str,
    settings: RetrievalSettings | None = None,
) -> list[str]:
    if not query.strip():
        return []
    return format_retrieved_docs(retrieve_documents(query, settings=settings))
