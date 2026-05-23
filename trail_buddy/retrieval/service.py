from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RetrievalTrace:
    retrievers: list[str]
    fusion: str | None
    rrf_rank_constant: int | None
    vector_k: int
    bm25_k: int | None
    collections: list[str]


_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


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


def _stored_documents(vectorstore: Any, *, collection_name: str) -> list[Document]:
    result = vectorstore.get(include=["documents", "metadatas"])
    ids = result.get("ids", [])
    texts = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    documents: list[Document] = []
    for doc_id, text, metadata in zip(ids, texts, metadatas, strict=False):
        if not isinstance(text, str) or not text.strip():
            continue
        resolved_metadata = dict(metadata or {})
        resolved_metadata.setdefault("id", doc_id)
        resolved_metadata.setdefault("collection", collection_name)
        documents.append(
            Document(page_content=text, metadata=resolved_metadata)
        )
    return documents


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


def _bm25_search(query: str, documents: list[Document], *, k: int) -> list[Document]:
    if not query.strip() or not documents or k <= 0:
        return []

    query_terms = _tokens(query)
    if not query_terms:
        return []

    tokenized_documents = [_tokens(document.page_content) for document in documents]
    document_count = len(tokenized_documents)
    average_length = (
        sum(len(tokens) for tokens in tokenized_documents) / document_count
        if document_count
        else 0.0
    )
    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequencies.update(set(tokens))

    k1 = 1.5
    b = 0.75
    scored_docs: list[tuple[Document, float]] = []
    for document, tokens in zip(documents, tokenized_documents, strict=True):
        term_frequencies = Counter(tokens)
        doc_length = len(tokens)
        score = 0.0
        for term in query_terms:
            term_frequency = term_frequencies.get(term, 0)
            if not term_frequency:
                continue
            docs_with_term = document_frequencies[term]
            idf = math.log(
                1 + (document_count - docs_with_term + 0.5) / (docs_with_term + 0.5)
            )
            denominator = term_frequency + k1 * (
                1 - b + b * (doc_length / average_length if average_length else 0)
            )
            score += idf * (term_frequency * (k1 + 1)) / denominator
        if score > 0:
            scored_docs.append((document, score))

    scored_docs.sort(key=lambda item: item[1], reverse=True)
    return [document for document, _score in scored_docs[:k]]


def _doc_key(doc: Document) -> str:
    source = doc.metadata.get("url") or doc.metadata.get("source") or ""
    if doc.page_content:
        raw = f"{source}\n{doc.page_content}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    stable_id = (
        getattr(doc, "id", None)
        or doc.metadata.get("id")
        or doc.metadata.get("chunk_id")
    )
    if stable_id:
        return str(stable_id)

    raw = repr(doc.metadata)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _reciprocal_rank_fusion(
    ranked_doc_lists: list[list[Document]],
    *,
    top_n: int,
    rank_constant: int,
) -> list[Document]:
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}

    for docs in ranked_doc_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            docs_by_id.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + rank_constant)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs_by_id[key] for key, _score in ranked[:top_n]]


def _format_doc(doc: Document, index: int) -> str:
    text = " ".join(doc.page_content.split())
    return f"[{index}] {text}"


def format_retrieved_docs(docs: list[Document]) -> list[str]:
    return [_format_doc(doc, index) for index, doc in enumerate(docs, start=1)]


def retrieve_with_trace(
    query: str,
    settings: RetrievalSettings | None = None,
) -> tuple[list[Document], RetrievalTrace]:
    resolved = settings or get_retrieval_settings()
    collection_names = (
        [resolved.collection_name]
        if resolved.collection_name
        else available_collection_names(resolved.rag_store_dir, resolved.chroma_dir)
    )
    if not collection_names:
        raise RetrievalUnavailable(
            "No Chroma collections found in the configured RAG store."
        )

    vector_k = (
        max(resolved.retriever_k, resolved.bm25_k)
        if resolved.use_bm25
        else resolved.retriever_k
    )
    trace = RetrievalTrace(
        retrievers=["vector", "bm25"] if resolved.use_bm25 else ["vector"],
        fusion="rrf" if resolved.use_bm25 else None,
        rrf_rank_constant=resolved.rrf_rank_constant if resolved.use_bm25 else None,
        vector_k=vector_k,
        bm25_k=resolved.bm25_k if resolved.use_bm25 else None,
        collections=collection_names,
    )
    scored_docs: list[tuple[Document, float]] = []
    bm25_corpus: list[Document] = []
    for collection_name in collection_names:
        vectorstore = _cached_vectorstore(
            str(resolved.chroma_dir),
            collection_name,
            resolved.embedding_model,
        )
        for doc, score in vectorstore.similarity_search_with_score(
            query,
            k=vector_k,
        ):
            doc.metadata.setdefault("collection", collection_name)
            scored_docs.append((doc, score))
        if resolved.use_bm25:
            bm25_corpus.extend(
                _stored_documents(vectorstore, collection_name=collection_name)
            )

    scored_docs.sort(key=lambda item: item[1])
    vector_docs = [doc for doc, _score in scored_docs[:vector_k]]
    if not resolved.use_bm25:
        return vector_docs[:resolved.retriever_k], trace

    bm25_docs = _bm25_search(query, bm25_corpus, k=resolved.bm25_k)
    if not bm25_docs:
        return vector_docs[:resolved.retriever_k], trace

    return (
        _reciprocal_rank_fusion(
            [vector_docs, bm25_docs],
            top_n=resolved.retriever_k,
            rank_constant=resolved.rrf_rank_constant,
        ),
        trace,
    )


def retrieve_documents(
    query: str,
    settings: RetrievalSettings | None = None,
) -> list[Document]:
    docs, _trace = retrieve_with_trace(query, settings=settings)
    return docs


def retrieve_context(
    query: str,
    settings: RetrievalSettings | None = None,
) -> list[str]:
    if not query.strip():
        return []
    return format_retrieved_docs(retrieve_documents(query, settings=settings))
