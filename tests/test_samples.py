import sqlite3

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from trail_buddy.graph import build_graph
from trail_buddy.nodes import _doc_summary, _trace_summary
from trail_buddy.prompts import render_system_prompt
from trail_buddy.retrieval import RetrievalTrace, format_retrieved_docs
from trail_buddy.retrieval.config import (
    PROJECT_ROOT,
    RetrievalSettings,
    available_collection_names,
    available_raw_document_paths,
    get_retrieval_settings,
)
from trail_buddy.retrieval.service import _bm25_search, _reciprocal_rank_fusion


def _invoke(graph, text: str, thread: str):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread}},
    )


def test_graph_builds_and_responds_with_fake_llm():
    llm = FakeListChatModel(responses=["Sure — what's the race name and your half marathon PR?"])
    graph = build_graph(llm=llm, retriever=lambda query: [])

    result = _invoke(
        graph,
        "I want to run a trail. Can I cope with 35 km / 2000 D+?",
        thread="t1",
    )

    last = result["messages"][-1]
    assert "race name" in last.content.lower()


def test_multi_turn_uses_checkpointer():
    llm = FakeListChatModel(responses=["Which race?", "Roughly 6:00–7:00 hours."])
    graph = build_graph(llm=llm, retriever=lambda query: [])

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


def test_system_prompt_avoids_source_citations_in_answers():
    """The prompt should use retrieved context without surfacing source citations."""
    rendered = render_system_prompt(retrieved=["Some retrieved chunk."])
    lowered = rendered.lower()
    assert "retrieved context" in lowered
    assert "do not add" in lowered
    assert "source citations" in lowered
    assert "urls" in lowered
    assert "cite the source" not in lowered
    assert "source (url" not in lowered


def test_system_prompt_uses_search_for_location_specific_route_recommendations():
    rendered = render_system_prompt()
    lowered = rendered.lower()
    assert "local trail/route" in lowered
    assert "specific" in lowered
    assert "country" in lowered
    assert "mountain range" in lowered
    assert "must call `tavily_search` before recommending" in lowered
    assert "where should i train in durmitor" in lowered
    assert "do not answer from" in lowered
    assert "route discovery" in lowered


def test_system_prompt_requires_weather_tool_for_exact_date_conditions():
    rendered = render_system_prompt()
    lowered = rendered.lower()
    normalized = " ".join(lowered.split())
    assert "specific place and exact date" in normalized
    assert "you must call `trail_weather_search`" in normalized
    assert "a month or season alone" in normalized
    assert "is not a planned date" in normalized


def test_advisor_answers_without_source_links_when_retrieval_present():
    llm = FakeListChatModel(responses=["Use poles on steep climbs."])
    graph = build_graph(
        llm=llm,
        retriever=lambda _query: [
            "[1] Title: Trail poles guide\n"
            "Source: https://example.test/poles\n"
            "Collection: trail_buddy_irunfar\n"
            "Chunk ID: chunk-1\n"
            "Plant poles below the line on steep descents."
        ],
        tools=[],
    )
    result = _invoke(graph, "How to choose trail poles?", thread="no-cite")
    answer = result["messages"][-1].content
    assert answer == "Use poles on steep climbs."
    assert "https://example.test/poles" not in answer
    assert "Source:" not in answer


def test_state_has_messages_after_invocation():
    llm = FakeListChatModel(responses=["Wash them."])
    graph = build_graph(llm=llm, retriever=lambda query: [])
    result = _invoke(graph, "А кроссовки надо стирать?", thread="ru")
    assert result["messages"][-1].content == "Wash them."
    assert result["retrieved"] == []


def test_graph_populates_retrieved_context_from_retriever():
    llm = FakeListChatModel(responses=["Use poles on steep climbs."])
    graph = build_graph(
        llm=llm,
        retriever=lambda query: [f"Retrieved context for: {query}"],
    )

    result = _invoke(graph, "How to choose trail poles?", thread="rag")

    assert result["retrieved"] == ["Retrieved context for: How to choose trail poles?"]
    assert result["messages"][-1].content == "Use poles on steep climbs."


def test_graph_continues_when_retriever_fails(caplog):
    def failing_retriever(_query):
        raise RuntimeError("embedding model unavailable")

    llm = FakeListChatModel(responses=["Answer without retrieved context."])
    graph = build_graph(llm=llm, retriever=failing_retriever)

    with caplog.at_level("INFO", logger="trail_buddy.nodes"):
        result = _invoke(graph, "How to choose trail poles?", thread="rag-error")

    assert result["retrieved"] == []
    assert result["messages"][-1].content == "Answer without retrieved context."
    assert "[RAG] retrieval_error" in caplog.text
    assert "RuntimeError: embedding model unavailable" in caplog.text
    assert "[RAG] retrievers=custom | fusion=unknown" in caplog.text
    assert "[RAG] retrieved_docs: 0" in caplog.text


def test_doc_summary_uses_document_text_not_collection_metadata():
    doc = Document(
        page_content="Actual article text starts here.",
        metadata={
            "title": "Trail poles guide",
            "source": "https://example.test/poles",
            "collection": "trail_buddy_irunfar",
            "chunk_id": "chunk-1",
        },
    )

    summary = _doc_summary(doc)

    assert "collection=trail_buddy_irunfar" in summary
    assert "title=Trail poles guide" in summary
    assert "source=https://example.test/poles" in summary
    assert "text=Actual article text starts here." in summary
    assert "text=Collection:" not in summary


def test_trace_summary_logs_vector_only_retriever():
    trace = RetrievalTrace(
        retrievers=["vector"],
        fusion=None,
        rrf_rank_constant=None,
        vector_k=5,
        bm25_k=None,
        collections=["trail_buddy_irunfar"],
    )

    assert _trace_summary(trace) == (
        "retrievers=vector | fusion=none | vector_k=5 | "
        "collections=trail_buddy_irunfar"
    )


def test_trace_summary_logs_bm25_and_rrf():
    trace = RetrievalTrace(
        retrievers=["vector", "bm25"],
        fusion="rrf",
        rrf_rank_constant=60,
        vector_k=10,
        bm25_k=10,
        collections=["trail_buddy_irunfar", "trail_buddy_gear"],
    )

    assert _trace_summary(trace) == (
        "retrievers=vector,bm25 | fusion=rrf | vector_k=10 | "
        "rrf_rank_constant=60 | bm25_k=10 | "
        "collections=trail_buddy_irunfar,trail_buddy_gear"
    )


def test_format_retrieved_docs_omits_source_metadata_from_prompt_context():
    doc = Document(
        page_content="Actual article text starts here.",
        metadata={
            "title": "Trail poles guide",
            "source": "https://example.test/poles",
            "collection": "trail_buddy_irunfar",
            "chunk_id": "chunk-1",
        },
    )

    formatted = format_retrieved_docs([doc])

    assert formatted == ["[1] Actual article text starts here."]
    assert "https://example.test/poles" not in formatted[0]
    assert "Source:" not in formatted[0]
    assert "Collection:" not in formatted[0]
    assert "Chunk ID:" not in formatted[0]


def test_retrieval_settings_resolve_external_store(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIL_BUDDY_RAG_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("TRAIL_BUDDY_RAG_COLLECTION", "test_collection")
    monkeypatch.setenv("TRAIL_BUDDY_RAG_EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("TRAIL_BUDDY_RAG_RETRIEVER_K", "7")
    monkeypatch.setenv("TRAIL_BUDDY_RAG_USE_BM25", "true")
    monkeypatch.setenv("TRAIL_BUDDY_RAG_BM25_K", "11")
    monkeypatch.setenv("TRAIL_BUDDY_RAG_RRF_RANK_CONSTANT", "42")

    settings = get_retrieval_settings()

    assert settings.rag_store_dir == tmp_path
    assert settings.raw_data_dir == tmp_path / "data" / "raw"
    assert settings.processed_data_dir == tmp_path / "data" / "processed"
    assert settings.chroma_dir == tmp_path / "indexes" / "chroma"
    assert settings.document_paths == []
    assert settings.collection_name == "test_collection"
    assert settings.embedding_model == "test-embedding"
    assert settings.retriever_k == 7
    assert settings.use_bm25 is True
    assert settings.bm25_k == 11
    assert settings.rrf_rank_constant == 42


def test_bm25_search_ranks_keyword_matches():
    documents = [
        Document(page_content="hydration vest bottles mountain race", metadata={"id": "a"}),
        Document(page_content="road shoes track intervals", metadata={"id": "b"}),
        Document(page_content="mountain race poles steep climb", metadata={"id": "c"}),
    ]

    results = _bm25_search("mountain race poles", documents, k=2)

    assert [doc.metadata["id"] for doc in results] == ["c", "a"]


def test_reciprocal_rank_fusion_merges_duplicate_docs():
    vector_only = Document(page_content="vector only", metadata={"id": "vector"})
    shared = Document(page_content="shared", metadata={"id": "shared"})
    bm25_only = Document(page_content="bm25 only", metadata={"id": "bm25"})

    results = _reciprocal_rank_fusion(
        [[vector_only, shared], [shared, bm25_only]],
        top_n=2,
        rank_constant=60,
    )

    assert [doc.metadata["id"] for doc in results] == ["shared", "vector"]


def test_relative_rag_store_path_resolves_from_project_root(monkeypatch):
    monkeypatch.setenv("TRAIL_BUDDY_RAG_STORE_DIR", "rag_store_for_tests")

    settings = get_retrieval_settings()

    assert settings.rag_store_dir == PROJECT_ROOT / "rag_store_for_tests"


def test_collection_names_can_be_read_from_rag_store(monkeypatch):
    monkeypatch.delenv("TRAIL_BUDDY_RAG_COLLECTION", raising=False)

    settings = get_retrieval_settings()
    names = available_collection_names(settings.rag_store_dir)
    if not names:
        pytest.skip(f"No Chroma collections found in configured RAG store: {settings.rag_store_dir}")

    assert settings.collection_name is None
    assert names


def test_single_named_chroma_index_is_discovered(tmp_path):
    chroma_dir = tmp_path / "indexes" / "irunfar_training"
    chroma_dir.mkdir(parents=True)
    with sqlite3.connect(chroma_dir / "chroma.sqlite3") as connection:
        connection.execute("create table collections (name text)")
        connection.execute("insert into collections values ('irunfar_training')")

    settings = RetrievalSettings(rag_store_dir=tmp_path)

    assert settings.chroma_dir == chroma_dir
    assert available_collection_names(settings.rag_store_dir, settings.chroma_dir) == [
        "irunfar_training"
    ]


def test_raw_document_paths_can_be_read_from_rag_store():
    settings = get_retrieval_settings()
    paths = available_raw_document_paths(settings.rag_store_dir)
    if not paths:
        pytest.skip(f"No raw JSONL documents found in configured RAG store: {settings.rag_store_dir}")

    assert settings.document_paths == paths
