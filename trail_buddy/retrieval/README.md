# Retrieval

This package contains Trail Buddy's local RAG reader. The chat app does not scrape
or build indexes; it reads a user-provided Chroma store and returns formatted
context to `retrieve_node` in `trail_buddy.nodes`.

## External RAG Store

Scraping and index refreshes stay outside the chat app. The app reads from one
external store root:

```text
TRAIL_BUDDY_RAG_STORE_DIR/
  data/
    raw/
      *.jsonl
    processed/
  indexes/
    chroma/
      chroma.sqlite3
      ...
```

Configure it with:

```bash
TRAIL_BUDDY_RAG_STORE_DIR=rag_store
TRAIL_BUDDY_RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
TRAIL_BUDDY_RAG_RETRIEVER_K=5
```

`TRAIL_BUDDY_RAG_STORE_DIR` defaults to `./rag_store` inside the project root.
The app uses collection names discovered in `rag_store/indexes/chroma/chroma.sqlite3`.
All raw `*.jsonl` files under `rag_store/data/raw/` are discovered by path; the
app does not require a specific filename.

## Check RAG

```bash
.venv/bin/python - <<'PY'
from trail_buddy.retrieval import retrieve_context

results = retrieve_context("How should I choose trail running poles?")
print(len(results))
print(results[0] if results else "No retrieved context")
PY
```

In the graph, inspect `result["retrieved"]`; non-empty means retrieved context
was added before the LLM answered. If the store, embedding model, or Chroma query
fails, the graph logs the retrieval error and answers without retrieved context.

Working assumptions for the next iteration:

- **Embeddings**: keep `all-MiniLM-L6-v2` while using the copied/existing Chroma
  index. Rebuild into a separate collection before switching to `BAAI/bge-m3`.
- **Vector store**: Chroma (file-backed, zero-ops) for the course project. Swap to
  Qdrant or pgvector if the corpus grows.
- **Sources**: race databases (UTMB index, ITRA), gear catalogues, training and
  sports-medicine references.
- **Routing**: skip retrieval for chit-chat / pure-opinion questions via a cheap
  classifier or keyword pre-filter — RAG is only useful when grounded facts beat
  parametric knowledge.
