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

```toml
# rag_config.toml
[rag]
store_dir = "rag_store"
collection = ""
embedding_model = "all-MiniLM-L6-v2"
retriever_k = 5
use_bm25 = false
bm25_k = 10
rrf_rank_constant = 60
```

`store_dir` defaults to `./rag_store` inside the project root. Empty
`collection` means the app searches all collections discovered in
`rag_store/indexes/chroma/chroma.sqlite3`, merges their matches, and keeps the
best `retriever_k` results overall. Set `collection` only if you want to force a
single collection. All raw `*.jsonl` files under `rag_store/data/raw/` are
discovered by path; the app does not require a specific filename.

Set `use_bm25 = true` to also rank the stored Chroma documents with BM25. When
BM25 is enabled, the app fetches vector candidates and BM25 candidates, merges
duplicate documents, scores them with Reciprocal Rank Fusion, and returns the
top `retriever_k` results:

```text
doc_score = 1 / (rank_vector + rrf_rank_constant)
          + 1 / (rank_bm25 + rrf_rank_constant)
```

Env vars still override the file: `TRAIL_BUDDY_RAG_STORE_DIR`,
`TRAIL_BUDDY_RAG_COLLECTION`, `TRAIL_BUDDY_RAG_EMBEDDING_MODEL`, and
`TRAIL_BUDDY_RAG_RETRIEVER_K`. BM25 can be controlled with
`TRAIL_BUDDY_RAG_USE_BM25`, `TRAIL_BUDDY_RAG_BM25_K`, and
`TRAIL_BUDDY_RAG_RRF_RANK_CONSTANT`. Set `TRAIL_BUDDY_RAG_CONFIG_FILE` to use a
different config file.

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
