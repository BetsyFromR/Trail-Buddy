# Trail Buddy

Conversational trail-running assistant. LangGraph + LiteLLM + Gradio. Final project for "Intro to AI Agents".

## What it does

Answers questions about race readiness, gear, training, recovery, health concerns, and race-day logistics. Multilingual (EN / RU / SR). Asks one clarifying question when the answer materially depends on missing facts (race name, fitness baseline, etc.).

## Setup

```bash
uv sync
cp .env.example .env
# fill in TRAIL_BUDDY_MODEL and the matching provider API key
```

`TRAIL_BUDDY_MODEL` is a LiteLLM model id — `anthropic/claude-sonnet-4-5`, `openai/gpt-4.1`, `ollama/qwen2.5:32b`, etc. Switching providers needs no code change.

RAG settings live in `rag_config.toml`. Set `store_dir` to the store root that
contains `data/raw/`, `data/processed/`, and `indexes/chroma/`. The default is
`rag_store` inside this project. Env vars such as `TRAIL_BUDDY_RAG_RETRIEVER_K`
still override the file when needed.
Set `use_bm25 = true` to fuse BM25 matches over the stored Chroma documents
with vector matches using Reciprocal Rank Fusion.

Logs are written to `logs/trail_buddy.log` and `logs/trail_buddy.error.log`.
Override the location or verbosity with `TRAIL_BUDDY_LOG_DIR` and
`TRAIL_BUDDY_LOG_LEVEL`.

## Run

```bash
uv run python app.py        # Gradio UI on http://127.0.0.1:7860
uv run pytest               # smoke tests (use a fake LLM, no API key needed)
```

## Layout

```
trail_buddy/
  state.py        State schema (messages, profile, retrieved)
  prompts.py      System prompt — clarification policy, language rule, health caveats
  llm.py          ChatLiteLLM factory (provider chosen via env)
  nodes.py        retrieve_node (local RAG lookup) + advisor_node
  graph.py        build_graph() — START → retrieve → advisor → END, with MemorySaver
  retrieval/      RAG path/config seam for external docs and indexes
app.py            Gradio ChatInterface
tests/            pytest smoke tests
```

## RAG

`trail_buddy/retrieval/` reads RAG settings from `rag_config.toml`, then reads a
user-provided Chroma store from the configured `store_dir`. If retrieval is
unavailable, the graph logs the failure and answers without retrieved context.

Current assumptions:

- Embeddings: `all-MiniLM-L6-v2` for the existing copied index; rebuild a new
  collection before switching to `BAAI/bge-m3`.
- Vector store: Chroma (file-backed, zero-ops).
- Sources: race databases (UTMB / ITRA), gear catalogues, sports-medicine references.
- Skip retrieval for chit-chat and pure-opinion questions.
