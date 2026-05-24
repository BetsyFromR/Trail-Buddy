# Trail Buddy

Conversational trail-running assistant. LangGraph + LiteLLM + Gradio, with local RAG and optional tool use. Final project for "Intro to AI Agents".

## What it does

Answers questions about race readiness, gear, training, recovery, health concerns, weather, current race information, and race-day logistics. Multilingual (EN / RU / SR). Asks one clarifying question when the answer materially depends on missing facts (race name, fitness baseline, date, location, etc.).

The graph always tries local RAG first when a user message is present. The advisor can then call tools for:

- weather and historical climatology via Open-Meteo;
- rough flat-kilometer effort estimates from distance and elevation gain;
- optional public-web search via Tavily when `TAVILY_API_KEY` is set.

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

The weather tool uses Open-Meteo and does not need an API key. Optional weather
settings are in `.env.example` under `TRAIL_BUDDY_WEATHER_*`.

The public-web search tool is enabled only when `TAVILY_API_KEY` is set. Without
that key, Trail Buddy starts normally and runs without web search.

Logs are written to `logs/trail_buddy.log` and `logs/trail_buddy.error.log`.
Override the location or verbosity with `TRAIL_BUDDY_LOG_DIR` and
`TRAIL_BUDDY_LOG_LEVEL`.

## Run

```bash
uv run python app.py        # Gradio UI with a temporary public *.gradio.live URL
uv run pytest               # smoke tests (use a fake LLM, no API key needed)
```

## Evaluation

Evaluation settings live in `evaluation/eval_config.toml`. Run the offline
evaluation pipeline with:

```bash
uv run python -m evaluation.run
uv run python -m evaluation.run --limit 5
uv run python -m evaluation.run --include-ungraded
uv run python -m evaluation.run --output evaluation/results/test.jsonl
```

The runner sends each English dataset query through the real Trail Buddy graph,
then makes two structured judge calls: one for Truthfulness and one for
Completeness. Results are written as JSONL plus a summary JSON file in
`evaluation/results/`.

Evaluation model selection is separate from the app `.env` model setting:
`agent_model` controls the Trail Buddy agent being evaluated, and `judge_model`
controls the judge. `TRAIL_BUDDY_MODEL` is for the app runtime. Provider API
keys are still read from the environment or `.env` in the standard LiteLLM
variables such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

If the provider returns `429 Too Many Requests`, increase
`request_delay_seconds` or `retry_backoff_seconds` in `evaluation/eval_config.toml`.

## Layout

```
trail_buddy/
  state.py        State schema (messages, profile, retrieved)
  prompts.py      System prompt — clarification policy, language rule, health caveats
  llm.py          ChatLiteLLM factory (provider chosen via env)
  nodes.py        retrieve_node (local RAG lookup) + advisor_node
  graph.py        build_graph() — START → retrieve → advisor, optional tools loop, MemorySaver
  retrieval/      RAG path/config seam for external docs and indexes
  effort/         flat-kilometer effort estimation tool
  weather/        Open-Meteo geocoding, forecast, and historical climatology tool
  web_search/     optional Tavily-backed public-web search tool
evaluation/       offline evaluation runner, prompts, dataset loading, result models
app.py            Gradio ChatInterface
tests/            pytest smoke tests
```

## RAG

`trail_buddy/retrieval/` reads RAG settings from `rag_config.toml`, then reads a
user-provided Chroma store from the configured `store_dir`. If retrieval is
unavailable, the graph logs the failure and answers without retrieved context.
Retrieval is currently attempted for every non-empty user message; there is no
classifier that skips chit-chat or pure-opinion questions yet.

Current assumptions:

- Embeddings: `all-MiniLM-L6-v2` for the existing copied index; rebuild a new
  collection before switching to `BAAI/bge-m3`.
- Vector store: Chroma (file-backed, zero-ops).
- Sources: race databases (UTMB / ITRA), gear catalogues, sports-medicine references.
