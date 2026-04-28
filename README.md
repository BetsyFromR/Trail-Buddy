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
  nodes.py        retrieve_node (RAG stub) + advisor_node
  graph.py        build_graph() — START → retrieve → advisor → END, with MemorySaver
  retrieval/      placeholder package for future RAG
app.py            Gradio ChatInterface
tests/            pytest smoke tests
```

## Roadmap — RAG

`trail_buddy/retrieval/` is the seam. Working assumptions:

- Embeddings: `BAAI/bge-m3` (multilingual, local).
- Vector store: Chroma (file-backed, zero-ops).
- Sources: race databases (UTMB / ITRA), gear catalogues, sports-medicine references.
- Skip retrieval for chit-chat and pure-opinion questions.
