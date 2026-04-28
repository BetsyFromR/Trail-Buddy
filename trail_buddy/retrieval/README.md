# Retrieval (planned)

This package is the seam for adding RAG to Trail Buddy. The MVP uses a no-op
`retrieve_node` in `trail_buddy.nodes`; once retrieval lands, replace that stub with
calls into this package.

Working assumptions for the next iteration:

- **Embeddings**: `BAAI/bge-m3` via `sentence-transformers` or `fastembed` —
  multilingual, runs locally, no provider lock-in.
- **Vector store**: Chroma (file-backed, zero-ops) for the course project. Swap to
  Qdrant or pgvector if the corpus grows.
- **Sources**: race databases (UTMB index, ITRA), gear catalogues, training and
  sports-medicine references.
- **Routing**: skip retrieval for chit-chat / pure-opinion questions via a cheap
  classifier or keyword pre-filter — RAG is only useful when grounded facts beat
  parametric knowledge.
