# Architecture Notes: RAG Regression Testing Platform

## Pipeline

```text
Docs -> Chunking -> Embedding -> Vector Store -> Retrieval -> Answer Gen -> Metrics vs Ground Truth
```

## Components

- Document upload and ingestion
- Automatic test-question generation from documents
- RAG query execution against configurable pipelines
- Expected vs. generated answer comparison
- Retrieval quality metrics (recall/precision @k)
- Answer quality scoring
- Embedding model comparison
- Chunking strategy comparison
- Vector store comparison (FAISS vs Chroma, etc.)

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
