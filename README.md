# RAG Regression Testing Platform

> LLM, RAG & Agentic AI portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-phase%201%20in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

RAG pipelines silently degrade when chunking, embeddings, or retrieval settings change. Teams need a lab to measure retrieval and answer quality objectively before shipping changes.

## 2. Architecture

```text
Docs -> Chunking -> Embedding -> Vector Store -> Retrieval -> Answer Gen -> Metrics vs Ground Truth
```

## 3. Technology Stack

- Python
- LangChain or LlamaIndex
- FAISS / Chroma
- OpenAI API
- FastAPI

## 4. Feature List

- Document upload and ingestion
- Automatic test-question generation from documents
- RAG query execution against configurable pipelines
- Expected vs. generated answer comparison
- Retrieval quality metrics (recall/precision @k)
- Answer quality scoring
- Embedding model comparison
- Chunking strategy comparison
- Vector store comparison (FAISS vs Chroma, etc.)

## 5. Implementation Plan

1. Phase 1: Document ingestion + auto question/answer-pair generation
2. Phase 2: Configurable pipeline (chunker, embedder, vector store) as swappable modules
3. Phase 3: Retrieval metrics (hit rate, MRR) and answer-quality scoring
4. Phase 4: Comparison dashboard across pipeline configurations

## Task Tracking

Work is broken into phase-tagged user stories tracked as GitHub Issues, not in this file. To see what's open:

```bash
gh issue list --repo faheemkhaskheli9/rag-evaluation-lab --state open --label type:user-story
```

Implement Phase 1 issues first (later phases depend on it). When you start one, add label `status:in-progress`. When you finish, close it referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## Phase 1 quickstart

```bash
pip install -r requirements.txt

# HTTP API
PYTHONPATH=src uvicorn ragel.api:app --reload
curl -s -F 'file=@examples/sample.txt' localhost:8000/documents

# CLI
PYTHONPATH=src python -m ragel.cli ingest examples/sample.txt
PYTHONPATH=src python -m ragel.cli list
```

VS Code: **RAGEL: FastAPI (uvicorn)**, **RAGEL: CLI (ingest)**, **RAGEL: pytest**
in `.vscode/launch.json`.

## 6. Repository Structure

```text
rag-evaluation-lab/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd rag-evaluation-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Document which public dataset(s) or synthetic data generators are used here.
No proprietary, employer-owned, or client-identifiable data is used in this project.

## 9. Training / Execution

Document the commands used to run training, ingestion, or the main pipeline, e.g.:

```bash
python -m src.main --config configs/default.yaml
```

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see `docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics tables, and
sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t rag-evaluation-lab .
docker run -p 8000:8000 rag-evaluation-lab
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.

## 16. Future Work

- Expand evaluation coverage and add CI-based regression checks.
- Add more configuration presets and deployment targets.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-08-18_
