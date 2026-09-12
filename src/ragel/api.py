"""FastAPI ingestion surface (Phase 1)."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .config import Settings, load_settings
from .ingest import CorruptDocumentError, IngestStore, UnsupportedTypeError
from .qa_generation import QAPair, QAStore, generate_qa_pairs

app = FastAPI(title="RAG Evaluation Lab", version="0.1.0")


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return load_settings()


def get_store(settings: Settings = Depends(_settings)) -> IngestStore:
    return IngestStore(settings.storage_dir, settings.max_upload_bytes)


def get_qa_store(settings: Settings = Depends(_settings)) -> QAStore:
    return QAStore(settings.storage_dir)


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    content_sha256: str
    char_count: int
    ingested_at: str
    raw_path: str
    text_path: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, status_code=201)
async def ingest_document(
    file: UploadFile, store: IngestStore = Depends(get_store)
) -> DocumentOut:
    content = await file.read()
    try:
        record = store.ingest(file.filename or "upload", content)
    except UnsupportedTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except CorruptDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:  # size limit
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return DocumentOut(**record.__dict__)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(store: IngestStore = Depends(get_store)) -> list[DocumentOut]:
    return [DocumentOut(**r.__dict__) for r in store.list_documents()]


@app.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(
    document_id: str, store: IngestStore = Depends(get_store)
) -> DocumentOut:
    record = store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentOut(**record.__dict__)


@app.get("/documents/{document_id}/text", response_class=PlainTextResponse)
def get_document_text(
    document_id: str, store: IngestStore = Depends(get_store)
) -> str:
    text = store.read_text(document_id)
    if text is None:
        raise HTTPException(status_code=404, detail="document not found")
    return text


class QAPairOut(BaseModel):
    pair_id: str
    document_id: str
    chunk_index: int
    source_passage: str
    question: str
    answer: str
    generated_at: str


class GenerateQARequest(BaseModel):
    max_pairs: int | None = None


@app.post("/documents/{document_id}/qa-pairs", response_model=list[QAPairOut])
def generate_document_qa_pairs(
    document_id: str,
    request: GenerateQARequest | None = None,
    store: IngestStore = Depends(get_store),
    qa_store: QAStore = Depends(get_qa_store),
) -> list[QAPairOut]:
    text = store.read_text(document_id)
    if text is None:
        raise HTTPException(status_code=404, detail="document not found")
    max_pairs = request.max_pairs if request is not None else None
    pairs: list[QAPair] = generate_qa_pairs(document_id, text, max_pairs=max_pairs)
    merged = qa_store.upsert_pairs(document_id, pairs)
    return [QAPairOut(**p.__dict__) for p in merged]


@app.get("/documents/{document_id}/qa-pairs", response_model=list[QAPairOut])
def list_document_qa_pairs(
    document_id: str, qa_store: QAStore = Depends(get_qa_store)
) -> list[QAPairOut]:
    return [QAPairOut(**p.__dict__) for p in qa_store.get_pairs(document_id)]


@app.get("/qa-pairs/{pair_id}", response_model=QAPairOut)
def get_qa_pair(pair_id: str, qa_store: QAStore = Depends(get_qa_store)) -> QAPairOut:
    """Traceability lookup (issue #3): the originating passage for any
    given Q/A pair, without the caller already knowing its document id."""
    pair = qa_store.get_pair(pair_id)
    if pair is None:
        raise HTTPException(status_code=404, detail="Q/A pair not found")
    return QAPairOut(**pair.__dict__)
