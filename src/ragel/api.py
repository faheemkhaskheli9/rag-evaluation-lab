"""FastAPI ingestion surface (Phase 1)."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .config import Settings, load_settings
from .ingest import CorruptDocumentError, IngestStore, UnsupportedTypeError

app = FastAPI(title="RAG Evaluation Lab", version="0.1.0")


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return load_settings()


def get_store(settings: Settings = Depends(_settings)) -> IngestStore:
    return IngestStore(settings.storage_dir, settings.max_upload_bytes)


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
