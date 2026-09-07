"""Document ingestion: parse to raw text + persist with a stable id.

Robustness rules applied:
* atomic temp-file + os.replace for every persisted file (raw blob, extracted
  text, metadata index) — an interrupted ingest never leaves a partial corpus
  entry;
* document id = sha256 of the original bytes, so re-ingesting the same file is
  idempotent instead of duplicating under a fresh counter/uuid;
* unsupported type and corrupt file raise distinct, explicit errors — a
  document is never silently dropped;
* a poisoned metadata index is discarded on read rather than wedging ingest.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst"}
_PDF_EXTS = {".pdf"}
_SUPPORTED = _TEXT_EXTS | _PDF_EXTS
_DOC_ID_LEN = 16


class UnsupportedTypeError(ValueError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            f"unsupported document type: {filename!r} (supported: {sorted(_SUPPORTED)})"
        )
        self.filename = filename


class CorruptDocumentError(ValueError):
    """File matched a supported type but could not be parsed to text."""


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    content_type: str  # "pdf" | "text"
    size_bytes: int
    content_sha256: str
    char_count: int
    ingested_at: str
    raw_path: str
    text_path: str


def _ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _extract_text(filename: str, content: bytes) -> tuple[str, str]:
    ext = _ext(filename)
    if ext in _TEXT_EXTS:
        try:
            return content.decode("utf-8"), "text"
        except UnicodeDecodeError as exc:
            raise CorruptDocumentError(f"{filename!r} is not valid UTF-8 text") from exc
    if ext in _PDF_EXTS:
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
        except (PyPdfError, ValueError, OSError) as exc:
            raise CorruptDocumentError(f"{filename!r} is not a readable PDF") from exc
        return "\n\n".join(pages), "pdf"
    raise UnsupportedTypeError(filename)


def _atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class IngestStore:
    def __init__(self, storage_dir: Path, max_upload_bytes: int) -> None:
        self.storage_dir = Path(storage_dir)
        self.raw_dir = self.storage_dir / "raw"
        self.text_dir = self.storage_dir / "text"
        self.index_path = self.storage_dir / "index.json"
        self.max_upload_bytes = max_upload_bytes

    def _read_index(self) -> dict[str, dict]:
        if not self.index_path.is_file():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_index(self, index: dict[str, dict]) -> None:
        _atomic_write(
            self.index_path,
            json.dumps(index, indent=2, sort_keys=True).encode("utf-8"),
        )

    def get(self, document_id: str) -> DocumentRecord | None:
        rec = self._read_index().get(document_id)
        return DocumentRecord(**rec) if rec else None

    def list_documents(self) -> list[DocumentRecord]:
        return [DocumentRecord(**r) for r in self._read_index().values()]

    def read_text(self, document_id: str) -> str | None:
        rec = self.get(document_id)
        if rec is None:
            return None
        return Path(rec.text_path).read_text(encoding="utf-8")

    def ingest(self, filename: str, content: bytes) -> DocumentRecord:
        if not content:
            raise CorruptDocumentError(f"{filename!r} is empty")
        if len(content) > self.max_upload_bytes:
            raise ValueError(
                f"{filename!r} is {len(content)} bytes; limit is {self.max_upload_bytes}"
            )
        if _ext(filename) not in _SUPPORTED:
            raise UnsupportedTypeError(filename)

        text, content_type = _extract_text(filename, content)

        sha = hashlib.sha256(content).hexdigest()
        document_id = sha[:_DOC_ID_LEN]
        index = self._read_index()
        existing = index.get(document_id)
        raw_path = self.raw_dir / f"{document_id}{_ext(filename)}"
        text_path = self.text_dir / f"{document_id}.txt"
        if existing is not None and raw_path.is_file() and text_path.is_file():
            return DocumentRecord(**existing)

        _atomic_write(raw_path, content)
        _atomic_write(text_path, text.encode("utf-8"))

        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            content_sha256=sha,
            char_count=len(text),
            ingested_at=datetime.now(timezone.utc).isoformat(),
            raw_path=str(raw_path),
            text_path=str(text_path),
        )
        index[document_id] = asdict(record)
        self._write_index(index)
        return record
