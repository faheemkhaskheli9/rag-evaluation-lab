"""Auto-generated question/answer ground-truth pairs from ingested documents.

Applies the knowledge-base "chunking-and-embedding-ingestion" pattern: a
pair's id is ``hash(document_id, chunk_index, passage_text)``, never a
counter, so re-running generation for an unchanged document upserts the same
ids instead of duplicating pairs, while an edited passage gets a new id
rather than silently overwriting stale text under a stale key.

Generation itself would call the OpenAI API per the README's tech stack;
this is a CPU-only sweep with no API key and no network budget, so
``QAGenerator`` is an interface with a deterministic offline
``MockQAGenerator`` standing in -- documented here, not hidden. Swap in a
real LLM-backed generator behind the same interface once a key is
available; nothing else in this module needs to change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

_PAIR_ID_LEN = 16
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DEFAULT_MIN_PASSAGE_CHARS = 200


def split_passages(text: str, min_chars: int = _DEFAULT_MIN_PASSAGE_CHARS) -> list[str]:
    """Group ``text`` into passages of at least ~``min_chars``, snapped to
    paragraph boundaries.

    This is not the general-purpose swappable chunking module (issues
    #5/#18) — just enough structure to give each generated Q/A pair a
    stable, citable source passage.
    """

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        return []

    passages: list[str] = []
    current = ""
    for para in paragraphs:
        current = f"{current}\n\n{para}".strip() if current else para
        if len(current) >= min_chars:
            passages.append(current)
            current = ""
    if current:
        if passages:
            passages[-1] = f"{passages[-1]}\n\n{current}".strip()
        else:
            passages.append(current)
    return passages


@runtime_checkable
class QAGenerator(Protocol):
    def generate(self, passage: str) -> tuple[str, str]:
        """Return ``(question, answer)`` grounded in ``passage``."""
        ...


@dataclass
class MockQAGenerator:
    """Deterministic offline stand-in for an LLM-backed generator.

    Takes the passage's first sentence as the "answer" and phrases a generic
    question around its leading word — enough to prove the ingest ->
    generate -> persist pipeline end-to-end with no API key, no network
    call, and a reproducible result for tests.
    """

    def generate(self, passage: str) -> tuple[str, str]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(passage) if s.strip()]
        answer = sentences[0] if sentences else passage.strip()
        words = answer.split()
        topic = words[0] if words else "this passage"
        question = f"What does the document say about {topic}?"
        return question, answer


@dataclass(frozen=True)
class QAPair:
    pair_id: str
    document_id: str
    chunk_index: int
    source_passage: str
    question: str
    answer: str
    generated_at: str

    def to_retrieval_reference(self) -> dict:
        """The subset of fields Phase 3's retrieval-metrics step needs to
        score recall/precision against the correct source chunk — the
        traceability data this issue adds must survive being passed
        downstream into that step."""
        return {
            "pair_id": self.pair_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "source_passage": self.source_passage,
            "expected_answer": self.answer,
        }


def _pair_id(document_id: str, chunk_index: int, passage: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}:{chunk_index}:{passage}".encode("utf-8")
    ).hexdigest()
    return digest[:_PAIR_ID_LEN]


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


def generate_qa_pairs(
    document_id: str,
    text: str,
    generator: QAGenerator | None = None,
    max_pairs: int | None = None,
) -> list[QAPair]:
    """Generate one Q/A pair per passage of ``text``, up to ``max_pairs``."""

    generator = generator or MockQAGenerator()
    passages = split_passages(text)
    if max_pairs is not None:
        passages = passages[:max_pairs]

    now = datetime.now(timezone.utc).isoformat()
    pairs: list[QAPair] = []
    for i, passage in enumerate(passages):
        question, answer = generator.generate(passage)
        pairs.append(
            QAPair(
                pair_id=_pair_id(document_id, i, passage),
                document_id=document_id,
                chunk_index=i,
                source_passage=passage,
                question=question,
                answer=answer,
                generated_at=now,
            )
        )
    return pairs


class QAStore:
    """Persists generated Q/A pairs per document, upserted by pair id.

    Re-running generation for an unchanged document reproduces the same
    pair ids and overwrites those records in place rather than duplicating
    them; an edited passage gets a new pair id under the same document, per
    the ingestion pattern's chunk-identity rule.
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.qa_dir = self.storage_dir / "qa_pairs"

    def _path(self, document_id: str) -> Path:
        return self.qa_dir / f"{document_id}.json"

    def _index_path(self) -> Path:
        return self.qa_dir / "_pair_index.json"

    def _load_index(self) -> dict[str, str]:
        path = self._index_path()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def get_pairs(self, document_id: str) -> list[QAPair]:
        path = self._path(document_id)
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Poisoned file — a partial legacy write. Treat as empty rather
            # than wedging every future generation for this document.
            return []
        if not isinstance(data, list):
            return []
        return [QAPair(**rec) for rec in data]

    def upsert_pairs(self, document_id: str, pairs: list[QAPair]) -> list[QAPair]:
        existing = {p.pair_id: p for p in self.get_pairs(document_id)}
        for pair in pairs:
            existing[pair.pair_id] = pair
        merged = sorted(existing.values(), key=lambda p: p.chunk_index)
        _atomic_write(
            self._path(document_id),
            json.dumps([asdict(p) for p in merged], indent=2, sort_keys=True).encode("utf-8"),
        )

        index = self._load_index()
        for pair in merged:
            index[pair.pair_id] = document_id
        _atomic_write(
            self._index_path(),
            json.dumps(index, indent=2, sort_keys=True).encode("utf-8"),
        )
        return merged

    def get_pair(self, pair_id: str) -> QAPair | None:
        """Look up a single Q/A pair by id alone (issue #3: a lookup for
        the originating passage of any given Q/A pair, without the caller
        already knowing which document it came from)."""
        index = self._load_index()
        document_id = index.get(pair_id)
        if document_id is not None:
            for pair in self.get_pairs(document_id):
                if pair.pair_id == pair_id:
                    return pair

        # Index miss (or stale) — fall back to scanning every document's
        # pairs rather than reporting "not found" for a pair that actually
        # exists; a stale/missing index must degrade, not poison lookups.
        if not self.qa_dir.is_dir():
            return None
        for path in self.qa_dir.glob("*.json"):
            if path.name == self._index_path().name:
                continue
            for pair in self.get_pairs(path.stem):
                if pair.pair_id == pair_id:
                    return pair
        return None
