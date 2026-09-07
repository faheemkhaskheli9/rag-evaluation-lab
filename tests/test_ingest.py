import pytest

from ragel.ingest import (
    CorruptDocumentError,
    IngestStore,
    UnsupportedTypeError,
)


def _store(tmp_path, max_bytes=5_000_000):
    return IngestStore(tmp_path / "corpus", max_bytes)


def test_ingest_text_file_extracts_and_persists(tmp_path, text_bytes):
    store = _store(tmp_path)
    rec = store.ingest("notes.txt", text_bytes)
    assert rec.content_type == "text"
    assert rec.char_count == len(text_bytes.decode())
    assert store.read_text(rec.document_id).startswith("The quick brown fox")
    assert (tmp_path / "corpus" / "raw" / f"{rec.document_id}.txt").read_bytes() == text_bytes


def test_ingest_pdf_file(tmp_path, pdf_bytes):
    store = _store(tmp_path)
    rec = store.ingest("paper.pdf", pdf_bytes)
    assert rec.content_type == "pdf"
    assert isinstance(store.read_text(rec.document_id), str)
    assert store.get(rec.document_id).filename == "paper.pdf"


def test_reingest_same_bytes_idempotent(tmp_path, text_bytes):
    store = _store(tmp_path)
    a = store.ingest("a.txt", text_bytes)
    b = store.ingest("a.txt", text_bytes)
    assert a.document_id == b.document_id
    assert len(store.list_documents()) == 1


def test_unsupported_type_raises(tmp_path):
    with pytest.raises(UnsupportedTypeError):
        _store(tmp_path).ingest("archive.zip", b"PK\x03\x04zip")


def test_corrupt_pdf_raises_not_silent_drop(tmp_path):
    with pytest.raises(CorruptDocumentError):
        _store(tmp_path).ingest("broken.pdf", b"%PDF-1.4 then total garbage \x00\x01")


def test_non_utf8_text_raises(tmp_path):
    with pytest.raises(CorruptDocumentError):
        _store(tmp_path).ingest("bad.txt", b"\xff\xfe\x00rubbish")


def test_empty_file_raises(tmp_path):
    with pytest.raises(CorruptDocumentError):
        _store(tmp_path).ingest("empty.txt", b"")


def test_oversize_raises(tmp_path, text_bytes):
    with pytest.raises(ValueError):
        _store(tmp_path, max_bytes=5).ingest("a.txt", text_bytes)


def test_corrupt_index_recovers(tmp_path, text_bytes):
    store = _store(tmp_path)
    store.index_path.parent.mkdir(parents=True, exist_ok=True)
    store.index_path.write_text("not json {{")
    rec = store.ingest("a.txt", text_bytes)
    assert store.get(rec.document_id) is not None
