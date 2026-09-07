import pytest
from fastapi.testclient import TestClient

from ragel.api import app, get_store
from ragel.ingest import IngestStore


@pytest.fixture
def client(tmp_path):
    store = IngestStore(tmp_path / "corpus", max_upload_bytes=5_000_000)
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_text_and_retrieve(client, text_bytes):
    resp = client.post("/documents", files={"file": ("notes.txt", text_bytes, "text/plain")})
    assert resp.status_code == 201
    doc_id = resp.json()["document_id"]

    assert client.get(f"/documents/{doc_id}").json()["content_type"] == "text"
    assert "quick brown fox" in client.get(f"/documents/{doc_id}/text").text
    assert any(d["document_id"] == doc_id for d in client.get("/documents").json())


def test_ingest_pdf(client, pdf_bytes):
    resp = client.post("/documents", files={"file": ("p.pdf", pdf_bytes, "application/pdf")})
    assert resp.status_code == 201
    assert resp.json()["content_type"] == "pdf"


def test_unsupported_type_415(client):
    resp = client.post("/documents", files={"file": ("a.zip", b"PK\x03\x04", "application/zip")})
    assert resp.status_code == 415


def test_corrupt_pdf_422(client):
    resp = client.post(
        "/documents", files={"file": ("b.pdf", b"%PDF-1.4 garbage\x00", "application/pdf")}
    )
    assert resp.status_code == 422


def test_missing_document_404(client):
    assert client.get("/documents/nope").status_code == 404
    assert client.get("/documents/nope/text").status_code == 404
