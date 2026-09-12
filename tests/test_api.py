import pytest
from fastapi.testclient import TestClient

from ragel.api import app, get_qa_store, get_store
from ragel.ingest import IngestStore
from ragel.qa_generation import QAStore


@pytest.fixture
def client(tmp_path):
    store = IngestStore(tmp_path / "corpus", max_upload_bytes=5_000_000)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_qa_store] = lambda: QAStore(tmp_path / "corpus")
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


def test_generate_and_list_qa_pairs(client, text_bytes):
    doc_id = client.post(
        "/documents", files={"file": ("notes.txt", text_bytes, "text/plain")}
    ).json()["document_id"]

    generated = client.post(f"/documents/{doc_id}/qa-pairs")
    assert generated.status_code == 200
    pairs = generated.json()
    assert len(pairs) >= 1
    assert all(p["document_id"] == doc_id for p in pairs)

    listed = client.get(f"/documents/{doc_id}/qa-pairs").json()
    assert {p["pair_id"] for p in listed} == {p["pair_id"] for p in pairs}


def test_generate_qa_pairs_rerun_is_idempotent(client, text_bytes):
    doc_id = client.post(
        "/documents", files={"file": ("notes.txt", text_bytes, "text/plain")}
    ).json()["document_id"]

    first = client.post(f"/documents/{doc_id}/qa-pairs").json()
    second = client.post(f"/documents/{doc_id}/qa-pairs").json()
    assert {p["pair_id"] for p in first} == {p["pair_id"] for p in second}


def test_generate_qa_pairs_respects_max_pairs(client, text_bytes):
    doc_id = client.post(
        "/documents", files={"file": ("notes.txt", text_bytes, "text/plain")}
    ).json()["document_id"]
    resp = client.post(f"/documents/{doc_id}/qa-pairs", json={"max_pairs": 1})
    assert len(resp.json()) <= 1


def test_generate_qa_pairs_missing_document_404(client):
    assert client.post("/documents/nope/qa-pairs").status_code == 404


def test_get_qa_pair_by_id_traces_back_to_source_passage(client, text_bytes):
    doc_id = client.post(
        "/documents", files={"file": ("notes.txt", text_bytes, "text/plain")}
    ).json()["document_id"]
    pairs = client.post(f"/documents/{doc_id}/qa-pairs").json()
    pair_id = pairs[0]["pair_id"]

    resp = client.get(f"/qa-pairs/{pair_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pair_id"] == pair_id
    assert body["document_id"] == doc_id
    assert body["source_passage"] == pairs[0]["source_passage"]


def test_get_qa_pair_missing_404(client):
    assert client.get("/qa-pairs/does-not-exist").status_code == 404
