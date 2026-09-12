import json

from ragel.qa_generation import (
    MockQAGenerator,
    QAStore,
    generate_qa_pairs,
    split_passages,
)

SAMPLE_TEXT = (
    "Paragraph one talks about widgets. Widgets are small and useful.\n\n"
    "Paragraph two talks about gadgets. Gadgets are shiny and expensive. "
    "They are often paired with widgets in practice.\n\n"
    "Paragraph three is a short closer."
)


def test_split_passages_groups_short_paragraphs_together():
    passages = split_passages(SAMPLE_TEXT, min_chars=10)
    assert len(passages) >= 1
    assert "".join(passages).replace("\n\n", " ").strip()


def test_split_passages_empty_text_returns_no_passages():
    assert split_passages("   ") == []


def test_generate_qa_pairs_grounds_answer_in_passage():
    pairs = generate_qa_pairs("doc1", SAMPLE_TEXT)
    assert len(pairs) >= 1
    for pair in pairs:
        assert pair.answer in pair.source_passage
        assert pair.document_id == "doc1"


def test_generate_qa_pairs_is_deterministic():
    first = generate_qa_pairs("doc1", SAMPLE_TEXT)
    second = generate_qa_pairs("doc1", SAMPLE_TEXT)
    assert [p.pair_id for p in first] == [p.pair_id for p in second]
    assert [p.question for p in first] == [p.question for p in second]


def test_generate_qa_pairs_respects_max_pairs():
    pairs = generate_qa_pairs("doc1", SAMPLE_TEXT, max_pairs=1)
    assert len(pairs) == 1


def test_edited_passage_gets_a_new_pair_id_not_a_silent_overwrite():
    original = generate_qa_pairs("doc1", "Some original passage text here. More text.")
    edited = generate_qa_pairs("doc1", "Some edited passage text here. More text.")
    assert original[0].pair_id != edited[0].pair_id


def test_mock_generator_is_deterministic_and_grounded():
    gen = MockQAGenerator()
    q1, a1 = gen.generate("Widgets are small. They are useful.")
    q2, a2 = gen.generate("Widgets are small. They are useful.")
    assert (q1, a1) == (q2, a2)
    assert a1 == "Widgets are small."


class _StubGenerator:
    def generate(self, passage: str) -> tuple[str, str]:
        return ("stub question", "stub answer")


def test_generate_qa_pairs_uses_injected_generator():
    pairs = generate_qa_pairs("doc1", SAMPLE_TEXT, generator=_StubGenerator())
    assert all(p.question == "stub question" for p in pairs)


# --- QAStore: idempotent persistence -------------------------------------


def test_upsert_pairs_is_idempotent_on_rerun(tmp_path):
    store = QAStore(tmp_path / "corpus")
    pairs = generate_qa_pairs("doc1", SAMPLE_TEXT)

    first = store.upsert_pairs("doc1", pairs)
    second = store.upsert_pairs("doc1", pairs)  # re-run: same input

    assert len(first) == len(second)
    assert {p.pair_id for p in first} == {p.pair_id for p in second}


def test_upsert_pairs_persists_across_store_instances(tmp_path):
    store_dir = tmp_path / "corpus"
    pairs = generate_qa_pairs("doc1", SAMPLE_TEXT)
    QAStore(store_dir).upsert_pairs("doc1", pairs)

    reloaded = QAStore(store_dir).get_pairs("doc1")
    assert len(reloaded) == len(pairs)


def test_upsert_pairs_does_not_duplicate_unchanged_passages(tmp_path):
    store = QAStore(tmp_path / "corpus")
    pairs = generate_qa_pairs("doc1", SAMPLE_TEXT)
    store.upsert_pairs("doc1", pairs)
    merged = store.upsert_pairs("doc1", pairs)
    assert len(merged) == len(pairs)


def test_get_pairs_on_poisoned_file_returns_empty_not_a_crash(tmp_path):
    store_dir = tmp_path / "corpus"
    qa_dir = store_dir / "qa_pairs"
    qa_dir.mkdir(parents=True)
    (qa_dir / "doc1.json").write_text("{not valid json", encoding="utf-8")
    store = QAStore(store_dir)
    assert store.get_pairs("doc1") == []


# --- Traceability: Q/A pair -> source chunk (issue #3) -------------------


def test_to_retrieval_reference_carries_traceability_fields():
    pair = generate_qa_pairs("doc1", SAMPLE_TEXT)[0]
    ref = pair.to_retrieval_reference()

    assert ref["document_id"] == "doc1"
    assert ref["chunk_index"] == pair.chunk_index
    assert ref["source_passage"] == pair.source_passage
    assert ref["expected_answer"] == pair.answer


def test_get_pair_looks_up_by_id_alone(tmp_path):
    store = QAStore(tmp_path / "corpus")
    pairs = generate_qa_pairs("doc1", SAMPLE_TEXT)
    store.upsert_pairs("doc1", pairs)

    found = store.get_pair(pairs[0].pair_id)
    assert found == pairs[0]


def test_get_pair_unknown_id_returns_none(tmp_path):
    store = QAStore(tmp_path / "corpus")
    assert store.get_pair("does-not-exist") is None


def test_get_pair_falls_back_to_scan_when_index_is_stale(tmp_path):
    store = QAStore(tmp_path / "corpus")
    pairs = generate_qa_pairs("doc1", SAMPLE_TEXT)
    store.upsert_pairs("doc1", pairs)

    # Corrupt the index on purpose: it now points the pair at a document
    # that doesn't have it. A stale index must degrade to a full scan, not
    # report "not found" for a pair that actually exists.
    index_path = store._index_path()
    index_path.write_text(
        json.dumps({pairs[0].pair_id: "some-other-doc"}), encoding="utf-8"
    )

    found = store.get_pair(pairs[0].pair_id)
    assert found == pairs[0]


def test_get_pair_works_across_multiple_documents(tmp_path):
    store = QAStore(tmp_path / "corpus")
    pairs_a = generate_qa_pairs("doc-a", SAMPLE_TEXT)
    pairs_b = generate_qa_pairs("doc-b", "Different text entirely. Second sentence here.")
    store.upsert_pairs("doc-a", pairs_a)
    store.upsert_pairs("doc-b", pairs_b)

    assert store.get_pair(pairs_a[0].pair_id).document_id == "doc-a"
    assert store.get_pair(pairs_b[0].pair_id).document_id == "doc-b"
