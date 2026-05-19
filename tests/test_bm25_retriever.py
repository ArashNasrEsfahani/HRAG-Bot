"""Tests for hrag.retrieval.bm25 — BM25Retriever.

rank_bm25 is optional; skip the whole file if absent.
Uses the tmp_db fixture for a real SQLite database.
"""

from __future__ import annotations

import pytest

rank_bm25 = pytest.importorskip("rank_bm25")

from hrag.retrieval.bm25 import BM25Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_doc(db, doc_id: str, user_id: str, source_type: str = "document"):
    db.execute(
        "INSERT OR IGNORE INTO documents(doc_id, user_id, source_path, title, source_type)"
        " VALUES (?, ?, ?, ?, ?)",
        (doc_id, user_id, f"/fake/{doc_id}.txt", doc_id, source_type),
    )
    db.commit()


def _insert_chunk(
    db,
    chunk_id: str,
    doc_id: str,
    user_id: str,
    text: str,
    source_type: str = "document",
    excluded: int = 0,
):
    db.execute(
        "INSERT INTO chunks(chunk_id, doc_id, user_id, text, source_type, excluded,"
        " chunk_index, token_count)"
        " VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        (chunk_id, doc_id, user_id, text, source_type, excluded, len(text.split())),
    )
    db.commit()


def _seed_alice_bob(db):
    """Insert 5 chunks for alice, 2 for bob."""
    db.ensure_user("alice")
    db.ensure_user("bob")

    # Alice's docs
    _insert_doc(db, "alice-doc1", "alice", "document")
    _insert_doc(db, "alice-doc2", "alice", "document")
    _insert_doc(db, "alice-episodic", "alice", "episodic")

    _insert_chunk(db, "a1", "alice-doc1", "alice",
                  "Python is a high-level programming language.", "document")
    _insert_chunk(db, "a2", "alice-doc1", "alice",
                  "Java and Python are popular programming languages.", "document")
    _insert_chunk(db, "a3", "alice-doc2", "alice",
                  "Machine learning uses statistical techniques.", "document")
    _insert_chunk(db, "a4", "alice-doc2", "alice",
                  "Neural networks are used in deep learning.", "document")
    _insert_chunk(db, "a5", "alice-episodic", "alice",
                  "Today I learned about Python decorators.", "episodic")

    # Bob's docs
    _insert_doc(db, "bob-doc1", "bob", "document")
    _insert_chunk(db, "b1", "bob-doc1", "bob",
                  "Bob prefers cooking over programming.", "document")
    _insert_chunk(db, "b2", "bob-doc1", "bob",
                  "Bob enjoys hiking and outdoor activities.", "document")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_python_query_returns_python_chunk_first(tmp_db):
    _seed_alice_bob(tmp_db)
    retriever = BM25Retriever(tmp_db)
    results = retriever.retrieve("Python programming", user_id="alice", top_k=5)
    assert len(results) > 0
    # The top result should mention Python
    top_text = results[0].chunk.text.lower()
    assert "python" in top_text, f"Top result does not mention python: {top_text}"


def test_user_filter_excludes_other_users_chunks(tmp_db):
    _seed_alice_bob(tmp_db)
    retriever = BM25Retriever(tmp_db)
    results = retriever.retrieve("programming", user_id="alice", top_k=10)
    for r in results:
        assert r.chunk.user_id == "alice", (
            f"Got chunk from unexpected user: {r.chunk.user_id}"
        )


def test_bob_query_excludes_alice_chunks(tmp_db):
    _seed_alice_bob(tmp_db)
    retriever = BM25Retriever(tmp_db)
    results = retriever.retrieve("cooking", user_id="bob", top_k=5)
    assert len(results) > 0
    for r in results:
        assert r.chunk.user_id == "bob"


def test_source_type_filter_episodic(tmp_db):
    _seed_alice_bob(tmp_db)
    retriever = BM25Retriever(tmp_db)
    results = retriever.retrieve(
        "Python decorators", user_id="alice", top_k=10, source_types=["episodic"]
    )
    assert len(results) > 0
    for r in results:
        assert r.chunk.source_type == "episodic", (
            f"Expected episodic, got {r.chunk.source_type}"
        )


def test_source_type_filter_document_only(tmp_db):
    _seed_alice_bob(tmp_db)
    retriever = BM25Retriever(tmp_db)
    results = retriever.retrieve(
        "Python", user_id="alice", top_k=10, source_types=["document"]
    )
    for r in results:
        assert r.chunk.source_type == "document"


def test_empty_corpus_returns_empty_list(tmp_db):
    """A fresh DB with no chunks → empty results."""
    retriever = BM25Retriever(tmp_db)  # no chunks inserted
    results = retriever.retrieve("Python", user_id="default", top_k=5)
    assert results == []


def test_excluded_chunks_not_returned(tmp_db):
    """Chunks with excluded=1 must not appear in results."""
    db = tmp_db
    db.ensure_user("alice")
    _insert_doc(db, "alice-doc-ex", "alice", "document")
    _insert_chunk(db, "excl-chunk", "alice-doc-ex", "alice",
                  "Python excluded content should not appear.", "document", excluded=1)
    _insert_chunk(db, "ok-chunk", "alice-doc-ex", "alice",
                  "Regular Python content is fine.", "document", excluded=0)

    retriever = BM25Retriever(db)
    results = retriever.retrieve("Python", user_id="alice", top_k=10)
    chunk_ids = [r.chunk.chunk_id for r in results]
    assert "excl-chunk" not in chunk_ids, "Excluded chunk should not be in results"
    assert "ok-chunk" in chunk_ids, "Non-excluded chunk should be in results"


def test_refresh_picks_up_new_chunks(tmp_db):
    """After inserting a chunk and calling .refresh(), it should appear in results."""
    db = tmp_db
    db.ensure_user("alice")
    _insert_doc(db, "alice-new-doc", "alice", "document")
    _insert_chunk(db, "new-chunk", "alice-new-doc", "alice",
                  "Quantum computing is fascinating and novel.", "document")

    retriever = BM25Retriever(db)
    # Verify it's present before the extra insert
    results_before = retriever.retrieve("quantum", user_id="alice", top_k=5)
    assert any(r.chunk.chunk_id == "new-chunk" for r in results_before)

    # Insert a second chunk
    _insert_chunk(db, "newer-chunk", "alice-new-doc", "alice",
                  "Quantum entanglement enables quantum teleportation.", "document")

    # Without refresh, the new chunk might not appear
    retriever.refresh()

    results_after = retriever.retrieve("quantum", user_id="alice", top_k=5)
    chunk_ids = [r.chunk.chunk_id for r in results_after]
    assert "newer-chunk" in chunk_ids, "refresh() should pick up newly inserted chunks"


def test_retriever_name_is_bm25(tmp_db):
    """Returned results should have retriever='bm25'."""
    db = tmp_db
    db.ensure_user("alice")
    _insert_doc(db, "alice-d", "alice", "document")
    _insert_chunk(db, "c1", "alice-d", "alice", "Python scripting automation", "document")

    retriever = BM25Retriever(db)
    results = retriever.retrieve("Python", user_id="alice", top_k=5)
    assert len(results) > 0
    for r in results:
        assert r.retriever == "bm25"


def test_top_k_limits_results(tmp_db):
    _seed_alice_bob(tmp_db)
    retriever = BM25Retriever(tmp_db)
    results = retriever.retrieve("python", user_id="alice", top_k=2)
    assert len(results) <= 2
