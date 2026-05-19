"""Tests for the document CRUD endpoints added to ``hrag.web.app``.

Covers:
1. DELETE /api/documents/{doc_id} with a non-existent id is idempotent (200, deleted_chunks=0).
2. DELETE /api/documents/{doc_id} with an ``episodic:`` id returns 400.
3. DELETE /api/documents/{doc_id} for a real seeded doc removes its chunks.
4. GET /api/documents/{doc_id} returns metadata + preview + n_chunks for a seeded doc.
5. GET /api/documents/{doc_id} for a missing id returns 404.
6. GET /api/docs rows include the ``node_id`` field (null for unfiled docs).
7. GET /api/docs includes ``source_type`` field.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from hrag.web.app import _State, app  # noqa: E402


# ---------------------------------------------------------------------------
# State management helpers
# ---------------------------------------------------------------------------


def _reset_state() -> None:
    with _State.lock:
        _State.cfg = None
        _State.orch = None


@pytest.fixture(autouse=True)
def reset_state():
    _reset_state()
    yield
    _reset_state()


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _get_uid(client: TestClient) -> str:
    return client.get("/api/config").json()["user_id"]


# ---------------------------------------------------------------------------
# Seed / cleanup helpers (bypass the full ingest pipeline)
# ---------------------------------------------------------------------------


def _seed_doc(
    client: TestClient,
    doc_id: str,
    title: str = "Test Doc",
    source_type: str = "document",
) -> None:
    """Insert a minimal document row directly into the DB."""
    from hrag.web.app import _get_orch  # noqa: PLC0415

    orch = _get_orch()
    uid = _get_uid(client)
    orch.db.ensure_user(uid)
    with orch.db.conn:
        orch.db.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, user_id, source_path, title, source_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_id, uid, f"/tmp/{doc_id}.txt", title, source_type),
        )
    orch.db.commit()


def _seed_chunks(
    client: TestClient,
    doc_id: str,
    texts: list[str],
) -> list[str]:
    """Insert chunk rows for a seeded document. Returns list of chunk_ids."""
    import uuid  # noqa: PLC0415

    from hrag.web.app import _get_orch  # noqa: PLC0415

    orch = _get_orch()
    uid = _get_uid(client)
    chunk_ids = []
    with orch.db.conn:
        for i, text in enumerate(texts):
            cid = f"chunk_{doc_id}_{i}_{uuid.uuid4().hex[:6]}"
            chunk_ids.append(cid)
            orch.db.execute(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id, doc_id, user_id, text, title, chunk_index, source_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cid, doc_id, uid, text, "Test Doc", i, "document"),
            )
    orch.db.commit()
    return chunk_ids


def _cleanup(client: TestClient, doc_id: str) -> None:
    """Hard-remove a seeded doc + its chunks from the DB."""
    from hrag.web.app import _get_orch  # noqa: PLC0415

    orch = _get_orch()
    with orch.db.conn:
        orch.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        orch.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    orch.db.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_delete_nonexistent_doc_is_idempotent(client: TestClient) -> None:
    """DELETE on a non-existent doc_id returns 200 with deleted_chunks=0."""
    r = client.delete("/api/documents/totally-bogus-doc-xyz-9999")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["doc_id"] == "totally-bogus-doc-xyz-9999"
    assert body["deleted_chunks"] == 0


def test_delete_episodic_id_returns_400(client: TestClient) -> None:
    """DELETE with an episodic: prefix is rejected — use /api/memories instead."""
    r = client.delete("/api/documents/episodic:some-memory-id")
    assert r.status_code == 400, r.text
    assert "memories" in r.json()["detail"].lower()


def test_delete_real_doc_removes_chunks(client: TestClient) -> None:
    """DELETE a seeded doc removes its chunk rows from the DB."""
    from hrag.web.app import _get_orch  # noqa: PLC0415

    doc_id = "crud_test_delete_real_doc"
    _seed_doc(client, doc_id, title="DeleteMe")
    chunk_ids = _seed_chunks(client, doc_id, ["Chunk alpha.", "Chunk beta.", "Chunk gamma."])

    try:
        r = client.delete(f"/api/documents/{doc_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["doc_id"] == doc_id
        assert body["deleted_chunks"] == 3

        # Verify chunks are gone from the DB.
        orch = _get_orch()
        remaining = orch.db.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
        assert remaining == 0, f"expected 0 remaining chunks, got {remaining}"

        # Verify document row is gone too.
        doc_row = orch.db.execute(
            "SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        assert doc_row is None, "document row should have been deleted"
    finally:
        # If the test failed before DELETE fired, clean up manually.
        _cleanup(client, doc_id)


def test_get_document_returns_metadata_and_preview(client: TestClient) -> None:
    """GET /api/documents/{doc_id} returns all expected fields for a seeded doc."""
    doc_id = "crud_test_get_doc"
    _seed_doc(client, doc_id, title="MyDocument")
    _seed_chunks(client, doc_id, ["Hello world.", "Second chunk content."])

    try:
        r = client.get(f"/api/documents/{doc_id}")
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["doc_id"] == doc_id
        assert body["title"] == "MyDocument"
        assert isinstance(body["source_path"], str)
        assert body["source_type"] == "document"
        assert body["n_chunks"] == 2
        assert isinstance(body["preview"], str)
        assert len(body["preview"]) > 0
        # node_id should be null since we haven't assigned a taxonomy node.
        assert "node_id" in body
        assert body["node_id"] is None
    finally:
        _cleanup(client, doc_id)


def test_get_document_missing_returns_404(client: TestClient) -> None:
    """GET /api/documents/{doc_id} returns 404 for an unknown doc_id."""
    r = client.get("/api/documents/does-not-exist-at-all-xyz")
    assert r.status_code == 404, r.text


def test_list_docs_includes_node_id_field(client: TestClient) -> None:
    """GET /api/docs rows include a ``node_id`` key (null for unfiled docs)."""
    doc_id = "crud_test_list_docs_node_id"
    _seed_doc(client, doc_id, title="NodeIdTest")

    try:
        r = client.get("/api/docs")
        assert r.status_code == 200, r.text
        rows = r.json()
        # Find our seeded doc.
        matching = [row for row in rows if row["doc_id"] == doc_id]
        assert matching, f"seeded doc {doc_id!r} not found in /api/docs response"
        row = matching[0]
        assert "node_id" in row, f"node_id key missing from row: {row}"
        # Unfiled doc should have node_id == null.
        assert row["node_id"] is None, f"expected null node_id, got {row['node_id']!r}"
    finally:
        _cleanup(client, doc_id)


def test_list_docs_includes_source_type_field(client: TestClient) -> None:
    """GET /api/docs rows include a ``source_type`` field."""
    doc_id = "crud_test_list_docs_source_type"
    _seed_doc(client, doc_id, title="SourceTypeTest", source_type="document")

    try:
        r = client.get("/api/docs")
        assert r.status_code == 200, r.text
        rows = r.json()
        matching = [row for row in rows if row["doc_id"] == doc_id]
        assert matching, f"seeded doc {doc_id!r} not found in /api/docs"
        row = matching[0]
        assert "source_type" in row, f"source_type key missing from row: {row}"
        assert row["source_type"] == "document"
    finally:
        _cleanup(client, doc_id)
