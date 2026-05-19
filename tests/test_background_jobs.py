"""Tests for the background-job system introduced for the global progress bar.

Five tests via FastAPI's ``TestClient``:

1. ``POST /api/memories?background=true`` returns immediately with a job_id;
   polling ``GET /api/jobs/{id}`` eventually shows status=done and the
   memory exists in the DB.
2. ``POST /api/memories`` (no ``background`` flag) returns the full memory
   dict synchronously — preserves the sync contract for CLI / smoke tests.
3. ``POST /api/ingest?background=true&taxonomy_mode=manual`` runs the worker
   with ``skip_taxonomy=True`` so the doc has no taxonomy assignment row.
4. Same upload with ``taxonomy_mode=auto`` DOES create an assignment row
   (or skips silently when the tree is empty — the contract is that the
   auto-assign hook *runs*, not that it always succeeds).
5. ``GET /api/jobs/unknown_id`` returns 404 — matches the existing
   ``_get_job`` behaviour.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from hrag.web.app import _State, _get_orch, app  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
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


def _wait_for_job(client: TestClient, job_id: str, timeout_s: float = 10.0) -> dict:
    """Poll ``GET /api/jobs/{id}`` until status is terminal or timeout."""
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        if r.status_code == 404:
            time.sleep(0.05)
            continue
        last = r.json()
        if last.get("status") in ("done", "failed"):
            return last
        time.sleep(0.05)
    return last


# ---------------------------------------------------------------------------
# 1. Memory create — background path
# ---------------------------------------------------------------------------


def test_memory_create_background_returns_job_id(client: TestClient) -> None:
    """POST /api/memories?background=true returns a job_id; polling reveals
    a done state and the memory lands in the DB."""
    text = "I prefer dark mode at night."
    r = client.post("/api/memories?background=true", json={"text": text})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("background") is True
    job_id = body.get("job_id")
    assert isinstance(job_id, str) and len(job_id) > 0
    assert body.get("kind") == "memory_embed"

    final = _wait_for_job(client, job_id, timeout_s=15.0)
    assert final.get("status") == "done", final
    # Result column should mention the new memory_id (echoed by the worker).
    result = final.get("result")
    if isinstance(result, dict):
        mem_id = result.get("memory_id")
    else:
        # Some serializers leave it as JSON-text — handle both shapes.
        import json
        mem_id = json.loads(result or "{}").get("memory_id") if result else None
    assert mem_id, f"worker did not surface memory_id in result; got {result!r}"

    # And the memory really exists in /api/memories.
    mems = client.get("/api/memories").json()
    assert any(m["memory_id"] == mem_id for m in mems), mems


# ---------------------------------------------------------------------------
# 2. Memory create — sync path is unchanged
# ---------------------------------------------------------------------------


def test_memory_create_sync_unchanged(client: TestClient) -> None:
    """POST /api/memories (no background flag) returns the memory dict directly."""
    text = "Synchronous memory write."
    r = client.post("/api/memories", json={"text": text})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "memory_id" in body, body
    assert body["memory_id"].startswith("episodic:"), body["memory_id"]
    # Must NOT have a job_id — preserving the original sync envelope.
    assert "job_id" not in body
    assert "background" not in body


# ---------------------------------------------------------------------------
# 3. Doc ingest — manual mode skips the auto-assign hook
# ---------------------------------------------------------------------------


def _spy_ingest_path(monkeypatch) -> dict:
    """Replace ``ingest.ingest_path`` with a tiny spy so we don't need a real
    PDF / DOCX loader pipeline.

    Returns a dict that the test can inspect: ``{"calls": [...]}``.

    Each call is recorded as ``(path, user_id, skip_taxonomy)``. The spy
    inserts a stub document row + chunk and (in auto mode) optionally a
    taxonomy assignment so the worker can read it back.
    """
    captured: dict = {"calls": []}

    def _fake_ingest_path(path, user_id, *, skip_taxonomy: bool = False, progress_cb=None):
        from hrag.types import Document
        from datetime import datetime, timezone

        orch = _get_orch()
        orch.db.ensure_user(user_id)
        doc_id = f"job-test-{Path(path).stem}"
        title = Path(path).stem
        with orch.db.conn:
            orch.db.execute(
                "INSERT OR REPLACE INTO documents "
                "(doc_id, user_id, source_path, title, source_type) "
                "VALUES (?, ?, ?, ?, 'document')",
                (doc_id, user_id, str(path), title),
            )
            orch.db.execute(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id, doc_id, user_id, text, title, chunk_index, source_type) "
                "VALUES (?, ?, ?, ?, ?, 0, 'document')",
                (f"{doc_id}_c0", doc_id, user_id, "stub chunk.", title),
            )
            # Simulate the auto-assign hook: in auto mode, drop a fake
            # assignment row so the test can assert it lands.
            if not skip_taxonomy:
                # Need a fake leaf node — create the root + one leaf if
                # neither exists for this user.
                root_id = f"root::{user_id}"
                leaf_id = "tx_jobtest_leaf"
                orch.db.execute(
                    "INSERT OR IGNORE INTO kg_taxonomy_nodes "
                    "(node_id, user_id, parent_id, label, depth, is_leaf) "
                    "VALUES (?, ?, NULL, 'Root', 0, 0)",
                    (root_id, user_id),
                )
                orch.db.execute(
                    "INSERT OR IGNORE INTO kg_taxonomy_nodes "
                    "(node_id, user_id, parent_id, label, depth, is_leaf) "
                    "VALUES (?, ?, ?, 'JobTestLeaf', 1, 1)",
                    (leaf_id, user_id, root_id),
                )
                orch.db.execute(
                    "INSERT OR IGNORE INTO kg_taxonomy_assignments "
                    "(user_id, doc_id, node_id, score, is_primary) "
                    "VALUES (?, ?, ?, 1.0, 1)",
                    (user_id, doc_id, leaf_id),
                )
        orch.db.commit()

        captured["calls"].append((str(path), user_id, skip_taxonomy))
        return Document(
            doc_id=doc_id,
            user_id=user_id,
            source_path=str(path),
            title=title,
            text="stub",
            source_type="document",
            ingested_at=datetime.now(tz=timezone.utc),
        )

    # Boot the orchestrator before swapping its ingest pipeline.
    orch = _get_orch()
    monkeypatch.setattr(orch.ingest, "ingest_path", _fake_ingest_path)
    return captured


def test_doc_ingest_manual_mode_skips_auto_assign(
    client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    """Upload with ``taxonomy_mode=manual`` must run with skip_taxonomy=True
    and the worker MUST NOT create a taxonomy assignment row."""
    spy = _spy_ingest_path(monkeypatch)
    uid = _get_uid(client)

    # Build a tiny in-memory text file.
    file_bytes = b"manual mode test content."
    files = {"file": ("manual-mode.txt", file_bytes, "text/plain")}
    r = client.post(
        "/api/ingest?background=true&taxonomy_mode=manual", files=files,
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    final = _wait_for_job(client, job_id, timeout_s=10.0)
    assert final.get("status") == "done", final
    # Worker ran with skip_taxonomy=True
    assert spy["calls"], "ingest_path spy was never invoked"
    _path, _uid, skip = spy["calls"][-1]
    assert skip is True
    assert _uid == uid

    # Result payload carries taxonomy_mode + doc_id for the frontend banner.
    result = final.get("result")
    if isinstance(result, str):
        import json
        result = json.loads(result)
    assert result.get("taxonomy_mode") == "manual", result
    doc_id = result.get("doc_id")
    assert doc_id, f"worker did not surface doc_id; got {result!r}"
    assert doc_id.startswith("job-test-manual-mode"), doc_id

    # No taxonomy assignment row was created for this doc.
    orch = _get_orch()
    row = orch.db.execute(
        "SELECT COUNT(*) AS n FROM kg_taxonomy_assignments "
        "WHERE user_id = ? AND doc_id = ?",
        (uid, doc_id),
    ).fetchone()
    assert int(row["n"]) == 0, "manual mode must not insert an assignment row"


def test_doc_ingest_auto_mode_assigns(
    client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    """Upload with ``taxonomy_mode=auto`` must run with skip_taxonomy=False
    and the resulting assignment row must exist."""
    spy = _spy_ingest_path(monkeypatch)
    uid = _get_uid(client)

    files = {"file": ("auto-mode.txt", b"auto mode content.", "text/plain")}
    r = client.post(
        "/api/ingest?background=true&taxonomy_mode=auto", files=files,
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    final = _wait_for_job(client, job_id, timeout_s=10.0)
    assert final.get("status") == "done", final

    _path, _uid, skip = spy["calls"][-1]
    assert skip is False
    assert _uid == uid

    result = final.get("result")
    if isinstance(result, str):
        import json
        result = json.loads(result)
    assert result.get("taxonomy_mode") == "auto", result
    doc_id = result.get("doc_id")
    assert doc_id, result
    assert doc_id.startswith("job-test-auto-mode"), doc_id

    # An assignment row was created (the spy inserted it under "tx_jobtest_leaf").
    orch = _get_orch()
    row = orch.db.execute(
        "SELECT node_id FROM kg_taxonomy_assignments "
        "WHERE user_id = ? AND doc_id = ?",
        (uid, doc_id),
    ).fetchone()
    assert row is not None and row["node_id"] == "tx_jobtest_leaf"
    assert result.get("assigned_node") == "tx_jobtest_leaf", result


# ---------------------------------------------------------------------------
# 5. Unknown job_id → 404
# ---------------------------------------------------------------------------


def test_jobs_endpoint_404_on_unknown(client: TestClient) -> None:
    """GET /api/jobs/{unknown} returns 404 (matches _get_job behaviour)."""
    r = client.get("/api/jobs/this-id-does-not-exist-xyz-9999")
    assert r.status_code == 404, r.text
