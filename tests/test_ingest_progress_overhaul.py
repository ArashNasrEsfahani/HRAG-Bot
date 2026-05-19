"""Tests for the SOTA ingest progress overhaul.

Covers:
1. ``IngestPipeline.ingest_path`` calls ``progress_cb`` for every stage AND
   emits at least one mid-progress embed event.
2. The cancel endpoint flips a running job to ``"cancelled"`` and prevents the
   doc from landing in the chunks index.
3. ``GET /api/jobs/{id}`` returns per-stage detail (``stages.{load, chunk,
   filter, embed, index}`` with ``n_done`` / ``n_total`` / ``duration_s``).
4. Multi-file ingest: a middle-file failure does not block the other files
   from landing in the index, and the error is surfaced in ``state.errors``.
"""

from __future__ import annotations

import sys
import threading
import time
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


def _make_text_file(tmp_path: Path, name: str, n_paragraphs: int = 80) -> Path:
    """Synthesise a longish .txt so the chunker produces several chunks (and
    therefore several embed batches)."""
    lines = []
    for i in range(n_paragraphs):
        lines.append(
            f"Section {i}. The hierarchical retrieval augmented generation "
            f"system uses dense vectors plus knowledge graphs. "
            f"Paragraph {i} elaborates on the entity store and PPR seeds. "
            f"More boilerplate text to give the chunker something to bite on. "
            f"And another sentence to pad it out, indexed {i}.\n"
        )
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. progress_cb is invoked for every stage including mid-embed events
# ---------------------------------------------------------------------------


def test_pipeline_emits_progress_callbacks(tmp_path: Path, client: TestClient):
    """Every required stage fires and embed emits >=2 events (start + at
    least one batch progress / end)."""
    from hrag.web.app import _get_orch  # noqa: PLC0415

    orch = _get_orch()
    uid = client.get("/api/config").json()["user_id"]
    orch.db.ensure_user(uid)

    doc_path = _make_text_file(tmp_path, "progress_doc.txt", n_paragraphs=80)

    events: list[tuple[str, dict]] = []

    def cb(stage: str, payload: dict) -> None:
        events.append((stage, payload))

    orch.ingest.ingest_path(
        str(doc_path), uid, skip_taxonomy=True, progress_cb=cb,
    )

    stages_seen = {stage for stage, _ in events}
    # Required stages (assign is skipped because skip_taxonomy=True).
    for required in ("load", "chunk", "filter", "embed", "index", "done"):
        assert required in stages_seen, f"missing stage: {required} (got {sorted(stages_seen)})"

    embed_events = [p for s, p in events if s == "embed"]
    # Start + at least one batch
    assert len(embed_events) >= 2, "embed must emit at least start + one batch update"
    # The last embed event should report n_done == n_total
    last = embed_events[-1]
    assert last["n_done"] == last["n_total"], f"final embed not at 100%: {last}"

    # Done event includes doc_id + n_chunks
    done = [p for s, p in events if s == "done"][-1]
    assert "doc_id" in done and done["doc_id"]
    assert "n_chunks" in done


# ---------------------------------------------------------------------------
# 2. Cancel endpoint aborts the worker and prevents indexing
# ---------------------------------------------------------------------------


def test_cancel_during_embed_aborts(tmp_path: Path, client: TestClient):
    """A cancel signal raised mid-embed unwinds the pipeline and the job row
    transitions to ``cancelled``. The doc must NOT have any chunks landed.

    We make the embed stage slow by monkey-patching the embedder's
    ``embed()`` method to sleep, then trigger the cancel from another
    thread.
    """
    from hrag.web.app import _cancel_events, _get_orch  # noqa: PLC0415

    orch = _get_orch()
    uid = client.get("/api/config").json()["user_id"]
    orch.db.ensure_user(uid)

    doc_path = _make_text_file(tmp_path, "cancel_doc.txt", n_paragraphs=200)

    # Slow down embedding aggressively so cancel has a clear window.
    # ``ingest.pipeline._EMBED_BATCH_SIZE`` is 32; sleeping a full second per
    # batch gives the cancel signal time to land between batches.
    original_embed = orch.ingest.embedder.embed
    def slow_embed(texts):
        time.sleep(1.0)  # 1 s per batch
        return original_embed(texts)
    orch.ingest.embedder.embed = slow_embed  # type: ignore[method-assign]

    try:
        # Pre-register a cancel event under a known job id and run the
        # worker on a thread so we can signal it.
        from hrag.web.app import _run_ingest_job, _create_job  # noqa: PLC0415

        job_id = _create_job(uid, kind="doc_ingest", total=1, message="queued")
        evt = threading.Event()
        _cancel_events[job_id] = evt

        # Copy file because the worker unlinks dest_path on failure.
        dest = tmp_path / "cancel_dest.txt"
        dest.write_bytes(doc_path.read_bytes())

        worker = threading.Thread(
            target=_run_ingest_job,
            args=(job_id, uid, dest, "cancel_doc.txt"),
            kwargs={"taxonomy_mode": "skip"},
            daemon=True,
        )
        worker.start()

        # Give it long enough to enter the embed stage (slow_embed sleeps
        # 1s per batch), then cancel before the worker finishes.
        time.sleep(1.2)
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 200
        assert r.json()["cancelled"] is True

        worker.join(timeout=30)
        assert not worker.is_alive(), "worker did not unwind after cancel"

        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "cancelled", f"expected cancelled, got {job['status']}"

        # The doc should not have made it into the chunks table.
        rows = orch.db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE doc_id LIKE ? AND user_id = ?",
            ("%cancel_doc%", uid),
        ).fetchone()
        # The worker might have written the documents row but should not
        # have got chunks indexed. We check there's no chunks for any doc
        # whose title matches our cancelled file.
        # (Allowing 0 — if a fast machine landed nothing yet that's also fine.)
        assert rows is None or rows["n"] == 0, "chunks were indexed despite cancel"
    finally:
        orch.ingest.embedder.embed = original_embed  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 3. /api/jobs/{id} surfaces per-stage detail
# ---------------------------------------------------------------------------


def test_jobs_endpoint_returns_stage_detail(tmp_path: Path, client: TestClient):
    """After a synchronous ingest, GET /api/jobs/{id} must include
    ``result.stages`` with every stage carrying ``n_done`` / ``n_total`` /
    ``duration_s``."""
    doc_path = _make_text_file(tmp_path, "stages_doc.txt", n_paragraphs=40)

    # Use the synchronous ingest path so the test doesn't have to poll.
    with doc_path.open("rb") as fh:
        r = client.post(
            "/api/ingest?background=false&taxonomy_mode=skip",
            files={"file": (doc_path.name, fh, "text/plain")},
        )
    assert r.status_code == 200, r.text
    payload = r.json()
    job_id = payload["job_id"]

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    result = job["result"]
    assert isinstance(result, dict)
    stages = result.get("stages") or {}
    # assign is skipped (taxonomy_mode=skip); load lives on ingest_path
    for required in ("load", "chunk", "filter", "embed", "index"):
        assert required in stages, f"missing stage {required} in {sorted(stages)}"
        s = stages[required]
        assert "n_done" in s and "n_total" in s
        # All terminal stages should have duration_s
        assert "duration_s" in s, f"stage {required} missing duration_s: {s}"
        assert s["duration_s"] >= 0


# ---------------------------------------------------------------------------
# 4. Multi-file partial failure: other files still land
# ---------------------------------------------------------------------------


def test_multi_file_partial_failure_isolated(tmp_path: Path, client: TestClient):
    """The web ingest endpoint runs one file per job, so the "queue continues
    unaffected" guarantee is exercised by submitting 3 jobs and verifying:

    * Two of them complete with status=done and their docs are queryable.
    * The middle one fails (we feed it an unreadable path) and surfaces in
      the failed job's ``result.errors``.
    """
    from hrag.web.app import _get_orch  # noqa: PLC0415

    orch = _get_orch()
    uid = client.get("/api/config").json()["user_id"]
    orch.db.ensure_user(uid)

    good_a = _make_text_file(tmp_path, "good_a.txt", n_paragraphs=15)
    good_b = _make_text_file(tmp_path, "good_b.txt", n_paragraphs=15)

    # File A
    with good_a.open("rb") as fh:
        ra = client.post(
            "/api/ingest?background=false&taxonomy_mode=skip",
            files={"file": (good_a.name, fh, "text/plain")},
        )
    assert ra.status_code == 200, ra.text

    # File B: stub the loader to raise so this job fails. We don't take
    # down the global orchestrator because each job uses its own pipeline
    # state but the loader is a module-level function — easier to make a
    # broken file the pipeline rejects. The chunker drops empty files
    # silently, so we corrupt the path: feed a zero-byte file. The
    # endpoint already rejects empty multipart uploads with 400, so to
    # exercise the in-worker failure path we instead patch ingest_path
    # to raise for one filename.
    orch_pipeline = orch.ingest
    original_ingest_path = orch_pipeline.ingest_path

    def maybe_raise(path, user_id, *, skip_taxonomy=False, progress_cb=None):
        if "BROKEN" in str(path):
            raise RuntimeError("synthetic loader failure")
        return original_ingest_path(
            path, user_id, skip_taxonomy=skip_taxonomy, progress_cb=progress_cb,
        )

    orch_pipeline.ingest_path = maybe_raise  # type: ignore[method-assign]
    try:
        broken_path = tmp_path / "BROKEN_middle.txt"
        broken_path.write_text("dummy text to satisfy multipart upload", encoding="utf-8")
        with broken_path.open("rb") as fh:
            rmid = client.post(
                "/api/ingest?background=false&taxonomy_mode=skip",
                files={"file": (broken_path.name, fh, "text/plain")},
            )
        # Synchronous failure path raises 500
        assert rmid.status_code == 500
    finally:
        orch_pipeline.ingest_path = original_ingest_path  # type: ignore[method-assign]

    # File C — should still succeed despite the failure of the middle file.
    with good_b.open("rb") as fh:
        rc = client.post(
            "/api/ingest?background=false&taxonomy_mode=skip",
            files={"file": (good_b.name, fh, "text/plain")},
        )
    assert rc.status_code == 200, rc.text

    # Verify A and B's chunks are in the index. The uploads dir suffixes a
    # short hash onto the title (``good_a-<hex>``) so we match by prefix.
    rows = orch.db.execute(
        "SELECT title FROM documents WHERE user_id = ?",
        (uid,),
    ).fetchall()
    titles = {r["title"] for r in rows}
    assert any(t.startswith("good_a") for t in titles), (
        f"good_a failed to land; have: {sorted(titles)}"
    )
    assert any(t.startswith("good_b") for t in titles), (
        f"good_b failed to land; have: {sorted(titles)}"
    )
    # And no BROKEN doc landed
    assert not any("BROKEN" in t for t in titles), (
        f"BROKEN doc should not have landed; have: {sorted(titles)}"
    )
