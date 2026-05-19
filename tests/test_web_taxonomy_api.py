"""Tests for the taxonomy REST + SSE surface added to ``hrag.web.app``.

Covers:
- GET /api/taxonomy/tree (empty + populated)
- GET /api/taxonomy/unfiled
- POST /api/taxonomy/nodes (create)
- PUT /api/taxonomy/nodes/{id} (rename + move + cycle rejection)
- DELETE /api/taxonomy/nodes/{id} (reparents children)
- POST /api/taxonomy/move-doc
- POST /api/taxonomy/clear
- POST /api/taxonomy/recompute (SSE happy path + error path)

The SSE tests inject a stub builder via ``_TaxonomyState.builder_factory`` so
the test doesn't have to run the real LLM-driven build pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from hrag.web.app import _State, _TaxonomyState, app  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _reset_state() -> None:
    _State.cfg = None
    _State.orch = None
    _TaxonomyState.builder_factory = None
    _TaxonomyState.assigner_factory = None


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


def _ensure_clean_tree(client: TestClient) -> None:
    """Wipe any taxonomy state lingering from a prior run on this DB."""
    r = client.post("/api/taxonomy/clear")
    assert r.status_code == 200, r.text


def _parse_sse(raw: str) -> list[dict[str, Any]]:
    """Parse SSE chunks separated by blank lines into [{event, data}]."""
    events: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev_type: Optional[str] = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if ev_type is None:
            continue
        raw_data = "\n".join(data_lines)
        try:
            data = json.loads(raw_data)
        except Exception:
            data = raw_data
        events.append({"event": ev_type, "data": data})
    return events


def _insert_doc(client: TestClient, doc_id: str, title: str = "Sample") -> None:
    """Insert a row directly into ``documents`` so move-doc has something
    to point at. We bypass the ingest pipeline (no chunks needed)."""
    from hrag.web.app import _get_orch  # noqa: PLC0415

    orch = _get_orch()
    uid = _get_uid(client)
    orch.db.ensure_user(uid)
    with orch.db.conn:
        orch.db.execute(
            "INSERT OR REPLACE INTO documents(doc_id, user_id, source_path, title) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, uid, f"/tmp/{doc_id}.txt", title),
        )
    orch.db.commit()


def _delete_doc(client: TestClient, doc_id: str) -> None:
    from hrag.web.app import _get_orch  # noqa: PLC0415

    orch = _get_orch()
    with orch.db.conn:
        orch.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    orch.db.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tree_empty_returns_null_root(client: TestClient) -> None:
    _ensure_clean_tree(client)
    r = client.get("/api/taxonomy/tree")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["root"] is None
    assert body["node_count"] == 0
    assert body["doc_count"] == 0


def test_create_two_nodes_and_tree_reflects_them(client: TestClient) -> None:
    _ensure_clean_tree(client)
    r1 = client.post(
        "/api/taxonomy/nodes",
        json={"label": "Research", "description": "papers"},
    )
    assert r1.status_code == 200, r1.text
    parent_id = r1.json()["node_id"]

    r2 = client.post(
        "/api/taxonomy/nodes",
        json={"label": "Dynamics", "description": "dyn", "parent_id": parent_id, "is_leaf": True},
    )
    assert r2.status_code == 200, r2.text
    leaf_id = r2.json()["node_id"]

    tree = client.get("/api/taxonomy/tree").json()
    assert tree["root"] is not None
    # Root has at least one child (Research) — ensure_root may add more.
    labels = [c["label"] for c in tree["root"]["children"]]
    assert "Research" in labels, labels
    research = next(c for c in tree["root"]["children"] if c["label"] == "Research")
    leaf_labels = [c["label"] for c in research["children"]]
    assert "Dynamics" in leaf_labels
    # leaf id appears as a direct child
    assert any(c["node_id"] == leaf_id for c in research["children"])


def test_rename_node(client: TestClient) -> None:
    _ensure_clean_tree(client)
    r = client.post("/api/taxonomy/nodes", json={"label": "Old"})
    nid = r.json()["node_id"]
    r2 = client.put(
        f"/api/taxonomy/nodes/{nid}",
        json={"label": "New", "description": "updated"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["label"] == "New"
    assert r2.json()["description"] == "updated"


def test_move_node(client: TestClient) -> None:
    _ensure_clean_tree(client)
    a = client.post("/api/taxonomy/nodes", json={"label": "A"}).json()["node_id"]
    b = client.post("/api/taxonomy/nodes", json={"label": "B"}).json()["node_id"]
    # move A under B
    r = client.put(f"/api/taxonomy/nodes/{a}", json={"parent_id": b})
    assert r.status_code == 200, r.text
    assert r.json()["parent_id"] == b


def test_move_cycle_rejected(client: TestClient) -> None:
    _ensure_clean_tree(client)
    a = client.post("/api/taxonomy/nodes", json={"label": "A"}).json()["node_id"]
    b = client.post(
        "/api/taxonomy/nodes", json={"label": "B", "parent_id": a}
    ).json()["node_id"]
    # Try to move A under B → cycle.
    r = client.put(f"/api/taxonomy/nodes/{a}", json={"parent_id": b})
    assert r.status_code == 400, r.text
    assert "cycle" in r.json()["detail"].lower() or "descendant" in r.json()["detail"].lower()


def test_delete_node_reparents_children(client: TestClient) -> None:
    _ensure_clean_tree(client)
    parent = client.post("/api/taxonomy/nodes", json={"label": "P"}).json()["node_id"]
    mid = client.post(
        "/api/taxonomy/nodes", json={"label": "Mid", "parent_id": parent}
    ).json()["node_id"]
    child = client.post(
        "/api/taxonomy/nodes", json={"label": "Child", "parent_id": mid, "is_leaf": True}
    ).json()["node_id"]

    r = client.delete(f"/api/taxonomy/nodes/{mid}")
    assert r.status_code == 200, r.text

    # The child should now be a direct child of parent.
    tree = client.get("/api/taxonomy/tree").json()

    def _find(node, target):
        if node["node_id"] == target:
            return node
        for c in node.get("children", []):
            hit = _find(c, target)
            if hit is not None:
                return hit
        return None

    found_child = _find(tree["root"], child)
    assert found_child is not None
    assert found_child["parent_id"] == parent


def test_cannot_delete_root(client: TestClient) -> None:
    _ensure_clean_tree(client)
    # Force-create the root via add then read.
    client.post("/api/taxonomy/nodes", json={"label": "X"})
    tree = client.get("/api/taxonomy/tree").json()
    root_id = tree["root"]["node_id"]
    r = client.delete(f"/api/taxonomy/nodes/{root_id}")
    assert r.status_code == 400


def test_move_doc_reassigns(client: TestClient) -> None:
    _ensure_clean_tree(client)
    a = client.post(
        "/api/taxonomy/nodes", json={"label": "BucketA", "is_leaf": True}
    ).json()["node_id"]
    b = client.post(
        "/api/taxonomy/nodes", json={"label": "BucketB", "is_leaf": True}
    ).json()["node_id"]
    doc_id = "tax_test_doc_1"
    _insert_doc(client, doc_id, title="MoveMe")
    try:
        # File it under A first via move-doc.
        r1 = client.post(
            "/api/taxonomy/move-doc",
            json={"doc_id": doc_id, "node_id": a},
        )
        assert r1.status_code == 200, r1.text
        # Confirm via /nodes/{a}/docs.
        ra = client.get(f"/api/taxonomy/nodes/{a}/docs").json()
        assert any(d["doc_id"] == doc_id for d in ra["docs"]), ra

        # Now move it to B.
        r2 = client.post(
            "/api/taxonomy/move-doc",
            json={"doc_id": doc_id, "node_id": b},
        )
        assert r2.status_code == 200, r2.text
        rb = client.get(f"/api/taxonomy/nodes/{b}/docs").json()
        ra = client.get(f"/api/taxonomy/nodes/{a}/docs").json()
        assert any(d["doc_id"] == doc_id for d in rb["docs"]), rb
        assert not any(d["doc_id"] == doc_id for d in ra["docs"]), ra
    finally:
        _delete_doc(client, doc_id)


def test_clear_drops_tree(client: TestClient) -> None:
    client.post("/api/taxonomy/nodes", json={"label": "Foo"})
    r = client.post("/api/taxonomy/clear")
    assert r.status_code == 200, r.text
    tree = client.get("/api/taxonomy/tree").json()
    assert tree["root"] is None
    assert tree["node_count"] == 0


def test_unfiled_lists_docs_without_assignment(client: TestClient) -> None:
    _ensure_clean_tree(client)
    doc_id = "tax_test_doc_unfiled"
    _insert_doc(client, doc_id, title="UnfiledDoc")
    try:
        r = client.get("/api/taxonomy/unfiled")
        assert r.status_code == 200, r.text
        ids = [d["doc_id"] for d in r.json()]
        assert doc_id in ids
    finally:
        _delete_doc(client, doc_id)


# ---------------------------------------------------------------------------
# SSE tests (use a stub builder so we don't run the LLM)
# ---------------------------------------------------------------------------


class _StubBuilder:
    """Emit a fixed sequence of progress stages."""

    def __init__(self, stages: Optional[list[tuple[str, dict]]] = None,
                 raise_after: Optional[int] = None) -> None:
        self.stages = stages or [
            ("start", {"user_id": "default"}),
            ("doc_summary", {"i": 1, "n": 1, "doc_id": "d"}),
            ("summaries_done", {"n_summarized": 1}),
            ("propose_tree_done", {"n_nodes": 1}),
            ("done", {"total_duration_s": 0.01, "n_nodes": 1}),
        ]
        self._raise_after = raise_after

    def build(self, user_id: str, *, progress: Callable[[str, dict], None]) -> None:
        for i, (stage, payload) in enumerate(self.stages):
            if self._raise_after is not None and i == self._raise_after:
                raise RuntimeError("simulated builder failure")
            progress(stage, payload)

    # Also expose the legacy name so the endpoint can fall back to either.
    def build_for_user(self, user_id: str, *, progress: Callable[[str, dict], None]) -> None:
        self.build(user_id, progress=progress)


def test_recompute_sse_emits_stage_events(client: TestClient) -> None:
    _TaxonomyState.builder_factory = lambda: _StubBuilder()
    with client.stream("POST", "/api/taxonomy/recompute", json={}) as resp:
        assert resp.status_code == 200, resp.read()
        raw = b"".join(resp.iter_bytes()).decode("utf-8")
    events = _parse_sse(raw)
    assert events, "no SSE events received"
    types = [ev["event"] for ev in events]
    assert types[0] == "open", types
    assert "stage" in types, types
    # The very first stage should be the "start" event we put in the stub.
    stage_events = [ev for ev in events if ev["event"] == "stage"]
    assert stage_events[0]["data"].get("stage") == "start", stage_events[0]
    # Final event is "done" so the client can close.
    assert types[-1] == "done", types


def test_recompute_sse_emits_error_on_builder_failure(client: TestClient) -> None:
    _TaxonomyState.builder_factory = lambda: _StubBuilder(raise_after=1)
    with client.stream("POST", "/api/taxonomy/recompute", json={}) as resp:
        assert resp.status_code == 200, resp.read()
        raw = b"".join(resp.iter_bytes()).decode("utf-8")
    events = _parse_sse(raw)
    types = [ev["event"] for ev in events]
    # The error event MUST appear and we still end with done.
    assert "error" in types, types
    err = next(ev for ev in events if ev["event"] == "error")
    assert "simulated builder failure" in err["data"].get("message", ""), err
    assert types[-1] == "done", types
