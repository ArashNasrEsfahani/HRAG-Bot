"""Tests for POST /api/memories/extract — the Smart Remember endpoint.

Five tests via FastAPI's ``TestClient``:

1. Empty DB (no sessions) → ``{items: [], session_id: null, n_turns_considered: 0}``.
2. Session with messages → extractor runs against a scripted LLM and the
   response contains reshaped items.
3. ``session_id`` that does not exist → ``{items: [], n_turns_considered: 0}``
   (no 404; the modal expects an empty list rather than an error).
4. ``max_items: 3`` caps the response length.
5. LLM that returns garbage does not crash the endpoint — 200 + ``items: []``.

Each test injects a tiny scripted LLM into ``orch.llm`` after the orchestrator
singleton is built, mirroring the pattern in tests/test_web_taxonomy_api.py
and tests/test_orchestrator.py.
"""

from __future__ import annotations

import sys
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


class _ScriptedLLM:
    """Returns a canned string from ``complete`` regardless of the prompt."""

    name = "scripted"

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[str] = []

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        self.calls.append(prompt)
        return self._response


def _install_llm(response: str) -> _ScriptedLLM:
    """Build the orchestrator, then swap its LLM for a scripted stub."""
    orch = _get_orch()
    stub = _ScriptedLLM(response)
    orch.llm = stub
    return stub


def _seed_session(uid: str, turns: list[tuple[str, str]]) -> str:
    """Insert a session + N messages directly. Returns the session_id."""
    import uuid  # noqa: PLC0415

    orch = _get_orch()
    orch.db.ensure_user(uid)
    sid = uuid.uuid4().hex
    with orch.db.conn:
        orch.db.execute(
            "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
            (sid, uid),
        )
        for i, (role, content) in enumerate(turns):
            # ``message_id`` is INTEGER AUTOINCREMENT — let SQLite assign it.
            # We force ``created_at`` so ORDER BY created_at ASC matches the
            # insert order even when adjacent rows would otherwise share the
            # same one-second resolution timestamp.
            orch.db.execute(
                "INSERT INTO messages (session_id, user_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, datetime('now', ?))",
                (sid, uid, role, content, f"+{i} seconds"),
            )
    orch.db.commit()
    return sid


def _purge_sessions(uid: str) -> None:
    """Drop every session row for this user — used by Test 1 because the
    project-root ``data/store.sqlite`` may already contain real sessions."""
    orch = _get_orch()
    rows = orch.db.execute(
        "SELECT session_id FROM sessions WHERE user_id = ?", (uid,)
    ).fetchall()
    with orch.db.conn:
        for r in rows:
            orch.db.execute(
                "DELETE FROM messages WHERE session_id = ?", (r["session_id"],)
            )
            orch.db.execute(
                "DELETE FROM sessions WHERE session_id = ?", (r["session_id"],)
            )
    orch.db.commit()


def _uid(client: TestClient) -> str:
    return client.get("/api/config").json()["user_id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_empty_db_returns_no_session(client: TestClient) -> None:
    """No sessions exist → endpoint short-circuits with an empty response."""
    _install_llm("[]")  # never reached, but keep the harness clean
    uid = _uid(client)
    _purge_sessions(uid)

    r = client.post("/api/memories/extract", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"items": [], "session_id": None, "n_turns_considered": 0}


def test_extract_from_4_message_session_returns_items(client: TestClient) -> None:
    """A scripted LLM returns 2 candidates; the endpoint reshapes them."""
    raw = (
        '[{"polarity": "fact",  "topic": "occupation", "value": "data engineer", "confidence": 0.95},'
        ' {"polarity": "like",  "topic": "language",   "value": "Python",        "confidence": 0.8}]'
    )
    _install_llm(raw)
    uid = _uid(client)
    sid = _seed_session(uid, [
        ("user", "I'm a data engineer based in Singapore."),
        ("assistant", "Got it. Anything else?"),
        ("user", "I prefer Python over R."),
        ("assistant", "Noted."),
    ])

    r = client.post("/api/memories/extract", json={"session_id": sid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == sid
    assert body["n_turns_considered"] == 4
    items = body["items"]
    assert len(items) == 2
    # First item — fact
    assert items[0]["category"] == "fact"
    assert "data engineer" in items[0]["text"]
    assert items[0]["confidence"] == pytest.approx(0.95)
    # Second item — like
    assert items[1]["category"] == "like"
    assert "Python" in items[1]["text"]


def test_extract_nonexistent_session_returns_empty(client: TestClient) -> None:
    """Unknown session_id → empty payload, NOT a 404 (modal renders empty state)."""
    _install_llm("[]")
    r = client.post(
        "/api/memories/extract",
        json={"session_id": "does-not-exist-12345"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["items"] == []
    assert body["n_turns_considered"] == 0
    assert body["session_id"] == "does-not-exist-12345"


def test_extract_respects_max_items(client: TestClient) -> None:
    """``max_items: 3`` caps the response to at most 3 items."""
    # Six candidates — extractor will return all 6; endpoint must trim to 3.
    raw = (
        "["
        '{"polarity":"fact","topic":"a","value":"v1","confidence":0.9},'
        '{"polarity":"fact","topic":"b","value":"v2","confidence":0.9},'
        '{"polarity":"like","topic":"c","value":"v3","confidence":0.9},'
        '{"polarity":"like","topic":"d","value":"v4","confidence":0.9},'
        '{"polarity":"style","topic":"e","value":"v5","confidence":0.9},'
        '{"polarity":"dislike","topic":"f","value":"v6","confidence":0.9}'
        "]"
    )
    _install_llm(raw)
    uid = _uid(client)
    sid = _seed_session(uid, [
        ("user", "Some user turn."),
        ("assistant", "Some assistant turn."),
    ])

    r = client.post(
        "/api/memories/extract",
        json={"session_id": sid, "max_items": 3},
    )
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert len(items) == 3


def test_extract_garbage_llm_output_returns_empty(client: TestClient) -> None:
    """LLM returns non-JSON garbage → endpoint returns 200 with items=[]."""
    _install_llm("definitely not json — sorry, I can't help with that")
    uid = _uid(client)
    sid = _seed_session(uid, [
        ("user", "Hello."),
        ("assistant", "Hi there."),
    ])

    r = client.post("/api/memories/extract", json={"session_id": sid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["items"] == []
    assert body["session_id"] == sid
    assert body["n_turns_considered"] == 2
