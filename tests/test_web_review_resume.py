"""Phase 8 — web /resume endpoint, /why_source endpoint, and SSE relay tests.

These tests exercise the FastAPI surface only — the orchestrator's chat()
loop is monkey-patched so we don't need a live LLM / vector store. The
``_isolate_web_app_state`` autouse fixture in ``tests/conftest.py`` already
points the singleton at a tmp_path SQLite + Chroma.

Coverage:

* ``test_resume_endpoint_unknown_turn_returns_accepted_false`` — POST a
  turn_id that was never registered; expect 200 with ``accepted=False``.
* ``test_resume_endpoint_idempotent`` — register a turn, POST resume twice;
  first ``accepted=True``, second ``accepted=False`` (retry-safe).
* ``test_resume_endpoint_action_validation`` — POST with an invalid action;
  pydantic Literal should produce a 422.
* ``test_resume_endpoint_submits_to_store`` — POST with ``action='filter'``
  and a chunk-id list; verify the store's ``get_decision`` reflects it.
* ``test_why_source_endpoint_disabled`` — flag off → empty explanation, no
  LLM call.
* ``test_why_source_endpoint_enabled_calls_llm`` — flag on + stub LLM →
  returns the LLM's text.
* ``test_sse_relay_review_required_passes_through`` — orchestrator emits a
  ``review_required`` progress event; verify the SSE stream carries it as
  ``event: review_required`` (not generic ``progress``).
* ``test_sse_relay_review_resolved_event`` — same for ``review_resolved``.
* ``test_sse_relay_followups_event`` — same for ``followups``.
* ``test_sse_relay_start_event_carries_turn_id`` — when the orchestrator
  emits a ``start`` event with a ``turn_id``, the relay must preserve it.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from hrag.interaction.store import InteractionStore  # noqa: E402
from hrag.web.app import _State, app  # noqa: E402


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
def client() -> TestClient:
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# /resume endpoint
# ---------------------------------------------------------------------------


def test_resume_endpoint_unknown_turn_returns_accepted_false(client: TestClient) -> None:
    """Unknown turn_id → 200 with accepted=False (no 404).

    The frontend must be able to retry a /resume after a flaky network blip
    without crashing on a 404.
    """
    r = client.post(
        "/api/chat/turns/turn_does_not_exist/resume",
        json={"action": "continue"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is False
    assert body["turn_id"] == "turn_does_not_exist"
    assert "reason" in body


def test_resume_endpoint_idempotent(client: TestClient) -> None:
    """First POST → accepted=True; second POST on the same turn → accepted=False."""
    # Force-create the orchestrator + interaction store.
    r0 = client.get("/api/config")
    assert r0.status_code == 200
    orch = _State.orch
    assert orch is not None
    store = getattr(orch, "interaction_store", None)
    if store is None:
        store = InteractionStore()
        orch.interaction_store = store

    turn_id = "turn_idempotent_1"
    store.create_turn(turn_id)

    r1 = client.post(
        f"/api/chat/turns/{turn_id}/resume",
        json={"action": "continue"},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["accepted"] is True

    r2 = client.post(
        f"/api/chat/turns/{turn_id}/resume",
        json={"action": "continue"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["accepted"] is False


def test_resume_endpoint_action_validation(client: TestClient) -> None:
    """An action outside the Literal set must be rejected with 422 by pydantic."""
    r = client.post(
        "/api/chat/turns/anything/resume",
        json={"action": "not_a_real_action"},
    )
    assert r.status_code == 422, r.text


def test_resume_endpoint_submits_to_store(client: TestClient) -> None:
    """A POST with action='filter' must land in the store with the full payload."""
    r0 = client.get("/api/config")
    assert r0.status_code == 200
    orch = _State.orch
    assert orch is not None
    store = getattr(orch, "interaction_store", None)
    if store is None:
        store = InteractionStore()
        orch.interaction_store = store

    turn_id = "turn_submit_2"
    store.create_turn(turn_id)

    payload = {
        "action": "filter",
        "selected_chunk_ids": ["c1", "c2", "c3"],
        "include_episodic": True,
        "remember_choice": True,
    }
    r = client.post(f"/api/chat/turns/{turn_id}/resume", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["accepted"] is True

    decision = store.get_decision(turn_id)
    assert decision is not None
    assert decision["action"] == "filter"
    assert decision["selected_chunk_ids"] == ["c1", "c2", "c3"]
    assert decision["include_episodic"] is True
    assert decision["remember_choice"] is True
    # Optional fields default to None / False but must still round-trip.
    assert decision["rewritten_query"] is None
    assert decision["expand_from_doc_id"] is None


# ---------------------------------------------------------------------------
# /why_source endpoint
# ---------------------------------------------------------------------------


def test_why_source_endpoint_disabled(client: TestClient) -> None:
    """When interaction.why_source_enabled=False, the endpoint returns "" and
    must NOT call the LLM (we replace ``orch.llm.complete`` with a tripwire)."""
    r0 = client.get("/api/config")
    assert r0.status_code == 200
    orch = _State.orch
    cfg = _State.cfg
    assert orch is not None and cfg is not None
    cfg.interaction.why_source_enabled = False

    def _explode(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("LLM must not be called when why_source is disabled")

    orch.llm.complete = _explode  # type: ignore[assignment]

    r = client.post(
        "/api/chat/turns/turn-1/why_source",
        json={"question": "q", "title": "t", "section": "s", "text": "p"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"explanation": ""}


def test_why_source_endpoint_enabled_calls_llm(client: TestClient) -> None:
    """When the flag is on, we stub LLM.complete and verify its output is returned."""
    r0 = client.get("/api/config")
    assert r0.status_code == 200
    orch = _State.orch
    cfg = _State.cfg
    assert orch is not None and cfg is not None
    cfg.interaction.why_source_enabled = True

    captured: dict[str, Any] = {}

    def _stub(prompt: str, *, temperature: float = 0.2, max_tokens: int = 80, **_kw: Any) -> str:
        captured["prompt"] = prompt
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return "  The passage defines the loss function. "

    orch.llm.complete = _stub  # type: ignore[assignment]

    r = client.post(
        "/api/chat/turns/turn-X/why_source",
        json={
            "question": "what is the loss function?",
            "title": "Paper Title",
            "section": "3.2 Loss",
            "text": "We define L(theta) = ...",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["explanation"] == "The passage defines the loss function."
    # Verify the LLM saw the rendered template (sanity check on placeholders).
    assert "what is the loss function?" in captured["prompt"]
    assert "Paper Title" in captured["prompt"]
    assert "3.2 Loss" in captured["prompt"]


# ---------------------------------------------------------------------------
# SSE relay — exercise via /api/chat with a monkey-patched Orchestrator.chat
# ---------------------------------------------------------------------------


@dataclass
class _FakeChatResult:
    session_id: str = "sess-fake"
    answer: str = "fake answer"
    sources: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sources is None:
            self.sources = []


def _drive_chat_with_events(
    client: TestClient,
    events_to_emit: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """POST /api/chat with a stubbed Orchestrator.chat that fires the given
    progress events in order, then returns. Parse the SSE response and return
    the list of {event, data} dicts."""
    # Force the orchestrator singleton to build now.
    r0 = client.get("/api/config")
    assert r0.status_code == 200
    orch = _State.orch
    assert orch is not None

    def _fake_chat(
        message: str,
        *,
        user_id: str = "u",
        session_id: str | None = None,
        progress=None,
        stream: bool = True,
    ) -> _FakeChatResult:
        for ev, payload in events_to_emit:
            if progress is not None:
                progress(ev, payload)
        return _FakeChatResult()

    orch.chat = _fake_chat  # type: ignore[assignment]

    with client.stream("POST", "/api/chat", json={"message": "hi"}, timeout=30) as stream:
        raw = stream.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    events: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev_type: str | None = None
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


def test_sse_relay_review_required_passes_through(client: TestClient) -> None:
    """A ``review_required`` progress event must surface as its OWN SSE event
    type (not buried inside a generic ``progress`` packet)."""
    payload = {
        "turn_id": "abc123",
        "reasons": ["score_floor", "ambiguity_delta"],
        "candidates": [{"chunk_id": "c1", "title": "T"}],
    }
    events = _drive_chat_with_events(client, [("review_required", payload)])
    event_types = [e["event"] for e in events]
    assert "review_required" in event_types, (
        f"review_required missing from SSE stream; got {event_types}"
    )
    # Confirm the payload survived the relay.
    rr = next(e for e in events if e["event"] == "review_required")
    assert rr["data"]["turn_id"] == "abc123"
    assert rr["data"]["reasons"] == ["score_floor", "ambiguity_delta"]


def test_sse_relay_review_resolved_event(client: TestClient) -> None:
    """A ``review_resolved`` event must surface as its own SSE event type."""
    payload = {"action": "filter", "timed_out": False, "reasons": ["score_floor"]}
    events = _drive_chat_with_events(client, [("review_resolved", payload)])
    event_types = [e["event"] for e in events]
    assert "review_resolved" in event_types, (
        f"review_resolved missing from SSE stream; got {event_types}"
    )
    rr = next(e for e in events if e["event"] == "review_resolved")
    assert rr["data"]["action"] == "filter"
    assert rr["data"]["timed_out"] is False


def test_sse_relay_followups_event(client: TestClient) -> None:
    """A ``followups`` event must surface as its own SSE event type."""
    payload = {"chips": ["What about X?", "Compare Y and Z", "Why so?"]}
    events = _drive_chat_with_events(client, [("followups", payload)])
    event_types = [e["event"] for e in events]
    assert "followups" in event_types, (
        f"followups missing from SSE stream; got {event_types}"
    )
    fu = next(e for e in events if e["event"] == "followups")
    assert fu["data"]["chips"] == ["What about X?", "Compare Y and Z", "Why so?"]


def test_sse_relay_start_event_carries_turn_id(client: TestClient) -> None:
    """When the orchestrator emits a ``start`` event with a ``turn_id`` field,
    the SSE relay must preserve it (wave 3 frontend reads turn_id from start)."""
    payload = {"turn_id": "turn-42", "session_id": "sess-1"}
    events = _drive_chat_with_events(client, [("start", payload)])
    # ``start`` is unknown to the relay → goes through as generic ``progress``
    # with the original event name + payload nested inside.
    progress_events = [e for e in events if e["event"] == "progress"]
    starts = [
        e for e in progress_events
        if isinstance(e["data"], dict) and e["data"].get("event") == "start"
    ]
    assert starts, f"start progress event missing from SSE stream; got {[e['event'] for e in events]}"
    payload_seen = starts[0]["data"]["payload"]
    assert payload_seen.get("turn_id") == "turn-42", (
        f"turn_id missing from start event payload: {payload_seen!r}"
    )
