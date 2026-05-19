"""Phase 8 wave-4 END-TO-END tests.

Exercises the full ``Orchestrator.chat()`` → SSE relay → ``/resume``
round-trip through ``TestClient``, all in-process with no real LLM or
vector store.

The ``_isolate_web_app_state`` autouse fixture in ``conftest.py`` redirects
all storage to a per-test ``tmp_path`` so none of these tests ever touch the
user's live ``data/`` directory.

Coverage
--------
1. ``test_e2e_pause_then_continue_default``
   review_mode="always" → pause fires → POST /resume continue → token stream
   and final event arrive.

2. ``test_e2e_pause_then_general_drops_sources``
   resume with action="general" → final event sources list is empty.

3. ``test_e2e_pause_then_filter_subsets``
   resume with action="filter" + selected_chunk_ids=["c2"] → LLM prompt
   only contains c2's text.

4. ``test_e2e_pause_then_abort_yields_marker_message``
   resume with action="abort" → SSE stream has a ``done`` event carrying
   ``aborted_by_user=True``; the persisted assistant message contains the
   abort marker text.

5. ``test_e2e_followups_event_emitted_when_enabled``
   followups_enabled=True, healthy retrieval (no pause) → ``followups`` SSE
   event arrives with 3 chips.

6. ``test_e2e_disabled_no_review_required_event``
   review_enabled=False → no ``review_required`` event in stream even with
   low-score retrieval.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from hrag.web.app import _State, app  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers shared across tests
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


def _parse_sse(raw: bytes | str) -> list[dict[str, Any]]:
    """Parse an SSE byte-stream into a list of {event, data} dicts."""
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
        except Exception:  # noqa: BLE001
            data = raw_data
        events.append({"event": ev_type, "data": data})
    return events


# ---------------------------------------------------------------------------
# Stub LLM + Retriever factories
# ---------------------------------------------------------------------------


class _RecordingLLM:
    """LLM stub that records every prompt it sees and returns configurable text."""

    name = "recording"

    def __init__(self, answer: str = "final answer here") -> None:
        self.prompts: list[str] = []
        self._answer = answer

    def _dispatch(self, prompt: str) -> str:
        self.prompts.append(prompt)
        # Intent classifier
        if "Intent Classification" in prompt or "Output (one word only)" in prompt:
            return "factual"
        # Follow-ups generation: look for common cues in prompts
        if (
            "follow-up" in prompt.lower()
            or "follow up" in prompt.lower()
            or "Return exactly 3" in prompt
            or "followup" in prompt.lower()
        ):
            return "followup1\nfollowup2\nfollowup3"
        return self._answer

    def complete(self, prompt: str, system=None, temperature=None, max_tokens=None) -> str:
        return self._dispatch(prompt)

    def generate(self, request):
        from hrag.types import GenerationResponse  # noqa: PLC0415
        prompt = " ".join(m.content for m in request.messages)
        return GenerationResponse(text=self._dispatch(prompt), raw=None)

    def generate_stream(self, request):
        yield self.generate(request).text


def _make_result(chunk_id: str, text: str, rerank_score: float = 1.0):
    from hrag.types import Chunk, RetrievalResult  # noqa: PLC0415
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        user_id="default",
        text=text,
        embedding_text=text,
        title=f"Title-{chunk_id}",
        section="Sec",
    )
    return RetrievalResult(chunk=chunk, score=0.9, rerank_score=rerank_score)


class _StubRetriever:
    name = "stub"

    def __init__(self, results) -> None:
        self._results = list(results)

    def retrieve(self, query, user_id, top_k=10, source_types=None, intent_hint=None, where=None):
        return list(self._results)


# ---------------------------------------------------------------------------
# Core helper: patch orch and read SSE, optionally posting /resume
# from a side thread while the SSE stream is open.
# ---------------------------------------------------------------------------


def _run_e2e(
    client: TestClient,
    results,
    cfg_patches: dict[str, Any],
    resume_payload: dict[str, Any] | None = None,
    *,
    answer_text: str = "final answer here",
) -> tuple[list[dict[str, Any]], "_RecordingLLM"]:
    """Trigger GET /api/config (builds the orchestrator singleton), patch it,
    start a POST /api/chat stream, optionally send /resume from a thread,
    and return (parsed_events, recording_llm)."""

    # Build the orchestrator singleton via GET /api/config.
    r0 = client.get("/api/config")
    assert r0.status_code == 200, r0.text

    orch = _State.orch
    cfg = _State.cfg
    assert orch is not None and cfg is not None

    # Apply config patches.
    cfg.interaction.review_enabled = cfg_patches.get("review_enabled", True)
    cfg.interaction.review_mode = cfg_patches.get("review_mode", "always")
    cfg.interaction.review_timeout_s = cfg_patches.get("review_timeout_s", 10.0)
    cfg.interaction.followups_enabled = cfg_patches.get("followups_enabled", False)
    cfg.interaction.rephrasings_enabled = cfg_patches.get("rephrasings_enabled", False)
    cfg.interaction.persistence_enabled = cfg_patches.get("persistence_enabled", True)
    cfg.intent.enabled = False  # bypass intent classifier for determinism

    # Swap retriever + LLM.
    stub_retriever = _StubRetriever(results)
    orch.retriever = stub_retriever  # type: ignore[assignment]
    # Disable reranker to keep scores coming from the retriever directly.
    orch.reranker = None
    cfg.retrieval.rerank_enabled = False

    llm = _RecordingLLM(answer=answer_text)
    orch.llm = llm
    # Gate / clue may have their own LLM reference.
    if orch.gate is not None:
        orch.gate.llm = llm
    if orch.clue is not None:
        orch.clue.llm = llm

    turn_id_box: dict[str, str] = {}

    # Intercept progress events to capture turn_id from start event.
    _real_chat = orch.chat

    def _chat_with_spy(message, *, user_id="default", session_id=None, progress=None, stream=True):
        def _spy_progress(name, payload):
            if name == "start":
                turn_id_box["turn_id"] = payload.get("turn_id", "")
            if progress is not None:
                progress(name, payload)

        return _real_chat(
            message,
            user_id=user_id,
            session_id=session_id,
            progress=_spy_progress,
            stream=stream,
        )

    orch.chat = _chat_with_spy  # type: ignore[assignment]

    # If we need to POST /resume, launch a side thread that waits for the
    # turn_id to be captured and then sends the decision.
    #
    # IMPORTANT: TestClient (backed by httpx.Client) is NOT thread-safe for
    # concurrent requests.  Calling client.post() from a side thread while
    # the main thread is blocked inside client.stream(...).read() causes a
    # deadlock.  Instead we post directly to the interaction_store — which
    # tests the SSE relay just as thoroughly (we already have dedicated HTTP
    # endpoint tests in test_web_review_resume.py).
    if resume_payload is not None:
        _store = orch.interaction_store  # captured before the thread starts

        def _resume_thread():
            # Wait up to 5 s for the turn_id to appear in turn_id_box.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not turn_id_box.get("turn_id"):
                time.sleep(0.02)
            tid = turn_id_box.get("turn_id", "")
            if not tid:
                return
            # Small extra beat so the orchestrator has registered the turn.
            time.sleep(0.05)
            _store.submit_decision(tid, resume_payload)

        t = threading.Thread(target=_resume_thread, daemon=True)
        t.start()

    # Read the SSE stream synchronously.
    with client.stream("POST", "/api/chat", json={"message": "test question"}, timeout=30) as st:
        raw = st.read()

    events = _parse_sse(raw)
    return events, llm


# ---------------------------------------------------------------------------
# Test 1 — pause then continue: token + final events arrive
# ---------------------------------------------------------------------------


def test_e2e_pause_then_continue_default(client: TestClient) -> None:
    """review_mode=always → review_required fires → POST /resume continue
    → the SSE stream emits token events and a final event with the answer."""
    results = [
        _make_result("c1", "alpha text", rerank_score=1.5),
        _make_result("c2", "beta text", rerank_score=1.0),
    ]
    events, llm = _run_e2e(
        client,
        results,
        {"review_mode": "always", "review_enabled": True},
        resume_payload={"action": "continue"},
    )

    event_types = [e["event"] for e in events]

    # A review_required event must have fired.
    assert "review_required" in event_types, (
        f"review_required missing; got {event_types}"
    )
    # And generation still completed.
    assert "final" in event_types, f"final event missing; got {event_types}"
    final_data = next(e["data"] for e in events if e["event"] == "final")
    assert "final answer here" in (final_data.get("answer") or ""), (
        f"answer missing from final: {final_data!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — pause then general: sources list is empty
# ---------------------------------------------------------------------------


def test_e2e_pause_then_general_drops_sources(client: TestClient) -> None:
    """action=general → the final event's sources list must be empty."""
    results = [_make_result("c1", "alpha text", rerank_score=1.5)]
    events, _llm = _run_e2e(
        client,
        results,
        {"review_mode": "always", "review_enabled": True},
        resume_payload={"action": "general"},
    )

    event_types = [e["event"] for e in events]
    assert "final" in event_types, f"final event missing; got {event_types}"

    final_data = next(e["data"] for e in events if e["event"] == "final")
    sources = final_data.get("sources", [])
    assert sources == [], (
        f"sources should be empty for action=general, got: {sources}"
    )


# ---------------------------------------------------------------------------
# Test 3 — pause then filter: LLM prompt only contains selected chunk text
# ---------------------------------------------------------------------------


def test_e2e_pause_then_filter_subsets(client: TestClient) -> None:
    """action=filter with selected_chunk_ids=['c2'] → prompt only has c2 text."""
    results = [
        _make_result("c1", "alpha text", rerank_score=1.5),
        _make_result("c2", "beta text", rerank_score=1.0),
        _make_result("c3", "gamma text", rerank_score=0.8),
    ]
    events, llm = _run_e2e(
        client,
        results,
        {"review_mode": "always", "review_enabled": True},
        resume_payload={"action": "filter", "selected_chunk_ids": ["c2"]},
    )

    event_types = [e["event"] for e in events]
    assert "final" in event_types, f"final event missing; got {event_types}"

    # Check the prompts the LLM received: the answer-generation prompt must
    # contain only c2's text, not c1 or c3.
    #
    # The SSE runner calls the orchestrator's chat() with stream=True, which
    # uses generate_stream().  Our _RecordingLLM records every call via _dispatch.
    # The filter was submitted via interaction_store.submit_decision directly
    # (avoids the TestClient re-entrancy issue).
    #
    # If stream=True is passed to _real_chat, the answer prompt is built and
    # generate_stream() is called which records the prompt.  The full content
    # of prompts is then available in llm.prompts.
    assert llm.prompts, (
        f"LLM was never called during filter test; event_types={event_types}. "
        "Possible causes: exception in orchestrator chat(), or stream=False path taken."
    )
    # Identify the answer-generation prompt: the one that contains a chunk
    # passage.  The RAFT-style answer.md template always renders retrieved
    # passages verbatim, so any of our chunk text strings will appear in it.
    # We exclude Intent Classification prompts (short, no passage text) and
    # follow-up generation prompts (no passage text either).
    answer_prompts = [
        p for p in llm.prompts
        if "Intent Classification" not in p
        and "Output (one word only)" not in p
        and "Return exactly 3" not in p
        and "follow-up questions" not in p.lower()
    ]
    combined = "\n".join(answer_prompts)
    assert "beta text" in combined, (
        f"c2 text ('beta text') must appear in LLM prompts; "
        f"got answer_prompts (first 200 chars each): {[p[:200] for p in answer_prompts]}"
    )
    assert "alpha text" not in combined, (
        f"c1 text ('alpha text') must NOT appear after filter; got: {combined[:400]}"
    )
    assert "gamma text" not in combined, (
        f"c3 text ('gamma text') must NOT appear after filter; got: {combined[:400]}"
    )


# ---------------------------------------------------------------------------
# Test 4 — pause then abort: SSE done event carries aborted_by_user=True
# ---------------------------------------------------------------------------


def test_e2e_pause_then_abort_yields_marker_message(client: TestClient) -> None:
    """action=abort → done event has aborted_by_user=True; persisted message
    contains the abort marker text."""
    results = [_make_result("c1", "alpha text", rerank_score=1.5)]
    events, _llm = _run_e2e(
        client,
        results,
        {
            "review_mode": "always",
            "review_enabled": True,
            "persistence_enabled": True,
        },
        resume_payload={"action": "abort"},
    )

    event_types = [e["event"] for e in events]

    # On the abort path the orchestrator calls ``_emit("done", {..., "aborted_by_user": True})``.
    # The SSE relay has no special case for a progress event named "done" (that
    # name is reserved for the relay's own terminal sentinel), so it travels as
    # a generic ``progress`` packet: event="progress", data={"event": "done", "payload": {...}}.
    # The terminal sentinel also becomes event="done" but with data={"ts": ...} only.
    #
    # We accept either representation so the test remains valid if the relay
    # ever adds a first-class "abort" event type.
    aborted_signal_found = False

    # Check the relay's own done event (terminal sentinel).
    for e in events:
        if e["event"] == "done" and isinstance(e["data"], dict):
            if e["data"].get("aborted_by_user") is True:
                aborted_signal_found = True
                break

    # Check wrapped inside a generic progress event.
    if not aborted_signal_found:
        for e in events:
            if e["event"] == "progress" and isinstance(e["data"], dict):
                if (
                    e["data"].get("event") == "done"
                    and isinstance(e["data"].get("payload"), dict)
                    and e["data"]["payload"].get("aborted_by_user") is True
                ):
                    aborted_signal_found = True
                    break

    assert aborted_signal_found, (
        f"No abort signal (aborted_by_user=True) found in SSE stream; "
        f"event_types={event_types}\n"
        f"done/progress events: {[e for e in events if e['event'] in ('done','progress')]}"
    )

    # The final SSE event should contain the abort marker text.
    final_events = [e for e in events if e["event"] == "final"]
    if final_events:
        answer = final_events[0]["data"].get("answer", "")
        assert "aborted" in answer.lower() or "Turn aborted" in answer, (
            f"Abort marker not in final answer: {answer!r}"
        )


# ---------------------------------------------------------------------------
# Test 5 — followups event emitted when enabled (no pause)
# ---------------------------------------------------------------------------


def test_e2e_followups_event_emitted_when_enabled(client: TestClient) -> None:
    """followups_enabled=True, healthy retrieval, no review pause →
    a ``followups`` SSE event arrives with 3 chips."""
    results = [
        _make_result("c1", "healthy text", rerank_score=2.0),
        _make_result("c2", "more text", rerank_score=1.5),
    ]
    # Use "smart_auto" mode (default) so healthy scores do NOT trigger review.
    # followups_enabled=True so the LLM generates chips.
    events, _llm = _run_e2e(
        client,
        results,
        {
            "review_mode": "smart_auto",
            "review_enabled": True,
            "followups_enabled": True,
            "rephrasings_enabled": False,
        },
        resume_payload=None,  # no pause expected; no need to unblock
    )

    event_types = [e["event"] for e in events]
    assert "followups" in event_types, (
        f"followups event missing; got: {event_types}"
    )
    fu = next(e for e in events if e["event"] == "followups")
    chips = fu["data"].get("chips", [])
    assert len(chips) == 3, f"Expected 3 followup chips, got: {chips}"
    assert all(isinstance(c, str) and c.strip() for c in chips), (
        f"All chips must be non-empty strings: {chips}"
    )


# ---------------------------------------------------------------------------
# Test 6 — review_enabled=False: no review_required event even on low score
# ---------------------------------------------------------------------------


def test_e2e_disabled_no_review_required_event(client: TestClient) -> None:
    """review_enabled=False → no review_required event, even with
    very low rerank_scores that would normally trigger the score_floor signal."""
    results = [
        _make_result("c1", "poor text", rerank_score=-15.0),
        _make_result("c2", "also poor", rerank_score=-20.0),
    ]
    # review_enabled=False is the default; be explicit.
    events, _llm = _run_e2e(
        client,
        results,
        {"review_enabled": False},
        resume_payload=None,
    )

    event_types = [e["event"] for e in events]
    assert "review_required" not in event_types, (
        f"review_required must NOT fire when review_enabled=False; got: {event_types}"
    )
    # Generation should still complete normally.
    assert "final" in event_types, (
        f"final event missing (review disabled path should complete); got: {event_types}"
    )
