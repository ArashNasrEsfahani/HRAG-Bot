"""Phase 8 wave-2 orchestrator-wiring tests for the interactive review pause.

These tests reuse the Phase 4 wiring pattern from ``test_orchestrator.py`` —
the orchestrator is constructed against ``sample_config``, the LLM is swapped
for a scripted stub, and a spy retriever feeds a deterministic result list.
No real chromadb / sentence-transformers is required (the conftest stubs are
sufficient).
"""

from __future__ import annotations

import json
import threading
import time


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


def _reset_db_singleton() -> None:
    import hrag.db.connection as _conn_mod

    _conn_mod._db_singleton = None


class _ReviewLLM:
    """Stub LLM that reacts to prompt fingerprints.

    Differentiates the answer call from the clarify / followups calls so
    each can return a stable string the tests can assert on.
    """

    name = "review-stub"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _dispatch(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Phase 8 follow-ups prompt
        if "follow-up questions" in prompt.lower() or "Return exactly 3 follow-ups" in prompt:
            return "How does it scale?\nWhat are the trade-offs?\nIs there a reference impl?"
        # Phase 8 clarify prompt
        if "clarifying question" in prompt.lower():
            return "Could you specify which aspect you mean?"
        # Intent classifier
        if "Intent Classification" in prompt or "Output (one word only)" in prompt:
            return "factual"
        # Default = answer body
        return "This is the answer body."

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        return self._dispatch(prompt)

    def generate(self, request):
        from hrag.types import GenerationResponse

        prompt = " ".join(m.content for m in request.messages)
        return GenerationResponse(text=self._dispatch(prompt), raw=None)

    def generate_stream(self, request):
        yield self.generate(request).text


class _SpyRetriever:
    """Records calls; returns a fixed list of RetrievalResult objects."""

    name = "spy"

    def __init__(self, results=None) -> None:
        self.calls: list[dict] = []
        self._results = list(results or [])

    def retrieve(
        self,
        query,
        user_id,
        top_k=10,
        source_types=None,
        intent_hint=None,
        where=None,
    ):
        self.calls.append(
            {
                "query": query,
                "user_id": user_id,
                "top_k": top_k,
                "source_types": source_types,
                "intent_hint": intent_hint,
                "where": where,
            }
        )
        return list(self._results)


def _result(chunk_id="c1", text="passage text", score=0.5, rerank_score=None):
    from hrag.types import Chunk, RetrievalResult

    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        user_id="default",
        text=text,
        embedding_text=text,
        title=f"Title-{chunk_id}",
        section="Section",
    )
    return RetrievalResult(chunk=chunk, score=score, rerank_score=rerank_score)


def _build_orch(sample_config, results=None, force_factual: bool = True):
    sample_config.retrieval.rerank_enabled = False
    if force_factual:
        sample_config.intent.enabled = False
    _reset_db_singleton()
    from hrag.orchestrator import Orchestrator

    orch = Orchestrator(sample_config)
    llm = _ReviewLLM()
    orch.llm = llm
    if orch.gate is not None:
        orch.gate.llm = llm
    if orch.clue is not None:
        orch.clue.llm = llm

    spy = _SpyRetriever(results=results)
    orch.retriever = spy
    return orch, spy, llm


# ---------------------------------------------------------------------------
# 1. Default-off invariant
# ---------------------------------------------------------------------------


def test_orchestrator_review_disabled_no_pause(sample_config):
    """review_enabled=False → no review_required event, no pause."""
    assert sample_config.interaction.review_enabled is False
    results = [_result("c1"), _result("c2"), _result("c3")]
    orch, _spy, _llm = _build_orch(sample_config, results=results)
    events: list[tuple[str, dict]] = []
    try:
        out = orch.chat(
            "what is foo?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    fired = {n for n, _ in events}
    assert "review_required" not in fired
    assert "review_resolved" not in fired
    assert out.answer  # generation completed


def test_orchestrator_review_enabled_no_trigger(sample_config):
    """review_enabled=True but every signal healthy → no pause."""
    sample_config.interaction.review_enabled = True
    # Score floor is -3.0; provide a comfortably-high score.
    results = [_result("c1", score=0.9, rerank_score=2.0),
               _result("c2", score=0.5, rerank_score=0.5)]
    # Single-leaf descend would not trigger BRANCH_THRESHOLD (branch_threshold=2).
    orch, _spy, _llm = _build_orch(sample_config, results=results)
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what is foo?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    fired = {n for n, _ in events}
    # mode=smart_auto by default → no automatic ALWAYS trigger.
    # AMBIGUITY_DELTA: rerank 2.0 vs 0.5 → spread 1.5 > 0.4 → no fire.
    # SCORE_FLOOR: max 2.0 > -3.0 → no fire.
    # No descend, no router, intent=FACTUAL → confident.
    # FACTUAL_GENERAL_SWAP: top score 0.9 vs floor 0.15 → no.
    assert "review_required" not in fired


# ---------------------------------------------------------------------------
# 2. Triggered pause + decision dispatch
# ---------------------------------------------------------------------------


def _submit_after(orch, turn_id_holder, decision_dict, delay=0.05):
    """Helper: from a background thread, submit a decision once the
    orchestrator has registered the turn."""
    def _go():
        # Poll for the turn_id to be set
        for _ in range(50):
            if turn_id_holder.get("turn_id"):
                break
            time.sleep(0.02)
        # Then wait a beat for the orchestrator to register in the store
        time.sleep(delay)
        tid = turn_id_holder.get("turn_id")
        if tid:
            orch.interaction_store.submit_decision(tid, decision_dict)

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    return t


def test_orchestrator_review_enabled_low_score_pauses(sample_config):
    """Low rerank scores trigger SCORE_FLOOR; chat completes when continue submitted."""
    sample_config.interaction.review_enabled = True
    sample_config.interaction.review_timeout_s = 5.0
    sample_config.interaction.followups_enabled = False  # keep events focused
    sample_config.interaction.rephrasings_enabled = False
    results = [
        _result("c1", score=0.5, rerank_score=-10.0),
        _result("c2", score=0.5, rerank_score=-11.0),
    ]
    orch, _spy, _llm = _build_orch(sample_config, results=results)
    events: list[tuple[str, dict]] = []
    turn_id_holder: dict = {}

    def cb(name, payload):
        events.append((name, payload))
        if name == "start":
            turn_id_holder["turn_id"] = payload.get("turn_id")

    _submit_after(orch, turn_id_holder, {"action": "continue"})

    try:
        out = orch.chat(
            "an obscure question",
            user_id="default",
            progress=cb,
        )
    finally:
        orch.close()
        _reset_db_singleton()

    review_events = [p for n, p in events if n == "review_required"]
    assert len(review_events) == 1
    assert "score_floor" in review_events[0]["reasons"]
    # turn_id surfaced on start AND in review_required, both equal
    assert turn_id_holder["turn_id"] == review_events[0]["turn_id"]
    # Generation still completed
    assert out.answer == "This is the answer body."


def test_orchestrator_review_general_action_swaps_intent(sample_config):
    """`general` action → no retrieved passages in prompt, intent rewritten."""
    sample_config.interaction.review_enabled = True
    sample_config.interaction.review_mode = "always"  # force pause
    sample_config.interaction.review_timeout_s = 5.0
    sample_config.interaction.followups_enabled = False
    sample_config.interaction.rephrasings_enabled = False
    results = [_result("c1", score=0.9, rerank_score=2.0)]
    orch, _spy, _llm = _build_orch(sample_config, results=results)
    events: list[tuple[str, dict]] = []
    turn_id_holder: dict = {}

    def cb(name, payload):
        events.append((name, payload))
        if name == "start":
            turn_id_holder["turn_id"] = payload.get("turn_id")

    _submit_after(orch, turn_id_holder, {"action": "general"})

    try:
        out = orch.chat(
            "what is foo?",
            user_id="default",
            progress=cb,
        )
    finally:
        orch.close()
        _reset_db_singleton()

    # The prompt must not include the chunk text (GENERAL prompt template has
    # no retrieved_passages slot).
    assert "passage text" not in out.prompt


def test_orchestrator_review_filter_action_subsets_results(sample_config):
    """`filter` action restricts the result set to selected chunk_ids."""
    sample_config.interaction.review_enabled = True
    sample_config.interaction.review_mode = "always"
    sample_config.interaction.review_timeout_s = 5.0
    sample_config.interaction.followups_enabled = False
    sample_config.interaction.rephrasings_enabled = False
    results = [
        _result("c1", text="alpha text", score=0.9, rerank_score=2.0),
        _result("c2", text="beta text",  score=0.8, rerank_score=1.5),
        _result("c3", text="gamma text", score=0.7, rerank_score=1.2),
    ]
    orch, _spy, _llm = _build_orch(sample_config, results=results)
    events: list[tuple[str, dict]] = []
    turn_id_holder: dict = {}

    def cb(name, payload):
        events.append((name, payload))
        if name == "start":
            turn_id_holder["turn_id"] = payload.get("turn_id")

    _submit_after(orch, turn_id_holder, {
        "action": "filter",
        "selected_chunk_ids": ["c2"],
    })

    try:
        out = orch.chat(
            "what is foo?",
            user_id="default",
            progress=cb,
        )
    finally:
        orch.close()
        _reset_db_singleton()

    # Only c2's text should appear in the prompt.
    assert "beta text" in out.prompt
    assert "alpha text" not in out.prompt
    assert "gamma text" not in out.prompt


# ---------------------------------------------------------------------------
# 3. Follow-ups
# ---------------------------------------------------------------------------


def test_orchestrator_followups_emitted_when_enabled(sample_config):
    sample_config.interaction.review_enabled = True
    sample_config.interaction.followups_enabled = True
    sample_config.interaction.rephrasings_enabled = False
    # Healthy scores → no pause needed.
    results = [
        _result("c1", score=0.9, rerank_score=2.0),
        _result("c2", score=0.5, rerank_score=0.2),
    ]
    orch, _spy, _llm = _build_orch(sample_config, results=results)
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what is foo?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    fu_events = [p for n, p in events if n == "followups"]
    assert len(fu_events) == 1
    chips = fu_events[0]["chips"]
    assert len(chips) == 3
    assert all(isinstance(c, str) and c for c in chips)


def test_orchestrator_followups_skipped_when_disabled(sample_config):
    sample_config.interaction.review_enabled = True
    sample_config.interaction.followups_enabled = False
    results = [_result("c1", score=0.9, rerank_score=2.0)]
    orch, _spy, _llm = _build_orch(sample_config, results=results)
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what is foo?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    fu_events = [p for n, p in events if n == "followups"]
    assert fu_events == []


# ---------------------------------------------------------------------------
# 4. Metadata persistence
# ---------------------------------------------------------------------------


def _read_assistant_metadata(db_path, session_id):
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT role, metadata FROM messages "
            "WHERE session_id=? AND role='assistant' "
            "ORDER BY message_id ASC",
            (session_id,),
        ).fetchall()
        return [(r["role"], r["metadata"]) for r in rows]
    finally:
        conn.close()


def test_orchestrator_metadata_written_on_paused_turn(sample_config, tmp_path):
    sample_config.interaction.review_enabled = True
    sample_config.interaction.review_mode = "always"
    sample_config.interaction.review_timeout_s = 5.0
    sample_config.interaction.followups_enabled = False
    sample_config.interaction.rephrasings_enabled = False
    results = [_result("c1", score=0.9, rerank_score=2.0)]
    orch, _spy, _llm = _build_orch(sample_config, results=results)
    events: list[tuple[str, dict]] = []
    turn_id_holder: dict = {}
    db_path = orch.db.path
    session_id_box: dict = {}

    def cb(name, payload):
        events.append((name, payload))
        if name == "start":
            turn_id_holder["turn_id"] = payload.get("turn_id")

    _submit_after(orch, turn_id_holder, {"action": "continue"})

    try:
        out = orch.chat(
            "what is foo?",
            user_id="default",
            progress=cb,
        )
        session_id_box["sid"] = out.session_id
    finally:
        orch.close()
        _reset_db_singleton()

    rows = _read_assistant_metadata(db_path, session_id_box["sid"])
    assert len(rows) == 1
    role, meta = rows[0]
    assert role == "assistant"
    assert meta is not None
    parsed = json.loads(meta)
    assert "phase8" in parsed
    assert parsed["phase8"]["action"] == "continue"
    assert parsed["phase8"]["reasons"]  # non-empty


def test_orchestrator_no_metadata_written_when_no_pause(sample_config):
    """Healthy turn → metadata column is NULL on the assistant message."""
    sample_config.interaction.review_enabled = True
    sample_config.interaction.followups_enabled = False
    results = [_result("c1", score=0.9, rerank_score=2.0)]
    orch, _spy, _llm = _build_orch(sample_config, results=results)
    db_path = orch.db.path
    try:
        out = orch.chat("what is foo?", user_id="default")
        sid = out.session_id
    finally:
        orch.close()
        _reset_db_singleton()

    rows = _read_assistant_metadata(db_path, sid)
    assert len(rows) == 1
    role, meta = rows[0]
    assert role == "assistant"
    assert meta is None
