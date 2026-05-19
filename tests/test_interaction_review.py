"""Unit tests for hrag.interaction.review."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from hrag.interaction.review import (
    PauseReason,
    ReviewDecision,
    build_review_payload,
    generate_rephrasings,
    maybe_pause,
    should_pause,
)
from hrag.interaction.store import InteractionStore


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> SimpleNamespace:
    """Build a SimpleNamespace standing in for InteractionConfig."""
    defaults = dict(
        review_enabled=True,
        review_mode="smart_auto",
        review_score_floor=-3.0,
        review_ambiguity_delta=0.4,
        review_branch_threshold=2,
        review_timeout_s=90.0,
        rephrasings_enabled=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _chunk(chunk_id: str = "c1", text: str = "passage", has_math: bool = False):
    """Build a minimal Chunk for RetrievalResult."""
    from hrag.types import Chunk

    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        user_id="default",
        text=text,
        embedding_text=text,
        title=f"Title-{chunk_id}",
        section="Section",
        metadata={"has_math": has_math},
    )


def _result(
    chunk_id: str = "c1",
    score: float = 0.5,
    rerank_score: float | None = None,
    text: str = "passage",
):
    from hrag.types import RetrievalResult

    return RetrievalResult(
        chunk=_chunk(chunk_id=chunk_id, text=text),
        score=score,
        rerank_score=rerank_score,
    )


def _intent_verdict(value: str):
    intent = SimpleNamespace(value=value)
    return SimpleNamespace(intent=intent, confidence=0.9)


# ---------------------------------------------------------------------------
# should_pause — trigger matrix
# ---------------------------------------------------------------------------


def test_should_pause_empty_when_disabled():
    reasons = should_pause(
        cfg=_cfg(review_enabled=False),
        results=[_result(rerank_score=-99.0)],
        descend=None,
        intent_verdict=None,
        router_label=None,
        factual_general_swap_imminent=True,
    )
    assert reasons == []


def test_should_pause_score_floor():
    reasons = should_pause(
        cfg=_cfg(review_score_floor=-3.0),
        results=[
            _result("a", rerank_score=-5.0),
            _result("b", rerank_score=-6.0),
        ],
        descend=None,
        intent_verdict=None,
        router_label=None,
        factual_general_swap_imminent=False,
    )
    # The score-floor reason must fire; ambiguity-delta will also fire
    # because the top-2 spread (1.0) is < default 0.4? Actually 1.0 > 0.4
    # so it should NOT fire. So just SCORE_FLOOR.
    assert PauseReason.SCORE_FLOOR in reasons
    assert PauseReason.AMBIGUITY_DELTA not in reasons


def test_should_pause_ambiguity_delta():
    reasons = should_pause(
        cfg=_cfg(review_score_floor=-100.0, review_ambiguity_delta=0.4),
        results=[
            _result("a", rerank_score=0.5),
            _result("b", rerank_score=0.4),
        ],
        descend=None,
        intent_verdict=None,
        router_label=None,
        factual_general_swap_imminent=False,
    )
    assert PauseReason.AMBIGUITY_DELTA in reasons
    assert PauseReason.SCORE_FLOOR not in reasons


def test_should_pause_branch_threshold():
    reasons = should_pause(
        cfg=_cfg(review_branch_threshold=2),
        results=[],
        descend={"stats": {"leaves_picked": 3}},
        intent_verdict=None,
        router_label=None,
        factual_general_swap_imminent=False,
    )
    assert reasons == [PauseReason.BRANCH_THRESHOLD]


def test_should_pause_branch_threshold_not_fired_at_equal():
    """leaves_picked == threshold should NOT fire (strictly greater than)."""
    reasons = should_pause(
        cfg=_cfg(review_branch_threshold=2),
        results=[],
        descend={"stats": {"leaves_picked": 2}},
        intent_verdict=None,
        router_label=None,
        factual_general_swap_imminent=False,
    )
    assert PauseReason.BRANCH_THRESHOLD not in reasons


def test_should_pause_intent_unclear():
    reasons = should_pause(
        cfg=_cfg(),
        results=[],
        descend=None,
        intent_verdict=_intent_verdict("unclear"),
        router_label=None,
        factual_general_swap_imminent=False,
    )
    assert reasons == [PauseReason.INTENT_UNCLEAR]


def test_should_pause_intent_factual_does_not_fire():
    reasons = should_pause(
        cfg=_cfg(),
        results=[],
        descend=None,
        intent_verdict=_intent_verdict("factual"),
        router_label=None,
        factual_general_swap_imminent=False,
    )
    assert reasons == []


def test_should_pause_router_ambiguous():
    reasons = should_pause(
        cfg=_cfg(),
        results=[],
        descend=None,
        intent_verdict=None,
        router_label="ambiguous",
        factual_general_swap_imminent=False,
    )
    assert reasons == [PauseReason.ROUTER_AMBIGUOUS]


def test_should_pause_factual_general_swap():
    reasons = should_pause(
        cfg=_cfg(),
        results=[],
        descend=None,
        intent_verdict=None,
        router_label=None,
        factual_general_swap_imminent=True,
    )
    assert reasons == [PauseReason.FACTUAL_GENERAL_SWAP]


def test_should_pause_always_mode():
    reasons = should_pause(
        cfg=_cfg(review_mode="always"),
        results=[_result("a", rerank_score=99.0)],  # nothing else would trigger
        descend=None,
        intent_verdict=_intent_verdict("factual"),
        router_label="entity",
        factual_general_swap_imminent=False,
    )
    assert PauseReason.ALWAYS_MODE in reasons


def test_should_pause_always_mode_additive():
    """always_mode should not suppress other reasons that also fire."""
    reasons = should_pause(
        cfg=_cfg(review_mode="always"),
        results=[_result("a", rerank_score=-99.0)],
        descend=None,
        intent_verdict=_intent_verdict("unclear"),
        router_label="ambiguous",
        factual_general_swap_imminent=True,
    )
    assert PauseReason.ALWAYS_MODE in reasons
    assert PauseReason.SCORE_FLOOR in reasons
    assert PauseReason.INTENT_UNCLEAR in reasons
    assert PauseReason.ROUTER_AMBIGUOUS in reasons
    assert PauseReason.FACTUAL_GENERAL_SWAP in reasons


def test_should_pause_multiple_reasons():
    """Assemble a case where exactly 3 triggers fire and assert all 3."""
    reasons = should_pause(
        cfg=_cfg(review_score_floor=-3.0, review_branch_threshold=2),
        results=[
            _result("a", rerank_score=-5.0),  # SCORE_FLOOR
            _result("b", rerank_score=-6.0),
        ],
        descend={"stats": {"leaves_picked": 5}},  # BRANCH_THRESHOLD
        intent_verdict=_intent_verdict("unclear"),  # INTENT_UNCLEAR
        router_label="entity",
        factual_general_swap_imminent=False,
    )
    assert PauseReason.SCORE_FLOOR in reasons
    assert PauseReason.BRANCH_THRESHOLD in reasons
    assert PauseReason.INTENT_UNCLEAR in reasons
    assert len(reasons) >= 3


def test_should_pause_uses_raw_score_when_no_rerank():
    """When rerank_score is None, fall back to RetrievalResult.score."""
    reasons = should_pause(
        cfg=_cfg(review_score_floor=0.0),
        results=[_result("a", score=-1.0, rerank_score=None)],
        descend=None,
        intent_verdict=None,
        router_label=None,
        factual_general_swap_imminent=False,
    )
    assert PauseReason.SCORE_FLOOR in reasons


# ---------------------------------------------------------------------------
# maybe_pause — orchestration & blocking behaviour
# ---------------------------------------------------------------------------


def test_maybe_pause_disabled_short_circuit():
    """review_enabled=False → no progress events, no turn registered, continue."""
    progress_calls: list[tuple[str, dict]] = []
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        decision = maybe_pause(
            cfg=_cfg(review_enabled=False),
            results=[_result("a", rerank_score=-99.0)],
            descend=None,
            intent_verdict=_intent_verdict("unclear"),
            router_label="ambiguous",
            factual_general_swap_imminent=True,
            clue=None,
            question="q?",
            retrieval_query="q?",
            user_id="default",
            session_id=None,
            progress=lambda e, p: progress_calls.append((e, p)),
            store=store,
        )
        assert decision.action == "continue"
        assert decision.reasons == []
        assert decision.timed_out is False
        # No events and no turn registered.
        assert progress_calls == []
        # Walk the store: nothing should be present.
        assert store.get("anything") is None
    finally:
        store.shutdown()


def test_maybe_pause_no_triggers_no_block():
    """No triggers fire → no events, no turn registered, returns continue."""
    progress_calls: list[tuple[str, dict]] = []
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        decision = maybe_pause(
            cfg=_cfg(),
            results=[
                _result("a", rerank_score=2.0),
                _result("b", rerank_score=0.5),  # spread 1.5 > 0.4
            ],
            descend=None,
            intent_verdict=_intent_verdict("factual"),
            router_label="entity",
            factual_general_swap_imminent=False,
            clue=None,
            question="What is X?",
            retrieval_query="X",
            user_id="default",
            session_id="s1",
            progress=lambda e, p: progress_calls.append((e, p)),
            store=store,
        )
        assert decision.action == "continue"
        assert decision.reasons == []
        assert decision.timed_out is False
        assert progress_calls == []
    finally:
        store.shutdown()


def test_maybe_pause_triggers_then_decision():
    """Spin a side thread that posts the decision; maybe_pause must return it."""
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        progress_calls: list[tuple[str, dict]] = []
        captured_turn_id: list[str] = []

        def _on_progress(event: str, payload: dict):
            progress_calls.append((event, payload))
            if event == "review_required":
                captured_turn_id.append(payload["turn_id"])

        # Submitter thread: wait until the orchestrator has registered the
        # turn (we peek at the captured_turn_id from the progress event),
        # then submit a "filter" decision.
        def _submitter():
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if captured_turn_id:
                    break
                time.sleep(0.01)
            assert captured_turn_id, "review_required never fired"
            store.submit_decision(
                captured_turn_id[0],
                {
                    "action": "filter",
                    "selected_chunk_ids": ["a", "b"],
                    "include_episodic": True,
                },
            )

        thread = threading.Thread(target=_submitter)
        thread.start()
        try:
            decision = maybe_pause(
                cfg=_cfg(review_timeout_s=3.0),
                results=[
                    _result("a", rerank_score=-5.0),
                    _result("b", rerank_score=-6.0),
                ],
                descend=None,
                intent_verdict=None,
                router_label=None,
                factual_general_swap_imminent=False,
                clue="clue text",
                question="q?",
                retrieval_query="q?",
                user_id="default",
                session_id="s1",
                progress=_on_progress,
                store=store,
            )
        finally:
            thread.join(timeout=2.0)

        assert decision.action == "filter"
        assert decision.selected_chunk_ids == ["a", "b"]
        assert decision.include_episodic is True
        assert decision.timed_out is False
        # SCORE_FLOOR fired.
        assert "score_floor" in decision.reasons

        # Two progress events emitted in order.
        events = [e for e, _ in progress_calls]
        assert events == ["review_required", "review_resolved"]
        resolved = progress_calls[1][1]
        assert resolved["action"] == "filter"
        assert resolved["timed_out"] is False
    finally:
        store.shutdown()


def test_maybe_pause_timeout_returns_continue():
    """No submission within timeout → continue with timed_out=True."""
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        progress_calls: list[tuple[str, dict]] = []
        decision = maybe_pause(
            cfg=_cfg(review_timeout_s=0.05),
            results=[_result("a", rerank_score=-99.0)],
            descend=None,
            intent_verdict=None,
            router_label=None,
            factual_general_swap_imminent=False,
            clue=None,
            question="q?",
            retrieval_query="q?",
            user_id="default",
            session_id=None,
            progress=lambda e, p: progress_calls.append((e, p)),
            store=store,
        )
        assert decision.action == "continue"
        assert decision.timed_out is True
        assert decision.reasons  # non-empty
        # Resolved event payload reflects the timeout.
        resolved = [p for e, p in progress_calls if e == "review_resolved"]
        assert resolved and resolved[0]["timed_out"] is True
    finally:
        store.shutdown()


def test_maybe_pause_no_store_degrades_gracefully():
    """No store → returns continue with timed_out=True (no hang)."""
    progress_calls: list[tuple[str, dict]] = []
    decision = maybe_pause(
        cfg=_cfg(review_timeout_s=5.0),
        results=[_result("a", rerank_score=-99.0)],
        descend=None,
        intent_verdict=None,
        router_label=None,
        factual_general_swap_imminent=False,
        clue=None,
        question="q?",
        retrieval_query="q?",
        user_id="default",
        session_id=None,
        progress=lambda e, p: progress_calls.append((e, p)),
        store=None,
    )
    assert decision.action == "continue"
    assert decision.timed_out is True
    # Both events still emitted.
    events = [e for e, _ in progress_calls]
    assert "review_required" in events
    assert "review_resolved" in events


def test_maybe_pause_uses_provided_turn_id():
    """If turn_id is supplied, it must be used (not regenerated)."""
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        captured: list[str] = []

        def _submitter():
            time.sleep(0.05)
            store.submit_decision("FIXED-ID", {"action": "continue"})

        thread = threading.Thread(target=_submitter)
        thread.start()
        try:
            maybe_pause(
                cfg=_cfg(review_timeout_s=2.0),
                results=[_result("a", rerank_score=-99.0)],
                descend=None,
                intent_verdict=None,
                router_label=None,
                factual_general_swap_imminent=False,
                clue=None,
                question="q?",
                retrieval_query="q?",
                user_id="default",
                session_id=None,
                turn_id="FIXED-ID",
                progress=lambda e, p: captured.append(p.get("turn_id", "")),
                store=store,
            )
        finally:
            thread.join(timeout=2.0)
        # review_required progress carried the supplied turn_id.
        assert "FIXED-ID" in captured
    finally:
        store.shutdown()


# ---------------------------------------------------------------------------
# build_review_payload
# ---------------------------------------------------------------------------


def test_build_review_payload_truncates_snippet():
    """A chunk with >240 chars must be truncated to exactly 240 in the payload."""
    from hrag.types import Chunk, RetrievalResult

    big_text = "x" * 1000
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        user_id="default",
        text=big_text,
        embedding_text=big_text,
        title="T",
        section="S",
    )
    payload = build_review_payload(
        turn_id="t",
        reasons=[PauseReason.SCORE_FLOOR],
        results=[RetrievalResult(chunk=chunk, score=0.1, rerank_score=-5.0)],
        descend=None,
        intent_verdict=None,
        router_label=None,
        retrieval_query="q",
        original_question="q?",
        clue=None,
        rephrasings=[],
        timeout_s=10.0,
    )
    assert len(payload.sources) == 1
    assert len(payload.sources[0]["snippet"]) == 240
    assert payload.sources[0]["chunk_id"] == "c1"
    assert payload.sources[0]["doc_id"] == "d1"


def test_build_review_payload_includes_required_fields():
    """Sources must include chunk_id, doc_id, snippet, score, rerank_score, has_math."""
    payload = build_review_payload(
        turn_id="t",
        reasons=[PauseReason.SCORE_FLOOR, PauseReason.AMBIGUITY_DELTA],
        results=[_result("a", score=0.5, rerank_score=-1.0, text="hello")],
        descend={"stats": {"leaves_picked": 1}},
        intent_verdict=_intent_verdict("factual"),
        router_label="entity",
        retrieval_query="rq",
        original_question="oq",
        clue="clue",
        rephrasings=["alt1"],
        timeout_s=90.0,
    )
    src = payload.sources[0]
    for key in (
        "chunk_id", "doc_id", "title", "section", "source_type",
        "score", "rerank_score", "snippet", "has_math",
    ):
        assert key in src
    d = payload.to_dict()
    assert d["reasons"] == ["score_floor", "ambiguity_delta"]
    assert d["intent"] == "factual"
    assert d["router_label"] == "entity"
    assert d["clue"] == "clue"
    assert d["rephrasings"] == ["alt1"]
    assert d["taxonomy_descend"] == {"stats": {"leaves_picked": 1}}
    assert d["timeout_s"] == 90.0


# ---------------------------------------------------------------------------
# generate_rephrasings
# ---------------------------------------------------------------------------


def test_generate_rephrasings_empty_on_no_llm():
    assert generate_rephrasings(llm=None, question="q?") == []


def test_generate_rephrasings_empty_on_no_complete_method():
    fake = SimpleNamespace()  # no .complete attr
    assert generate_rephrasings(llm=fake, question="q?") == []


def test_generate_rephrasings_parses_numbered_list():
    """Numbered output → cleaned, bullet-stripped list."""

    class StubLLM:
        def complete(self, prompt, **kw):
            return "1. Alt one\n2. Alt two\n3. Alt three"

    out = generate_rephrasings(llm=StubLLM(), question="q?", n=3)
    assert out == ["Alt one", "Alt two", "Alt three"]


def test_generate_rephrasings_strips_bullets_and_quotes():
    class StubLLM:
        def complete(self, prompt, **kw):
            return '- "first alt"\n* second alt\n• third alt'

    out = generate_rephrasings(llm=StubLLM(), question="q?", n=3)
    assert out == ["first alt", "second alt", "third alt"]


def test_generate_rephrasings_caps_to_n():
    class StubLLM:
        def complete(self, prompt, **kw):
            return "\n".join(f"alt {i}" for i in range(10))

    out = generate_rephrasings(llm=StubLLM(), question="q?", n=2)
    assert len(out) == 2


def test_generate_rephrasings_empty_on_exception():
    class StubLLM:
        def complete(self, prompt, **kw):
            raise RuntimeError("boom")

    assert generate_rephrasings(llm=StubLLM(), question="q?") == []


def test_generate_rephrasings_empty_on_blank_output():
    class StubLLM:
        def complete(self, prompt, **kw):
            return ""

    assert generate_rephrasings(llm=StubLLM(), question="q?") == []


# ---------------------------------------------------------------------------
# ReviewDecision.from_dict
# ---------------------------------------------------------------------------


def test_review_decision_from_dict_defaults():
    d = ReviewDecision.from_dict({}, reasons=["score_floor"])
    assert d.action == "continue"
    assert d.selected_chunk_ids == []
    assert d.rewritten_query is None
    assert d.expand_from_doc_id is None
    assert d.redirect_taxonomy_node_id is None
    assert d.include_episodic is False
    assert d.remember_choice is False
    assert d.reasons == ["score_floor"]
    assert d.timed_out is False


def test_review_decision_from_dict_unknown_action_normalised():
    d = ReviewDecision.from_dict({"action": "ROFLCOPTER"})
    assert d.action == "continue"


def test_review_decision_from_dict_all_fields():
    payload = {
        "action": "filter",
        "selected_chunk_ids": ["a", "b"],
        "rewritten_query": "new q",
        "expand_from_doc_id": "doc123",
        "redirect_taxonomy_node_id": "node5",
        "include_episodic": True,
        "remember_choice": True,
    }
    d = ReviewDecision.from_dict(payload, reasons=["ambiguity_delta"])
    assert d.action == "filter"
    assert d.selected_chunk_ids == ["a", "b"]
    assert d.rewritten_query == "new q"
    assert d.expand_from_doc_id == "doc123"
    assert d.redirect_taxonomy_node_id == "node5"
    assert d.include_episodic is True
    assert d.remember_choice is True
    assert d.reasons == ["ambiguity_delta"]


# ---------------------------------------------------------------------------
# Import-time hygiene
# ---------------------------------------------------------------------------


def test_interaction_module_imports_light():
    """`import hrag.interaction` must not pull heavy deps."""
    import importlib
    mod = importlib.import_module("hrag.interaction")
    assert hasattr(mod, "maybe_pause")
    assert hasattr(mod, "should_pause")
    assert hasattr(mod, "InteractionStore")
    assert hasattr(mod, "PendingTurn")
    assert hasattr(mod, "PauseReason")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
