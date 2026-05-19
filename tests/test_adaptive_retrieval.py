"""Tests for the Phase 6 adaptive-retrieval-per-intent layer.

Two layers of coverage:

* Pure-function tests of ``_adaptive_top_k`` — fast, deterministic, exercise
  every config branch without spinning up an Orchestrator.
* Integration tests through ``Orchestrator.chat`` with a stub retriever and a
  scripted intent classifier — confirm the greeting-skips-retrieval shortcut
  and the personal-episodic-bias re-sort actually fire.
"""

from __future__ import annotations

from typing import Optional

import pytest

from hrag.config import Config, RetrievalConfig
from hrag.intent import Intent, IntentVerdict
from hrag.orchestrator import _adaptive_top_k
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Resolver unit tests
# ---------------------------------------------------------------------------


def _cfg(**retrieval_kwargs) -> Config:
    c = Config()
    for k, v in retrieval_kwargs.items():
        setattr(c.retrieval, k, v)
    return c


def test_adaptive_disabled_returns_global_defaults() -> None:
    """With adaptive_enabled=False, the resolver is a no-op."""
    cfg = _cfg(adaptive_enabled=False, top_k_vector=10, top_k_final=6)
    for intent in (Intent.GREETING, Intent.PERSONAL, Intent.FACTUAL):
        assert _adaptive_top_k(cfg, intent) == (10, 6)


def test_adaptive_greeting_returns_none_pair_to_signal_skip() -> None:
    """Greeting's default mapping is 0 → (None, None) so the orchestrator skips."""
    cfg = _cfg(adaptive_enabled=True)
    assert _adaptive_top_k(cfg, Intent.GREETING) == (None, None)


def test_adaptive_factual_uses_dict_and_widens_vec_k() -> None:
    """Factual is 6 by default → final=6, vec=max(12, 12) for reranker slack."""
    cfg = _cfg(adaptive_enabled=True)
    vec_k, final_k = _adaptive_top_k(cfg, Intent.FACTUAL)
    assert final_k == 6
    assert vec_k == 12  # max(6*2, 12) == 12


def test_adaptive_personal_dict_value() -> None:
    """Personal default is 8 → final=8, vec=max(16, 12)=16."""
    cfg = _cfg(adaptive_enabled=True)
    vec_k, final_k = _adaptive_top_k(cfg, Intent.PERSONAL)
    assert final_k == 8
    assert vec_k == 16


def test_adaptive_general_and_unclear_default_to_4() -> None:
    cfg = _cfg(adaptive_enabled=True)
    # Note: GENERAL is in the str-enum but not directly emittable; the resolver
    # still handles it (the orchestrator rewrites FACTUAL→GENERAL post-retrieval).
    assert _adaptive_top_k(cfg, Intent.UNCLEAR) == (12, 4)


def test_adaptive_custom_top_k_dict_respected() -> None:
    """User-provided per-intent overrides override the defaults."""
    cfg = _cfg(
        adaptive_enabled=True,
        adaptive_top_k={
            "greeting": 2,        # not zero anymore — should retrieve a tiny bit
            "personal": 20,
            "factual": 10,
            "general": 3,
            "unclear": 5,
        },
    )
    assert _adaptive_top_k(cfg, Intent.GREETING) == (12, 2)
    assert _adaptive_top_k(cfg, Intent.PERSONAL) == (40, 20)
    assert _adaptive_top_k(cfg, Intent.FACTUAL) == (20, 10)


def test_adaptive_unknown_intent_falls_back_to_global_final() -> None:
    """An intent value missing from the dict falls back to top_k_final."""
    cfg = _cfg(
        adaptive_enabled=True,
        top_k_final=5,
        adaptive_top_k={"factual": 6},  # everything else absent
    )
    vec_k, final_k = _adaptive_top_k(cfg, Intent.UNCLEAR)
    assert final_k == 5
    assert vec_k == 12  # max(5*2, 12)


# ---------------------------------------------------------------------------
# Integration tests through Orchestrator.chat()
# ---------------------------------------------------------------------------


class _ScriptedClassifier:
    """Stand-in intent classifier — returns a fixed verdict."""

    def __init__(self, intent: Intent) -> None:
        self._intent = intent

    def classify(self, text: str) -> IntentVerdict:
        return IntentVerdict(
            intent=self._intent,
            confidence=1.0,
            source="test",
            raw_label=self._intent.value,
        )


class _RecordingRetriever:
    """Spy retriever that records every retrieve() call and returns a stub list.

    The ``results`` constructor arg lets a test seed mixed source-types so the
    episodic-bias sort has something to actually reorder.
    """

    name = "spy"

    def __init__(self, results: Optional[list[RetrievalResult]] = None) -> None:
        self.calls: list[dict] = []
        self._results = results or []

    def retrieve(self, query, user_id, top_k=10, source_types=None, intent_hint=None, where=None):
        self.calls.append({
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
            "source_types": source_types,
            "intent_hint": intent_hint,
            "where": where,
        })
        return list(self._results)


def _make_orch(sample_config, classifier: _ScriptedClassifier, retriever: _RecordingRetriever):
    """Build an Orchestrator with the scripted classifier + recording retriever.

    Mirrors the ``_make_phase4_orch`` pattern in ``test_orchestrator.py``.
    """
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None
    sample_config.retrieval.rerank_enabled = False

    from hrag.orchestrator import Orchestrator
    orch = Orchestrator(sample_config)

    # Stub out the LLM so nothing calls a real provider.
    from tests.conftest import FakeLLM
    fake_llm = FakeLLM()
    orch.llm = fake_llm
    if orch.gate is not None:
        orch.gate.llm = fake_llm
    if orch.clue is not None:
        orch.clue.llm = fake_llm

    orch.intent_classifier = classifier  # type: ignore[assignment]
    orch.retriever = retriever  # type: ignore[assignment]
    return orch


def _chunk(chunk_id: str, source_type: str = "document") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        user_id="default",
        text=f"text for {chunk_id}",
        embedding_text=f"emb for {chunk_id}",
        source_type=source_type,
    )


def _result(chunk_id: str, score: float, source_type: str = "document") -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(chunk_id, source_type), score=score)


def test_greeting_skips_retrieval_when_adaptive_enabled(sample_config) -> None:
    """adaptive_enabled=True + greeting -> retriever is never called."""
    sample_config.retrieval.adaptive_enabled = True
    spy = _RecordingRetriever()
    orch = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.GREETING),
        spy,
    )
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "hi",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    # The spy was never called.
    assert spy.calls == []
    # The skip event fired with reason=greeting.
    skips = [p for n, p in events if n == "retrieval_skipped"]
    assert len(skips) == 1
    assert skips[0]["reason"] == "greeting"
    # And the resolver event was emitted with (None, None).
    adaptive_events = [p for n, p in events if n == "adaptive_top_k"]
    assert len(adaptive_events) == 1
    assert adaptive_events[0]["top_k_vector"] is None
    assert adaptive_events[0]["top_k_final"] is None
    assert adaptive_events[0]["intent"] == "greeting"


def test_greeting_does_not_skip_when_adaptive_disabled(sample_config) -> None:
    """Regression guard: with adaptive_enabled=False, greetings still hit the
    retriever (legacy behaviour). Intent routing may still keep ``scope="none"``
    on its own, but the skip event added by Phase 6 must not fire."""
    sample_config.retrieval.adaptive_enabled = False
    spy = _RecordingRetriever()
    orch = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.GREETING),
        spy,
    )
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "hello",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    # The Phase 6 skip event MUST NOT fire when adaptive is off.
    skips = [p for n, p in events if n == "retrieval_skipped"]
    assert skips == []


def test_factual_uses_adaptive_top_k(sample_config) -> None:
    """When adaptive is on for a FACTUAL turn, retrieve() is called with the
    widened top_k_vector (12 = max(6*2, 12))."""
    sample_config.retrieval.adaptive_enabled = True
    spy = _RecordingRetriever(results=[_result("c1", 0.9)])
    orch = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.FACTUAL),
        spy,
    )
    try:
        orch.chat("what is hipporag?", user_id="default")
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    assert len(spy.calls) == 1
    assert spy.calls[0]["top_k"] == 12  # widened vec_k for the reranker


def test_personal_episodic_bias_reorders_results(sample_config) -> None:
    """Personal intent + episodic bias on: episodic results sorted to the top."""
    sample_config.retrieval.adaptive_enabled = True
    sample_config.retrieval.adaptive_personal_episodic_bias = True
    # Seed mixed-source-type results — episodic should float up.
    spy = _RecordingRetriever(results=[
        _result("doc1", 0.95, source_type="document"),
        _result("ep1", 0.80, source_type="episodic"),
        _result("doc2", 0.70, source_type="document"),
        _result("ep2", 0.60, source_type="episodic"),
    ])
    orch = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.PERSONAL),
        spy,
    )
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what do I prefer for testing?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    # The retriever was asked for both source types.
    assert spy.calls and spy.calls[0]["source_types"] == ["document", "episodic"]

    # The episodic-bias event fired.
    bias_events = [p for n, p in events if n == "episodic_bias_applied"]
    assert len(bias_events) == 1
    assert bias_events[0]["episodic_count"] == 2
    assert bias_events[0]["total"] == 4


def test_personal_no_bias_when_flag_off(sample_config) -> None:
    """With adaptive_personal_episodic_bias=False, no reorder + no event."""
    sample_config.retrieval.adaptive_enabled = True
    sample_config.retrieval.adaptive_personal_episodic_bias = False
    spy = _RecordingRetriever(results=[
        _result("doc1", 0.95, source_type="document"),
        _result("ep1", 0.80, source_type="episodic"),
    ])
    orch = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.PERSONAL),
        spy,
    )
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what do I prefer?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    bias_events = [p for n, p in events if n == "episodic_bias_applied"]
    assert bias_events == []
