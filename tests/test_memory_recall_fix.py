"""Phase 8.1 — memories surface for FACTUAL intent, not only PERSONAL.

Regression test for the "the app doesn't seem to remember my memories" bug.
Before the fix, FACTUAL turns passed ``source_types=plan.source_types`` (None
on the "full" path) or were scoped down by retrievers like TaxonomyRetriever
that only return chunks under taxonomy-tree leaves. Either way, episodic
memories were invisible to FACTUAL queries.

The fix flips ``cfg.retrieval.always_include_episodic = True`` (the new
default) and makes the orchestrator explicitly pass ``["document",
"episodic"]`` for every intent.
"""

from __future__ import annotations

from typing import Optional

from hrag.intent import Intent, IntentVerdict
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Local test doubles (mirror the pattern in tests/test_adaptive_retrieval.py)
# ---------------------------------------------------------------------------


class _ScriptedClassifier:
    """Stand-in intent classifier returning a fixed verdict."""

    def __init__(self, intent: Intent) -> None:
        self._intent = intent

    def classify(self, text: str, **kwargs) -> IntentVerdict:
        return IntentVerdict(
            intent=self._intent,
            confidence=1.0,
            source="test",
            raw_label=self._intent.value,
        )


class _RecordingRetriever:
    """Spy retriever that records every retrieve() call."""

    name = "recording"

    def __init__(self, results: Optional[list[RetrievalResult]] = None) -> None:
        self.calls: list[dict] = []
        self._results = results or []

    def retrieve(
        self,
        query,
        user_id,
        top_k=10,
        source_types=None,
        intent_hint=None,
        where=None,
    ):
        self.calls.append({
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
            "source_types": source_types,
            "intent_hint": intent_hint,
            "where": where,
        })
        return list(self._results)


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


def _make_orch(sample_config, classifier: _ScriptedClassifier, retriever: _RecordingRetriever):
    """Build an Orchestrator with the scripted classifier + recording retriever."""
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None
    sample_config.retrieval.rerank_enabled = False

    from hrag.orchestrator import Orchestrator
    orch = Orchestrator(sample_config)

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factual_intent_includes_episodic_by_default(sample_config) -> None:
    """With always_include_episodic=True (default), FACTUAL turns see episodic chunks."""
    # always_include_episodic defaults to True; assert the default explicitly so
    # the test is self-documenting if someone flips it later.
    assert sample_config.retrieval.always_include_episodic is True

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

    # The retriever was called at least once on the FACTUAL "full" path.
    assert len(spy.calls) >= 1
    # The recorded source_types contained "episodic".
    src = spy.calls[0]["source_types"]
    assert src is not None, "source_types must not be None when always_include_episodic=True"
    assert "episodic" in src
    # And "document" is still present so docs still compete on relevance.
    assert "document" in src


def test_personal_intent_still_lifts_episodic_to_top(sample_config) -> None:
    """PERSONAL intent still applies the stable-sort bias on top of the fix."""
    sample_config.retrieval.adaptive_enabled = True
    sample_config.retrieval.adaptive_personal_episodic_bias = True
    # always_include_episodic defaults to True; confirm the PERSONAL path still
    # sorts episodic chunks above document chunks.
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
    assert spy.calls
    src = spy.calls[0]["source_types"]
    assert src is not None
    assert "episodic" in src and "document" in src

    # The episodic-bias event still fires for PERSONAL turns.
    bias_events = [p for n, p in events if n == "episodic_bias_applied"]
    assert len(bias_events) == 1
    assert bias_events[0]["episodic_count"] == 2
    assert bias_events[0]["total"] == 4


def test_opt_out_via_flag(sample_config) -> None:
    """Setting always_include_episodic=False reverts to strict per-intent source_types."""
    sample_config.retrieval.always_include_episodic = False
    # Also keep adaptive off so neither path injects "episodic".
    sample_config.retrieval.adaptive_enabled = False

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

    # With the flag off + adaptive off, the FACTUAL "full" path falls back to
    # plan.source_types, which is None for FACTUAL (no filter applied at the
    # retriever boundary).
    assert len(spy.calls) >= 1
    assert spy.calls[0]["source_types"] is None
