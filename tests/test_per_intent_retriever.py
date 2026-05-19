"""Tests for the Phase 6-B1 per-intent retriever override layer.

The resolver (``Orchestrator._pick_retriever_for_intent``) lets each Intent
pick a different *retriever* — not just a different top_k. It must:

* No-op when ``retrieval.adaptive_enabled`` is False.
* No-op when every entry in ``adaptive_retriever_per_intent`` is "default".
* Build + cache an alternative retriever via :func:`build_retriever` when the
  intent maps to a non-"default", non-global retriever name, and emit the
  ``adaptive_retriever_picked`` progress event.
* Silently fall back to the global retriever (with a logger.warning) when the
  build fails — e.g. unknown name, or missing pre-requisite (taxonomy_store).

Patterns mirror ``tests/test_adaptive_retrieval.py`` (spy retriever + scripted
classifier wired into a real Orchestrator).
"""

from __future__ import annotations

from typing import Optional

import pytest  # noqa: F401  (collected via fixtures)

from hrag.intent import Intent, IntentVerdict
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers (cloned from test_adaptive_retrieval.py — keep the spy minimal)
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
    """Spy retriever — records every retrieve() call. Carries a ``name`` so
    the override-detection compare against ``self.retriever.name`` resolves."""

    def __init__(self, name: str = "spy", results: Optional[list[RetrievalResult]] = None) -> None:
        self.name = name
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
    """Build an Orchestrator with the scripted classifier + recording retriever.

    NOTE: ``cfg.retrieval.retriever`` is left at its existing legal value
    ("vector" by default) so the orchestrator's own ``build_retriever`` call
    inside ``__init__`` succeeds. We then swap ``self.retriever`` for the spy
    AND rewrite ``cfg.retrieval.retriever`` to the spy's name so the resolver
    short-circuits when an intent maps to the same name as the spy.
    """
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
    # Re-anchor the cfg.retrieval.retriever string so the resolver compares
    # against the new spy name; otherwise a mapping equal to the spy's name
    # would still try to build a fresh retriever instead of reusing self.retriever.
    sample_config.retrieval.retriever = retriever.name
    return orch


def _close(orch) -> None:
    try:
        orch.close()
    finally:
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None


# ---------------------------------------------------------------------------
# Resolver unit-level tests (no chat() call)
# ---------------------------------------------------------------------------


def test_resolver_no_op_when_adaptive_disabled(sample_config) -> None:
    """adaptive_enabled=False → resolver returns self.retriever for every intent."""
    sample_config.retrieval.adaptive_enabled = False
    sample_config.retrieval.adaptive_retriever_per_intent = {
        "greeting": "bm25",
        "personal": "kg_ppr",
        "factual": "hybrid",
        "general": "vector",
        "unclear": "router",
    }
    spy = _RecordingRetriever(name="vector")
    orch = _make_orch(sample_config, _ScriptedClassifier(Intent.FACTUAL), spy)
    try:
        for intent in (
            Intent.GREETING, Intent.PERSONAL, Intent.FACTUAL,
            Intent.GENERAL, Intent.UNCLEAR,
        ):
            assert orch._pick_retriever_for_intent(intent) is spy
        # No per-intent retrievers were built.
        assert orch._per_intent_retrievers == {}
    finally:
        _close(orch)


def test_resolver_all_defaults_returns_global(sample_config) -> None:
    """Every intent maps to "default" → always returns self.retriever."""
    sample_config.retrieval.adaptive_enabled = True
    sample_config.retrieval.adaptive_retriever_per_intent = {
        "greeting": "default",
        "personal": "default",
        "factual": "default",
        "general": "default",
        "unclear": "default",
    }
    spy = _RecordingRetriever(name="vector")
    orch = _make_orch(sample_config, _ScriptedClassifier(Intent.FACTUAL), spy)
    try:
        for intent in (
            Intent.GREETING, Intent.PERSONAL, Intent.FACTUAL,
            Intent.GENERAL, Intent.UNCLEAR,
        ):
            assert orch._pick_retriever_for_intent(intent) is spy
        assert orch._per_intent_retrievers == {}
    finally:
        _close(orch)


def test_resolver_mapping_matches_global_short_circuits(sample_config) -> None:
    """If the intent maps to the same name as ``retrieval.retriever``, the
    resolver reuses self.retriever without going through the factory."""
    sample_config.retrieval.adaptive_enabled = True
    # _make_orch rewrites cfg.retrieval.retriever to "spy" post-init.
    sample_config.retrieval.adaptive_retriever_per_intent = {
        "greeting": "default",
        "personal": "spy",  # same as the post-init global name
        "factual": "default",
        "general": "default",
        "unclear": "default",
    }
    spy = _RecordingRetriever(name="spy")
    orch = _make_orch(sample_config, _ScriptedClassifier(Intent.FACTUAL), spy)
    try:
        assert orch._pick_retriever_for_intent(Intent.PERSONAL) is spy
        # No alternative retriever built.
        assert orch._per_intent_retrievers == {}
    finally:
        _close(orch)


def test_resolver_builds_and_caches_alternative(sample_config, monkeypatch) -> None:
    """PERSONAL→bm25 while global=vector → resolver builds BM25 once, caches it.

    A second call returns the SAME instance (no rebuild).
    """
    sample_config.retrieval.adaptive_enabled = True
    sample_config.retrieval.adaptive_retriever_per_intent = {
        "greeting": "default",
        "personal": "bm25",
        "factual": "default",
        "general": "default",
        "unclear": "default",
    }
    spy = _RecordingRetriever(name="vector")
    orch = _make_orch(sample_config, _ScriptedClassifier(Intent.PERSONAL), spy)

    # Intercept the factory AFTER orchestrator init (so its own build of the
    # global "vector" retriever uses the real factory). Only the resolver's
    # subsequent call for "bm25" hits the stub.
    built_for: list[str] = []
    bm25_spy = _RecordingRetriever(name="bm25")

    def fake_build_retriever(retrieval_cfg, db, vector_store, embedder, **kwargs):
        built_for.append(retrieval_cfg.retriever)
        return bm25_spy

    monkeypatch.setattr("hrag.orchestrator.build_retriever", fake_build_retriever)

    try:
        # First call builds + caches.
        first = orch._pick_retriever_for_intent(Intent.PERSONAL)
        assert first is bm25_spy
        assert built_for == ["bm25"]
        assert "bm25" in orch._per_intent_retrievers

        # Second call returns the cached instance — factory not invoked again.
        second = orch._pick_retriever_for_intent(Intent.PERSONAL)
        assert second is bm25_spy
        assert built_for == ["bm25"]  # unchanged
    finally:
        _close(orch)


# ---------------------------------------------------------------------------
# Integration tests through Orchestrator.chat()
# ---------------------------------------------------------------------------


def test_chat_emits_adaptive_retriever_picked_event(sample_config, monkeypatch) -> None:
    """PERSONAL turn with override="bm25" while global="vector" — chat() fires
    the ``adaptive_retriever_picked`` event AND routes through the BM25 spy."""
    sample_config.retrieval.adaptive_enabled = True
    sample_config.retrieval.retriever = "vector"
    sample_config.retrieval.adaptive_retriever_per_intent = {
        "greeting": "default",
        "personal": "bm25",
        "factual": "default",
        "general": "default",
        "unclear": "default",
    }
    # Disable doc-scope so the spy doesn't get wrapped + name-shifted.
    sample_config.retrieval.doc_scope_enabled = False

    global_spy = _RecordingRetriever(name="vector")
    bm25_spy = _RecordingRetriever(
        name="bm25",
        results=[_result("ep1", 0.9, source_type="episodic")],
    )

    # Build the orchestrator FIRST so its __init__ goes through the real
    # build_retriever (which works for "vector"). Only AFTER that do we
    # monkeypatch the factory — so the only call we intercept is the one the
    # per-intent resolver makes for "bm25".
    orch = _make_orch(sample_config, _ScriptedClassifier(Intent.PERSONAL), global_spy)

    def fake_build_retriever(retrieval_cfg, db, vector_store, embedder, **kwargs):
        assert retrieval_cfg.retriever == "bm25"
        return bm25_spy

    monkeypatch.setattr("hrag.orchestrator.build_retriever", fake_build_retriever)

    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what do I prefer for testing?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        _close(orch)

    # The override event fired once with the right payload.
    picked = [p for n, p in events if n == "adaptive_retriever_picked"]
    assert len(picked) == 1, f"expected one adaptive_retriever_picked event, got {picked}"
    assert picked[0]["intent"] == "personal"
    assert picked[0]["retriever"] == "bm25"
    assert picked[0]["global"] == "vector"

    # The BM25 spy was used for retrieval, not the global spy.
    assert bm25_spy.calls, "bm25 spy should have been called"
    assert global_spy.calls == [], "global spy must not have been called"


def test_chat_does_not_emit_when_mapping_is_default(sample_config) -> None:
    """All-default mapping → the override event MUST NOT fire (silent path)."""
    sample_config.retrieval.adaptive_enabled = True
    # default-everywhere mapping (the default config).
    spy = _RecordingRetriever(
        name="vector",
        results=[_result("c1", 0.9)],
    )
    orch = _make_orch(sample_config, _ScriptedClassifier(Intent.FACTUAL), spy)
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what is hipporag?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        _close(orch)

    picked = [p for n, p in events if n == "adaptive_retriever_picked"]
    assert picked == []
    # And the global spy was used.
    assert spy.calls


def test_invalid_retriever_name_falls_back_silently(sample_config) -> None:
    """A mapping value that build_retriever rejects → resolver logs a warning
    and returns self.retriever; chat() does not crash."""
    sample_config.retrieval.adaptive_enabled = True
    sample_config.retrieval.adaptive_retriever_per_intent = {
        "greeting": "default",
        "personal": "default",
        "factual": "not_a_real_retriever",  # build_retriever raises ValueError
        "general": "default",
        "unclear": "default",
    }
    spy = _RecordingRetriever(
        name="vector",
        results=[_result("c1", 0.9)],
    )
    orch = _make_orch(sample_config, _ScriptedClassifier(Intent.FACTUAL), spy)

    try:
        # Resolver directly — confirm graceful fallback.
        picked = orch._pick_retriever_for_intent(Intent.FACTUAL)
        assert picked is spy
        # Nothing was cached (the build failed).
        assert "not_a_real_retriever" not in orch._per_intent_retrievers
    finally:
        _close(orch)


def test_missing_taxonomy_dep_falls_back_silently(sample_config) -> None:
    """A mapping of "taxonomy" with no taxonomy_store on the Orchestrator must
    NOT crash — the resolver swallows the ValueError and returns the global."""
    # Ensure taxonomy is disabled so self.taxonomy_store is None.
    sample_config.taxonomy.enabled = False
    sample_config.retrieval.adaptive_enabled = True
    sample_config.retrieval.retriever = "vector"
    sample_config.retrieval.adaptive_retriever_per_intent = {
        "greeting": "default",
        "personal": "default",
        "factual": "taxonomy",  # needs taxonomy_store, which is None
        "general": "default",
        "unclear": "default",
    }
    spy = _RecordingRetriever(name="vector")
    orch = _make_orch(sample_config, _ScriptedClassifier(Intent.FACTUAL), spy)
    try:
        # taxonomy_store is None → build_retriever for "taxonomy" raises;
        # the resolver must catch + fall back to the global.
        assert orch.taxonomy_store is None
        picked = orch._pick_retriever_for_intent(Intent.FACTUAL)
        assert picked is spy
        assert "taxonomy" not in orch._per_intent_retrievers
    finally:
        _close(orch)
