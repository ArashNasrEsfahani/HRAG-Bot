"""Tests for Phase 9.11 — QueryRouter speculative short-circuit.

When ``short_circuit=True``, the router skips the multi-retriever RRF fusion
for ``entity`` and ``global`` routes and calls ONLY the primary retriever:

  entity  -> kg_ppr (falls back to bm25, then vector)
  global  -> community (falls back to vector)

``cross_document`` and ``ambiguous`` always use the full RRF fan-out.

These tests are additive: the 24 pre-Phase-9.11 tests in ``test_router.py``
all exercise the fusion path (``short_circuit=False`` default) and must remain
byte-identical pass.
"""

from __future__ import annotations

from typing import Optional

import pytest

from hrag.config import RetrievalConfig
from hrag.retrieval.router import QueryRouter
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Stubs (duplicated locally to keep this file self-contained)
# ---------------------------------------------------------------------------


class _StubLLM:
    def __init__(self, output: str = "entity") -> None:
        self._output = output
        self.calls: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        return self._output


class _StubRetriever:
    def __init__(self, name: str, results: list[RetrievalResult]) -> None:
        self.name = name
        self._results = results
        self.calls = 0

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 30,
        source_types: Optional[list[str]] = None,
        intent_hint=None,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        self.calls += 1
        return list(self._results)


def _make_result(chunk_id: str, retriever: str = "stub") -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        user_id="u1",
        text=f"text-{chunk_id}",
        embedding_text=f"text-{chunk_id}",
    )
    return RetrievalResult(chunk=chunk, score=0.9, retriever=retriever)


def _make_results(ids: list[str], retriever: str = "stub") -> list[RetrievalResult]:
    return [_make_result(cid, retriever) for cid in ids]


# ---------------------------------------------------------------------------
# 1. Config default
# ---------------------------------------------------------------------------


def test_short_circuit_default_on() -> None:
    """RetrievalConfig.router_short_circuit must default to True."""
    assert RetrievalConfig().router_short_circuit is True


# ---------------------------------------------------------------------------
# 2. entity label -> ONLY kg_ppr called
# ---------------------------------------------------------------------------


def test_short_circuit_entity_calls_only_kg_ppr() -> None:
    """With short_circuit=True and label=entity, only kg_ppr.retrieve() is called.

    bm25 and vector must NOT be called. The returned results come straight
    from kg_ppr (no RRF fusion, no router re-tagging).
    """
    llm = _StubLLM(output="entity")
    kg = _StubRetriever("kg_ppr", _make_results(["kg1", "kg2"], retriever="kg_ppr"))
    bm25 = _StubRetriever("bm25", _make_results(["bm1"], retriever="bm25"))
    vec = _StubRetriever("vector", _make_results(["v1"], retriever="vector"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        bm25_retriever=bm25,
        short_circuit=True,
    )
    results = router.retrieve("what is X?", user_id="u1", top_k=10)

    assert kg.calls == 1, "kg_ppr must be called exactly once"
    assert bm25.calls == 0, "bm25 must NOT be called when short-circuiting entity"
    assert vec.calls == 0, "vector must NOT be called when short-circuiting entity"

    chunk_ids = {r.chunk.chunk_id for r in results}
    assert chunk_ids == {"kg1", "kg2"}, "results must come from kg_ppr only"


# ---------------------------------------------------------------------------
# 3. global label -> ONLY community called
# ---------------------------------------------------------------------------


def test_short_circuit_global_calls_only_community() -> None:
    """With short_circuit=True and label=global, only community.retrieve() is called.

    kg_ppr, bm25, and vector must NOT be called.
    """
    llm = _StubLLM(output="global")
    kg = _StubRetriever("kg_ppr", _make_results(["kg1"]))
    vec = _StubRetriever("vector", _make_results(["v1"]))
    com = _StubRetriever("community", _make_results(["c1", "c2"], retriever="community"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        community_retriever=com,
        short_circuit=True,
    )
    results = router.retrieve("summarize the corpus", user_id="u1", top_k=10)

    assert com.calls == 1, "community must be called exactly once"
    assert kg.calls == 0, "kg_ppr must NOT be called when short-circuiting global"
    assert vec.calls == 0, "vector must NOT be called when short-circuiting global"

    chunk_ids = {r.chunk.chunk_id for r in results}
    assert chunk_ids == {"c1", "c2"}, "results must come from community only"


# ---------------------------------------------------------------------------
# 4. cross_document still fuses all retrievers
# ---------------------------------------------------------------------------


def test_short_circuit_cross_document_still_fuses() -> None:
    """cross_document always uses the full RRF fusion, even when short_circuit=True."""
    llm = _StubLLM(output="cross_document")
    kg = _StubRetriever("kg_ppr", _make_results(["A", "B"], retriever="kg_ppr"))
    bm25 = _StubRetriever("bm25", _make_results(["B", "C"], retriever="bm25"))
    vec = _StubRetriever("vector", _make_results(["C", "D"], retriever="vector"))
    com = _StubRetriever("community", _make_results(["D", "E"], retriever="community"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        community_retriever=com,
        bm25_retriever=bm25,
        short_circuit=True,
    )
    results = router.retrieve("compare A and B", user_id="u1", top_k=20)

    # All four retrievers must be called for cross_document.
    assert kg.calls == 1
    assert bm25.calls == 1
    assert vec.calls == 1
    assert com.calls == 1

    chunk_ids = {r.chunk.chunk_id for r in results}
    assert chunk_ids == {"A", "B", "C", "D", "E"}
    assert all(r.retriever == "router" for r in results), "fused results must be tagged router"


# ---------------------------------------------------------------------------
# 5. ambiguous still fuses
# ---------------------------------------------------------------------------


def test_short_circuit_ambiguous_still_fuses() -> None:
    """ambiguous always uses the full RRF fusion, even when short_circuit=True."""
    llm = _StubLLM(output="ambiguous")
    kg = _StubRetriever("kg_ppr", _make_results(["A"], retriever="kg_ppr"))
    bm25 = _StubRetriever("bm25", _make_results(["B"], retriever="bm25"))
    vec = _StubRetriever("vector", _make_results(["C"], retriever="vector"))
    com = _StubRetriever("community", _make_results(["X"]))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        community_retriever=com,
        bm25_retriever=bm25,
        short_circuit=True,
    )
    results = router.retrieve("tell me more", user_id="u1", top_k=10)

    # ambiguous fuses kg + bm25 + vector; community is NOT called.
    assert kg.calls == 1
    assert bm25.calls == 1
    assert vec.calls == 1
    assert com.calls == 0

    chunk_ids = {r.chunk.chunk_id for r in results}
    assert chunk_ids == {"A", "B", "C"}
    assert all(r.retriever == "router" for r in results)


# ---------------------------------------------------------------------------
# 6. Disabled flag -> fusion for all labels
# ---------------------------------------------------------------------------


def test_short_circuit_disabled_uses_fusion_for_all_labels() -> None:
    """When short_circuit=False, all labels go through full fusion."""
    llm = _StubLLM(output="entity")
    kg = _StubRetriever("kg_ppr", _make_results(["kg1"], retriever="kg_ppr"))
    bm25 = _StubRetriever("bm25", _make_results(["bm1"], retriever="bm25"))
    vec = _StubRetriever("vector", _make_results(["v1"], retriever="vector"))

    # short_circuit=False is the constructor default; pass it explicitly here.
    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        bm25_retriever=bm25,
        short_circuit=False,
    )
    results = router.retrieve("what is X?", user_id="u1", top_k=10)

    # Entity fusion path: all three must be called.
    assert kg.calls == 1
    assert bm25.calls == 1
    assert vec.calls == 1

    chunk_ids = {r.chunk.chunk_id for r in results}
    assert {"kg1", "bm1", "v1"} == chunk_ids
    assert all(r.retriever == "router" for r in results)


# ---------------------------------------------------------------------------
# 7. Progress event emitted on short-circuit
# ---------------------------------------------------------------------------


def test_short_circuit_emits_progress_event() -> None:
    """When short_circuit=True and label is entity or global, a
    ``router_short_circuit`` progress event is emitted with the correct payload.
    """
    events: list[tuple[str, dict]] = []

    def capture(event: str, payload: dict) -> None:
        events.append((event, payload))

    # Test entity short-circuit progress event.
    llm = _StubLLM(output="entity")
    kg = _StubRetriever("kg_ppr", _make_results(["kg1"]))

    router = QueryRouter(
        llm=llm,
        kg_ppr_retriever=kg,
        short_circuit=True,
        progress=capture,
    )
    router.retrieve("what is X?", user_id="u1", top_k=10)

    sc_events = [(e, p) for e, p in events if e == "router_short_circuit"]
    assert len(sc_events) == 1, "exactly one router_short_circuit event expected"
    event_name, payload = sc_events[0]
    assert payload["label"] == "entity"
    assert payload["retriever"] == "kg_ppr"

    # Test global short-circuit progress event.
    events.clear()
    llm2 = _StubLLM(output="global")
    com = _StubRetriever("community", _make_results(["c1"]))

    router2 = QueryRouter(
        llm=llm2,
        community_retriever=com,
        short_circuit=True,
        progress=capture,
    )
    router2.retrieve("summarize", user_id="u1", top_k=10)

    sc_events2 = [(e, p) for e, p in events if e == "router_short_circuit"]
    assert len(sc_events2) == 1
    _, payload2 = sc_events2[0]
    assert payload2["label"] == "global"
    assert payload2["retriever"] == "community"


def test_short_circuit_no_event_when_no_progress_callback() -> None:
    """When short_circuit=True but no progress callback was given, no error raised."""
    llm = _StubLLM(output="entity")
    kg = _StubRetriever("kg_ppr", _make_results(["kg1"]))

    router = QueryRouter(
        llm=llm,
        kg_ppr_retriever=kg,
        short_circuit=True,
        progress=None,  # explicit None
    )
    # Must not raise even though no callback is wired.
    results = router.retrieve("what is X?", user_id="u1", top_k=10)
    assert len(results) == 1


def test_short_circuit_no_event_for_cross_document() -> None:
    """cross_document does NOT emit a router_short_circuit event."""
    events: list[tuple[str, dict]] = []

    def capture(event: str, payload: dict) -> None:
        events.append((event, payload))

    llm = _StubLLM(output="cross_document")
    kg = _StubRetriever("kg_ppr", _make_results(["A"]))
    vec = _StubRetriever("vector", _make_results(["B"]))

    router = QueryRouter(
        llm=llm,
        kg_ppr_retriever=kg,
        vector_retriever=vec,
        short_circuit=True,
        progress=capture,
    )
    router.retrieve("compare A and B", user_id="u1", top_k=10)

    sc_events = [e for e, _ in events if e == "router_short_circuit"]
    assert sc_events == [], "router_short_circuit must NOT fire for cross_document"


# ---------------------------------------------------------------------------
# 8. Classifier cache still works with short-circuit
# ---------------------------------------------------------------------------


def test_short_circuit_preserves_classifier_cache() -> None:
    """Calling retrieve() twice with the same query only invokes the LLM once.

    The classify() method caches results per query string. Short-circuit must
    not bypass or invalidate the cache.
    """
    llm = _StubLLM(output="entity")
    kg = _StubRetriever("kg_ppr", _make_results(["kg1"]))

    router = QueryRouter(
        llm=llm,
        kg_ppr_retriever=kg,
        short_circuit=True,
    )

    router.retrieve("the same question", user_id="u1", top_k=10)
    router.retrieve("the same question", user_id="u1", top_k=10)

    # LLM classify call must happen exactly once (cache hit on 2nd call).
    assert len(llm.calls) == 1, (
        f"LLM was called {len(llm.calls)} times; expected 1 (cache should hit on 2nd call)"
    )
    # kg_ppr.retrieve() is called both times (short-circuit fires both times,
    # but only the LLM is cached, not the retrieval results).
    assert kg.calls == 2


# ---------------------------------------------------------------------------
# 9. Fallback when primary retriever is absent (entity -> bm25)
# ---------------------------------------------------------------------------


def test_short_circuit_entity_falls_back_to_bm25_when_no_kg() -> None:
    """When kg_ppr is absent, entity short-circuit falls back to bm25."""
    llm = _StubLLM(output="entity")
    bm25 = _StubRetriever("bm25", _make_results(["bm1", "bm2"]))
    vec = _StubRetriever("vector", _make_results(["v1"]))

    router = QueryRouter(
        llm=llm,
        bm25_retriever=bm25,
        vector_retriever=vec,
        short_circuit=True,
    )
    results = router.retrieve("what is X?", user_id="u1", top_k=10)

    assert bm25.calls == 1
    assert vec.calls == 0  # only first-available, not all
    assert {r.chunk.chunk_id for r in results} == {"bm1", "bm2"}


def test_short_circuit_global_falls_back_to_vector_when_no_community() -> None:
    """When community is absent, global short-circuit falls back to vector."""
    llm = _StubLLM(output="global")
    vec = _StubRetriever("vector", _make_results(["v1", "v2"]))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        short_circuit=True,
    )
    results = router.retrieve("summarize the corpus", user_id="u1", top_k=10)

    assert vec.calls == 1
    assert {r.chunk.chunk_id for r in results} == {"v1", "v2"}


# ---------------------------------------------------------------------------
# 10. where kwarg is propagated through short-circuit path
# ---------------------------------------------------------------------------


def test_short_circuit_propagates_where_kwarg() -> None:
    """Phase 7-A contract 16: the where kwarg must thread through short-circuit."""

    class _WhereCapture:
        name = "kg_ppr"
        calls: list[dict | None] = []

        def retrieve(self, query, user_id, top_k=30, source_types=None,
                     intent_hint=None, where=None):
            _WhereCapture.calls.append(where)
            return []

    llm = _StubLLM(output="entity")
    kg = _WhereCapture()

    router = QueryRouter(
        llm=llm,
        kg_ppr_retriever=kg,  # type: ignore[arg-type]
        short_circuit=True,
    )
    router.retrieve("what is X?", user_id="u1", top_k=10, where={"has_math": True})

    assert _WhereCapture.calls == [{"has_math": True}], (
        "where kwarg must be forwarded to the inner retriever via short-circuit"
    )
