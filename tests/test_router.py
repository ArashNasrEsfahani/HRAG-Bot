"""Tests for hrag.retrieval.router — QueryRouter classification + dispatch."""

from __future__ import annotations

from typing import Optional

import pytest

from hrag.retrieval.router import QueryRouter, _clean_llm_output, _rrf_fuse
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal LLMProvider stand-in for tests."""

    def __init__(self, output: str = "ambiguous", raise_exc: Exception | None = None) -> None:
        self._output = output
        self._raise = raise_exc
        self.calls: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        if self._raise is not None:
            raise self._raise
        return self._output


class _StubRetriever:
    """Returns canned results; counts how many times retrieve() is called."""

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
        intent_hint=None,  # ignored; required for Retriever contract compat
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        self.calls += 1
        return list(self._results)


class _RaisingRetriever:
    name = "boom"

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("kaboom")


# ---------------------------------------------------------------------------
# Result-list builders
# ---------------------------------------------------------------------------


def _make_result(chunk_id: str, score: float = 1.0, retriever: str = "vector") -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        user_id="u1",
        text=f"text-{chunk_id}",
        embedding_text=f"text-{chunk_id}",
    )
    return RetrievalResult(chunk=chunk, score=score, retriever=retriever)


def _make_results(ids: list[str], retriever: str = "vector") -> list[RetrievalResult]:
    return [_make_result(cid, score=1.0 - i * 0.01, retriever=retriever) for i, cid in enumerate(ids)]


# ---------------------------------------------------------------------------
# _clean_llm_output
# ---------------------------------------------------------------------------


def test_clean_llm_output_strips_quotes_and_prefixes() -> None:
    assert _clean_llm_output('  "entity"  ') == "entity"
    assert _clean_llm_output("Answer: entity") == "entity"
    assert _clean_llm_output("A: 'global'") == "global"
    assert _clean_llm_output("```ambiguous```") == "ambiguous"


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["entity", "global", "cross_document", "ambiguous"])
def test_classify_returns_label_for_clean_output(label: str) -> None:
    llm = _StubLLM(output=label)
    router = QueryRouter(llm=llm, vector_retriever=_StubRetriever("v", []))
    assert router.classify("any question") == label


def test_classify_strips_quotes_and_prose_around_label() -> None:
    llm = _StubLLM(output='  "entity"  ')
    router = QueryRouter(llm=llm)
    assert router.classify("what is X?") == "entity"


def test_classify_falls_back_to_entity_on_malformed() -> None:
    # Updated 2026-05: precision-favoring fallback flipped from "ambiguous"
    # to "entity" — when the classifier emits garbage, route to the KG/vector
    # path rather than fanning out across all retrievers.
    llm = _StubLLM(output="I don't know")
    router = QueryRouter(llm=llm)
    assert router.classify("what is X?") == "entity"


def test_classify_falls_back_to_entity_on_exception() -> None:
    # Updated 2026-05: precision-favoring fallback flipped from "ambiguous"
    # to "entity" so LLM failures don't trigger broad multi-retriever fanout.
    llm = _StubLLM(raise_exc=RuntimeError("boom"))
    router = QueryRouter(llm=llm)
    assert router.classify("what is X?") == "entity"


def test_classify_is_cached() -> None:
    llm = _StubLLM(output="entity")
    router = QueryRouter(llm=llm)
    assert router.classify("the same query") == "entity"
    assert router.classify("the same query") == "entity"
    assert len(llm.calls) == 1


def test_classify_cross_document_takes_priority_over_entity_substring() -> None:
    """LLM may emit something like 'cross_document'; substring search must
    not collapse it to 'entity' just because 'entity' is a label too."""
    llm = _StubLLM(output="cross_document")
    router = QueryRouter(llm=llm)
    assert router.classify("compare A vs B") == "cross_document"


def test_classify_handles_dashed_variant() -> None:
    llm = _StubLLM(output="cross-document")
    router = QueryRouter(llm=llm)
    assert router.classify("compare A vs B") == "cross_document"


def test_classify_empty_query_returns_entity_without_llm_call() -> None:
    # Updated 2026-05: precision-favoring fallback flipped from "ambiguous"
    # to "entity" for empty/whitespace queries. No LLM call still expected.
    llm = _StubLLM(output="entity")
    router = QueryRouter(llm=llm)
    assert router.classify("") == "entity"
    assert router.classify("    ") == "entity"
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_routing_entity_rrf_fuses_kg_bm25_vector() -> None:
    # updated for iter-3: entity route now RRF-fuses kg_ppr+bm25+vector instead
    # of calling kg_ppr only. community is NOT called for entity queries.
    llm = _StubLLM(output="entity")
    kg = _StubRetriever("kg_ppr", _make_results(["A", "B"], retriever="kg_ppr"))
    bm25 = _StubRetriever("bm25", _make_results(["B", "C"], retriever="bm25"))
    vec = _StubRetriever("vector", _make_results(["C", "D"], retriever="vector"))
    com = _StubRetriever("community", _make_results(["C1"], retriever="community"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        community_retriever=com,
        bm25_retriever=bm25,
    )
    results = router.retrieve("what is X", user_id="u1", top_k=10)
    assert kg.calls == 1
    assert bm25.calls == 1
    assert vec.calls == 1
    assert com.calls == 0
    chunk_ids = {r.chunk.chunk_id for r in results}
    # Fusion must include the union across all three entity retrievers.
    assert chunk_ids == {"A", "B", "C", "D"}
    # Fused results carry the router tag.
    assert all(r.retriever == "router" for r in results)


def test_routing_global_uses_community_only() -> None:
    llm = _StubLLM(output="global")
    kg = _StubRetriever("kg_ppr", _make_results(["A"], retriever="kg_ppr"))
    vec = _StubRetriever("vector", _make_results(["X"], retriever="vector"))
    com = _StubRetriever("community", _make_results(["C1", "C2"], retriever="community"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        community_retriever=com,
    )
    results = router.retrieve("summarize the corpus", user_id="u1", top_k=10)
    assert com.calls == 1
    assert kg.calls == 0
    assert vec.calls == 0
    assert {r.chunk.chunk_id for r in results} == {"C1", "C2"}


def test_routing_cross_document_calls_all_three_and_fuses() -> None:
    llm = _StubLLM(output="cross_document")
    kg = _StubRetriever("kg_ppr", _make_results(["A", "B"], retriever="kg_ppr"))
    vec = _StubRetriever("vector", _make_results(["B", "C"], retriever="vector"))
    com = _StubRetriever("community", _make_results(["A", "D"], retriever="community"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        community_retriever=com,
    )
    results = router.retrieve("compare A and B", user_id="u1", top_k=10)
    assert kg.calls == 1
    assert vec.calls == 1
    assert com.calls == 1
    chunk_ids = {r.chunk.chunk_id for r in results}
    # Fusion must include unique union of all three lists.
    assert chunk_ids == {"A", "B", "C", "D"}
    # Fused results carry the router tag.
    assert all(r.retriever == "router" for r in results)


def test_routing_ambiguous_calls_kg_and_vector() -> None:
    llm = _StubLLM(output="ambiguous")
    kg = _StubRetriever("kg_ppr", _make_results(["A"], retriever="kg_ppr"))
    vec = _StubRetriever("vector", _make_results(["X"], retriever="vector"))
    com = _StubRetriever("community", _make_results(["C1"], retriever="community"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        community_retriever=com,
    )
    results = router.retrieve("tell me more", user_id="u1", top_k=10)
    assert kg.calls == 1
    assert vec.calls == 1
    assert com.calls == 0
    assert {r.chunk.chunk_id for r in results} == {"A", "X"}


def test_routing_entity_falls_back_to_vector_when_no_kg() -> None:
    llm = _StubLLM(output="entity")
    vec = _StubRetriever("vector", _make_results(["X", "Y"], retriever="vector"))

    router = QueryRouter(llm=llm, vector_retriever=vec)
    results = router.retrieve("what is the threshold?", user_id="u1", top_k=10)
    assert vec.calls == 1
    assert {r.chunk.chunk_id for r in results} == {"X", "Y"}


def test_routing_no_retrievers_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    llm = _StubLLM(output="ambiguous")
    router = QueryRouter(llm=llm)
    with caplog.at_level("WARNING"):
        results = router.retrieve("anything", user_id="u1", top_k=10)
    assert results == []
    assert any("no retrievers" in rec.message.lower() for rec in caplog.records)


def test_routing_one_retriever_raises_others_complete(caplog: pytest.LogCaptureFixture) -> None:
    llm = _StubLLM(output="cross_document")
    kg = _RaisingRetriever()
    vec = _StubRetriever("vector", _make_results(["X", "Y"], retriever="vector"))
    com = _StubRetriever("community", _make_results(["C1"], retriever="community"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        community_retriever=com,
    )
    with caplog.at_level("WARNING"):
        results = router.retrieve("compare A and B", user_id="u1", top_k=10)
    # vector + community still produced results
    assert kg.calls == 1
    assert vec.calls == 1
    assert com.calls == 1
    chunk_ids = {r.chunk.chunk_id for r in results}
    assert chunk_ids == {"X", "Y", "C1"}


def test_top_k_honored_after_rrf() -> None:
    llm = _StubLLM(output="cross_document")
    kg = _StubRetriever("kg_ppr", _make_results(["A", "B", "C", "D", "E"], retriever="kg_ppr"))
    vec = _StubRetriever("vector", _make_results(["F", "G", "H", "I", "J"], retriever="vector"))
    com = _StubRetriever("community", _make_results(["K", "L", "M", "N", "O"], retriever="community"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        community_retriever=com,
    )
    results = router.retrieve("compare", user_id="u1", top_k=3)
    assert len(results) <= 3


def test_rrf_fuses_consistent_top_first() -> None:
    """If chunk A is rank-1 in both retrievers, it must top the fused list."""
    list_a = _make_results(["A", "B", "C"], retriever="r1")
    list_b = _make_results(["A", "X", "Y"], retriever="r2")
    fused = _rrf_fuse([list_a, list_b], k=60, top_k=10)
    assert fused[0].chunk.chunk_id == "A"


def test_rrf_returns_empty_for_empty_input() -> None:
    assert _rrf_fuse([], k=60, top_k=10) == []
    assert _rrf_fuse([[]], k=60, top_k=10) == []


def test_rrf_results_tagged_router() -> None:
    list_a = _make_results(["A"], retriever="r1")
    fused = _rrf_fuse([list_a], k=60, top_k=10)
    assert fused[0].retriever == "router"


def test_routing_entity_all_three_wired_returns_results_from_each() -> None:
    """iter-3: entity route with kg_ppr+bm25+vector all wired returns RRF union.

    Verifies that each of the three entity retrievers is called exactly once and
    that the fused result set contains chunks from all three sources.
    """
    llm = _StubLLM(output="entity")
    kg = _StubRetriever("kg_ppr", _make_results(["kg1", "kg2"], retriever="kg_ppr"))
    bm25 = _StubRetriever("bm25", _make_results(["bm1", "bm2"], retriever="bm25"))
    vec = _StubRetriever("vector", _make_results(["v1", "v2"], retriever="vector"))

    router = QueryRouter(
        llm=llm,
        vector_retriever=vec,
        kg_ppr_retriever=kg,
        bm25_retriever=bm25,
    )
    results = router.retrieve("what is the synonymy threshold?", user_id="u1", top_k=10)

    # All three entity retrievers must be called.
    assert kg.calls == 1
    assert bm25.calls == 1
    assert vec.calls == 1

    # Fused result set contains chunks from all three.
    chunk_ids = {r.chunk.chunk_id for r in results}
    assert {"kg1", "kg2"} <= chunk_ids, "kg_ppr chunks missing from entity fusion"
    assert {"bm1", "bm2"} <= chunk_ids, "bm25 chunks missing from entity fusion"
    assert {"v1", "v2"} <= chunk_ids, "vector chunks missing from entity fusion"

    # All results carry the router tag (RRF was applied).
    assert all(r.retriever == "router" for r in results)
