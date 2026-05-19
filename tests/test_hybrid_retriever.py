"""Tests for hrag.retrieval.hybrid — HybridRetriever (RRF fusion).

No external deps; uses stub retrievers.
"""

from __future__ import annotations

from typing import Optional

import pytest

from hrag.retrieval.base import Retriever
from hrag.retrieval.hybrid import HybridRetriever
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc001",
        user_id="tester",
        text=text or chunk_id,
        embedding_text=text or chunk_id,
        chunk_index=0,
        token_count=3,
    )


def _make_result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(chunk_id),
        score=score,
        retriever="test",
    )


class StubRetriever(Retriever):
    """Returns a fixed list of results regardless of query."""

    name = "stub"

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 30,
        source_types: Optional[list[str]] = None,
        intent_hint=None,  # ignored; required for Retriever contract compat
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rrf_fusion_basic():
    """Items from both retrievers appear in fused output."""
    r1 = StubRetriever([_make_result("A"), _make_result("B")])
    r2 = StubRetriever([_make_result("C"), _make_result("D")])
    hybrid = HybridRetriever([r1, r2])
    results = hybrid.retrieve("q", user_id="u", top_k=4)
    ids = {r.chunk.chunk_id for r in results}
    assert {"A", "B", "C", "D"} == ids


def test_chunk_in_both_retrievers_scores_higher():
    """Chunk X appears rank-1 in retriever-1 and rank-1 in retriever-2;
    chunk Y appears only in retriever-2 at rank-1. X should outscore Y."""
    r1 = StubRetriever([_make_result("X", 1.0), _make_result("A", 0.9)])
    r2 = StubRetriever([_make_result("X", 1.0), _make_result("Y", 0.8)])
    hybrid = HybridRetriever([r1, r2])
    results = hybrid.retrieve("q", user_id="u", top_k=10)
    scores = {r.chunk.chunk_id: r.score for r in results}
    assert scores["X"] > scores["Y"], (
        f"X (both lists) should outscore Y (one list): X={scores['X']}, Y={scores['Y']}"
    )


def test_weights_move_top_result():
    """With weights=[2.0, 1.0], the first retriever's top item should rank first."""
    # Retriever-1 has chunk "HEAVY" at rank-1
    # Retriever-2 has chunk "LIGHT" at rank-1
    # Equal weights would make them tied if they're only in their own list;
    # here HEAVY gets double contribution so it should end up on top.
    r1 = StubRetriever([_make_result("HEAVY", 1.0)])
    r2 = StubRetriever([_make_result("LIGHT", 1.0)])
    hybrid = HybridRetriever([r1, r2], weights=[2.0, 1.0])
    results = hybrid.retrieve("q", user_id="u", top_k=10)
    top = results[0].chunk.chunk_id
    assert top == "HEAVY", f"Expected HEAVY at top, got {top}"


def test_top_k_truncation():
    """HybridRetriever must not return more than top_k results."""
    r1 = StubRetriever([_make_result(f"A{i}") for i in range(10)])
    r2 = StubRetriever([_make_result(f"B{i}") for i in range(10)])
    hybrid = HybridRetriever([r1, r2])
    results = hybrid.retrieve("q", user_id="u", top_k=3)
    assert len(results) == 3


def test_retriever_field_is_hybrid():
    """All returned RetrievalResult objects must have retriever='hybrid'."""
    r1 = StubRetriever([_make_result("A")])
    r2 = StubRetriever([_make_result("B")])
    hybrid = HybridRetriever([r1, r2])
    results = hybrid.retrieve("q", user_id="u", top_k=10)
    for r in results:
        assert r.retriever == "hybrid", f"Expected 'hybrid', got {r.retriever!r}"


def test_rrf_scores_are_positive():
    """RRF scores must be strictly positive for any retrieved chunk."""
    r1 = StubRetriever([_make_result("X"), _make_result("Y")])
    hybrid = HybridRetriever([r1])
    results = hybrid.retrieve("q", user_id="u", top_k=10)
    for r in results:
        assert r.score > 0.0, f"RRF score should be positive, got {r.score}"


def test_empty_retrievers_return_empty():
    """Retrievers returning nothing → empty hybrid output."""
    r1 = StubRetriever([])
    r2 = StubRetriever([])
    hybrid = HybridRetriever([r1, r2])
    results = hybrid.retrieve("q", user_id="u", top_k=10)
    assert results == []


def test_single_retriever_passes_through():
    """A single-element retriever list still works."""
    items = [_make_result(f"item{i}") for i in range(5)]
    r1 = StubRetriever(items)
    hybrid = HybridRetriever([r1])
    results = hybrid.retrieve("q", user_id="u", top_k=5)
    assert len(results) == 5


def test_mismatched_weights_raises():
    """len(weights) != len(retrievers) must raise ValueError."""
    r1 = StubRetriever([])
    r2 = StubRetriever([])
    with pytest.raises(ValueError, match="len\\(weights\\)"):
        HybridRetriever([r1, r2], weights=[1.0])


def test_empty_retrievers_list_raises():
    """Empty retriever list must raise ValueError."""
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_results_sorted_by_rrf_score_descending():
    """Returned results must be ordered by RRF score descending."""
    # chunk "TOP" appears at rank-1 in both retrievers → highest score
    r1 = StubRetriever([
        _make_result("TOP"),
        _make_result("MID"),
        _make_result("LOW"),
    ])
    r2 = StubRetriever([
        _make_result("TOP"),
        _make_result("OTHER"),
    ])
    hybrid = HybridRetriever([r1, r2])
    results = hybrid.retrieve("q", user_id="u", top_k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), f"Not sorted: {scores}"
    assert results[0].chunk.chunk_id == "TOP"
