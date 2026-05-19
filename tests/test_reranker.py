"""Tests for hrag.retrieval.reranker — LLMReranker."""

from __future__ import annotations

import pytest

from hrag.retrieval.reranker import LLMReranker, _parse_score
from hrag.types import Chunk, RetrievalResult

# FakeLLM is imported from conftest implicitly via the fake_llm fixture,
# but we also import the class directly for type annotations.
from tests.conftest import FakeLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, idx: int = 0, doc_id: str = "doc001") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{idx:04d}",
        doc_id=doc_id,
        user_id="tester",
        text=text,
        embedding_text=text,
        chunk_index=idx,
        token_count=len(text.split()),
    )


def _make_result(text: str, score: float = 0.8, idx: int = 0) -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(text, idx=idx), score=score)


def _make_results(n: int = 5) -> list[RetrievalResult]:
    texts = [
        "The capital of France is Paris.",
        "Photosynthesis converts sunlight into energy.",
        "Python is a programming language.",
        "The moon orbits the Earth.",
        "Water boils at 100 degrees Celsius.",
    ]
    return [
        _make_result(texts[i % len(texts)], score=0.9 - i * 0.05, idx=i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _parse_score unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("3", 3),
    ("0", 0),
    ("Score: 2", 2),
    ("The relevance is 1.", 1),
    ("no digit here", 0),
    ("", 0),
    ("  \n3\n  ", 3),
    ("4",  0),   # 4 is out of range 0-3; regex only matches 0-3
])
def test_parse_score(raw: str, expected: int) -> None:
    assert _parse_score(raw) == expected


# ---------------------------------------------------------------------------
# rerank_score is set on results
# ---------------------------------------------------------------------------

def test_rerank_sets_rerank_score(fake_llm: FakeLLM) -> None:
    """rerank() must populate rerank_score on each result that passes threshold."""
    reranker = LLMReranker(fake_llm)
    results = _make_results(3)
    # FakeLLM rerank cycle: 0,1,2,3,2,1,...
    # With threshold=0, all pass
    ranked = reranker.rerank("What is the capital of France?", results, threshold=0)
    assert all(r.rerank_score is not None for r in ranked)
    assert all(isinstance(r.rerank_score, int) for r in ranked)


def test_rerank_score_in_valid_range(fake_llm: FakeLLM) -> None:
    """Scores from FakeLLM parsing must be in 0-3."""
    reranker = LLMReranker(fake_llm)
    results = _make_results(6)
    ranked = reranker.rerank("test query", results, threshold=0)
    for r in ranked:
        assert 0 <= r.rerank_score <= 3, f"Score out of range: {r.rerank_score}"


# ---------------------------------------------------------------------------
# Threshold filtering
# ---------------------------------------------------------------------------

def test_rerank_filters_below_threshold(fake_llm: FakeLLM) -> None:
    """Results with rerank_score < threshold must be excluded."""
    reranker = LLMReranker(fake_llm)
    results = _make_results(6)
    # FakeLLM cycle: 0,1,2,3,2,1 → 0 and 1 are below threshold=2
    ranked = reranker.rerank("query", results, threshold=2)
    for r in ranked:
        assert r.rerank_score >= 2, (
            f"Result with score {r.rerank_score} should have been filtered out"
        )


def test_rerank_threshold_zero_keeps_all_nonzero_or_above(fake_llm: FakeLLM) -> None:
    """threshold=0 keeps everything (score 0 >= 0)."""
    reranker = LLMReranker(fake_llm)
    results = _make_results(4)
    ranked = reranker.rerank("query", results, threshold=0)
    # All 4 should survive since 0 >= 0
    assert len(ranked) == 4


def test_rerank_threshold_high_filters_all(fake_llm: FakeLLM) -> None:
    """threshold=3 should only keep results scored 3."""
    reranker = LLMReranker(fake_llm)
    # Reset counter to ensure we know the cycle position
    fake_llm._counter = 0
    results = _make_results(6)
    # Cycle: 0,1,2,3,2,1 → only index 3 (score=3) passes threshold=3
    ranked = reranker.rerank("query", results, threshold=3)
    assert all(r.rerank_score == 3 for r in ranked)


# ---------------------------------------------------------------------------
# top_k truncation
# ---------------------------------------------------------------------------

def test_rerank_truncates_to_top_k(fake_llm: FakeLLM) -> None:
    """top_k must limit the number of returned results."""
    fake_llm._counter = 0
    reranker = LLMReranker(fake_llm)
    results = _make_results(6)
    ranked = reranker.rerank("query", results, threshold=0, top_k=2)
    assert len(ranked) <= 2


def test_rerank_top_k_none_returns_all_passing(fake_llm: FakeLLM) -> None:
    """When top_k is None, all results above threshold are returned."""
    fake_llm._counter = 0
    reranker = LLMReranker(fake_llm)
    results = _make_results(6)
    # threshold=0, top_k=None → all 6 pass
    ranked = reranker.rerank("query", results, threshold=0, top_k=None)
    assert len(ranked) == 6


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_rerank_sorted_descending_by_score(fake_llm: FakeLLM) -> None:
    """Results must be sorted by rerank_score descending."""
    fake_llm._counter = 0
    reranker = LLMReranker(fake_llm)
    results = _make_results(6)
    ranked = reranker.rerank("query", results, threshold=0)
    scores = [r.rerank_score for r in ranked]
    assert scores == sorted(scores, reverse=True), f"Not sorted descending: {scores}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_rerank_empty_input_returns_empty(fake_llm: FakeLLM) -> None:
    reranker = LLMReranker(fake_llm)
    assert reranker.rerank("query", [], threshold=2) == []


def test_rerank_llm_called_once_per_result(fake_llm: FakeLLM) -> None:
    """The LLM must be called exactly once per input result."""
    reranker = LLMReranker(fake_llm)
    results = _make_results(4)
    fake_llm.calls.clear()
    reranker.rerank("query", results, threshold=0)
    # complete() → generate() builds a single merged prompt; check call count
    assert len(fake_llm.calls) == 4
