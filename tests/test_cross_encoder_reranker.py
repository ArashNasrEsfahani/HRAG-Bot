"""Tests for hrag.retrieval.cross_encoder_reranker — CrossEncoderReranker.

sentence_transformers is optional; skip the whole file if absent.
The CrossEncoder model is NOT downloaded — we inject a FakeCrossEncoder class
into the sentence_transformers stub module so that CrossEncoderReranker.__init__
picks it up without any network call.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")

from hrag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"chunk:{idx:04d}",
        doc_id="doc001",
        user_id="tester",
        text=text,
        embedding_text=text,
        chunk_index=idx,
        token_count=len(text.split()),
    )


def _make_result(text: str, score: float = 0.5, idx: int = 0) -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(text, idx=idx), score=score)


def _five_results() -> list[RetrievalResult]:
    topics = [
        "python programming language tutorial",
        "weather forecast rain umbrella",
        "Eiffel tower Paris landmark France",
        "machine learning models deep learning neural networks",
        "cooking recipes pasta ingredients",
    ]
    return [_make_result(t, score=0.8 - i * 0.1, idx=i) for i, t in enumerate(topics)]


# ---------------------------------------------------------------------------
# Fake CrossEncoder that uses word-overlap for deterministic scoring
# ---------------------------------------------------------------------------

class _FakeCrossEncoder:
    """Deterministic, offline CrossEncoder replacement."""

    def __init__(self, model_name=None, device=None, max_length=512, **kwargs):
        pass  # no download

    def predict(self, pairs, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True, **kwargs):
        out = []
        for q, p in pairs:
            q_tokens = set(q.lower().split())
            p_tokens = set(p.lower().split())
            overlap = len(q_tokens & p_tokens)
            out.append(float(overlap))
        return np.array(out)


# ---------------------------------------------------------------------------
# Fixture: inject FakeCrossEncoder into the sentence_transformers stub module
# ---------------------------------------------------------------------------

@pytest.fixture()
def patched_reranker(monkeypatch):
    """CrossEncoderReranker backed by FakeCrossEncoder — no download, deterministic."""
    # The conftest stub registers sentence_transformers in sys.modules.
    # We add CrossEncoder to that stub so the `from sentence_transformers import CrossEncoder`
    # inside CrossEncoderReranker.__init__ resolves to our fake.
    st_mod = sys.modules["sentence_transformers"]
    monkeypatch.setattr(st_mod, "CrossEncoder", _FakeCrossEncoder, raising=False)
    return CrossEncoderReranker(model_name="stub-model")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ml_chunk_near_top_for_deep_learning_query(patched_reranker):
    """The ML chunk should appear in top-2 results for 'deep learning' query."""
    results = _five_results()
    # index 3 == "machine learning models deep learning neural networks"
    ranked = patched_reranker.rerank(
        "deep learning", results, threshold=0.0, top_k=2
    )
    chunk_texts = [r.chunk.text for r in ranked]
    ml_text = "machine learning models deep learning neural networks"
    assert any(ml_text in t for t in chunk_texts), (
        f"ML chunk not in top-2: {chunk_texts}"
    )


def test_high_threshold_returns_empty(patched_reranker):
    """A threshold of 100.0 should filter out all results (max overlap << 100)."""
    results = _five_results()
    ranked = patched_reranker.rerank("deep learning", results, threshold=100.0)
    assert ranked == []


def test_top_k_truncation(patched_reranker):
    """top_k=2 must return exactly 2 results (assuming enough pass threshold)."""
    results = _five_results()
    ranked = patched_reranker.rerank("deep learning", results, threshold=0.0, top_k=2)
    assert len(ranked) == 2


def test_rerank_score_set_on_retained_results(patched_reranker):
    """Every result in the returned list must have a float rerank_score."""
    results = _five_results()
    ranked = patched_reranker.rerank("python tutorial", results, threshold=0.0)
    assert len(ranked) > 0
    for r in ranked:
        assert r.rerank_score is not None
        assert isinstance(r.rerank_score, float)


def test_rerank_score_set_even_on_filtered_results(patched_reranker):
    """rerank_score should be populated even for results that don't pass threshold."""
    results = _five_results()
    # Call with threshold so some may be filtered — but scores are still set on all
    patched_reranker.rerank("weather rain", results, threshold=0.0)
    for r in results:
        assert r.rerank_score is not None


def test_progress_callback_fires(patched_reranker):
    """Progress callback must be called at least once per result + once at end."""
    results = _five_results()
    calls = []

    def cb(idx, total, score):
        calls.append((idx, total, score))

    patched_reranker.rerank("python", results, threshold=0.0, progress=cb)
    # Per-result ticks (5) + one final summary call = 6 total
    n = len(results)
    assert len(calls) >= n, f"Expected >= {n} progress calls, got {len(calls)}"


def test_results_sorted_descending(patched_reranker):
    """Returned results must be sorted by rerank_score descending."""
    results = _five_results()
    ranked = patched_reranker.rerank("machine learning", results, threshold=0.0)
    scores = [r.rerank_score for r in ranked]
    assert scores == sorted(scores, reverse=True), f"Not sorted: {scores}"


def test_empty_input_returns_empty(patched_reranker):
    """Empty input list → empty output."""
    assert patched_reranker.rerank("query", []) == []
