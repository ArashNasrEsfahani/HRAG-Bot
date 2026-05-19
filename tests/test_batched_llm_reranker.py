"""Tests for hrag.retrieval.batched_llm_reranker — BatchedLLMReranker.

Uses the fake_llm fixture; monkeypatches complete() for deterministic responses.
"""

from __future__ import annotations


from hrag.retrieval.batched_llm_reranker import BatchedLLMReranker, _parse_scores
from hrag.types import Chunk, RetrievalResult
from tests.conftest import FakeLLM


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


def _make_result(text: str, score: float = 0.8, idx: int = 0) -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(text, idx=idx), score=score)


def _make_results(n: int = 5) -> list[RetrievalResult]:
    texts = [
        "The capital of France is Paris.",
        "Python is a dynamic programming language.",
        "Deep learning requires large datasets.",
        "The Eiffel tower stands 330 metres tall.",
        "Pasta is a staple of Italian cooking.",
    ]
    return [_make_result(texts[i % len(texts)], idx=i) for i in range(n)]


# ---------------------------------------------------------------------------
# _parse_scores unit tests (internal helper)
# ---------------------------------------------------------------------------

class TestParseScores:
    def test_valid_array(self):
        assert _parse_scores("[0, 2, 3, 1, 0]", 5) == [0, 2, 3, 1, 0]

    def test_wrong_length_returns_none(self):
        assert _parse_scores("[0, 1]", 5) is None

    def test_malformed_json_returns_none(self):
        assert _parse_scores("not json", 3) is None

    def test_out_of_range_value_returns_none(self):
        assert _parse_scores("[0, 4, 1]", 3) is None

    def test_markdown_fence_stripped(self):
        result = _parse_scores("```json\n[2, 0, 3]\n```", 3)
        assert result == [2, 0, 3]

    def test_float_integers_accepted(self):
        # 1.0 is technically a float that equals int 1
        assert _parse_scores("[1.0, 2.0, 3.0]", 3) == [1, 2, 3]

    def test_empty_array_wrong_count(self):
        assert _parse_scores("[]", 3) is None

    def test_non_list_returns_none(self):
        assert _parse_scores('{"a": 1}', 1) is None


# ---------------------------------------------------------------------------
# Happy-path test
# ---------------------------------------------------------------------------

def test_happy_path_scores_and_threshold(fake_llm: FakeLLM, monkeypatch):
    """complete() returns '[0, 2, 3, 1, 0]'; threshold=2 keeps scores 2 and 3."""
    monkeypatch.setattr(fake_llm, "complete", lambda prompt, **kw: "[0, 2, 3, 1, 0]")
    reranker = BatchedLLMReranker(
        fake_llm, prompt_template="{query}\n{numbered_passages}", max_passages_per_batch=12
    )
    results = _make_results(5)
    ranked = reranker.rerank("query", results, threshold=2, top_k=None)
    scores = [int(r.rerank_score) for r in ranked]
    assert all(s >= 2 for s in scores), f"Scores below threshold slipped through: {scores}"
    # Expect exactly the chunks that got scores [2, 3]
    assert set(scores) <= {2, 3}


def test_scores_set_on_results(fake_llm: FakeLLM, monkeypatch):
    """rerank_score should be set on every input result."""
    monkeypatch.setattr(fake_llm, "complete", lambda prompt, **kw: "[1, 2, 3, 0, 1]")
    reranker = BatchedLLMReranker(
        fake_llm, prompt_template="{query}\n{numbered_passages}"
    )
    results = _make_results(5)
    reranker.rerank("query", results, threshold=0)
    for r in results:
        assert r.rerank_score is not None
        assert 0 <= r.rerank_score <= 3


def test_sorted_descending_by_score(fake_llm: FakeLLM, monkeypatch):
    """Returned results must be sorted by rerank_score descending."""
    monkeypatch.setattr(fake_llm, "complete", lambda prompt, **kw: "[0, 2, 3, 1, 0]")
    reranker = BatchedLLMReranker(
        fake_llm, prompt_template="{query}\n{numbered_passages}"
    )
    results = _make_results(5)
    ranked = reranker.rerank("query", results, threshold=0)
    scores = [r.rerank_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

def test_markdown_fences_are_stripped(fake_llm: FakeLLM, monkeypatch):
    """Response wrapped in ```json ... ``` should be parsed correctly."""
    monkeypatch.setattr(
        fake_llm, "complete", lambda prompt, **kw: "```json\n[2, 0, 3]\n```"
    )
    reranker = BatchedLLMReranker(
        fake_llm, prompt_template="{query}\n{numbered_passages}"
    )
    results = _make_results(3)
    ranked = reranker.rerank("query", results, threshold=0)
    scores = [int(r.rerank_score) for r in ranked]
    assert sorted(scores, reverse=True) == scores
    assert set(scores) == {0, 2, 3}


# ---------------------------------------------------------------------------
# Malformed JSON fallback
# ---------------------------------------------------------------------------

def test_malformed_json_falls_back_to_zero_scores(fake_llm: FakeLLM, monkeypatch):
    """Malformed JSON → all scores 0 → everything filtered by threshold=1."""
    monkeypatch.setattr(fake_llm, "complete", lambda prompt, **kw: "not json")
    reranker = BatchedLLMReranker(
        fake_llm, prompt_template="{query}\n{numbered_passages}"
    )
    results = _make_results(4)
    ranked = reranker.rerank("query", results, threshold=1)
    assert ranked == [], "Malformed JSON should fall back to zero scores, all filtered"


def test_malformed_json_threshold_zero_keeps_all(fake_llm: FakeLLM, monkeypatch):
    """With threshold=0, zero-score fallback results are all kept."""
    monkeypatch.setattr(fake_llm, "complete", lambda prompt, **kw: "not json")
    reranker = BatchedLLMReranker(
        fake_llm, prompt_template="{query}\n{numbered_passages}"
    )
    results = _make_results(3)
    ranked = reranker.rerank("query", results, threshold=0)
    assert len(ranked) == 3
    for r in ranked:
        assert r.rerank_score == 0.0


# ---------------------------------------------------------------------------
# Partial-batch (length-matched) results
# ---------------------------------------------------------------------------

def test_partial_batch_3_results(fake_llm: FakeLLM, monkeypatch):
    """3-result batch with response '[1, 2, 3]' → scores set correctly."""
    monkeypatch.setattr(fake_llm, "complete", lambda prompt, **kw: "[1, 2, 3]")
    reranker = BatchedLLMReranker(
        fake_llm, prompt_template="{query}\n{numbered_passages}"
    )
    results = _make_results(3)
    ranked = reranker.rerank("query", results, threshold=0)
    scores = {int(r.rerank_score) for r in ranked}
    assert scores == {1, 2, 3}


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

def test_progress_callback_fires_once_per_chunk(fake_llm: FakeLLM, monkeypatch):
    """Progress callback must be called exactly once per input chunk."""
    monkeypatch.setattr(fake_llm, "complete", lambda prompt, **kw: "[2, 2, 2, 2, 2]")
    reranker = BatchedLLMReranker(
        fake_llm, prompt_template="{query}\n{numbered_passages}", max_passages_per_batch=12
    )
    results = _make_results(5)
    calls = []

    def cb(idx, total, score):
        calls.append((idx, total, score))

    reranker.rerank("query", results, threshold=0, progress=cb)
    assert len(calls) == 5, f"Expected 5 callback calls, got {len(calls)}"
    # idx should go 1..5 and total=5 throughout
    for i, (idx, total, _) in enumerate(calls, start=1):
        assert idx == i
        assert total == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty(fake_llm: FakeLLM):
    reranker = BatchedLLMReranker(fake_llm, prompt_template="{query}\n{numbered_passages}")
    assert reranker.rerank("query", [], threshold=0) == []


def test_top_k_truncation(fake_llm: FakeLLM, monkeypatch):
    """top_k limits the final list length."""
    monkeypatch.setattr(fake_llm, "complete", lambda prompt, **kw: "[3, 3, 3, 3, 3]")
    reranker = BatchedLLMReranker(
        fake_llm, prompt_template="{query}\n{numbered_passages}"
    )
    results = _make_results(5)
    ranked = reranker.rerank("query", results, threshold=0, top_k=2)
    assert len(ranked) == 2


def test_batching_splits_large_input(fake_llm: FakeLLM, monkeypatch):
    """With max_passages_per_batch=3 and 7 results, LLM called at least twice."""
    call_count = {"n": 0}

    def counting_complete(prompt, **kw):
        n = prompt.count("[") or 1  # rough heuristic; just count the call
        batch_size = prompt.count("\n\n") + 1  # passages separated by blank lines
        # Return a valid array matching the batch size
        call_count["n"] += 1
        # We don't know the exact batch size from prompt alone; return 3 items
        # Use a safe fallback: count [N] passage markers
        import re
        n_passages = len(re.findall(r"\[\d+\]", prompt))
        return "[" + ", ".join(["2"] * max(n_passages, 1)) + "]"

    monkeypatch.setattr(fake_llm, "complete", counting_complete)
    reranker = BatchedLLMReranker(
        fake_llm,
        prompt_template="{query}\n\n{numbered_passages}",
        max_passages_per_batch=3,
    )
    results = _make_results(7)
    ranked = reranker.rerank("query", results, threshold=0)
    assert call_count["n"] >= 2, "Expected at least 2 LLM calls for 7 results in batches of 3"
    assert len(ranked) == 7
