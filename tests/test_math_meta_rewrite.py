"""Tests for math-meta query expansion in the heuristic rewriter (Phase 7-A Method 2)."""

from __future__ import annotations

import pytest

from hrag.retrieval.query_rewriter import (
    HeuristicRewriter,
    _expand_math_meta,
    _MATH_EXPANSION_TOKENS,
)

# ---------------------------------------------------------------------------
# Unit tests for _expand_math_meta
# ---------------------------------------------------------------------------


def test_plain_query_unchanged():
    q = "what is personalized pagerank?"
    assert _expand_math_meta(q) == q


def test_formulas_and_math_expanded():
    q = "give me some formulas and math"
    result = _expand_math_meta(q)
    assert result.startswith(q)
    assert "equation" in result
    assert "gradient" in result


def test_equations_trigger_expansion():
    q = "what equations does it use?"
    result = _expand_math_meta(q)
    assert q in result
    assert "θ" in result


def test_formula_in_unrelated_context_triggers_expansion():
    # Limitation: "formula" is a broad trigger word that also matches
    # unrelated contexts like "formula one racing". This is accepted
    # behaviour — the expansion is harmless on non-math corpora because
    # the appended tokens won't match any passage.
    q = "tell me about formula one racing"
    result = _expand_math_meta(q)
    # The regex DOES match "formula" here — document this known limitation.
    assert result != q, (
        "Limitation: 'formula one racing' triggers math-meta expansion. "
        "This is accepted — the added tokens are harmless on non-math corpora."
    )


def test_case_insensitive_formulas():
    q = "FORMULAS used in the paper"
    result = _expand_math_meta(q)
    assert "equation" in result


def test_derivation_of_loss_function():
    q = "derivation of loss function in the model"
    result = _expand_math_meta(q)
    assert result.startswith(q)
    assert "∑" in result
    assert "∫" in result


# ---------------------------------------------------------------------------
# Integration test: HeuristicRewriter applies math-meta expansion
# ---------------------------------------------------------------------------


def test_heuristic_rewriter_math_meta_integration():
    rewriter = HeuristicRewriter()
    # Use an empty history so the follow-up path is not taken.
    query = "show me the math and equations"
    result = rewriter.rewrite(query, history=[])
    assert query in result, "Original query must be preserved"
    # Check at least three vocab tokens from the expansion are present.
    vocab_tokens = ["equation", "gradient", "θ"]
    hits = [tok for tok in vocab_tokens if tok in result]
    assert len(hits) >= 3, f"Expected ≥3 math vocab tokens, got: {result!r}"


def test_heuristic_rewriter_non_math_unchanged():
    rewriter = HeuristicRewriter()
    query = "explain the architecture of hipporag"
    result = rewriter.rewrite(query, history=[])
    # No math-meta expansion should be appended.
    assert _MATH_EXPANSION_TOKENS not in result
