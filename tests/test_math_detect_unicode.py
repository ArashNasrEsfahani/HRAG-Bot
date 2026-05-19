"""Tests for the Unicode-glyph math detector.

PyMuPDF strips LaTeX delimiters from PDF text, so PDF-derived chunks reach the
chunker as Unicode glyphs (Θ, ∑, 𝑌). These tests pin the heuristic that flips
``has_math`` based on independent Unicode signal categories.
"""

from __future__ import annotations

import pytest

from hrag.ingest.math_detect import has_math, has_unicode_math


def test_pure_prose_is_not_math() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    assert has_math(text) is False
    assert has_unicode_math(text) is False


def test_single_greek_letter_is_not_math() -> None:
    # One signal (Greek letter only) is below the default min_signals=2 floor.
    text = "the alpha α test"
    assert has_unicode_math(text) is False
    assert has_math(text) is False


def test_hipporag_chunk_text_is_math() -> None:
    # Real-world PDF-derived sample using Unicode mathematical italics +
    # Greek capital theta + function-of-variable form.
    text = (
        "The generation process of a LLM Θ(·) can be succinctly "
        "represented as \U0001D44C= Θ(\U0001D45E| \U0001D703)"
    )
    assert has_unicode_math(text) is True
    assert has_math(text) is True


def test_equals_heavy_kv_lines_are_not_math() -> None:
    # Many '=' signs but no Greek, no math symbols, no sub/super — should NOT
    # flip the tag. Equation density alone is one signal.
    text = "a=1 b=2 c=3 d=4"
    assert has_unicode_math(text) is False
    assert has_math(text) is False


def test_loss_formula_with_sigma_and_superscript_is_math() -> None:
    text = "loss = ∑ x_i² / N"  # "loss = ∑ x_i² / N"
    assert has_unicode_math(text) is True
    assert has_math(text) is True


def test_min_signals_three_filters_borderline() -> None:
    # HippoRAG sample has exactly two signals (greek+mathalpha, func-of-var).
    # Raising min_signals to 3 should drop it.
    text = (
        "The generation process of a LLM Θ(·) can be succinctly "
        "represented as \U0001D44C= Θ(\U0001D45E| \U0001D703)"
    )
    assert has_unicode_math(text, min_signals=2) is True
    assert has_unicode_math(text, min_signals=3) is False


def test_chunker_tags_unicode_math_chunk() -> None:
    """Integration: a synthetic doc with Unicode math gets at least one
    chunk flagged with ``has_math=True`` via the standard chunker path."""
    pytest.importorskip("tiktoken")
    from hrag.config import ChunkingConfig
    from hrag.ingest.chunker import chunk_document
    from hrag.types import Document

    text = (
        "Introduction\n\n"
        "This paper considers a generative model.\n\n"
        "Method\n\n"
        "The generation process of a LLM Θ(·) can be succinctly "
        "represented as \U0001D44C= Θ(\U0001D45E| \U0001D703), where "
        "\U0001D703 are the parameters and \U0001D45E is the query.\n\n"
        "We optimise the loss ∑_i x_i² over the training set.\n"
    )
    doc = Document(
        doc_id="utest",
        user_id="tester",
        source_path="/fake/unicode_math.md",
        title="UnicodeMath",
        text=text,
    )
    cfg = ChunkingConfig(max_tokens=400, overlap_tokens=60, metadata_fusion=False)
    chunks = chunk_document(doc, cfg)
    math_flagged = [c for c in chunks if c.metadata.get("has_math") is True]
    assert math_flagged, (
        "Expected at least one chunk tagged has_math=True from Unicode math content; "
        f"got {[c.metadata for c in chunks]}"
    )
