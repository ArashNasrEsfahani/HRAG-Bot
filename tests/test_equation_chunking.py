"""Tests for equation-aware chunking and the quality-filter math carve-out.

Covers:
  * has_math tagging for display math, LaTeX environments, and inline math
  * Equations are not split across chunk boundaries
  * Pure-equation chunks survive the quality filter (alpha-ratio / too-short)
  * Plain-prose chunks are NOT flagged has_math
"""

from __future__ import annotations

import pytest

tiktoken = pytest.importorskip("tiktoken")

from hrag.config import ChunkingConfig, QualityConfig  # noqa: E402
from hrag.ingest.chunker import chunk_document  # noqa: E402
from hrag.ingest.math_detect import find_display_math_spans, has_math  # noqa: E402
from hrag.ingest.quality import is_low_quality_chunk  # noqa: E402
from hrag.types import Chunk, Document  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str) -> Document:
    return Document(
        doc_id="mathdoc",
        user_id="tester",
        source_path="/fake/math.md",
        title="MathDoc",
        text=text,
    )


def _default_cfg(**kwargs) -> ChunkingConfig:
    defaults = dict(max_tokens=400, overlap_tokens=60, metadata_fusion=False)
    defaults.update(kwargs)
    return ChunkingConfig(**defaults)


def _make_chunk(text: str, *, has_math_flag: bool | None = None) -> Chunk:
    meta: dict = {}
    if has_math_flag is not None:
        meta["has_math"] = has_math_flag
    return Chunk(
        chunk_id="mathdoc:0000",
        doc_id="mathdoc",
        user_id="tester",
        text=text,
        embedding_text=text,
        token_count=0,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# 1. $$ ... $$ display math
# ---------------------------------------------------------------------------

def test_dollar_display_math_tagged_and_intact() -> None:
    text = (
        "Pythagoras tells us about right triangles.\n\n"
        "$$x^2 + y^2 = z^2$$\n\n"
        "This identity is foundational."
    )
    chunks = chunk_document(_make_doc(text), _default_cfg())
    # All chunks combined must still contain the equation.
    combined = "\n".join(c.text for c in chunks)
    assert "x^2 + y^2 = z^2" in combined
    # The chunk holding the equation has has_math == True.
    math_chunks = [c for c in chunks if "x^2 + y^2 = z^2" in c.text]
    assert math_chunks, "Expected a chunk containing the equation"
    assert all(c.metadata.get("has_math") is True for c in math_chunks)
    # The $$ delimiters survive — equation wasn't truncated.
    assert all("$$x^2 + y^2 = z^2$$" in c.text for c in math_chunks)


# ---------------------------------------------------------------------------
# 2. \begin{align} ... \end{align}
# ---------------------------------------------------------------------------

def test_latex_align_environment_tagged_and_intact() -> None:
    text = (
        "Consider the following system:\n\n"
        "\\begin{align}\n"
        "a + b &= c \\\\\n"
        "d - e &= f\n"
        "\\end{align}\n\n"
        "Both equations must hold simultaneously."
    )
    chunks = chunk_document(_make_doc(text), _default_cfg())
    math_chunks = [c for c in chunks if "\\begin{align}" in c.text]
    assert math_chunks, "Expected a chunk containing the align block"
    for c in math_chunks:
        assert c.metadata.get("has_math") is True
        # Both begin and end markers stay together.
        assert "\\end{align}" in c.text
        # And the inner equation lines too.
        assert "a + b" in c.text and "d - e" in c.text


# ---------------------------------------------------------------------------
# 3. Inline $ E = mc^2 $ mixed into prose
# ---------------------------------------------------------------------------

def test_inline_math_flag_set_but_no_split_disruption() -> None:
    text = (
        "Einstein's famous equation $E = mc^2$ relates mass and energy. "
        "It appears in special relativity and many textbooks."
    )
    chunks = chunk_document(_make_doc(text), _default_cfg())
    # Should produce a single chunk for this short input.
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata.get("has_math") is True
    assert "$E = mc^2$" in chunk.text


# ---------------------------------------------------------------------------
# 4. Pure-equation chunk survives the quality filter
# ---------------------------------------------------------------------------

def test_pure_equation_chunk_survives_quality_filter() -> None:
    pure_eq = "$$\\sum_{i=0}^{n} a_i$$"
    chunk = _make_chunk(pure_eq, has_math_flag=True)
    # Sanity: alpha ratio is well below 0.4
    low, reason = is_low_quality_chunk(chunk, QualityConfig())
    assert not low, f"Math chunk wrongly dropped: {reason!r}"


def test_pure_equation_without_flag_is_still_dropped() -> None:
    # Same text but no has_math flag — confirms the carve-out is what saved it
    # above, not some other lenient default.
    pure_eq = "$$\\sum_{i=0}^{n} a_i$$"
    chunk = _make_chunk(pure_eq, has_math_flag=False)
    low, _ = is_low_quality_chunk(chunk, QualityConfig())
    assert low, "Without has_math tag, low-alpha math should be dropped"


# ---------------------------------------------------------------------------
# 5. Plain prose chunk has has_math == False
# ---------------------------------------------------------------------------

def test_plain_prose_chunk_has_no_math_flag() -> None:
    text = (
        "This is ordinary prose. It contains no formulas, no LaTeX, "
        "no dollar signs of any kind. Just words and punctuation."
    )
    chunks = chunk_document(_make_doc(text), _default_cfg())
    assert len(chunks) == 1
    assert chunks[0].metadata.get("has_math") is False


# ---------------------------------------------------------------------------
# 6. Unit tests on math_detect helpers
# ---------------------------------------------------------------------------

def test_find_display_math_spans_basic() -> None:
    text = "intro\n\n$$a=b$$\n\nmiddle\n\n\\begin{equation}c=d\\end{equation}\n\nend"
    spans = find_display_math_spans(text)
    assert len(spans) == 2
    # Spans should be sorted by start.
    assert spans[0][0] < spans[1][0]
    # Spans should bound actual math text.
    for s, e in spans:
        assert "=" in text[s:e]


def test_has_math_inline_paren_form() -> None:
    assert has_math("Let \\(x\\) be a real number.") is True
    assert has_math("Plain prose with no math.") is False


def test_has_math_equation_star_environment() -> None:
    text = "\\begin{equation*}\n\\alpha + \\beta = \\gamma\n\\end{equation*}"
    assert has_math(text) is True
    assert len(find_display_math_spans(text)) == 1
