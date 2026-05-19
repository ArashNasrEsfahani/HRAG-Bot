"""Tests for hrag.ingest.quality — chunk quality filter."""

from __future__ import annotations


from hrag.ingest.quality import (
    QualityConfig,
    dedupe_chunks,
    filter_chunks,
    is_low_quality_chunk,
)
from hrag.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOC_ID = "testdoc01"
_USER_ID = "tester"


def _make_chunk(
    text: str,
    section: str = "",
    chunk_index: int = 0,
    token_count: int = 0,
) -> Chunk:
    """Build a minimal Chunk for testing. token_count=0 means auto-counted."""
    return Chunk(
        chunk_id=f"{_DOC_ID}:{chunk_index:04d}",
        doc_id=_DOC_ID,
        user_id=_USER_ID,
        text=text,
        embedding_text=text,
        section=section,
        chunk_index=chunk_index,
        token_count=token_count,
    )


def _default_cfg(**kwargs) -> QualityConfig:
    defaults: dict = {}
    defaults.update(kwargs)
    return QualityConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Too-short chunk dropped
# ---------------------------------------------------------------------------

def test_too_short_chunk_dropped() -> None:
    """A chunk with very few tokens AND very few chars must be dropped."""
    chunk = _make_chunk("Hi.", token_count=2)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected low-quality, got reason={reason!r}"
    assert "too_short" in reason


def test_too_short_requires_both_conditions() -> None:
    """If only one threshold is breached the chunk should NOT be dropped."""
    # Many chars but very few tokens (simulate code/equation with long identifiers)
    long_text = "x" * 200   # 200 chars but chunk.token_count set low
    chunk = _make_chunk(long_text, token_count=5)
    # chars >= min_chars (200 >= 80) so NOT both conditions met → keep
    low, _ = is_low_quality_chunk(chunk, _default_cfg(min_tokens=30, min_chars=80))
    assert not low, "Should be kept because char count is above threshold"


def test_chunk_above_both_thresholds_kept() -> None:
    """Normal-length chunk must pass the too-short filter."""
    text = "This is a reasonably long sentence that should pass. " * 3
    chunk = _make_chunk(text, token_count=40)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert not low, f"Expected kept, got dropped with reason={reason!r}"


# ---------------------------------------------------------------------------
# 2. Mostly-numeric / low-alpha chunk dropped
# ---------------------------------------------------------------------------

def test_mostly_numeric_chunk_dropped() -> None:
    """A chunk that is almost all digits and punctuation must be dropped."""
    # Table row: numbers separated by pipes
    text = "| 1.23 | 4.56 | 7.89 | 10.11 | 0.01 | 99.9 | 1234 | 5678 |"
    chunk = _make_chunk(text, token_count=50)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected low-quality (low alpha ratio), got reason={reason!r}"
    assert "alpha" in reason


def test_normal_text_alpha_passes() -> None:
    """A chunk with mostly alphabetic text must pass the alpha ratio filter."""
    text = "The results showed significant improvement in all metrics tested."
    chunk = _make_chunk(text, token_count=50)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert not low, f"Unexpectedly dropped: {reason!r}"


# ---------------------------------------------------------------------------
# 3. Bibliography / references section title dropped
# ---------------------------------------------------------------------------

def test_references_section_title_dropped() -> None:
    """Chunk whose section is 'References' must be dropped."""
    text = "Smith, J. (2021). A study. Journal of Things, 10(2), 45-67."
    chunk = _make_chunk(text, section="References", token_count=35)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected references section drop, got reason={reason!r}"
    assert "references_section" in reason


def test_bibliography_section_title_dropped() -> None:
    """Chunk whose section is 'Bibliography' must be dropped."""
    text = "Author, A. (2020). Paper title. Conf., 1-10."
    chunk = _make_chunk(text, section="Bibliography", token_count=30)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected bibliography section drop, got reason={reason!r}"


def test_works_cited_section_title_dropped() -> None:
    """Chunk whose section is 'Works Cited' must be dropped."""
    text = "Doe, J. (1999). Another paper. Journal, 5(1)."
    chunk = _make_chunk(text, section="Works Cited", token_count=30)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected works_cited drop, got reason={reason!r}"


def test_references_section_case_insensitive() -> None:
    """Section-title matching must be case-insensitive."""
    text = "Author A. (2022). Study. Nature, 12, 3-7."
    chunk = _make_chunk(text, section="REFERENCES", token_count=32)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low


def test_references_section_disabled() -> None:
    """Setting drop_references_sections=False must keep the chunk."""
    text = "Author A. (2022). Study. Nature, 12, 3-7."
    chunk = _make_chunk(text, section="References", token_count=32)
    low, _ = is_low_quality_chunk(chunk, _default_cfg(drop_references_sections=False))
    # May still fail other filters; we only assert the section filter is off
    # by using a chunk that would otherwise only fail on section title
    cfg = _default_cfg(
        drop_references_sections=False,
        drop_bibliography_chunks=False,
        min_tokens=10,
        min_chars=20,
    )
    low, reason = is_low_quality_chunk(chunk, cfg)
    assert not low, f"Should be kept when section filter disabled, got {reason!r}"


# ---------------------------------------------------------------------------
# 4. Bibliography-pattern detection (year mentions heuristic)
# ---------------------------------------------------------------------------

def test_bib_pattern_detected_multiple_years() -> None:
    """A chunk with >= 3 year-mentions and <= 3 complete sentences → bib."""
    text = (
        "Smith, J. (2019). Paper one. Nature.\n"
        "Jones, A. (2020). Paper two. Science.\n"
        "Brown, C. (2021). Paper three. Cell.\n"
    )
    chunk = _make_chunk(text, section="", token_count=50)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected bib chunk drop, got reason={reason!r}"
    assert "bibliography_chunk" in reason


def test_bib_pattern_not_triggered_for_narrative_prose() -> None:
    """Chunk with years inline in prose (single paragraph) → narrative, not bibliography.

    The bib heuristic requires >= 3 *lines* that each contain a year reference.
    A prose paragraph with multiple years on one line does not qualify.
    """
    text = (
        "In 1990, researchers began studying neural networks. "
        "By 2000, the field had expanded significantly. "
        "Research in 2010 showed further advances. "
        "Subsequently, work in 2015 introduced deep learning at scale. "
        "The culmination in 2020 demonstrated state-of-the-art results."
    )
    chunk = _make_chunk(text, token_count=80)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    # All years are on a single prose line → year_line_count=1 < 3 → kept
    assert not low, f"Narrative text with inline years should be kept, got {reason!r}"


def test_bib_pattern_disabled() -> None:
    """Setting drop_bibliography_chunks=False keeps bib-like chunks."""
    text = (
        "Smith, J. (2019). Paper one.\n"
        "Jones, A. (2020). Paper two.\n"
        "Brown, C. (2021). Paper three.\n"
    )
    chunk = _make_chunk(text, section="", token_count=50)
    cfg = _default_cfg(
        drop_bibliography_chunks=False,
        drop_references_sections=False,
        min_tokens=10,
        min_chars=20,
    )
    low, reason = is_low_quality_chunk(chunk, cfg)
    assert not low, f"Bib filter disabled; should be kept, got {reason!r}"


# ---------------------------------------------------------------------------
# 5. Page artifacts dropped
# ---------------------------------------------------------------------------

def test_page_artifact_plain_number() -> None:
    """A chunk that is just a page number must be dropped."""
    chunk = _make_chunk("42", token_count=2)
    # Token/char filters will also fire here; test the artifact filter too
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low


def test_page_artifact_page_x_of_y() -> None:
    """'Page 12 of 50' style artifact must be dropped."""
    chunk = _make_chunk("Page 12 of 50", token_count=4)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected page artifact drop, got reason={reason!r}"


def test_page_artifact_just_number_x_of_y() -> None:
    """'3 of 25' — page-like artifact without the word 'Page'."""
    chunk = _make_chunk("3 of 25", token_count=3)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low


def test_page_artifact_disabled() -> None:
    """Setting drop_page_artifacts=False should not fire the artifact filter."""
    # Use a text that ONLY the artifact filter would catch (not too_short etc.)
    chunk = _make_chunk("Page 12 of 50", token_count=4)
    cfg = _default_cfg(
        drop_page_artifacts=False,
        min_tokens=2,   # allow very short
        min_chars=5,
    )
    low, reason = is_low_quality_chunk(chunk, cfg)
    # alpha ratio on "Page 12 of 50" — 'P','a','g','e','o','f' = 6 alpha out of 12 non-space
    # May or may not pass alpha; the key thing is NOT reason=page_artifact
    if low:
        assert "page_artifact" not in reason


# ---------------------------------------------------------------------------
# 6. Deduplication drops second copy
# ---------------------------------------------------------------------------

def test_dedupe_drops_duplicate() -> None:
    """Two identical chunks — second must be removed."""
    text = "This is some important content about retrieval methods."
    chunk1 = _make_chunk(text, chunk_index=0, token_count=40)
    chunk2 = _make_chunk(text, chunk_index=1, token_count=40)
    deduped = dedupe_chunks([chunk1, chunk2])
    assert len(deduped) == 1
    assert deduped[0].chunk_index == 0  # first one kept


def test_dedupe_keeps_distinct_chunks() -> None:
    """Distinct chunks must all survive deduplication."""
    c1 = _make_chunk("First chunk text.", chunk_index=0, token_count=40)
    c2 = _make_chunk("Second chunk text.", chunk_index=1, token_count=40)
    c3 = _make_chunk("Third chunk text.", chunk_index=2, token_count=40)
    deduped = dedupe_chunks([c1, c2, c3])
    assert len(deduped) == 3


def test_filter_chunks_dedupe_via_filter() -> None:
    """filter_chunks with dedupe=True must report the duplicate as dropped."""
    text = "Shared content about machine learning and neural networks."
    chunk1 = _make_chunk(text, chunk_index=0, token_count=40)
    chunk2 = _make_chunk(text, chunk_index=1, token_count=40)
    kept, dropped = filter_chunks([chunk1, chunk2], _default_cfg(dedupe=True))
    assert len(kept) == 1
    drop_reasons = [r for _, r in dropped]
    assert any("duplicate" in r for r in drop_reasons)


def test_dedupe_disabled() -> None:
    """With dedupe=False, duplicate chunks must both survive filter_chunks."""
    text = "Shared content about machine learning and neural networks."
    chunk1 = _make_chunk(text, chunk_index=0, token_count=40)
    chunk2 = _make_chunk(text, chunk_index=1, token_count=40)
    kept, dropped = filter_chunks([chunk1, chunk2], _default_cfg(dedupe=False))
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# 7. Normal narrative chunk passes through
# ---------------------------------------------------------------------------

def test_normal_narrative_chunk_kept() -> None:
    """A well-formed narrative paragraph must survive all filters."""
    text = (
        "The proposed hierarchical retrieval approach leverages both dense "
        "and sparse representations to improve recall on domain-specific queries. "
        "Our experiments on three benchmark datasets demonstrate consistent "
        "improvements over the baseline methods, with statistically significant "
        "gains on the precision metric. We attribute the gains to the multi-level "
        "indexing scheme described in Section 3."
    )
    chunk = _make_chunk(text, section="Results", token_count=90)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert not low, f"Good narrative chunk was dropped with reason: {reason!r}"


def test_normal_chunk_passes_filter_chunks() -> None:
    """filter_chunks on a list of good chunks returns all kept."""
    texts = [
        (
            "Hierarchical retrieval combines different granularities "
            "of document representation to improve downstream question answering "
            "performance on diverse document collections."
        ),
        (
            "The encoder is pre-trained on a large multilingual corpus using "
            "contrastive learning objectives. Fine-tuning on domain data yields "
            "further improvements in retrieval quality and semantic coverage."
        ),
    ]
    chunks = [_make_chunk(t, token_count=60, chunk_index=i) for i, t in enumerate(texts)]
    kept, dropped = filter_chunks(chunks, _default_cfg())
    assert len(kept) == 2
    assert len(dropped) == 0


# ---------------------------------------------------------------------------
# 8. enabled=False keeps everything
# ---------------------------------------------------------------------------

def test_enabled_false_keeps_everything() -> None:
    """With enabled=False, filter_chunks must return all chunks unchanged."""
    junk_chunks = [
        _make_chunk("Hi.", token_count=1),            # too short
        _make_chunk("42", token_count=1),              # page artifact
        _make_chunk(
            "| 1.2 | 3.4 | 5.6 |", token_count=5
        ),  # low alpha
        _make_chunk(
            "Smith J. (2021). Nature.\nDoe A. (2019). Science.\nLee B. (2020). Cell.\n",
            section="References",
            token_count=35,
        ),
    ]
    cfg = _default_cfg(enabled=False)
    kept, dropped = filter_chunks(junk_chunks, cfg)
    assert len(kept) == len(junk_chunks)
    assert len(dropped) == 0


# ---------------------------------------------------------------------------
# 9. filter_chunks breakdown accounting
# ---------------------------------------------------------------------------

def test_filter_chunks_returns_correct_counts() -> None:
    """filter_chunks must correctly account all kept + dropped chunks."""
    good = _make_chunk(
        "The model achieves state-of-the-art results on all benchmarks tested.",
        token_count=50,
        chunk_index=0,
    )
    bad_short = _make_chunk("Hi.", token_count=1, chunk_index=1)
    bad_ref = _make_chunk(
        "Smith J. (2019). Nature.", section="References", token_count=32, chunk_index=2
    )
    chunks = [good, bad_short, bad_ref]
    kept, dropped = filter_chunks(chunks, _default_cfg())
    assert len(kept) + len(dropped) == len(chunks)
    assert good in kept


# ---------------------------------------------------------------------------
# 10. Leading-page-artifact: page number glued onto a tiny remainder
# ---------------------------------------------------------------------------

def test_leading_page_number_with_short_remainder_dropped() -> None:
    """Chunks like '250\\nshort note.' — page number + tiny content — dropped.

    The chunk's `token_count` is set high enough to bypass `_check_too_short`,
    so the only thing that can drop it is the leading-page-artifact filter.
    """
    text = "250\nshort note."
    chunk = _make_chunk(text, token_count=50)  # bypass too-short
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected leading page artifact drop, got reason={reason!r}"
    assert "leading_page_artifact" in reason


def test_leading_page_number_with_substantive_remainder_kept() -> None:
    """A chunk that starts with a page number but has real content after must be kept."""
    body = (
        "We use a single thread to query the OpenAI API for online retrieval. "
        "Since IRCoT is an iterative method, the per-question cost is several "
        "times that of a single retrieval call. We therefore amortise across "
        "questions and report the average wall time for a representative sample."
    )
    text = "250\n" + body
    chunk = _make_chunk(text, token_count=80)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert not low, f"Substantive remainder should keep the chunk, got {reason!r}"


# ---------------------------------------------------------------------------
# 11. Single-citation chunk dropped
# ---------------------------------------------------------------------------

def test_single_citation_chunk_dropped() -> None:
    """A chunk consisting of one bibliography entry with [N], year, URL → dropped."""
    text = (
        "[13] G. Csardi and T. Nepusz. The igraph software package for "
        "complex network research. 2006. URL https://igraph.org/."
    )
    chunk = _make_chunk(text, token_count=40)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected single_citation drop, got reason={reason!r}"
    assert "single_citation" in reason


def test_single_citation_with_doi_dropped() -> None:
    """[N] author year doi:... is the canonical bibliography pattern."""
    text = (
        "[55] OpenAI. GPT-3.5 Turbo, 2024. Association for Computational "
        "Linguistics. doi:10.18653/v1/2022.naacl-main.341."
    )
    chunk = _make_chunk(text, token_count=40)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    assert low, f"Expected single_citation drop, got reason={reason!r}"


def test_single_citation_without_locator_kept() -> None:
    """[N] author year without DOI/URL/arxiv — could be regular prose mentioning a citation."""
    text = (
        "As shown in [13], the approach achieves strong performance. We extend "
        "this work in 2024 by introducing a new training objective that further "
        "improves results on the multi-hop QA benchmark by a substantial margin."
    )
    chunk = _make_chunk(text, token_count=80)
    low, reason = is_low_quality_chunk(chunk, _default_cfg())
    # No URL, no DOI, no arxiv → must NOT trigger single_citation
    assert "single_citation" not in (reason or "")


def test_single_citation_disabled() -> None:
    """drop_bibliography_chunks=False also disables _check_single_citation."""
    text = "[1] A. Author. Title. 2020. URL https://example.com/paper.pdf."
    chunk = _make_chunk(text, token_count=40)
    cfg = _default_cfg(
        drop_bibliography_chunks=False,
        drop_references_sections=False,
        min_tokens=10,
        min_chars=20,
        min_alpha_ratio=0.0,
    )
    low, reason = is_low_quality_chunk(chunk, cfg)
    assert "single_citation" not in (reason or "")
