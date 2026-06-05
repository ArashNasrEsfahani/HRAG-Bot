"""Tests for hrag.ingest.metadata — detect_sections(), page_for_offset(), normalize_heading()."""

from __future__ import annotations

from hrag.ingest.metadata import detect_sections, normalize_heading, page_for_offset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spans_tile(text: str, spans: list[tuple[int, int, str, str]]) -> bool:
    """Return True iff the spans cover [0, len(text)] with no gaps or overlaps."""
    if not spans:
        return len(text) == 0
    if spans[0][0] != 0:
        return False
    for i in range(len(spans) - 1):
        if spans[i][1] != spans[i + 1][0]:
            return False
    return spans[-1][1] == len(text)


# ---------------------------------------------------------------------------
# Tiling invariant
# ---------------------------------------------------------------------------

def test_spans_tile_markdown_doc() -> None:
    """Spans must be contiguous and cover the entire text."""
    text = "# Intro\n\nHello world.\n\n## Details\n\nMore text.\n"
    spans = detect_sections(text)
    assert _spans_tile(text, spans), f"Spans do not tile: {spans}"


def test_spans_tile_no_headings() -> None:
    text = "Just some plain text.\nNo headings here at all.\n"
    spans = detect_sections(text)
    assert _spans_tile(text, spans)


def test_spans_tile_empty_string() -> None:
    spans = detect_sections("")
    assert spans == [(0, 0, "", "")]


# ---------------------------------------------------------------------------
# Markdown heading detection
# ---------------------------------------------------------------------------

def test_markdown_h1_detected() -> None:
    text = "# Section One\n\nContent under section one.\n"
    spans = detect_sections(text)
    # There should be a span with section="Section One"
    section_titles = {s[2] for s in spans}
    assert "Section One" in section_titles


def test_markdown_h2_sets_subsection() -> None:
    text = "# Main\n\nIntro text.\n\n## Sub\n\nSubsection content.\n"
    spans = detect_sections(text)
    subsection_titles = {s[3] for s in spans}
    assert "Sub" in subsection_titles


def test_markdown_h1_resets_subsection() -> None:
    """After a second H1, the subsection should reset to empty."""
    text = (
        "# Chapter 1\n\n## Part A\n\nText A.\n\n"
        "# Chapter 2\n\nText B.\n"
    )
    spans = detect_sections(text)
    # The span for Chapter 2 content should have subsection == ""
    chap2_spans = [s for s in spans if s[2] == "Chapter 2"]
    assert chap2_spans, "Expected a span with section='Chapter 2'"
    for sp in chap2_spans:
        assert sp[3] == "", f"Expected empty subsection after H1, got {sp[3]!r}"


def test_markdown_multiple_sections_tiling() -> None:
    text = "# A\n\nText A.\n\n# B\n\nText B.\n\n# C\n\nText C.\n"
    spans = detect_sections(text)
    assert _spans_tile(text, spans)
    sections = [s[2] for s in spans if s[2]]
    assert "A" in sections
    assert "B" in sections
    assert "C" in sections


# ---------------------------------------------------------------------------
# Fallback — no headings
# ---------------------------------------------------------------------------

def test_no_headings_returns_single_full_span() -> None:
    text = "Plain paragraph one.\n\nPlain paragraph two.\n"
    spans = detect_sections(text)
    assert len(spans) == 1
    assert spans[0][0] == 0
    assert spans[0][1] == len(text)
    assert spans[0][2] == ""
    assert spans[0][3] == ""


# ---------------------------------------------------------------------------
# ALL-CAPS heading detection
# ---------------------------------------------------------------------------

def test_allcaps_heading_detected() -> None:
    text = "INTRODUCTION\n\nThis section covers the intro.\n\nMETHODS\n\nDetails here.\n"
    spans = detect_sections(text)
    section_titles = {s[2] for s in spans}
    assert "INTRODUCTION" in section_titles or "METHODS" in section_titles


def test_allcaps_heading_tiles_text() -> None:
    text = "ABSTRACT\n\nAbstract content.\n\nCONCLUSION\n\nConclusion content.\n"
    spans = detect_sections(text)
    assert _spans_tile(text, spans)


def test_allcaps_long_line_not_detected() -> None:
    """ALL-CAPS lines >= 80 chars should NOT be treated as headings."""
    long_line = "A" * 80 + "\n"
    text = long_line + "\nSome body text.\n"
    spans = detect_sections(text)
    # The long line should not be a heading — expect single span or unlabelled spans
    section_titles = {s[2] for s in spans if s[2]}
    assert "A" * 80 not in section_titles


# ---------------------------------------------------------------------------
# Numbered heading detection
# ---------------------------------------------------------------------------

def test_numbered_h1_detected() -> None:
    text = "1. Introduction\n\nThis is the intro.\n"
    spans = detect_sections(text)
    section_titles = {s[2] for s in spans}
    assert "Introduction" in section_titles


def test_numbered_h2_detected() -> None:
    text = "1. Introduction\n\nText.\n\n1.1 Background\n\nBackground text.\n"
    spans = detect_sections(text)
    subsection_titles = {s[3] for s in spans}
    assert "Background" in subsection_titles


def test_numbered_heading_tiles_text() -> None:
    text = "1. First\n\nFirst content.\n\n2. Second\n\nSecond content.\n"
    spans = detect_sections(text)
    assert _spans_tile(text, spans)


def test_numbered_deep_heading_is_subsection() -> None:
    """Headings like '2.1 Methods' (depth > 1) map to level=2 (subsection)."""
    text = "1. Top Level\n\nTop text.\n\n2.1 Sub Level\n\nSub text.\n"
    spans = detect_sections(text)
    # 2.1 should produce a subsection entry, not a top-level section reset
    subsection_titles = {s[3] for s in spans if s[3]}
    assert subsection_titles, "Expected at least one subsection from '2.1 Sub Level'"


# ---------------------------------------------------------------------------
# Span content correctness
# ---------------------------------------------------------------------------

def test_span_text_reconstruction() -> None:
    """Concatenating all span slices must reproduce the original text."""
    text = "# Alpha\n\nAlpha content.\n\n## Beta\n\nBeta content.\n"
    spans = detect_sections(text)
    reconstructed = "".join(text[s[0]:s[1]] for s in spans)
    assert reconstructed == text


# ---------------------------------------------------------------------------
# find_references_offset — boundary detection for academic papers
# ---------------------------------------------------------------------------


def test_find_references_offset_markdown_heading() -> None:
    from hrag.ingest.metadata import find_references_offset

    text = (
        "# Introduction\n\nIntro body text.\n\n"
        "# Methods\n\nMethod body.\n\n"
        "# References\n\n[1] Smith J. (2019). Paper.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    assert text[off:].lstrip().startswith("# References")


def test_find_references_offset_allcaps_bibliography() -> None:
    from hrag.ingest.metadata import find_references_offset

    text = (
        "1. Introduction\n\nText here.\n\n"
        "BIBLIOGRAPHY\n\nSmith, A. (2020). Paper title. Journal.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    # Everything before the offset should NOT contain the keyword as a heading
    head = text[:off]
    assert "BIBLIOGRAPHY" not in head


def test_find_references_offset_works_cited() -> None:
    from hrag.ingest.metadata import find_references_offset

    text = "## Discussion\n\nFindings.\n\n## Works Cited\n\n[1] Author. 2021.\n"
    off = find_references_offset(text)
    assert off is not None


def test_find_references_offset_none_when_absent() -> None:
    from hrag.ingest.metadata import find_references_offset

    text = "# Introduction\n\nIntro.\n\n# Methods\n\nMethods.\n\n# Conclusion\n\nDone.\n"
    assert find_references_offset(text) is None


def test_find_references_offset_first_match_wins() -> None:
    """If 'References' appears multiple times, the FIRST one is the cut point."""
    from hrag.ingest.metadata import find_references_offset

    text = (
        "# Methods\n\nWe cite References below.\n\n"  # the word appears in body — must NOT trigger
        "# References\n\n[1] Smith. 2020.\n\n"
        "# References (continued)\n\n[2] Jones. 2021.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    # The first heading should be the one the function picks.
    assert text[off:].lstrip().startswith("# References")


def test_find_references_offset_empty_text() -> None:
    from hrag.ingest.metadata import find_references_offset

    assert find_references_offset("") is None


# ---------------------------------------------------------------------------
# find_references_offset — strengthened detection (numbered / title-case /
# all-caps / spaced caps / citation-run fallback)
# ---------------------------------------------------------------------------


def test_plain_title_case_references_heading() -> None:
    """A standalone 'References' line (no markdown, no numbering) is caught."""
    from hrag.ingest.metadata import find_references_offset

    text = (
        "Some closing paragraph about limitations.\n\n"
        "References\n"
        "[1] Smith, J. (2020). A paper title. Journal.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    assert text[off:].lstrip().startswith("References")


def test_numbered_references_heading() -> None:
    """A heading like '## 7 References' (numbered prefix) is caught."""
    from hrag.ingest.metadata import find_references_offset

    text = (
        "## 6 Conclusion\n\nWrap-up.\n\n"
        "## 7 References\n\n[1] Smith, J. (2020). foo.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    # The match should land on the heading line, before any citation.
    assert "[1] Smith" not in text[:off]


def test_dotted_numbered_references_heading() -> None:
    from hrag.ingest.metadata import find_references_offset

    text = "5.0 References\n\n[1] Smith, J. (2019). foo.\n"
    off = find_references_offset(text)
    assert off is not None
    assert off == 0


def test_caps_references_heading() -> None:
    from hrag.ingest.metadata import find_references_offset

    text = (
        "Final discussion of the work.\n\n"
        "REFERENCES\n\n"
        "[1] Author, A. (2024). Foo.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    head = text[:off]
    assert "REFERENCES" not in head


def test_spaced_caps_references_heading() -> None:
    """'R E F E R E N C E S' (some PDFs render headings this way)."""
    from hrag.ingest.metadata import find_references_offset

    text = (
        "Final paragraph.\n\n"
        "R E F E R E N C E S\n\n"
        "[1] Smith, J. (2020). Foo.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    head = text[:off]
    assert "R E F E R E N C E S" not in head


def test_section_colon_references_heading() -> None:
    from hrag.ingest.metadata import find_references_offset

    text = (
        "Conclusion text.\n\n"
        "Section 7: References\n\n"
        "[1] Smith, J. (2020). foo.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    head = text[:off]
    assert "Section 7: References" not in head


def test_citations_keyword_heading() -> None:
    from hrag.ingest.metadata import find_references_offset

    text = (
        "Results discussion.\n\n"
        "Citations\n\n"
        "Smith, J. (2020). Foo. Journal.\n"
    )
    off = find_references_offset(text)
    assert off is not None


def test_no_heading_falls_back_to_citation_run() -> None:
    """When there's NO 'References' heading but a long run of citation-shaped
    lines, return the offset of the first citation in the run.
    """
    from hrag.ingest.metadata import find_references_offset

    text = (
        "Concluding remarks about the experiments and limitations.\n\n"
        "Smith, J. (2020). A study of foo. Journal of Foo, 12(3).\n"
        "Jones, A. (2021). Bar systems explored. Conf on Bar.\n"
        "Lee, K. (2019). A novel approach. arxiv:1901.01234.\n"
        "Brown, T. (2022). Things and stuff. doi:10.1234/abcd.\n"
        "Doe, J. (2018). Yet another paper. Journal of Bar, 5(2).\n"
    )
    off = find_references_offset(text)
    assert off is not None
    # The offset should land at the start of the first citation line.
    assert text[off:].lstrip().startswith("Smith, J.")


def test_no_heading_no_citation_run_returns_none() -> None:
    """Plain document text with no bibliography should return None."""
    from hrag.ingest.metadata import find_references_offset

    text = (
        "# Introduction\n\nSome intro text.\n\n"
        "# Methods\n\nWe did stuff.\n\n"
        "# Conclusion\n\nWe concluded things.\n"
    )
    assert find_references_offset(text) is None


def test_existing_paper_pattern_still_works() -> None:
    """The original markdown '## References' behaviour must not regress."""
    from hrag.ingest.metadata import find_references_offset

    text = (
        "# Introduction\n\nIntro.\n\n"
        "# Methods\n\nMethods.\n\n"
        "## References\n\n[1] Smith. 2020.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    # Heading-prefix '## References' should appear at-or-after the offset.
    assert "## References" in text[off:]
    assert "## References" not in text[:off]


def test_short_citation_run_does_not_trigger() -> None:
    """A handful (< min_run) of citation-shaped lines is not enough to trigger
    the fallback — those may be inline references inside body text.
    """
    from hrag.ingest.metadata import find_references_offset

    text = (
        "We build on Smith, J. (2020) and Jones, A. (2021), among others.\n"
        "The approach is similar to Lee, K. (2019). However, we differ.\n\n"
        "More body text here without bibliographic patterns.\n"
    )
    # Only 2 citation-shaped lines — below the 5-line floor.
    assert find_references_offset(text) is None


def test_legacy_and_regex_pick_earliest() -> None:
    """When both detectors fire, the earliest offset wins."""
    from hrag.ingest.metadata import find_references_offset

    text = (
        "## Results\n\nFindings.\n\n"
        "References\n\n"  # plain title-case (regex layer)
        "[1] Smith. 2020.\n\n"
        "## References\n\n"  # markdown (legacy layer) — later
        "[2] Jones. 2021.\n"
    )
    off = find_references_offset(text)
    assert off is not None
    # Should match the FIRST 'References' (plain title case), not the markdown one.
    assert text[off:].startswith("References\n")


# ---------------------------------------------------------------------------
# page_for_offset — Phase 13.1
# ---------------------------------------------------------------------------

def _spans() -> list[tuple[int, int, int]]:
    """Build two synthetic page spans:
      Page 1: chars [0, 10)
      Page 2: chars [12, 22)   (gap [10,12) simulates the \\n\\n separator)
    """
    return [(0, 10, 1), (12, 22, 2)]


def test_page_for_offset_inside_page1() -> None:
    spans = _spans()
    assert page_for_offset(5, spans) == 1


def test_page_for_offset_inside_page2() -> None:
    spans = _spans()
    assert page_for_offset(15, spans) == 2


def test_page_for_offset_in_separator_gap() -> None:
    """Offset in the separator gap (between pages) should return the preceding page."""
    spans = _spans()
    # offset 11 is between end-of-page-1 (10) and start-of-page-2 (12)
    assert page_for_offset(11, spans) == 1


def test_page_for_offset_past_end_returns_last() -> None:
    spans = _spans()
    # offset 999 is beyond the last span
    assert page_for_offset(999, spans) == 2


def test_page_for_offset_empty_spans_returns_none() -> None:
    assert page_for_offset(0, []) is None


def test_page_for_offset_at_boundary() -> None:
    """Offset exactly at end of a span is NOT in that span (half-open); falls
    into the gap or the next span."""
    spans = _spans()
    # offset 10 == end of page 1 span → falls in the gap → returns page 1
    assert page_for_offset(10, spans) == 1


# ---------------------------------------------------------------------------
# normalize_heading — Phase 13.1
# ---------------------------------------------------------------------------

def test_normalize_heading_strips_markdown_hashes() -> None:
    assert normalize_heading("## Introduction") == "Introduction"


def test_normalize_heading_strips_triple_hash() -> None:
    assert normalize_heading("### Deep Section") == "Deep Section"


def test_normalize_heading_strips_numbered_prefix_simple() -> None:
    assert normalize_heading("1. Introduction") == "Introduction"


def test_normalize_heading_strips_numbered_prefix_dotted() -> None:
    assert normalize_heading("1.2 Methods") == "Methods"


def test_normalize_heading_strips_roman_numeral_prefix() -> None:
    result = normalize_heading("IV. Results")
    assert result == "Results"


def test_normalize_heading_strips_trailing_colon() -> None:
    assert normalize_heading("Introduction:") == "Introduction"


def test_normalize_heading_strips_trailing_punctuation() -> None:
    assert normalize_heading("Summary.") == "Summary"


def test_normalize_heading_collapses_internal_whitespace() -> None:
    assert normalize_heading("Some  Heading  Here") == "Some Heading Here"


def test_normalize_heading_rejects_mojibake() -> None:
    assert normalize_heading("Introduct�on") == ""


def test_normalize_heading_rejects_too_long() -> None:
    long = "A" * 81
    assert normalize_heading(long, max_len=80) == ""


def test_normalize_heading_rejects_digit_only() -> None:
    assert normalize_heading("12345") == ""


def test_normalize_heading_rejects_no_alpha() -> None:
    assert normalize_heading("--- ...") == ""


def test_normalize_heading_accepts_clean_title() -> None:
    assert normalize_heading("Liber Primus") == "Liber Primus"


def test_normalize_heading_empty_string() -> None:
    assert normalize_heading("") == ""


def test_normalize_heading_just_hashes() -> None:
    # After stripping hashes there's nothing left — no alpha → ""
    assert normalize_heading("##") == ""
