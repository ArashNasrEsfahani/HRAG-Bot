"""Heuristic section/heading detection from raw document text.

detect_sections(text) -> list of (start_offset, end_offset, section_title, subsection_title)
covering the entire text with no gaps and no overlaps.

Heading patterns detected (in priority order):
  1. Markdown headings:  # Section   ## Subsection   ### deeper -> subsection
  2. Numbered headings:  "1. Intro"  "2.1 Methods"  "1.1.2 Detail"
  3. ALL-CAPS lines:     lines < 80 chars that are entirely uppercase letters/punct/spaces
"""

from __future__ import annotations

import re
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Markdown heading: leading # signs
_RE_MD = re.compile(r"^(#{1,6})\s+(.+)$")

# Numbered heading: optional leading digits/dots then text
# Examples: "1. Introduction"  "2.1 Methods"  "10.3.4 Detail"
_RE_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.\s]+([A-Z].{2,})$")

# ALL-CAPS sentinel: line is all uppercase (letters, digits, spaces, basic punct)
_RE_ALLCAPS = re.compile(r"^[A-Z0-9\s\-–—:,./()]{3,79}$")

# Keywords that mark the start of a bibliography/references section.
# Mirrored in quality._REF_SECTION_KEYWORDS for the per-chunk filter; kept in
# sync intentionally — both layers want to recognise the same boundaries.
_REF_SECTION_KEYWORDS = ("references", "bibliography", "works cited")


class _Heading(NamedTuple):
    offset: int        # byte/char offset of the *start of that line* in text
    level: int         # 1 = section, 2 = subsection
    title: str


def _detect_headings(text: str) -> list[_Heading]:
    """Scan every line; return sorted list of detected headings with offsets."""
    headings: list[_Heading] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n\r")
        sstripped = stripped.strip()

        if sstripped:
            # 1. Markdown
            m = _RE_MD.match(sstripped)
            if m:
                level = 1 if len(m.group(1)) == 1 else 2
                headings.append(_Heading(offset=pos, level=level, title=m.group(2).strip()))
                pos += len(line)
                continue

            # 2. Numbered heading
            m2 = _RE_NUMBERED.match(sstripped)
            if m2:
                num_part = m2.group(1)
                # depth = number of dots + 1 in the numeric prefix
                depth = num_part.count(".") + 1
                level = 1 if depth == 1 else 2
                headings.append(_Heading(offset=pos, level=level, title=m2.group(2).strip()))
                pos += len(line)
                continue

            # 3. ALL-CAPS line (under 80 chars)
            if len(sstripped) < 80 and _RE_ALLCAPS.match(sstripped):
                headings.append(_Heading(offset=pos, level=1, title=sstripped))
                pos += len(line)
                continue

        pos += len(line)

    return headings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sections(text: str) -> list[tuple[int, int, str, str]]:
    """Return list of (start_offset, end_offset, section_title, subsection_title).

    The returned spans tile the entire text: the first span starts at 0, the
    last ends at len(text), with no gaps or overlaps.
    """
    total = len(text)

    if not text:
        return [(0, 0, "", "")]

    headings = _detect_headings(text)

    if not headings:
        return [(0, total, "", "")]

    # Build spans from headings, tracking current section/subsection state
    spans: list[tuple[int, int, str, str]] = []
    current_section = ""
    current_subsection = ""
    current_start = 0

    for h in headings:
        span_end = h.offset
        if span_end > current_start:
            spans.append((current_start, span_end, current_section, current_subsection))

        if h.level == 1:
            current_section = h.title
            current_subsection = ""
        else:
            current_subsection = h.title

        current_start = h.offset

    # Final span from last heading to end of text
    if current_start < total:
        spans.append((current_start, total, current_section, current_subsection))
    elif current_start == total:
        # Heading is right at the end - add a zero-length cap (edge case)
        spans.append((current_start, total, current_section, current_subsection))

    # If every heading was at offset 0 and there's only the preamble before them,
    # we may have produced zero spans; guarantee coverage:
    if not spans:
        spans = [(0, total, "", "")]

    # Ensure we start at 0 (the preamble before the first heading).
    if spans and spans[0][0] > 0:
        spans.insert(0, (0, spans[0][0], "", ""))

    return spans


# Standalone references heading: matches a single line that is just one of
# the keywords (case-insensitive), optionally with a numbered/section prefix
# and optional trailing punctuation. Captures the start-of-line offset.
# Examples that match:
#   "References"
#   "REFERENCES"
#   "## References"
#   "## 7 References"
#   "5. References"
#   "5.0 References"
#   "Section 7: References"
#   "Bibliography"
#   "Works Cited"
#   "Citations"
_RE_REFERENCES_HEADING = re.compile(
    r"(?im)^[ \t]*"                                # optional leading whitespace
    r"(?:#{1,6}[ \t]+)?"                           # optional markdown prefix
    r"(?:"
        r"\d+(?:\.\d+)*\.?[ \t]+"                  # "5 ", "5. ", "7.1 "
        r"|Section[ \t]+\d+(?:\.\d+)*[ \t]*[:.\-][ \t]+"   # "Section 7: "
    r")?"
    r"(?:References|Bibliography|Works[ \t]+Cited|Citations)"
    r"[ \t]*[:.\-]?[ \t]*$"                        # optional trailing punctuation
)

# Spaced caps variant: "R E F E R E N C E S" on a line by itself.
_RE_SPACED_CAPS_REFERENCES = re.compile(
    r"(?im)^[ \t]*(?:R[ \t]E[ \t]F[ \t]E[ \t]R[ \t]E[ \t]N[ \t]C[ \t]E[ \t]?S?)[ \t]*$"
)

# Citation-line indicators (reused/aligned with hrag.ingest.quality patterns).
# A "citation line" is one that strongly looks like a bibliography entry.
_RE_CITATION_OPENER = re.compile(r"^\s*\[\d+(?:\s*,\s*\d+)*\]\s+\S")
_RE_AUTHOR_INITIAL = re.compile(r"\b[A-Z][a-z]+,\s+[A-Z]\.")
_RE_YEAR_PAREN = re.compile(r"[\(\[](?:19|20)\d{2}[a-z]?[\)\]]")
_RE_DOI_INLINE = re.compile(r"\bdoi\s*:\s*10\.\d{4,}", re.IGNORECASE)
_RE_ARXIV_INLINE = re.compile(r"\barxiv\s*:\s*\d{4}\.\d{4,}", re.IGNORECASE)


def _looks_like_citation_line(line: str) -> bool:
    """Return True if the line looks like a single bibliography entry.

    Heuristic: must contain a year-in-brackets/parens OR a DOI/arXiv id, AND
    have an author-with-initial pattern OR a numbered citation opener.
    """
    if not line.strip():
        return False
    has_locator = bool(
        _RE_YEAR_PAREN.search(line)
        or _RE_DOI_INLINE.search(line)
        or _RE_ARXIV_INLINE.search(line)
    )
    has_author = bool(
        _RE_CITATION_OPENER.match(line)
        or _RE_AUTHOR_INITIAL.search(line)
    )
    return has_locator and has_author


def _find_citation_run_offset(
    text: str,
    *,
    min_run: int = 5,
    gap_tolerance: int = 2,
) -> int | None:
    """Find the offset of a long run of citation-shaped lines.

    Used as a fallback when no References/Bibliography heading is found.
    Walks the text line by line; whenever it encounters a citation-shaped
    line, it begins (or extends) a candidate run. Up to ``gap_tolerance``
    consecutive non-citation lines may appear inside a run (citations often
    wrap across lines, leaving the wrapped continuation looking less
    citation-y on its own). When a run reaches ``min_run`` citation hits,
    the start-of-line offset of the FIRST citation in that run is returned.
    """
    if not text:
        return None

    offset = 0
    in_run = False
    run_start: int | None = None
    hits = 0
    gap = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        is_cite = _looks_like_citation_line(stripped)

        if is_cite:
            if not in_run:
                in_run = True
                run_start = offset
                hits = 1
                gap = 0
            else:
                hits += 1
                gap = 0
            if hits >= min_run and run_start is not None:
                return run_start
        else:
            if in_run:
                if not stripped:
                    # Blank lines don't count toward the gap budget — they're
                    # the natural inter-entry separator in many PDFs.
                    pass
                else:
                    gap += 1
                    if gap > gap_tolerance:
                        in_run = False
                        run_start = None
                        hits = 0
                        gap = 0

        offset += len(line)

    return None


def find_references_offset(text: str) -> int | None:
    """Return the offset of the start of the bibliography section, or None.

    Used by the chunker to truncate academic-paper text before chunking when
    `chunking.drop_after_references` is enabled. Detection runs in three
    layers, picking the earliest candidate offset:

    1. The legacy heading-detector path (markdown / numbered / all-caps lines
       whose title equals one of `_REF_SECTION_KEYWORDS`). This still catches
       cases where the loader normalised the heading to ``# References``.
    2. A direct line-level regex (`_RE_REFERENCES_HEADING`) that recognises
       plain title-case standalone words like ``References`` and numbered
       prefixes like ``## 7 References`` or ``Section 7: References``.
    3. A spaced-caps regex for ``R E F E R E N C E S`` patterns.

    If none of those find a heading, a citation-run fallback scans for a
    long stretch of bibliography-shaped lines and returns the offset of the
    first such line in the run.
    """
    if not text:
        return None

    candidates: list[int] = []

    # Layer 1: legacy heading detector (preserves existing behaviour).
    for h in _detect_headings(text):
        normalised = h.title.strip().lower().rstrip(":.")
        for kw in _REF_SECTION_KEYWORDS:
            if normalised == kw or normalised.startswith(kw + " "):
                candidates.append(h.offset)
                break
        else:
            # Also catch "Citations" as a heading title.
            if normalised == "citations":
                candidates.append(h.offset)

    # Layer 2: direct line-level regex covers title-case / numbered headings.
    m = _RE_REFERENCES_HEADING.search(text)
    if m is not None:
        candidates.append(m.start())

    # Layer 3: spaced caps "R E F E R E N C E S".
    m = _RE_SPACED_CAPS_REFERENCES.search(text)
    if m is not None:
        candidates.append(m.start())

    if candidates:
        return min(candidates)

    # Fallback: long run of citation-shaped lines without any heading.
    return _find_citation_run_offset(text)
