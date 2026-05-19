"""Chunk quality filter for the HRAG-Bot ingestion pipeline.

Pure-function module: no I/O, no side effects.
Only stdlib + pydantic + count_tokens from chunker.

QualityConfig is defined in hrag.config to avoid circular imports
(config -> ingest.chunker -> config).  This module re-exports it for
callers who want a single import location.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

# QualityConfig lives in config.py (no hrag.ingest.* deps there)
from hrag.config import QualityConfig  # noqa: F401  (re-export)

if TYPE_CHECKING:
    from hrag.types import Chunk


# ---------------------------------------------------------------------------
# Compiled regexes (module-level for performance)
# ---------------------------------------------------------------------------

# Year pattern (1900-2099, optionally followed by a letter for disambiguation)
_RE_YEAR = re.compile(r"\b(19|20)\d{2}[a-z]?\b")

# Author initials pattern: comma followed by one or more initials like "A. B."
_RE_AUTHOR_INITIALS = re.compile(r",\s+(?:[A-Z]\.\s*)+")

# DOI pattern
_RE_DOI = re.compile(r"\bdoi\s*:\s*10\.\d{4,}", re.IGNORECASE)

# arXiv pattern
_RE_ARXIV = re.compile(r"\barxiv\s*:\s*\d{4}\.\d{4,}", re.IGNORECASE)

# Page artifact: the whole text (stripped) is just a page number / page X of Y
_RE_PAGE_ARTIFACT = re.compile(
    r"^(?:Page\s*)?\d+(?:\s*of\s*\d+)?\s*$",
    re.IGNORECASE,
)

# Reference section title keywords
_REF_SECTION_KEYWORDS = ("references", "bibliography", "works cited")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alpha_ratio(text: str) -> float:
    """Fraction of characters in *text* that are ASCII letters."""
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


def _is_url_soup(text: str) -> bool:
    """True when the text is dominated by URL/identifier noise.

    Heuristic: most 'words' look like hex strings, base64, or are URLs.
    Conservative — only fires on very extreme cases.
    """
    words = text.split()
    if len(words) < 3:
        return False
    url_like = sum(
        1 for w in words
        if w.startswith(("http://", "https://", "ftp://", "www."))
        or re.fullmatch(r"[0-9a-fA-F\-]{20,}", w)   # long hex / UUID
    )
    return url_like / len(words) >= 0.8


# ---------------------------------------------------------------------------
# Per-chunk filters
# ---------------------------------------------------------------------------

def _check_too_short(chunk: "Chunk", cfg: QualityConfig) -> str:
    """Drop if BOTH token count and char count are below their thresholds."""
    from hrag.ingest.chunker import count_tokens  # lazy import — avoids circular dep

    # Carve-out: a chunk that's just a single equation may legitimately
    # be very short (e.g. "$$E = mc^2$$"); keep it.
    if chunk.metadata.get("has_math"):
        return ""

    text = chunk.text.strip()
    chars = len(text)
    tokens = chunk.token_count if chunk.token_count > 0 else count_tokens(text)

    if tokens < cfg.min_tokens and chars < cfg.min_chars:
        return f"too_short (tokens={tokens}, chars={chars})"
    return ""


def _check_alpha_ratio(chunk: "Chunk", cfg: QualityConfig) -> str:
    """Drop if the fraction of alphabetic chars is too low."""
    # Carve-out: LaTeX math has a very low alpha ratio (lots of symbols,
    # backslashes, digits). Chunks tagged has_math bypass this check.
    if chunk.metadata.get("has_math"):
        return ""

    text = chunk.text.strip()
    if not text:
        return ""
    ratio = _alpha_ratio(text)
    if ratio < cfg.min_alpha_ratio:
        return f"low_alpha_ratio ({ratio:.2f} < {cfg.min_alpha_ratio})"
    return ""


def _check_references_section(chunk: "Chunk", cfg: QualityConfig) -> str:
    """Drop if the chunk's section title indicates a reference/bibliography section."""
    if not cfg.drop_references_sections:
        return ""
    section_lower = (chunk.section or "").lower().strip()
    if any(kw in section_lower for kw in _REF_SECTION_KEYWORDS):
        return f"references_section (section={chunk.section!r})"
    return ""


def _check_bibliography_chunk(chunk: "Chunk", cfg: QualityConfig) -> str:
    """Drop chunks that look like dense bibliographic lists.

    Heuristic:
    - Count non-empty lines that contain a year reference.
    - If >= 3 year-containing lines AND the ratio of year-lines to total lines
      is high (>= 0.5), treat the chunk as a bibliography list.
    Both conditions must be true (conservative).
    """
    if not cfg.drop_bibliography_chunks:
        return ""

    text = chunk.text.strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    year_lines = [ln for ln in lines if _RE_YEAR.search(ln)]
    year_line_count = len(year_lines)

    if year_line_count < 3:
        return ""

    year_line_ratio = year_line_count / len(lines)
    if year_line_ratio >= 0.5:
        return f"bibliography_chunk (year_lines={year_line_count}/{len(lines)})"
    return ""


def _check_page_artifact(chunk: "Chunk", cfg: QualityConfig) -> str:
    """Drop page-number headers/footers and URL-soup identifiers."""
    if not cfg.drop_page_artifacts:
        return ""

    text = chunk.text.strip()
    if not text:
        return ""

    if _RE_PAGE_ARTIFACT.match(text):
        return f"page_artifact ({text!r})"

    if _is_url_soup(text):
        return "url_soup"

    return ""


# Bibliography-entry opener: starts with "[N]" or "[N, N, ...]" then text.
_RE_CITATION_OPENER = re.compile(r"^\s*\[\d+(?:\s*,\s*\d+)*\]\s+\S")

# URL anywhere in the text.
_RE_URL_ANY = re.compile(r"https?://\S+", re.IGNORECASE)


def _check_leading_page_artifact(chunk: "Chunk", cfg: QualityConfig) -> str:
    """Drop chunks whose first non-empty line is a bare page number.

    PyMuPDF often glues a page-number line onto the first paragraph of the
    next page. The loader strips most of these now, but some survive (e.g.
    section-numbered headings like '6.3' that look like page numbers). When
    the page-number line is *all* that's giving the chunk substance, drop it.
    """
    if not cfg.drop_page_artifacts:
        return ""

    lines = [ln for ln in chunk.text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return ""

    if not _RE_PAGE_ARTIFACT.match(lines[0].strip()):
        return ""

    # If the rest of the chunk would be too short to keep on its own, drop it.
    remainder = "\n".join(lines[1:]).strip()
    rem_chars = len(remainder)

    from hrag.ingest.chunker import count_tokens  # lazy — circular dep avoidance
    rem_tokens = count_tokens(remainder)

    if rem_tokens < cfg.min_tokens and rem_chars < cfg.min_chars:
        return f"leading_page_artifact (head={lines[0]!r}, rem_tokens={rem_tokens})"
    return ""


def _check_single_citation(chunk: "Chunk", cfg: QualityConfig) -> str:
    """Drop chunks that look like a single bibliography entry.

    A single bibliography entry typically: starts with [N], has a year, and
    has either a DOI / arXiv id / URL. The "≥3 year-lines" filter above only
    catches dense reference lists; this catches one-entry chunks that survive.
    """
    if not cfg.drop_bibliography_chunks:
        return ""

    text = chunk.text.strip()
    if not text:
        return ""

    has_opener = bool(_RE_CITATION_OPENER.match(text))
    has_year = bool(_RE_YEAR.search(text))
    has_locator = bool(
        _RE_DOI.search(text) or _RE_ARXIV.search(text) or _RE_URL_ANY.search(text)
    )

    if has_opener and has_year and has_locator:
        return "single_citation"
    return ""


# ---------------------------------------------------------------------------
# Corpus-level deduplication
# ---------------------------------------------------------------------------

def dedupe_chunks(chunks: list["Chunk"]) -> list["Chunk"]:
    """Remove exact-duplicate chunks (same `.text`); keep first occurrence."""
    seen: set[str] = set()
    unique: list["Chunk"] = []
    for chunk in chunks:
        key = chunk.text.strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_low_quality_chunk(
    chunk: "Chunk", cfg: QualityConfig
) -> tuple[bool, str]:
    """Return (is_low_quality, reason).

    reason is "" when the chunk passes all filters.
    Checks are applied in order; the first match short-circuits.
    """
    for check_fn in (
        _check_too_short,
        _check_alpha_ratio,
        _check_references_section,
        _check_bibliography_chunk,
        _check_single_citation,
        _check_page_artifact,
        _check_leading_page_artifact,
    ):
        reason = check_fn(chunk, cfg)
        if reason:
            return True, reason
    return False, ""


def filter_chunks(
    chunks: list["Chunk"],
    cfg: QualityConfig,
) -> tuple[list["Chunk"], list[tuple["Chunk", str]]]:
    """Apply all quality filters to a list of chunks.

    Returns
    -------
    kept : list[Chunk]
        Chunks that passed all per-chunk filters (and, if cfg.dedupe, deduped).
    dropped_with_reasons : list[tuple[Chunk, str]]
        Chunks that were filtered out, paired with a reason string.

    Order: per-chunk filters first, then deduplication across the kept set.
    """
    if not cfg.enabled:
        return list(chunks), []

    kept: list["Chunk"] = []
    dropped: list[tuple["Chunk", str]] = []

    for chunk in chunks:
        low_quality, reason = is_low_quality_chunk(chunk, cfg)
        if low_quality:
            dropped.append((chunk, reason))
        else:
            kept.append(chunk)

    if cfg.dedupe:
        deduped = dedupe_chunks(kept)
        # Identify which chunks were dropped by dedup
        deduped_ids = {id(c) for c in deduped}
        for chunk in kept:
            if id(chunk) not in deduped_ids:
                dropped.append((chunk, "duplicate"))
        kept = deduped

    return kept, dropped
