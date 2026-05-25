"""Post-process answers that contain Phase 4 `[UNCERTAIN]` markers.

The answer prompt instructs the LLM to write `[UNCERTAIN]` after any sub-claim
it cannot back with a verbatim quote. The orchestrator optionally renders these
markers into a visible glyph + italicised label so the user sees model
self-doubt rather than confident hallucination.

This module is pure: no I/O, no LLM. Just a regex replacement.
"""
from __future__ import annotations

import re

_UNCERTAIN_RE = re.compile(r"\[UNCERTAIN\]")
_VISIBLE_MARKER = " ⚠️*(uncertain)*"


def render_uncertain(answer: str) -> tuple[str, int]:
    """Replace every `[UNCERTAIN]` token in *answer* with a visible marker.

    Returns (rendered_answer, count_of_markers).

    Idempotent: applying the function twice does not double-render, because
    the regex only matches the literal `[UNCERTAIN]` token, not the rendered
    visible-marker string.
    """
    if not answer:
        return answer, 0
    count = len(_UNCERTAIN_RE.findall(answer))
    if count == 0:
        return answer, 0
    rendered = _UNCERTAIN_RE.sub(_VISIBLE_MARKER, answer)
    return rendered, count


def strip_uncertain(answer: str) -> str:
    """Remove `[UNCERTAIN]` tokens entirely (used when masking is disabled
    but the LLM still emitted them — strip rather than leak raw tokens to
    end users). Returns the cleaned answer.
    """
    if not answer:
        return answer
    return _UNCERTAIN_RE.sub("", answer).rstrip()


def extract_uncertain_spans(answer: str, *, max_chars_per_span: int = 240) -> list[str]:
    """Return text fragments preceding each ``[UNCERTAIN]`` marker.

    Used by the Phase 9.17 Self-RAG pass: for every flagged sub-claim, we
    take the chunk of natural-language text the LLM wrote immediately before
    the marker as the query for an extra retrieval pass. Looks back to the
    previous sentence boundary (``.``/``!``/``?``/newline) or
    ``max_chars_per_span`` characters, whichever comes first.

    Empty / span-less inputs return ``[]`` with no allocation.
    """
    if not answer or "[UNCERTAIN]" not in answer:
        return []
    spans: list[str] = []
    for m in _UNCERTAIN_RE.finditer(answer):
        start = m.start()
        # Walk back to the nearest sentence boundary or N chars, whichever first.
        lookback_start = max(0, start - max_chars_per_span)
        snippet = answer[lookback_start:start]
        # Trim to last sentence ending; if none, keep the full snippet.
        for boundary in (". ", "! ", "? ", "\n"):
            idx = snippet.rfind(boundary)
            if idx != -1:
                snippet = snippet[idx + len(boundary):]
                break
        snippet = snippet.strip()
        if snippet:
            spans.append(snippet)
    return spans
