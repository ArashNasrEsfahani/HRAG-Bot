"""Tests for hrag.gating.uncertain — render_uncertain and strip_uncertain.

Pure-function module: no external deps, no importorskip needed.
"""

from __future__ import annotations

from hrag.gating.uncertain import render_uncertain, strip_uncertain


# ---------------------------------------------------------------------------
# render_uncertain
# ---------------------------------------------------------------------------


def test_no_marker_passthrough():
    """Input with no [UNCERTAIN] token is returned unchanged, count=0."""
    text = "The sky is blue and the grass is green."
    rendered, count = render_uncertain(text)
    assert rendered == text
    assert count == 0


def test_single_marker_rendered():
    """A single [UNCERTAIN] token is replaced with the visible glyph; count=1."""
    text = "Paris is the capital of France [UNCERTAIN] and also very beautiful."
    rendered, count = render_uncertain(text)
    assert count == 1
    assert "[UNCERTAIN]" not in rendered
    assert "⚠️" in rendered
    assert "uncertain" in rendered


def test_multiple_markers_rendered():
    """Three [UNCERTAIN] tokens all get replaced; count=3."""
    text = (
        "Claim A [UNCERTAIN] is supported. "
        "Claim B [UNCERTAIN] is also supported. "
        "Claim C [UNCERTAIN] might be wrong."
    )
    rendered, count = render_uncertain(text)
    assert count == 3
    assert "[UNCERTAIN]" not in rendered
    assert rendered.count("⚠️") == 3


def test_idempotent():
    """Applying render_uncertain twice does not double-render the visible marker."""
    text = "Some claim [UNCERTAIN] here."
    first, count1 = render_uncertain(text)
    second, count2 = render_uncertain(first)
    assert first == second, "Second application should not change the rendered output"
    assert count2 == 0, "No literal [UNCERTAIN] tokens remain after first render"


def test_empty_string():
    """Empty string returns ('', 0)."""
    rendered, count = render_uncertain("")
    assert rendered == ""
    assert count == 0


# ---------------------------------------------------------------------------
# strip_uncertain
# ---------------------------------------------------------------------------


def test_strip_uncertain_removes_tokens():
    """strip_uncertain removes [UNCERTAIN] and returns the rest intact."""
    text = "Foo [UNCERTAIN] bar."
    result = strip_uncertain(text)
    assert "[UNCERTAIN]" not in result
    # Core words should still be present
    assert "Foo" in result
    assert "bar" in result


def test_strip_uncertain_empty_string():
    """strip_uncertain handles empty input gracefully."""
    assert strip_uncertain("") == ""


# ---------------------------------------------------------------------------
# Literal token matching (case-sensitive)
# ---------------------------------------------------------------------------


def test_marker_only_matches_literal_token():
    """Lowercase [uncertain] must NOT be replaced — only the exact uppercase form."""
    text = "This is [uncertain] and should not be touched."
    rendered, count = render_uncertain(text)
    assert count == 0
    assert rendered == text
    assert "[uncertain]" in rendered  # token must survive unchanged
