"""Detect math content in raw text.

Used by the chunker to keep equations intact across chunk boundaries, and by
the quality filter to carve out math-only chunks from the alpha-ratio /
min-length checks.

Conventions:
  * "Display math" = block-level math that we must not split across chunks.
    Patterns: ``$$ ... $$`` and ``\\begin{X} ... \\end{X}`` for the standard
    LaTeX display environments (equation, align, eqnarray, gather, with or
    without the starred variants).
  * "Inline math" = ``$ ... $`` or ``\\( ... \\)``. We only use inline math to
    flip the ``has_math`` tag; inline spans are never used to nudge chunk
    boundaries (they fit within a paragraph anyway).
  * "Unicode math" = Greek letters, math symbols, sub/superscripts, equation
    density and function-of-variable patterns. PDF extractors (PyMuPDF) do not
    reconstruct LaTeX delimiters, so PDF-derived chunks rely entirely on the
    Unicode signal to get the ``has_math`` tag.

The detector is best-effort: a stray unmatched ``$`` in body text (e.g. a
price like "$5") may produce a false positive on the ``has_math`` tag, but
will not corrupt chunk boundaries because we only nudge on display math
which requires the doubled ``$$`` delimiter.
"""

from __future__ import annotations

import re

# Display math: $$ ... $$ across multiple lines (non-greedy).
_RE_DOLLAR_DISPLAY = re.compile(r"\$\$.*?\$\$", re.DOTALL)

# LaTeX display environments. Star variants share the same name in the
# closing tag, so we capture the optional star and require matching it.
_DISPLAY_ENV_NAMES = ("equation", "align", "eqnarray", "gather")
_RE_LATEX_ENV = re.compile(
    r"\\begin\{(" + "|".join(_DISPLAY_ENV_NAMES) + r")(\*?)\}"
    r".*?"
    r"\\end\{\1\2\}",
    re.DOTALL,
)

# Inline math: $...$ (no $$ — already handled) or \( ... \). Only used for
# the has_math flag; never for boundary decisions.
_RE_DOLLAR_INLINE = re.compile(r"(?<!\$)\$(?!\$)[^\n$]+?(?<!\$)\$(?!\$)")
_RE_PAREN_INLINE = re.compile(r"\\\(.+?\\\)", re.DOTALL)


# ---------------------------------------------------------------------------
# Unicode math signals
# ---------------------------------------------------------------------------
#
# Heuristic: a chunk is "unicode math" if at least `min_signals` (default 2)
# of these five independent categories fire. One Greek letter alone is too
# noisy (β-blockers, alpha test, etc.); two distinct categories almost always
# mean a real equation. We don't try to localise spans — only to flip the
# has_math tag.

# (a) Greek letters commonly used as variables in math/physics. BMP only.
_GREEK_MATH_LETTERS = set("αβγδεζηθικλμνξοπρστυφχψω" "ΓΔΘΛΞΠΣΦΨΩ" "∂∇")
# Unicode "Mathematical Alphanumeric Symbols" block: U+1D400..U+1D7FF.
_MATH_ALPHANUMERIC_LO = 0x1D400
_MATH_ALPHANUMERIC_HI = 0x1D7FF

# (b) Math operator / relation / bracket / dot symbols.
_MATH_SYMBOLS = set("∑∫∏√∞≤≥≠≈⟨⟩⊕⊗⋅")

# (c) Subscripts (₀-₉) and superscripts (⁰-⁹, ², ³ inherited from Latin-1).
_RE_SUBSCRIPT = re.compile(r"[₀-₉]")
_RE_SUPERSCRIPT = re.compile(r"[⁰-⁹²³¹]")

# (d) Equation density: 2+ '=' signs within a 200-char window where at least
# one '=' has a letter immediately adjacent (so "= 1 b = 2" hits, but bare
# "==" delimiters or YAML-ish "k = v" prose lines without clustering miss).
# Implemented procedurally inside has_unicode_math.

# (e) Function-of-variable: a single letter followed by '(' and a single-
# letter argument, e.g. "f(x)", "Θ(q)", or "Θ(𝑞|". One occurrence is enough.
# Includes Unicode "Mathematical Alphanumeric Symbols" (U+1D400..U+1D7FF) so
# PDF-derived math-italic glyphs count as letters.
_LETTER_CLASS = "A-Za-zΑ-Ωα-ω\U0001d400-\U0001d7ff"
_RE_FUNCTION_OF_VAR = re.compile(
    rf"(?<![{_LETTER_CLASS}])[{_LETTER_CLASS}]\([{_LETTER_CLASS}·][\s,)·|]"
)


def _has_greek_or_mathalpha(text: str) -> bool:
    for ch in text:
        if ch in _GREEK_MATH_LETTERS:
            return True
        code = ord(ch)
        if _MATH_ALPHANUMERIC_LO <= code <= _MATH_ALPHANUMERIC_HI:
            return True
    return False


def _has_math_symbol(text: str) -> bool:
    return any(ch in _MATH_SYMBOLS for ch in text)


def _has_sub_or_super(text: str) -> bool:
    return bool(_RE_SUBSCRIPT.search(text) or _RE_SUPERSCRIPT.search(text))


def _has_equation_density(text: str) -> bool:
    """2+ '=' signs in any 200-char window, with at least one letter-adjacent '='."""
    eq_positions = [i for i, ch in enumerate(text) if ch == "="]
    if len(eq_positions) < 2:
        return False
    # Letter-adjacent test once globally — if no '=' is touched by a letter,
    # this is probably a config block (k=v lines), not an equation.
    has_letter_adj = False
    for i in eq_positions:
        left = text[i - 1] if i > 0 else " "
        right = text[i + 1] if i + 1 < len(text) else " "
        if left.isalpha() or right.isalpha():
            has_letter_adj = True
            break
    if not has_letter_adj:
        return False
    # Sliding window of 200 chars: any pair of '=' within 200 chars wins.
    for j in range(1, len(eq_positions)):
        if eq_positions[j] - eq_positions[j - 1] <= 200:
            return True
    return False


def _has_function_of_var(text: str) -> bool:
    return bool(_RE_FUNCTION_OF_VAR.search(text))


def has_unicode_math(text: str, *, min_signals: int = 2) -> bool:
    """True if *text* shows >= ``min_signals`` independent Unicode math signals.

    Signals are: (a) Greek/math-alphanumeric letters, (b) math operator
    symbols, (c) subscripts/superscripts, (d) equation density, (e)
    function-of-variable. Two-of-five is enough to flip the tag for PDF
    chunks that PyMuPDF stripped of LaTeX delimiters.
    """
    signals = 0
    if _has_greek_or_mathalpha(text):
        signals += 1
    if _has_math_symbol(text):
        signals += 1
    if signals >= min_signals:
        return True
    if _has_sub_or_super(text):
        signals += 1
        if signals >= min_signals:
            return True
    if _has_equation_density(text):
        signals += 1
        if signals >= min_signals:
            return True
    if _has_function_of_var(text):
        signals += 1
    return signals >= min_signals


# ---------------------------------------------------------------------------
# LaTeX path (unchanged behaviour, renamed)
# ---------------------------------------------------------------------------


def _has_latex_math(text: str) -> bool:
    """True if *text* contains any LaTeX display OR inline math delimiter."""
    if _RE_DOLLAR_DISPLAY.search(text):
        return True
    if _RE_LATEX_ENV.search(text):
        return True
    if _RE_PAREN_INLINE.search(text):
        return True
    if _RE_DOLLAR_INLINE.search(text):
        return True
    return False


def find_display_math_spans(text: str) -> list[tuple[int, int]]:
    """Return sorted list of ``(start, end)`` spans for *display* math blocks.

    These are the spans the chunker must NOT split across.
    """
    spans: list[tuple[int, int]] = []
    for rx in (_RE_DOLLAR_DISPLAY, _RE_LATEX_ENV):
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))
    spans.sort()
    return spans


def has_math(text: str) -> bool:
    """True if *text* contains LaTeX OR Unicode math content.

    Cheap to call — short-circuits on the first match.
    """
    return _has_latex_math(text) or has_unicode_math(text)


def position_in_span(pos: int, spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    """If ``pos`` falls strictly inside any span, return that span; else None."""
    for s, e in spans:
        if s < pos < e:
            return (s, e)
        if pos < s:
            break
    return None
