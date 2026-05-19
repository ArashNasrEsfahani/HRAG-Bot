"""Query rewriting for conversational follow-up turns.

Stateless retrieval and reranking only see the bare user message. For follow-up
questions like "explain its architecture" or "tell me more", that bare query has
no antecedent — the cross-encoder rejects everything and the LLM ends up with
no context. This module rewrites the retrieval query (only) using conversation
history, leaving the original question intact for the answer prompt.

Three modes are exposed via factory:
  - "heuristic" (default): cheap rules, no LLM call.
  - "llm": one LLM call, falls back to heuristic on failure.
  - "none": passthrough; useful for benchmarking or when retrieval is fine.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from hrag.providers.llm import LLMProvider


# ---------------------------------------------------------------------------
# Trigger detection (used by HeuristicRewriter and as the LLM fallback)
# ---------------------------------------------------------------------------

# Bare pronouns that almost always need an antecedent from the prior turn.
_PRONOUNS = {"it", "its", "this", "that", "they", "them", "these", "those"}

# Question openers that signal an elaboration on the previous turn.
# Restricted to phrases that are *strong* continuation markers — "how" or "why"
# alone are common first-question words, so they're not included here. Short
# questions (≤6 tokens) and pronoun presence still independently trigger the
# follow-up rule.
_RE_FOLLOWUP_OPENER = re.compile(
    r"^\s*(tell me more|what about|how about|elaborate|continue|also|more details)\b",
    re.IGNORECASE,
)

# Token used to split a question into rough words for pronoun checks.
_RE_TOKEN = re.compile(r"[A-Za-z']+")

# ---------------------------------------------------------------------------
# Math-meta expansion
# ---------------------------------------------------------------------------

_RE_MATH_META = re.compile(
    r"\b(formula|formulae|equation|equations|math|maths|"
    r"mathematical|derivation|theorem|proof|loss\s+function|"
    r"objective\s+function)s?\b",
    re.IGNORECASE,
)

# Tokens that *actually appear* in equation-bearing chunks. The query
# rewriter appends these so the dense embedding lands closer to math
# passages, where the prose typically uses these words.
# Limitation: broad trigger words like "formula" will also match unrelated
# contexts (e.g. "formula one racing"). The expansion is harmless on
# non-math corpora — the extra tokens simply don't match anything.
_MATH_EXPANSION_TOKENS = (
    "equation parameter θ Θ loss function objective gradient "
    "∑ ∫ derivation variable"
)


def _expand_math_meta(query: str) -> str:
    """If the query asks about math/formulas as a content type, append
    a short bag of math-vocabulary tokens so the query embedding is
    closer to actual equation passages. No-op otherwise."""
    if _RE_MATH_META.search(query):
        return f"{query} {_MATH_EXPANSION_TOKENS}"
    return query


def _looks_like_followup(question: str) -> bool:
    """True when the question probably depends on the prior turn."""
    if not question.strip():
        return False

    tokens = _RE_TOKEN.findall(question.lower())
    if len(tokens) <= 6:
        return True
    if _RE_FOLLOWUP_OPENER.match(question):
        return True
    if any(tok in _PRONOUNS for tok in tokens):
        return True
    return False


def _last_user_message(history: list[tuple[str, str]]) -> Optional[str]:
    """Return the most recent prior user message, or None."""
    for role, content in reversed(history):
        if role == "user" and content.strip():
            return content.strip()
    return None


# ---------------------------------------------------------------------------
# QueryRewriter API
# ---------------------------------------------------------------------------


class QueryRewriter(ABC):
    """Rewrite a user question into a self-contained retrieval query."""

    name: str = "abstract"

    @abstractmethod
    def rewrite(self, question: str, history: list[tuple[str, str]]) -> str:
        """Return the query to use for retrieval. May equal `question`."""


class NoopRewriter(QueryRewriter):
    """Return the question unchanged. Useful as a kill-switch."""

    name = "noop"

    def rewrite(self, question: str, history: list[tuple[str, str]]) -> str:
        return question


class HeuristicRewriter(QueryRewriter):
    """Cheap rule-based rewriter; no LLM call.

    When the question looks like a follow-up and there is a prior user message
    in the history, return `"{prev_user}\\n{question}"` so the embedder and
    reranker see the antecedent. Otherwise return the question unchanged.
    """

    name = "heuristic"

    def rewrite(self, question: str, history: list[tuple[str, str]]) -> str:
        if not history:
            return _expand_math_meta(question)
        if not _looks_like_followup(question):
            return _expand_math_meta(question)
        prev = _last_user_message(history)
        if not prev:
            return _expand_math_meta(question)
        return _expand_math_meta(f"{prev}\n{question}")


class LLMRewriter(QueryRewriter):
    """Ask the LLM to fold conversation history into a self-contained query.

    On any failure (empty output, exception) falls back to the heuristic so a
    bad LLM call never breaks retrieval.
    """

    name = "llm"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        prompt_path = Path(__file__).parent.parent / "prompts" / "query_rewrite.md"
        self._template = prompt_path.read_text(encoding="utf-8")
        self._fallback = HeuristicRewriter()

    def rewrite(self, question: str, history: list[tuple[str, str]]) -> str:
        # No history → nothing to ground; skip the LLM call.
        if not history:
            return question

        history_text = _format_history(history)
        prompt = self._template.format(
            conversation_history=history_text,
            question=question,
        )

        try:
            raw = self._llm.complete(prompt, temperature=0.0).strip()
        except Exception:
            return self._fallback.rewrite(question, history)

        rewritten = _clean_llm_output(raw)
        if not rewritten:
            return self._fallback.rewrite(question, history)
        return rewritten


# ---------------------------------------------------------------------------
# Helpers for the LLM path
# ---------------------------------------------------------------------------


def _format_history(history: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, content in history:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _clean_llm_output(raw: str) -> str:
    """Strip common wrappers an LLM may add around the rewritten query."""
    text = raw.strip().strip("`").strip()
    # Run prefix and quote stripping until idempotent — the LLM may produce
    # combinations like 'Rewritten query: "..."' that need both passes.
    for _ in range(3):
        prev = text
        for prefix in ("rewritten query:", "rewritten:", "query:"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        if text == prev:
            break
    return text
