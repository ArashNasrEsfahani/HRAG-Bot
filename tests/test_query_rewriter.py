"""Tests for hrag.retrieval.query_rewriter — heuristic, llm, and noop rewriters."""

from __future__ import annotations


import pytest

from hrag.retrieval.query_rewriter import (
    HeuristicRewriter,
    LLMRewriter,
    NoopRewriter,
    _looks_like_followup,
)


# ---------------------------------------------------------------------------
# _looks_like_followup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        "explain its architecture",
        "tell me more",
        "why?",
        "how about that",
        "what about the second one",
        "elaborate",
        "and continue",
        "more details please",
    ],
)
def test_followup_openers_detected(q: str) -> None:
    assert _looks_like_followup(q)


@pytest.mark.parametrize(
    "q",
    [
        "describe it",
        "is this correct",
        "are these results valid",
        "do they match",
    ],
)
def test_pronouns_detected(q: str) -> None:
    assert _looks_like_followup(q)


def test_short_question_treated_as_followup() -> None:
    """Very short questions (≤ 6 tokens) are likely follow-ups."""
    assert _looks_like_followup("what is hipporag")  # 3 tokens
    assert _looks_like_followup("describe the system")  # 3 tokens


def test_long_self_contained_question_not_followup() -> None:
    """A long, content-rich question without pronouns or follow-up openers."""
    q = (
        "Describe the HippoRAG personalized PageRank algorithm operating over "
        "the knowledge graph constructed during offline indexing of passages."
    )
    assert not _looks_like_followup(q)


def test_empty_question_not_followup() -> None:
    assert not _looks_like_followup("")
    assert not _looks_like_followup("   ")


# ---------------------------------------------------------------------------
# NoopRewriter
# ---------------------------------------------------------------------------


def test_noop_returns_question_unchanged() -> None:
    r = NoopRewriter()
    assert r.rewrite("anything", []) == "anything"
    assert r.rewrite("anything", [("user", "prior"), ("assistant", "ans")]) == "anything"


# ---------------------------------------------------------------------------
# HeuristicRewriter
# ---------------------------------------------------------------------------


def test_heuristic_no_history_returns_question() -> None:
    r = HeuristicRewriter()
    assert r.rewrite("explain its architecture", []) == "explain its architecture"


def test_heuristic_self_contained_question_unchanged() -> None:
    """Long, content-rich questions without pronouns are not rewritten."""
    r = HeuristicRewriter()
    history = [("user", "what is hipporag?"), ("assistant", "It is...")]
    long_q = (
        "What evaluation benchmarks did HippoRAG use compared to "
        "baselines on the MuSiQue multi-hop reasoning dataset?"
    )
    assert r.rewrite(long_q, history) == long_q


def test_heuristic_followup_prepends_prev_user_message() -> None:
    r = HeuristicRewriter()
    history = [
        ("user", "what is hipporag?"),
        ("assistant", "It's a RAG framework using a knowledge graph."),
    ]
    out = r.rewrite("explain its architecture", history)
    assert "what is hipporag?" in out
    assert "explain its architecture" in out


def test_heuristic_skips_assistant_only_history() -> None:
    """If the only prior turn is from the assistant, fall back to passthrough."""
    r = HeuristicRewriter()
    history = [("assistant", "Some standalone preface.")]
    assert r.rewrite("explain it", history) == "explain it"


def test_heuristic_picks_most_recent_user_message() -> None:
    r = HeuristicRewriter()
    history = [
        ("user", "tell me about RAG generally"),
        ("assistant", "RAG is..."),
        ("user", "now tell me about hipporag"),
        ("assistant", "HippoRAG is..."),
    ]
    out = r.rewrite("explain it", history)
    # Most recent user message should be used as antecedent
    assert "now tell me about hipporag" in out


# ---------------------------------------------------------------------------
# LLMRewriter — uses a stub LLMProvider so no network call is needed.
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal LLMProvider stand-in for tests."""

    def __init__(self, output: str = "", raise_exc: Exception | None = None) -> None:
        self._output = output
        self._raise = raise_exc
        self.calls: list[str] = []

    # The LLMRewriter only calls .complete().
    def complete(self, prompt: str, **_kwargs) -> str:  # noqa: D401
        self.calls.append(prompt)
        if self._raise is not None:
            raise self._raise
        return self._output


def test_llm_rewriter_uses_llm_output() -> None:
    llm = _StubLLM(output="What is the architecture of HippoRAG?")
    r = LLMRewriter(llm)
    history = [("user", "what is hipporag?"), ("assistant", "...")]
    out = r.rewrite("explain its architecture", history)
    assert out == "What is the architecture of HippoRAG?"
    assert len(llm.calls) == 1


def test_llm_rewriter_strips_quotes_and_prefixes() -> None:
    llm = _StubLLM(output='Rewritten query: "what is hipporag\'s architecture"')
    r = LLMRewriter(llm)
    history = [("user", "what is hipporag?"), ("assistant", "...")]
    out = r.rewrite("explain its architecture", history)
    # Both the prefix and the surrounding quotes must be stripped
    assert "rewritten query" not in out.lower()
    assert not out.startswith('"')
    assert "hipporag" in out.lower()


def test_llm_rewriter_falls_back_on_empty() -> None:
    llm = _StubLLM(output="")
    r = LLMRewriter(llm)
    history = [("user", "what is hipporag?"), ("assistant", "...")]
    out = r.rewrite("explain its architecture", history)
    # Empty LLM output → heuristic fallback prepends prev_user
    assert "what is hipporag?" in out
    assert "explain its architecture" in out


def test_llm_rewriter_falls_back_on_exception() -> None:
    llm = _StubLLM(raise_exc=RuntimeError("boom"))
    r = LLMRewriter(llm)
    history = [("user", "what is hipporag?"), ("assistant", "...")]
    out = r.rewrite("explain its architecture", history)
    # Exception → heuristic fallback prepends prev_user
    assert "what is hipporag?" in out


def test_llm_rewriter_no_history_skips_call() -> None:
    """No history means nothing to ground; LLMRewriter must skip the LLM call."""
    llm = _StubLLM(output="should not be used")
    r = LLMRewriter(llm)
    out = r.rewrite("explain it", [])
    assert out == "explain it"
    assert llm.calls == []
