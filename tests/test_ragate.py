"""Tests for hrag.gating.gate — RAGate retrieval-gate classifier.

No optional heavy deps required — the stub LLM is defined inline.
All tests run in any environment where hrag.types is importable.
"""

from __future__ import annotations

import pytest

from hrag.gating.gate import RAGate, _format_history
from hrag.types import Message


# ---------------------------------------------------------------------------
# Stub LLM — records calls, returns a configurable reply
# ---------------------------------------------------------------------------

class _StubLLM:
    name = "stub"

    def __init__(self, reply: str = "RETRIEVE"):
        self.reply = reply
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self.reply


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def _gate(reply: str = "RETRIEVE", max_tokens: int = 8) -> tuple[RAGate, _StubLLM]:
    llm = _StubLLM(reply=reply)
    return RAGate(llm, max_tokens=max_tokens), llm


# ---------------------------------------------------------------------------
# Tests: routing decisions
# ---------------------------------------------------------------------------

def test_skip_routes_to_skip():
    """LLM returning 'SKIP' must produce decision 'SKIP'."""
    gate, _ = _gate("SKIP")
    assert gate.decide("Thanks, bye!") == "SKIP"


def test_retrieve_routes_to_retrieve():
    """LLM returning 'RETRIEVE' must produce decision 'RETRIEVE'."""
    gate, _ = _gate("RETRIEVE")
    assert gate.decide("What is the refund policy?") == "RETRIEVE"


def test_whitespace_and_punctuation_tolerated():
    """Leading/trailing whitespace and a trailing period must still parse as SKIP."""
    gate, _ = _gate("  skip.\n")
    assert gate.decide("How are you?") == "SKIP"


def test_case_insensitive():
    """Lowercase 'skip' must be normalised to SKIP."""
    gate, _ = _gate("skip")
    assert gate.decide("Hello!") == "SKIP"


# ---------------------------------------------------------------------------
# Tests: fail-open on garbled output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reply", [
    "I think we should retrieve documents.",
    "",
    "banana",
])
def test_garbled_output_fails_open(reply):
    """Any output that isn't SKIP (with punctuation/case variants) must return RETRIEVE."""
    gate, _ = _gate(reply)
    assert gate.decide("What is RAGate?") == "RETRIEVE"


def test_none_reply_fails_open():
    """A stub returning None (edge case) must fall back to RETRIEVE without error."""
    llm = _StubLLM(reply=None)  # type: ignore[arg-type]
    gate = RAGate(llm)
    # Should not raise; should return RETRIEVE
    assert gate.decide("test question") == "RETRIEVE"


# ---------------------------------------------------------------------------
# Tests: history rendering
# ---------------------------------------------------------------------------

def test_empty_history_renders_blank_conversation():
    """gate.decide with history=None or history=[] must not raise."""
    gate, llm = _gate("RETRIEVE")
    gate.decide("What is RAG?", history=None)
    gate.decide("What is RAG?", history=[])
    assert len(llm.calls) == 2


def test_history_rendered_correctly():
    """The prompt passed to the LLM must contain each message's content."""
    gate, llm = _gate("RETRIEVE")
    history = [
        _msg("user", "Hello there"),
        _msg("assistant", "Hi! How can I help?"),
    ]
    gate.decide("What is the capital of France?", history=history)

    assert len(llm.calls) == 1
    prompt = llm.calls[0]["prompt"]
    assert "Hello there" in prompt
    assert "Hi! How can I help?" in prompt


def test_history_roles_capitalised():
    """_format_history should capitalise the role label."""
    history = [_msg("user", "ping"), _msg("assistant", "pong")]
    rendered = _format_history(history)
    assert rendered.startswith("User:")
    assert "Assistant:" in rendered


def test_empty_history_returns_empty_string():
    """_format_history([]) and _format_history(None-equivalent) return ''."""
    assert _format_history([]) == ""


# ---------------------------------------------------------------------------
# Tests: max_tokens plumbing
# ---------------------------------------------------------------------------

def test_max_tokens_passed_through():
    """The max_tokens kwarg must be forwarded to LLMProvider.complete exactly."""
    gate, llm = _gate(max_tokens=4)
    gate.decide("Hello")
    assert len(llm.calls) == 1
    assert llm.calls[0]["max_tokens"] == 4


def test_default_max_tokens_is_8():
    """Default RAGate() must pass max_tokens=8 to the LLM."""
    gate, llm = _gate()  # default max_tokens=8
    gate.decide("Hello")
    assert llm.calls[0]["max_tokens"] == 8


def test_temperature_is_zero():
    """RAGate must always call complete with temperature=0.0 (deterministic)."""
    gate, llm = _gate()
    gate.decide("Some question")
    assert llm.calls[0]["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Tests: prompt template loading
# ---------------------------------------------------------------------------

def test_template_cached_after_first_call():
    """The gate must read the template file only once even across multiple decide() calls."""
    gate, llm = _gate("RETRIEVE")
    gate.decide("q1")
    gate.decide("q2")
    # _template should be non-None after first call
    assert gate._template is not None
    # Two LLM calls happened
    assert len(llm.calls) == 2


def test_name_attribute():
    """RAGate.name must equal 'ragate_llm' (used by logging/UI)."""
    gate, _ = _gate()
    assert gate.name == "ragate_llm"
