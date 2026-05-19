"""Tests for hrag.gating.clue — ClueGenerator.

All tests use a lightweight _StubLLM that avoids any real LLM calls.
"""
from __future__ import annotations

from hrag.gating.clue import ClueGenerator, _format_history
from hrag.types import Message


# ---------------------------------------------------------------------------
# Stub LLM
# ---------------------------------------------------------------------------

class _StubLLM:
    """Minimal LLMProvider stand-in that records the last complete() call."""

    name = "stub"

    def __init__(self, return_value: str = "Hypothesis text.", raise_exc: bool = False):
        self._return_value = return_value
        self._raise_exc = raise_exc
        self.last_prompt: str = ""
        self.last_temperature: float | None = None
        self.last_max_tokens: int | None = None

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        if self._raise_exc:
            raise RuntimeError("LLM backend failure")
        return self._return_value


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_generator(return_value: str = "Hypothesis text.", raise_exc: bool = False, max_tokens: int = 200):
    llm = _StubLLM(return_value=return_value, raise_exc=raise_exc)
    gen = ClueGenerator(llm, max_tokens=max_tokens)
    return gen, llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_hypothesis_text():
    """Stub returns a multi-sentence string; generator returns it stripped."""
    gen, _ = _make_generator(return_value="  First sentence. Second sentence.  ")
    result = gen.generate("What is quantum entanglement?")
    assert result == "First sentence. Second sentence."


def test_empty_output_falls_back_to_question():
    """Stub returns empty string; generator returns the original question."""
    question = "What safety protocols apply to lithium batteries?"
    gen, _ = _make_generator(return_value="")
    result = gen.generate(question)
    assert result == question


def test_whitespace_output_falls_back_to_question():
    """Stub returns whitespace-only string; generator returns the original question."""
    question = "How does gradient descent converge?"
    gen, _ = _make_generator(return_value="   \n  ")
    result = gen.generate(question)
    assert result == question


def test_llm_exception_falls_back_to_question():
    """Stub raises; generator catches and returns the original question without crashing."""
    question = "What is the capital of France?"
    gen, _ = _make_generator(raise_exc=True)
    result = gen.generate(question)
    assert result == question


def test_history_rendered_into_prompt():
    """Generator called with two Messages; captured prompt contains both messages."""
    gen, llm = _make_generator()
    history = [
        Message(role="user", content="I'm studying neural networks."),
        Message(role="assistant", content="Great, I can help with that."),
    ]
    gen.generate("What is backpropagation?", history=history)
    assert "I'm studying neural networks." in llm.last_prompt
    assert "Great, I can help with that." in llm.last_prompt


def test_max_tokens_passed_through():
    """Custom max_tokens value flows through to LLMProvider.complete."""
    gen, llm = _make_generator(max_tokens=350)
    gen.generate("Some question?")
    assert llm.last_max_tokens == 350


def test_temperature_is_nonzero():
    """Generator uses temperature=0.2 (slightly creative; different from gate's 0.0)."""
    gen, llm = _make_generator()
    gen.generate("What is reinforcement learning?")
    assert llm.last_temperature == 0.2


def test_no_history_renders_empty_conversation():
    """When no history is given, the conversation block in the prompt is empty."""
    gen, llm = _make_generator()
    gen.generate("What is entropy?")
    # The prompt template has "Conversation:\n{conversation}" — with no history,
    # the placeholder resolves to an empty string.
    assert "Conversation:\n\n" in llm.last_prompt or "Conversation:\n" in llm.last_prompt


def test_format_history_empty():
    """_format_history returns empty string for an empty list."""
    assert _format_history([]) == ""


def test_format_history_capitalises_role():
    """_format_history capitalises the role prefix."""
    msgs = [Message(role="user", content="Hello")]
    out = _format_history(msgs)
    assert out == "User: Hello"


def test_template_cached_after_first_load(tmp_path, monkeypatch):
    """Template is read from disk only once; subsequent calls use the cache."""
    import hrag.gating.clue as clue_module

    # Point _PROMPT_PATH at a temp file we control.
    fake_prompt = tmp_path / "clue.md"
    fake_prompt.write_text("Conversation:\n{conversation}\n\nQuestion: {question}\n\nHypothesis:", encoding="utf-8")
    monkeypatch.setattr(clue_module, "_PROMPT_PATH", fake_prompt)

    gen, llm = _make_generator()
    gen._template = None  # ensure fresh load

    gen.generate("First call?")
    # Overwrite the file — generator should still use cached template.
    fake_prompt.write_text("REPLACED CONTENT {conversation} {question}", encoding="utf-8")
    gen.generate("Second call?")

    # Both prompts should contain "Hypothesis:" from the original template.
    assert "Hypothesis:" in llm.last_prompt
