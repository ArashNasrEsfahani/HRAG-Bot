"""Phase 9.6 — Combined gate+clue+intent preflight tests.

The CombinedPreflight class is a pure parse-the-LLM-output helper, so we can
test it with a tiny stub LLM that returns canned text. No Orchestrator
instantiation needed for unit coverage.
"""
from __future__ import annotations

from hrag.config import CompactionConfig
from hrag.gating.combined import (
    CombinedPreflight,
    PreflightDecision,
    _extract_json,
)


class _StubLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple] = []

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None):
        self.calls.append((prompt, temperature, max_tokens))
        return self.payload


def test_combined_preflight_default_off():
    cfg = CompactionConfig()
    assert cfg.combined_preflight_enabled is False


def test_extract_json_naked_object():
    out = _extract_json('{"intent":"factual","gate":"RETRIEVE","clue":"..."}')
    assert out == {"intent": "factual", "gate": "RETRIEVE", "clue": "..."}


def test_extract_json_with_fence():
    raw = '```json\n{"intent":"factual","gate":"RETRIEVE","clue":"x"}\n```'
    out = _extract_json(raw)
    assert out is not None
    assert out["intent"] == "factual"


def test_extract_json_with_leading_prose():
    raw = 'Sure, here is the JSON:\n{"intent":"greeting","gate":"SKIP","clue":""}\nThanks!'
    out = _extract_json(raw)
    assert out == {"intent": "greeting", "gate": "SKIP", "clue": ""}


def test_extract_json_malformed_returns_none():
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None


def test_combined_preflight_happy_path():
    llm = _StubLLM('{"intent":"factual","gate":"RETRIEVE","clue":"A draft of an answer."}')
    pf = CombinedPreflight(llm)
    out = pf.decide("what is HippoRAG?", history=None)
    assert isinstance(out, PreflightDecision)
    assert out.intent == "factual"
    assert out.gate == "RETRIEVE"
    assert "draft" in out.clue
    assert len(llm.calls) == 1


def test_combined_preflight_skip_with_empty_clue():
    llm = _StubLLM('{"intent":"greeting","gate":"SKIP","clue":""}')
    out = CombinedPreflight(llm).decide("hi!")
    assert out is not None
    assert out.gate == "SKIP"
    assert out.intent == "greeting"
    assert out.clue == ""


def test_combined_preflight_bad_intent_returns_none():
    llm = _StubLLM('{"intent":"sports","gate":"RETRIEVE","clue":"x"}')
    assert CombinedPreflight(llm).decide("x") is None


def test_combined_preflight_bad_gate_returns_none():
    llm = _StubLLM('{"intent":"factual","gate":"MAYBE","clue":"x"}')
    assert CombinedPreflight(llm).decide("x") is None


def test_combined_preflight_llm_error_returns_none():
    class _Raiser:
        def complete(self, *a, **kw):
            raise RuntimeError("boom")

    assert CombinedPreflight(_Raiser()).decide("x") is None
