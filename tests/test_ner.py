"""Tests for hrag.kg.ner — SpacyNER, LLMNER, and build_ner factory."""

from __future__ import annotations

import types
import warnings
from unittest.mock import patch

import pytest

from hrag.kg.ner import LLMNER, NER, SpacyNER, build_ner


# ---------------------------------------------------------------------------
# Stub LLM (mirrors tests/test_query_rewriter.py)
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal LLMProvider stand-in for tests."""

    def __init__(self, output: str = "", raise_exc: Exception | None = None) -> None:
        self._output = output
        self._raise = raise_exc
        self.calls: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        if self._raise is not None:
            raise self._raise
        return self._output


# ---------------------------------------------------------------------------
# SpacyNER — regex fallback path (spaCy not installed)
# ---------------------------------------------------------------------------


def test_spacy_ner_regex_fallback_extracts_entities() -> None:
    """When spaCy is unavailable, the regex fallback finds capitalized phrases."""
    with patch.dict("sys.modules", {"spacy": None}):
        with pytest.warns(UserWarning, match="spaCy"):
            ner = SpacyNER()
            result = ner.extract("HippoRAG uses Personalized PageRank")

    # Both capitalized multi-word phrases should appear (lowercased)
    assert "hipporag" in result
    assert "personalized pagerank" in result


def test_spacy_ner_fallback_warns_once() -> None:
    """Only one warning is emitted — subsequent calls are silent."""
    with patch.dict("sys.modules", {"spacy": None}):
        with pytest.warns(UserWarning, match="spaCy"):
            ner = SpacyNER()
            # Trigger the first extraction (and the warning)
            ner.extract("OpenAI GPT Model")

        # Subsequent calls must not emit another warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ner.extract("Google DeepMind Research")  # should not raise


def test_spacy_ner_empty_query_returns_empty() -> None:
    """Empty or whitespace-only queries return []."""
    with patch.dict("sys.modules", {"spacy": None}):
        with pytest.warns(UserWarning):
            ner = SpacyNER()
            ner.extract("Bootstrap Warning")  # trigger fallback + warning

    assert ner.extract("") == []
    assert ner.extract("   ") == []


def test_spacy_ner_oserror_fallback() -> None:
    """OSError (model not installed) also triggers the regex fallback."""
    fake_spacy = types.ModuleType("spacy")

    def _raise_oserror(*args, **kwargs):
        raise OSError("model not found")

    fake_spacy.load = _raise_oserror

    with patch.dict("sys.modules", {"spacy": fake_spacy}):
        with pytest.warns(UserWarning, match="spaCy"):
            ner = SpacyNER()
            result = ner.extract("NASA Apollo Mission")

    assert "nasa" in result or "apollo" in result or "apollo mission" in result


# ---------------------------------------------------------------------------
# LLMNER
# ---------------------------------------------------------------------------


def test_llmner_parses_clean_json_list() -> None:
    """LLMNER correctly parses a clean JSON list."""
    llm = _StubLLM(output='["hipporag", "personalized pagerank"]')
    ner = LLMNER(llm)
    result = ner.extract("How does HippoRAG use Personalized PageRank?")
    assert result == ["hipporag", "personalized pagerank"]


def test_llmner_strips_code_fences() -> None:
    """LLMNER handles ```json ... ``` fences."""
    llm = _StubLLM(output='```json\n["bert", "transformer"]\n```')
    ner = LLMNER(llm)
    result = ner.extract("What is the architecture of BERT?")
    assert "bert" in result
    assert "transformer" in result


def test_llmner_strips_plain_code_fences() -> None:
    """LLMNER handles ``` ... ``` fences without json specifier."""
    llm = _StubLLM(output='```\n["openai", "gpt-4"]\n```')
    ner = LLMNER(llm)
    result = ner.extract("Tell me about OpenAI GPT-4")
    assert "openai" in result


def test_llmner_returns_empty_on_exception() -> None:
    """LLMNER must not propagate LLM exceptions — returns [] instead."""
    llm = _StubLLM(raise_exc=RuntimeError("network error"))
    ner = LLMNER(llm)
    result = ner.extract("Who founded DeepMind?")
    assert result == []


def test_llmner_returns_empty_on_malformed_json() -> None:
    """LLMNER returns [] when the LLM output is not valid JSON."""
    llm = _StubLLM(output="Sorry, I cannot extract entities from this query.")
    ner = LLMNER(llm)
    result = ner.extract("some question")
    assert result == []


def test_llmner_lowercases_and_dedupes() -> None:
    """LLMNER normalises case and removes duplicates (preserving first-seen order)."""
    llm = _StubLLM(output='["BERT", "bert", "Transformer", "BERT"]')
    ner = LLMNER(llm)
    result = ner.extract("BERT Transformer")
    assert result.count("bert") == 1
    assert result.count("transformer") == 1
    assert result[0] == "bert"


def test_llmner_empty_query_returns_empty() -> None:
    """Empty query skips the LLM call entirely."""
    llm = _StubLLM(output='["something"]')
    ner = LLMNER(llm)
    assert ner.extract("") == []
    assert ner.extract("   ") == []
    assert llm.calls == []


def test_llmner_prose_before_json_array() -> None:
    """LLMNER finds the JSON array even when preceded by prose."""
    llm = _StubLLM(output='Here are the entities: ["rag", "llm"] as requested.')
    ner = LLMNER(llm)
    result = ner.extract("Compare RAG and LLM approaches")
    assert "rag" in result
    assert "llm" in result


# ---------------------------------------------------------------------------
# build_ner factory
# ---------------------------------------------------------------------------


def test_build_ner_returns_spacy_by_default() -> None:
    """build_ner with cfg.ner='spacy' returns a SpacyNER instance."""
    cfg = types.SimpleNamespace(ner="spacy")
    llm = _StubLLM()
    ner = build_ner(cfg, llm)
    assert isinstance(ner, SpacyNER)


def test_build_ner_returns_llm_ner() -> None:
    """build_ner with cfg.ner='llm' returns an LLMNER instance."""
    cfg = types.SimpleNamespace(ner="llm")
    llm = _StubLLM(output="[]")
    ner = build_ner(cfg, llm)
    assert isinstance(ner, LLMNER)


def test_build_ner_raises_for_unknown_mode() -> None:
    """build_ner raises ValueError for unrecognised modes."""
    cfg = types.SimpleNamespace(ner="neural_net")
    llm = _StubLLM()
    with pytest.raises(ValueError, match="neural_net"):
        build_ner(cfg, llm)


def test_build_ner_uses_getattr_fallback() -> None:
    """build_ner works even when cfg has no .ner attribute (defaults to 'spacy')."""
    cfg = types.SimpleNamespace()  # no .ner attribute
    llm = _StubLLM()
    ner = build_ner(cfg, llm)
    assert isinstance(ner, SpacyNER)


def test_build_ner_simple_namespace_stub() -> None:
    """KGConfigStub via SimpleNamespace works correctly for 'spacy' mode."""
    KGConfigStub = types.SimpleNamespace(ner="spacy")
    llm = _StubLLM()
    ner = build_ner(KGConfigStub, llm)
    assert isinstance(ner, SpacyNER)
    assert ner.name == "spacy"


def test_build_ner_case_insensitive() -> None:
    """build_ner normalises mode strings (case + whitespace)."""
    cfg = types.SimpleNamespace(ner="  LLM  ")
    llm = _StubLLM(output="[]")
    ner = build_ner(cfg, llm)
    assert isinstance(ner, LLMNER)


# ---------------------------------------------------------------------------
# NER ABC
# ---------------------------------------------------------------------------


def test_ner_is_abstract() -> None:
    """NER cannot be instantiated directly."""
    with pytest.raises(TypeError):
        NER()  # type: ignore[abstract]
