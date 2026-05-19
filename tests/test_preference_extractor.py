"""PreferenceExtractor: JSON parsing + defensive fallbacks."""

from __future__ import annotations

import pytest


class _CannedLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        return self._response


def test_extract_parses_clean_json():
    from hrag.memory.extractor import PreferenceExtractor

    raw = (
        '[{"polarity": "fact", "topic": "occupation", "value": "engineer", "confidence": 0.9}, '
        '{"polarity": "like", "topic": "tools", "value": "vim", "confidence": 0.8}]'
    )
    extractor = PreferenceExtractor(_CannedLLM(raw))
    out = extractor.extract([("user", "I'm an engineer who loves vim.")])
    assert len(out) == 2
    assert out[0].polarity == "fact"
    assert out[0].topic == "occupation"
    assert out[0].confidence == pytest.approx(0.9)


def test_extract_handles_markdown_fences():
    from hrag.memory.extractor import PreferenceExtractor

    raw = '```json\n[{"polarity": "like", "topic": "lang", "value": "Python", "confidence": 1.0}]\n```'
    out = PreferenceExtractor(_CannedLLM(raw)).extract([("user", "I love Python.")])
    assert len(out) == 1
    assert out[0].topic == "lang"


def test_extract_handles_prose_before_json():
    from hrag.memory.extractor import PreferenceExtractor

    raw = (
        'Sure, here are the extracted preferences:\n'
        '[{"polarity": "fact", "topic": "city", "value": "Berlin", "confidence": 0.95}]'
    )
    out = PreferenceExtractor(_CannedLLM(raw)).extract([("user", "I live in Berlin.")])
    assert len(out) == 1
    assert out[0].topic == "city"


def test_extract_returns_empty_on_malformed():
    from hrag.memory.extractor import PreferenceExtractor

    out = PreferenceExtractor(_CannedLLM("not json at all")).extract(
        [("user", "Hi")]
    )
    assert out == []


def test_extract_returns_empty_on_empty_response():
    from hrag.memory.extractor import PreferenceExtractor

    out = PreferenceExtractor(_CannedLLM("")).extract([("user", "Hi")])
    assert out == []


def test_extract_drops_items_with_invalid_polarity():
    from hrag.memory.extractor import PreferenceExtractor

    raw = (
        '[{"polarity": "neutral", "topic": "x", "value": "y", "confidence": 0.9}, '
        '{"polarity": "fact", "topic": "a", "value": "b", "confidence": 0.9}]'
    )
    out = PreferenceExtractor(_CannedLLM(raw)).extract([("user", "...")])
    assert len(out) == 1
    assert out[0].topic == "a"


def test_extract_drops_items_missing_topic():
    from hrag.memory.extractor import PreferenceExtractor

    raw = (
        '[{"polarity": "fact", "topic": "", "value": "y", "confidence": 0.9}, '
        '{"polarity": "fact", "topic": "ok", "value": "v", "confidence": 0.9}]'
    )
    out = PreferenceExtractor(_CannedLLM(raw)).extract([("user", "...")])
    assert len(out) == 1
    assert out[0].topic == "ok"


def test_extract_clamps_bad_confidence_to_default():
    from hrag.memory.extractor import PreferenceExtractor

    raw = '[{"polarity": "fact", "topic": "t", "value": "v", "confidence": "not-a-number"}]'
    out = PreferenceExtractor(_CannedLLM(raw)).extract([("user", "...")])
    assert len(out) == 1
    assert out[0].confidence == pytest.approx(0.5)


def test_extract_on_empty_conversation():
    from hrag.memory.extractor import PreferenceExtractor

    assert PreferenceExtractor(_CannedLLM("anything")).extract([]) == []


def test_extract_handles_llm_exception():
    from hrag.memory.extractor import PreferenceExtractor

    class _BoomLLM:
        def complete(self, *a, **kw):
            raise RuntimeError("ollama down")

    assert PreferenceExtractor(_BoomLLM()).extract([("user", "x")]) == []
