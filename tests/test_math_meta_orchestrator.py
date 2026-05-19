"""Tests for the Phase 7-A math-meta + formula-extraction orchestrator wiring.

Covers:

* Unit tests on the standalone ``_is_math_meta_query`` detector.
* Integration tests through ``Orchestrator.chat()`` with a scripted classifier
  and a recording retriever — confirm the ``where={"has_math": True}`` filter
  is plumbed through when the flag is on, the empty-results fallback fires,
  and the formula-extraction LLM pass runs (and appends its block) only when
  both flags trigger together.
"""

from __future__ import annotations

from typing import Optional

import pytest  # noqa: F401 — keep available for future skips

from hrag.intent import Intent, IntentVerdict
from hrag.orchestrator import _is_math_meta_query
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Detector unit tests
# ---------------------------------------------------------------------------


def test_detector_formulas_true() -> None:
    """The bare plural 'formulas' should trigger."""
    assert _is_math_meta_query("give me some formulas") is True


def test_detector_equations_true() -> None:
    """'equations' triggers, plus the meta-question 'what equations...'"""
    assert _is_math_meta_query("what equations are used") is True


def test_detector_unrelated_false() -> None:
    """A regular factual query about a topic must NOT trigger."""
    assert _is_math_meta_query("tell me about hipporag") is False


def test_detector_show_math_true() -> None:
    """The standalone token 'math' must trigger."""
    assert _is_math_meta_query("show the math") is True


def test_detector_case_insensitive() -> None:
    """All-caps must match the same as lowercase."""
    assert _is_math_meta_query("FORMULAS") is True
    assert _is_math_meta_query("Equation") is True


# ---------------------------------------------------------------------------
# Orchestrator integration tests
# ---------------------------------------------------------------------------


class _ScriptedClassifier:
    """Returns a fixed intent verdict for every query."""

    def __init__(self, intent: Intent) -> None:
        self._intent = intent

    def classify(self, text: str) -> IntentVerdict:
        return IntentVerdict(
            intent=self._intent,
            confidence=1.0,
            source="test",
            raw_label=self._intent.value,
        )


class _RecordingRetriever:
    """Spy retriever recording each call (including the ``where`` kwarg).

    A second-call override lets tests simulate the empty-first-call fallback:
    when ``second_call_results`` is provided, the second invocation returns
    those instead of the first-call seed (which is treated as ``[]``).
    """

    name = "spy"

    def __init__(
        self,
        results: Optional[list[RetrievalResult]] = None,
        empty_first_then: Optional[list[RetrievalResult]] = None,
    ) -> None:
        self.calls: list[dict] = []
        self._results = results or []
        self._empty_first_then = empty_first_then

    def retrieve(
        self,
        query,
        user_id,
        top_k=10,
        source_types=None,
        intent_hint=None,
        where=None,
    ):
        self.calls.append({
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
            "source_types": source_types,
            "intent_hint": intent_hint,
            "where": where,
        })
        if self._empty_first_then is not None:
            if len(self.calls) == 1:
                return []
            return list(self._empty_first_then)
        return list(self._results)


def _reset_db_singleton() -> None:
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None


def _make_orch(sample_config, classifier, retriever):
    """Build an Orchestrator with the scripted classifier + recording retriever."""
    _reset_db_singleton()
    sample_config.retrieval.rerank_enabled = False

    from hrag.orchestrator import Orchestrator
    orch = Orchestrator(sample_config)

    # Stub LLM so nothing calls a real provider.
    from tests.conftest import FakeLLM
    fake_llm = FakeLLM()
    orch.llm = fake_llm
    if orch.gate is not None:
        orch.gate.llm = fake_llm
    if orch.clue is not None:
        orch.clue.llm = fake_llm

    orch.intent_classifier = classifier  # type: ignore[assignment]
    orch.retriever = retriever  # type: ignore[assignment]
    return orch, fake_llm


def _chunk(chunk_id: str, source_type: str = "document") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        user_id="default",
        text=f"text for {chunk_id}",
        embedding_text=f"emb for {chunk_id}",
        source_type=source_type,
    )


def _result(chunk_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(chunk_id), score=score)


# ---------------------------------------------------------------------------


def test_no_where_filter_when_flag_off(sample_config) -> None:
    """Regression guard: ``where`` is None when math_meta_filter_enabled is False."""
    sample_config.retrieval.math_meta_filter_enabled = False
    spy = _RecordingRetriever(results=[_result("c1")])
    orch, _ = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.FACTUAL),
        spy,
    )
    try:
        orch.chat("give me some formulas", user_id="default")
    finally:
        orch.close()
        _reset_db_singleton()

    assert len(spy.calls) == 1
    assert spy.calls[0]["where"] is None


def test_where_filter_passed_when_flag_on_and_meta_query(sample_config) -> None:
    """Flag on + meta query: ``where={"has_math": True}`` and event fires."""
    sample_config.retrieval.math_meta_filter_enabled = True
    spy = _RecordingRetriever(results=[_result("c1")])
    orch, _ = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.FACTUAL),
        spy,
    )
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what equations are used?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    # The filter was passed on the first call.
    assert spy.calls
    assert spy.calls[0]["where"] == {"has_math": True}
    # The event fired with the correct payload shape.
    filter_events = [p for n, p in events if n == "math_meta_filter"]
    assert len(filter_events) == 1
    assert filter_events[0]["where"] == {"has_math": True}
    assert filter_events[0]["query"] == "what equations are used?"


def test_fallback_when_filter_empty(sample_config) -> None:
    """Filtered retrieval returns []; orchestrator retries unfiltered and emits the fallback event."""
    sample_config.retrieval.math_meta_filter_enabled = True
    spy = _RecordingRetriever(
        empty_first_then=[_result("c1"), _result("c2")],
    )
    orch, _ = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.FACTUAL),
        spy,
    )
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "show the math behind it",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    # Two retrieve calls — first filtered (returned []), second unfiltered.
    assert len(spy.calls) == 2
    assert spy.calls[0]["where"] == {"has_math": True}
    assert spy.calls[1]["where"] is None
    # Fallback event fired.
    fb_events = [p for n, p in events if n == "math_meta_filter_fallback"]
    assert len(fb_events) == 1
    assert fb_events[0]["reason"] == "no_matches"


def test_formula_extraction_fires_when_enabled(sample_config) -> None:
    """formula_extraction.enabled=True + meta query + results: event fires and answer is appended."""
    sample_config.retrieval.math_meta_filter_enabled = True
    sample_config.formula_extraction.enabled = True
    spy = _RecordingRetriever(results=[_result("c1"), _result("c2")])
    orch, fake_llm = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.FACTUAL),
        spy,
    )
    events: list[tuple[str, dict]] = []
    try:
        result = orch.chat(
            "give me the formulas",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    # The extraction event fired.
    extract_events = [p for n, p in events if n == "formula_extract"]
    assert len(extract_events) == 1
    assert "duration_s" in extract_events[0]
    assert "chars" in extract_events[0]
    # The final answer carries the extracted-formulas block.
    assert "Extracted formulas:" in result.answer


def test_formula_extraction_skipped_when_non_meta_query(sample_config) -> None:
    """formula_extraction.enabled=True but a non-meta query: no event, no extra LLM call."""
    sample_config.retrieval.math_meta_filter_enabled = True
    sample_config.formula_extraction.enabled = True
    spy = _RecordingRetriever(results=[_result("c1")])
    orch, fake_llm = _make_orch(
        sample_config,
        _ScriptedClassifier(Intent.FACTUAL),
        spy,
    )
    events: list[tuple[str, dict]] = []
    # Count baseline LLM calls (intent classifier + answer pass).
    pre_calls = len(fake_llm.calls)
    try:
        result = orch.chat(
            "tell me about hipporag",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    extract_events = [p for n, p in events if n == "formula_extract"]
    assert extract_events == []
    # No "Extracted formulas:" block appended.
    assert "Extracted formulas:" not in result.answer
    # The number of LLM calls is bounded — at most one (the answer pass)
    # beyond the baseline. The extraction pass would be a second call.
    post_calls = len(fake_llm.calls)
    assert (post_calls - pre_calls) <= 1
