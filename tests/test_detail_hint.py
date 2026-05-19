"""Tests for the _detail_hint helper in orchestrator.py."""

from __future__ import annotations

from pathlib import Path

from hrag.orchestrator import _detail_hint


def test_detail_hint_triggers_on_in_detail() -> None:
    result = _detail_hint("explain its architecture in detail.")
    assert any(kw in result for kw in ("thorough", "multi-paragraph", "depth"))


def test_detail_hint_default_for_simple_question() -> None:
    result = _detail_hint("what is hipporag?")
    assert "Match length to question complexity" in result


def test_detail_hint_case_insensitive() -> None:
    result = _detail_hint("WALK ME THROUGH the algorithm")
    assert any(kw in result for kw in ("thorough", "multi-paragraph", "depth"))


def test_detail_hint_substring_match() -> None:
    result = _detail_hint("describe it step-by-step please")
    assert any(kw in result for kw in ("thorough", "multi-paragraph", "depth"))


def test_detail_hint_no_false_positive() -> None:
    # "indepth.com" contains "indepth" but not the token "in depth" (with a space)
    # and not any other pattern from _DETAIL_PATTERNS.
    result = _detail_hint("tell me about indepth.com the website")
    assert "Match length to question complexity" in result


def test_answer_prompt_renders_with_detail_hint() -> None:
    prompt_path = Path(__file__).parent.parent / "src" / "hrag" / "prompts" / "answer.md"
    template = prompt_path.read_text(encoding="utf-8")
    rendered = template.format(
        user_profile="",
        conversation_history="",
        retrieved_passages="",
        question="x",
        detail_hint="HINT",
    )
    assert "HINT" in rendered
