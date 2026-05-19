"""PreferenceExtractor: LLM-driven extraction of preferences from a conversation.

Renders ``src/hrag/prompts/preference_extract.md`` (already drafted with the
target JSON shape and few-shot example) and parses the LLM's JSON output.
Defensive on every step: malformed JSON returns ``[]``, items missing
required keys are dropped, polarity is validated against the schema enum.

Never invoked from the chat hot path. Two call sites:
  * ``hrag memory extract --session SID`` CLI subcommand (offline, manual).
  * ``SessionAutoExtractor.on_session_close`` daemon (opt-in via
    ``memory.auto_extract: true``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.providers.llm import LLMProvider


logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "preference_extract.md"
_VALID_POLARITIES = {"like", "dislike", "fact", "style"}

# Fence pattern that strips ```json``` / ``` ``` wrappers small models often emit.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class PreferenceCandidate:
    polarity: str
    topic: str
    value: str
    confidence: float


class PreferenceExtractor:
    def __init__(self, llm: "LLMProvider") -> None:
        self._llm = llm
        self._template = _PROMPT_PATH.read_text(encoding="utf-8")

    def extract(
        self,
        conversation: list[tuple[str, str]],
    ) -> list[PreferenceCandidate]:
        """Extract preference candidates from a list of (role, content) turns.

        Returns ``[]`` on any parse failure or empty input.
        """
        if not conversation:
            return []

        rendered = _format_conversation(conversation)
        # Prompt contains literal `{...}` JSON examples — use replace, not
        # .format(), so the braces aren't misinterpreted as field references.
        prompt = self._template.replace("{conversation}", rendered)
        try:
            raw = self._llm.complete(prompt, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PreferenceExtractor: LLM call failed: %s", exc)
            return []

        return _parse_candidates(raw)


def _format_conversation(conversation: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"{role.capitalize()}: {content}" for role, content in conversation
    )


def _parse_candidates(raw: str) -> list[PreferenceCandidate]:
    if not raw or not raw.strip():
        return []
    text = _FENCE_RE.sub("", raw.strip()).strip()
    if not text:
        return []

    try:
        items = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fall back: try to grab the first JSON array substring.
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return []

    if not isinstance(items, list):
        return []

    out: list[PreferenceCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        polarity = (item.get("polarity") or "").strip().lower()
        topic = (item.get("topic") or "").strip()
        value = (item.get("value") or "").strip()
        confidence_raw = item.get("confidence", 0.5)
        if polarity not in _VALID_POLARITIES or not topic:
            continue
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        out.append(
            PreferenceCandidate(
                polarity=polarity,
                topic=topic,
                value=value,
                confidence=confidence,
            )
        )
    return out
