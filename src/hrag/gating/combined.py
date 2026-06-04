"""CombinedPreflight — Phase 9.6 single-call gate + clue + intent.

When all three Phase-4 / intent flags are on, the orchestrator's pre-retrieval
block normally runs three serial LLM calls (intent classifier, RAGate, then
ClueGenerator). On Ollama-class providers each call is hundreds of ms, so the
combined call cuts ~2x off the pre-retrieval phase.

This module renders ``prompts/combined_preflight.md``, parses one JSON
response, and returns the three decisions. Failure modes degrade gracefully:
malformed JSON returns ``None`` so the caller can fall back to the three
separate calls.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from hrag.providers.llm import LLMProvider
    from hrag.types import Message


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "combined_preflight.md"
_VALID_INTENTS = {"factual", "personal", "greeting", "unclear"}
_VALID_GATES = {"RETRIEVE", "SKIP"}


@dataclass(frozen=True)
class PreflightDecision:
    intent: Literal["factual", "personal", "greeting", "unclear"]
    gate: Literal["RETRIEVE", "SKIP"]
    clue: str
    # Phase 11.1 — True when the personal question asks for an impression/
    # opinion of the user. None when the model omitted the field (older
    # prompt / malformed output), so callers can distinguish "not reflective"
    # from "unknown" and fall back to the standalone reflective check.
    reflective: Optional[bool] = None


def _format_history(history: list["Message"]) -> str:
    if not history:
        return "(no prior conversation)"
    lines: list[str] = []
    for m in history:
        role = (m.role or "user").capitalize()
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    s = raw.strip()
    # Strip common markdown fences a chatty model might wrap around the JSON.
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    m = _JSON_OBJ_RE.search(raw)
    if m is None:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


class CombinedPreflight:
    """One LLM call that returns ``{intent, gate, clue}`` together."""

    name = "combined_preflight"

    def __init__(self, llm: "LLMProvider", *, max_tokens: int = 280) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self._template: Optional[str] = None

    def _load_template(self) -> str:
        if self._template is None:
            self._template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._template

    def decide(
        self,
        question: str,
        history: Optional[list["Message"]] = None,
    ) -> Optional[PreflightDecision]:
        """Run the combined prompt. Returns None on parse failure."""
        prompt = self._load_template().format(
            conversation=_format_history(history or []),
            question=question,
        )
        try:
            raw = self.llm.complete(
                prompt, temperature=0.1, max_tokens=self.max_tokens
            )
        except Exception:  # noqa: BLE001
            return None
        data = _extract_json(raw or "")
        if data is None:
            return None
        intent = str(data.get("intent", "")).strip().lower()
        gate = str(data.get("gate", "")).strip().upper()
        clue = str(data.get("clue", "") or "").strip()
        if intent not in _VALID_INTENTS:
            return None
        if gate not in _VALID_GATES:
            return None
        # Phase 11.1 — tolerate a missing/odd `reflective` field. Coerce common
        # truthy spellings ("true"/true/"yes"); leave None when absent.
        reflective: Optional[bool] = None
        if "reflective" in data:
            rv = data.get("reflective")
            if isinstance(rv, bool):
                reflective = rv
            else:
                reflective = str(rv).strip().lower() in {"true", "yes", "1"}
        return PreflightDecision(  # type: ignore[arg-type]
            intent=intent, gate=gate, clue=clue, reflective=reflective,
        )
