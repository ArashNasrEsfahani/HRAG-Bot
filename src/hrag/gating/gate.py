"""RAGate — Phase 4 retrieval gate.

Single LLM call against prompts/gate.md that decides RETRIEVE vs SKIP.
Used by Orchestrator.chat() to short-circuit retrieval on small-talk that
slipped past the intent classifier.

Failure mode: anything other than the literal string 'SKIP' (case-insensitive,
ignoring whitespace and punctuation) is treated as RETRIEVE. We fail open so
the project never silently stops retrieving.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.providers.llm import LLMProvider
    from hrag.types import Message


GateDecision = Literal["RETRIEVE", "SKIP"]

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "gate.md"


def _format_history(history: list["Message"]) -> str:
    """Render a Message list as the {conversation} block expected by gate.md.

    Empty history -> empty string (the prompt handles that gracefully).
    """
    if not history:
        return ""
    lines: list[str] = []
    for m in history:
        role = (m.role or "user").capitalize()
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


class RAGate:
    """Wraps gate.md as a callable gate over (question, history)."""

    name = "ragate_llm"

    def __init__(self, llm: "LLMProvider", *, max_tokens: int = 8) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self._template: str | None = None

    def _load_template(self) -> str:
        if self._template is None:
            self._template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._template

    def decide(self, question: str, history: list["Message"] | None = None) -> GateDecision:
        """Return 'SKIP' iff the LLM output's first token (case-insensitive)
        is literally 'SKIP'. Everything else -> 'RETRIEVE' (fail-open).
        """
        prompt = self._load_template().format(
            conversation=_format_history(history or []),
            question=question,
        )
        raw = self.llm.complete(prompt, temperature=0.0, max_tokens=self.max_tokens)
        first = (raw or "").strip().split()
        token = first[0].upper().strip(".,!?:;'\"`") if first else ""
        return "SKIP" if token == "SKIP" else "RETRIEVE"
