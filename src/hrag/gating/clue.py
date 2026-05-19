"""ClueGenerator — Phase 4 MemoRAG-style retrieval hypothesis.

Single LLM call against prompts/clue.md producing a 2–4 sentence draft outline
of what a good answer would look like. The output is used as the retrieval
query (better vocabulary-bridging than the raw user question). The original
question is still passed to the generation prompt.

Failure mode: empty or whitespace-only output -> return the original question
unchanged so the caller has no special-case path.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.providers.llm import LLMProvider
    from hrag.types import Message


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "clue.md"


def _format_history(history: list["Message"]) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for m in history:
        role = (m.role or "user").capitalize()
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


class ClueGenerator:
    """Generate a 2–4 sentence retrieval hypothesis for *question*."""

    name = "clue_llm"

    def __init__(self, llm: "LLMProvider", *, max_tokens: int = 200) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self._template: str | None = None

    def _load_template(self) -> str:
        if self._template is None:
            self._template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._template

    def generate(self, question: str, history: list["Message"] | None = None) -> str:
        """Produce a hypothesis. Fallback to *question* on empty output."""
        prompt = self._load_template().format(
            conversation=_format_history(history or []),
            question=question,
        )
        try:
            raw = self.llm.complete(prompt, temperature=0.2, max_tokens=self.max_tokens)
        except Exception:
            return question
        clue = (raw or "").strip()
        return clue if clue else question
