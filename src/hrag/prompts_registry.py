"""Prompt template registry indexed by Intent.

Templates are eagerly loaded at construction time so a missing file surfaces at
boot rather than on the first user turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.intent import Intent


class PromptRegistry:
    """Maps each Intent to its raw prompt template string.

    Args:
        prompts_dir: Directory that contains the four answer .md files.

    Raises:
        FileNotFoundError: Immediately at construction time if any template file
            is absent.
    """

    def __init__(self, prompts_dir: Path) -> None:
        # Local import to avoid a circular-import cycle: intent.py may itself
        # import from config.py, which must not import prompts_registry at
        # module level.
        from hrag.intent import Intent  # local import to avoid circular

        self._prompts_dir = prompts_dir
        self._files: dict[Intent, str] = {
            Intent.FACTUAL:  "answer.md",
            Intent.GREETING: "answer_greeting.md",
            Intent.PERSONAL: "answer_personal.md",
            Intent.UNCLEAR:  "answer_unclear.md",
            Intent.GENERAL:  "answer_general.md",
        }
        self._templates: dict[Intent, str] = {
            intent: (prompts_dir / fname).read_text(encoding="utf-8")
            for intent, fname in self._files.items()
        }
        # Phase 7-A: extra prompt loaded lazily on first call so instances
        # that never trigger the math-meta path don't pay the file-read cost.
        self._extract_formulas_template: str | None = None
        # Phase 8.2: sibling template for the PERSONAL intent that fires when
        # there is nothing on file — no profile, no memories. Kept as a
        # separate file (not under Intent.PERSONAL) so the main personal
        # template never contains the fallback example string that small
        # Gemma-family models were copy-pasting verbatim into every answer.
        # Loaded eagerly so a missing file surfaces at boot.
        self._personal_empty_template: str = (
            prompts_dir / "answer_personal_empty.md"
        ).read_text(encoding="utf-8")
        # Phase 8.3: friendly memory-led template — fires when intent is
        # PERSONAL AND there is at least one episodic memory AND no document
        # chunk earned a meaningfully-positive rerank score. The previous
        # behaviour rendered a flat "I know X" reply; this template tells
        # the LLM to lead with the fact, admit the limit honestly, and
        # offer to dig further in one conversational beat.
        self._personal_known_template: str = (
            prompts_dir / "answer_personal_known.md"
        ).read_text(encoding="utf-8")

    def get(self, intent: Intent) -> str:
        """Return the raw (un-rendered) template string for *intent*."""
        return self._templates[intent]

    def render(self, intent: Intent, **kwargs: object) -> str:
        """Render the template for *intent* via ``str.format(**kwargs)``.

        Raises:
            KeyError: If a placeholder required by the template is absent from
                *kwargs*.
        """
        return self._templates[intent].format(**kwargs)

    def render_personal_empty(
        self,
        *,
        conversation_history: str,
        question: str,
    ) -> str:
        """Render the PERSONAL "nothing on file yet" sibling template.

        Phase 8.2: the main ``answer_personal.md`` no longer carries the
        literal fallback string as an example, because small Gemma-family
        models were copy-pasting it verbatim on every PERSONAL turn — even
        when memories WERE retrieved. The orchestrator picks this template
        instead when there are zero retrieval hits AND the profile is empty,
        and asks the LLM to phrase the "nothing yet" answer in its own words.
        """
        return self._personal_empty_template.format(
            conversation_history=conversation_history,
            question=question,
        )

    def render_personal_known(
        self,
        *,
        retrieved_memories: str,
        retrieved_docs_summary: str,
        conversation_history: str,
        question: str,
    ) -> str:
        """Render the Phase 8.3 memory-led friendly PERSONAL template.

        Fires when the orchestrator has at least one episodic memory hit
        AND no document chunk reached a positive rerank score — the bot
        leads with what it remembers, admits the limit, and offers to
        search the documents further in one conversational beat.
        """
        return self._personal_known_template.format(
            retrieved_memories=retrieved_memories,
            retrieved_docs_summary=retrieved_docs_summary,
            conversation_history=conversation_history,
            question=question,
        )

    def render_extract_formulas(self, retrieved_passages: str) -> str:
        """Render the Phase 7-A formula-extraction prompt.

        Loaded lazily on first call so existing instances that never trigger
        the math-meta path don't pay the file-read cost.
        """
        if self._extract_formulas_template is None:
            self._extract_formulas_template = (
                self._prompts_dir / "extract_formulas.md"
            ).read_text(encoding="utf-8")
        return self._extract_formulas_template.format(
            retrieved_passages=retrieved_passages,
        )
