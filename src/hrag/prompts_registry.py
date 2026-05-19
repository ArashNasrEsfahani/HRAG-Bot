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
