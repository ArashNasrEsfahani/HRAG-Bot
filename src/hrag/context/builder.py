"""ContextBuilder: renders the {user_profile} string for the answer prompt.

The orchestrator's chat path uses this as the single source of the profile
text passed into ``answer.md.format(user_profile=...)`` (replacing the
hard-coded empty string at ``orchestrator.py``). Anything that wants to
extend what the prompt sees about the user goes here.

Read-path only — no LLM calls, no Chroma queries. The profile is one
SQLite read (~1 ms) per chat turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.memory.profile import ProfileStore


class ContextBuilder:
    def __init__(
        self,
        profile_store: "ProfileStore",
        max_items: int = 12,
        min_confidence: float = 0.5,
    ) -> None:
        self._profile_store = profile_store
        self._max_items = int(max_items)
        self._min_confidence = float(min_confidence)

    def build(self, user_id: str) -> dict:
        """Return the prompt-kwargs dict. Currently {'user_profile': str}.

        Kept as a dict so future additions (recent memories, gate verdict,
        clue strings) can be returned in a single call without changing the
        orchestrator's render block.
        """
        return {
            "user_profile": self._profile_store.render(
                user_id,
                max_items=self._max_items,
                min_confidence=self._min_confidence,
            ),
        }
