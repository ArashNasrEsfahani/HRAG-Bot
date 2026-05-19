"""Retriever interface. All retrievers (vector, KG, community) implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from hrag.types import RetrievalResult

if TYPE_CHECKING:
    from hrag.intent import Intent


class Retriever(ABC):
    name: str = "abstract"

    @abstractmethod
    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 30,
        source_types: Optional[list[str]] = None,
        intent_hint: Optional["Intent"] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Return up to `top_k` chunks ranked by relevance.

        `source_types` filters by the Chunk.source_type field
        (e.g. ["document"], ["episodic"], or both).

        `intent_hint` is an optional pre-classified intent from the orchestrator.
        Most retrievers ignore it; QueryRouter uses it to skip its own LLM
        classification call.

        `where` is an optional Chroma-style metadata filter (Phase 7-A). Only
        retrievers that ultimately call a vector backend with metadata
        filtering honour it; sparse / KG / community retrievers accept the
        kwarg and silently ignore it.
        """
