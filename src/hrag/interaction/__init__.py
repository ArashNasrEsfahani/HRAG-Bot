"""Phase 8 — interactive retrieval-review module.

Public API for pausing the orchestrator between retrieval and answer
generation so the user can review / filter / rephrase the candidate sources.

The orchestrator imports :func:`maybe_pause` once between rerank and answer
rendering. Everything else (the SSE relay, the modal markup, the resume
endpoint) is built on top of this module.
"""

from __future__ import annotations

from .review import (
    PauseReason,
    ReviewDecision,
    ReviewRequired,
    build_review_payload,
    generate_rephrasings,
    maybe_pause,
    should_pause,
)
from .store import InteractionStore, PendingTurn

__all__ = [
    "ReviewDecision",
    "ReviewRequired",
    "PauseReason",
    "maybe_pause",
    "should_pause",
    "build_review_payload",
    "generate_rephrasings",
    "InteractionStore",
    "PendingTurn",
]
