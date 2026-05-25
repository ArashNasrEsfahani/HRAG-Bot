"""Retrieval-policy table: maps an Intent to a RetrievalPlan.

The orchestrator calls ``RetrievalPolicy.plan(intent)`` after intent
classification to decide what scope and parameters to pass to the retriever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.intent import Intent
    from hrag.config import IntentConfig

# The four retrieval scopes understood by the orchestrator:
#   "none"        — skip retrieval entirely (e.g. pure greetings)
#   "episodic"    — only query episodic memory chunks
#   "full"        — run the full retriever (vector / taxonomy / router / …)
#   "ask_clarify" — retrieval skipped; orchestrator should ask the user to
#                   rephrase before attempting to answer
Scope = Literal["none", "episodic", "full", "ask_clarify"]


@dataclass(frozen=True)
class RetrievalPlan:
    """Immutable descriptor of what the retriever should do for one query.

    Attributes:
        scope:           High-level retrieval mode.
        top_k_override:  When set, overrides the config's ``top_k_vector``;
                         ``None`` means "use whatever the config says".
        source_types:    Forwarded verbatim to ``Retriever.retrieve``'s
                         ``source_types`` parameter; ``None`` means no filter
                         (all source types).
    """

    scope: Scope
    top_k_override: Optional[int]       # None = use config default
    source_types: Optional[list[str]]   # forwarded to retriever


class RetrievalPolicy:
    """Stateless policy table that converts an Intent into a RetrievalPlan.

    Args:
        cfg: The ``IntentConfig`` section from the global ``Config``.  Provides
             intent-specific knobs such as ``personal_top_k``.  The type is
             forward-referenced to avoid a circular import — ``IntentConfig``
             is added by Wave 2A and will be present by the time the
             orchestrator instantiates this class.
    """

    def __init__(self, cfg: IntentConfig) -> None:
        self._cfg = cfg

    def plan(self, intent: Intent) -> RetrievalPlan:
        """Return a :class:`RetrievalPlan` appropriate for *intent*.

        Routing table:
        - ``GREETING``  → scope ``"none"`` (no retrieval; canned response)
        - ``PERSONAL``  → scope ``"episodic"`` (episodic memory only, limited k)
        - ``FACTUAL``   → scope ``"full"`` (normal retrieval pipeline)
        - ``GENERAL``   → scope ``"none"`` (substantive but off-corpus; answer
                          from LLM general knowledge, no retrieval)
        - ``UNCLEAR``   → scope ``"ask_clarify"`` (prompt user to rephrase)
        """
        # Local import to avoid a circular-import cycle: intent.py must not be
        # imported at module level from within the retrieval package.
        from hrag.intent import Intent  # local import to avoid circular

        if intent == Intent.GREETING:
            return RetrievalPlan(scope="none", top_k_override=None, source_types=None)
        if intent == Intent.PERSONAL:
            # Search BOTH uploaded documents and episodic memories.
            # Previously this was episodic-only, which meant uploaded CVs /
            # notes were never searched when the question was phrased personally
            # (e.g. "find my master GPA" → PERSONAL → episodic-only → miss).
            # Using scope="full" gives us reranking, which is necessary to
            # surface a relevant document chunk when memories also match.
            return RetrievalPlan(
                scope="full",
                top_k_override=None,          # use global top_k_vector
                source_types=["document", "episodic"],
            )
        if intent == Intent.FACTUAL:
            return RetrievalPlan(scope="full", top_k_override=None, source_types=None)
        if intent == Intent.GENERAL:
            # Off-corpus factual question — no retrieval. The answer prompt
            # tells the LLM to answer from world knowledge with a small note.
            return RetrievalPlan(scope="none", top_k_override=None, source_types=None)
        # UNCLEAR or any future intent variant we haven't mapped yet
        return RetrievalPlan(scope="ask_clarify", top_k_override=None, source_types=None)
