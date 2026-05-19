"""Phase 8 — interactive retrieval review.

This module is the decision layer of the review loop:

* :func:`should_pause` — pure function over the seven triggers; returns the
  full list of reasons that fired (empty when no pause is needed).
* :func:`build_review_payload` — pure builder for the ``review_required``
  SSE event payload.
* :func:`generate_rephrasings` — best-effort LLM call producing up to N
  alternative phrasings of the user's question. Never raises.
* :func:`maybe_pause` — the blocking call site. The orchestrator invokes
  this once between rerank and answer rendering. Default-off: when
  ``cfg.review_enabled`` is False it returns ``ReviewDecision(action="continue")``
  immediately with zero side effects.

Heavy deps (the orchestrator, the web app, the LLM provider) are referenced
via :pydata:`typing.TYPE_CHECKING` only, so ``import hrag.interaction`` stays
cheap.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover - import guards
    from hrag.config import InteractionConfig
    from hrag.intent import IntentVerdict
    from hrag.providers.llm import LLMProvider
    from hrag.types import RetrievalResult

    from .store import InteractionStore


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class PauseReason(str, Enum):
    """Every signal that can cause :func:`maybe_pause` to block on a user
    decision. Strings are stable wire values — frontend uses them as
    classification labels in the modal."""

    SCORE_FLOOR = "score_floor"
    AMBIGUITY_DELTA = "ambiguity_delta"
    BRANCH_THRESHOLD = "branch_threshold"
    INTENT_UNCLEAR = "intent_unclear"
    ROUTER_AMBIGUOUS = "router_ambiguous"
    FACTUAL_GENERAL_SWAP = "factual_general_swap"
    ALWAYS_MODE = "always_mode"


@dataclass
class ReviewRequired:
    """Payload emitted as the ``review_required`` SSE event."""

    turn_id: str
    reasons: list[str]
    sources: list[dict[str, Any]]
    clue: Optional[str]
    intent: str
    router_label: Optional[str]
    retrieval_query: str
    original_question: str
    rephrasings: list[str]
    taxonomy_descend: Optional[dict[str, Any]]
    timeout_s: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the progress channel / SSE relay."""
        return {
            "turn_id": self.turn_id,
            "reasons": list(self.reasons),
            "sources": list(self.sources),
            "clue": self.clue,
            "intent": self.intent,
            "router_label": self.router_label,
            "retrieval_query": self.retrieval_query,
            "original_question": self.original_question,
            "rephrasings": list(self.rephrasings),
            "taxonomy_descend": self.taxonomy_descend,
            "timeout_s": self.timeout_s,
        }


# The known action strings the frontend may submit. Unknown actions fall
# back to "continue" (treat as approval).
_KNOWN_ACTIONS: frozenset[str] = frozenset(
    {
        "continue",
        "filter",
        "rephrase",
        "general",
        "clarify",
        "expand_doc",
        "redescend",
        "abort",
    }
)


@dataclass
class ReviewDecision:
    """The structured outcome of a single review pause.

    Always returned by :func:`maybe_pause` — never raises. The orchestrator
    inspects ``action`` and the optional fields to decide how to mutate the
    pipeline state before producing the answer.
    """

    action: str = "continue"
    selected_chunk_ids: list[str] = field(default_factory=list)
    rewritten_query: Optional[str] = None
    expand_from_doc_id: Optional[str] = None
    redirect_taxonomy_node_id: Optional[str] = None
    include_episodic: bool = False
    remember_choice: bool = False
    reasons: list[str] = field(default_factory=list)
    timed_out: bool = False

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        reasons: Optional[list[str]] = None,
    ) -> "ReviewDecision":
        """Best-effort parse of the JSON payload the frontend POSTed.

        Unknown actions are normalised to ``"continue"``. Lists default to
        ``[]``; bool fields coerce via ``bool(...)``; ``None`` is preserved
        for the optional string fields.
        """
        action_raw = d.get("action") or "continue"
        action = action_raw if action_raw in _KNOWN_ACTIONS else "continue"

        sel = d.get("selected_chunk_ids") or []
        if not isinstance(sel, list):
            sel = []
        selected_chunk_ids = [str(x) for x in sel if x is not None]

        def _opt_str(key: str) -> Optional[str]:
            val = d.get(key)
            if val is None:
                return None
            s = str(val).strip()
            return s or None

        return cls(
            action=action,
            selected_chunk_ids=selected_chunk_ids,
            rewritten_query=_opt_str("rewritten_query"),
            expand_from_doc_id=_opt_str("expand_from_doc_id"),
            redirect_taxonomy_node_id=_opt_str("redirect_taxonomy_node_id"),
            include_episodic=bool(d.get("include_episodic", False)),
            remember_choice=bool(d.get("remember_choice", False)),
            reasons=list(reasons) if reasons else [],
            timed_out=bool(d.get("timed_out", False)),
        )


# ---------------------------------------------------------------------------
# Pure decision logic
# ---------------------------------------------------------------------------


def _cfg_get(cfg: Any, attr: str, default: Any) -> Any:
    """Tolerant attribute fetch — works with both pydantic models and the
    ``SimpleNamespace`` fakes used in tests. Returns *default* on AttributeError."""
    return getattr(cfg, attr, default)


def should_pause(
    *,
    cfg: "InteractionConfig",
    results: list["RetrievalResult"],
    descend: Optional[dict[str, Any]],
    intent_verdict: Optional["IntentVerdict"],
    router_label: Optional[str],
    factual_general_swap_imminent: bool,
) -> list[PauseReason]:
    """Pure function: evaluate every trigger and return all that fired.

    Returns ``[]`` when no trigger fires OR when ``cfg.review_enabled`` is
    False. Callers must short-circuit on the empty list (the modal would
    have nothing to render).
    """
    if not _cfg_get(cfg, "review_enabled", False):
        return []

    # Phase 8.3 — when the user asked a personal question and we DID find
    # at least one episodic memory with a reasonable cosine score, skip the
    # review pause entirely. The orchestrator will route to the memory-led
    # answer template and the bot will offer to dig further conversationally.
    # The user's feedback was explicit: a follow-up about a friend they
    # already mentioned should never trigger a low-confidence modal — the
    # bot already has something to say (and should admit the limit + offer
    # to search more in the same reply).
    intent_obj = getattr(intent_verdict, "intent", None) if intent_verdict else None
    intent_value = getattr(intent_obj, "value", intent_obj)
    if intent_value == "personal" and results:
        # raw retriever score (not the cross-encoder logit). Episodic
        # cosine-derived scores are typically in [0, 1]; 0.10 is a soft
        # floor that excludes pure noise but keeps anything mildly relevant.
        has_strong_episodic = any(
            (getattr(r.chunk, "source_type", "") == "episodic" and r.score >= 0.10)
            for r in results
        )
        if has_strong_episodic:
            return []  # no pause: we have something to say

    reasons: list[PauseReason] = []
    mode = str(_cfg_get(cfg, "review_mode", "smart_auto"))

    # ALWAYS mode is additive: every other signal still gets evaluated so
    # the modal can surface them, but the pause itself is guaranteed.
    if mode == "always":
        reasons.append(PauseReason.ALWAYS_MODE)

    # --- 1. SCORE_FLOOR -----------------------------------------------------
    score_floor = float(_cfg_get(cfg, "review_score_floor", -3.0))
    if results:
        # Prefer rerank_score (set when a reranker ran), fall back to the
        # raw retriever score for retrievers that don't rerank.
        rerank_scores = [
            r.rerank_score if r.rerank_score is not None else r.score
            for r in results
        ]
        max_score = max(rerank_scores)
        if max_score < score_floor:
            reasons.append(PauseReason.SCORE_FLOOR)

    # --- 2. AMBIGUITY_DELTA -------------------------------------------------
    ambiguity_delta = float(_cfg_get(cfg, "review_ambiguity_delta", 0.4))
    if len(results) >= 2:
        sorted_scores = sorted(
            (
                r.rerank_score if r.rerank_score is not None else r.score
                for r in results
            ),
            reverse=True,
        )
        spread = sorted_scores[0] - sorted_scores[1]
        if spread < ambiguity_delta:
            reasons.append(PauseReason.AMBIGUITY_DELTA)

    # --- 3. BRANCH_THRESHOLD ------------------------------------------------
    branch_threshold = int(_cfg_get(cfg, "review_branch_threshold", 2))
    if descend:
        stats = descend.get("stats") if isinstance(descend, dict) else None
        if isinstance(stats, dict):
            leaves_picked = stats.get("leaves_picked")
            if isinstance(leaves_picked, int) and leaves_picked > branch_threshold:
                reasons.append(PauseReason.BRANCH_THRESHOLD)

    # --- 4. INTENT_UNCLEAR --------------------------------------------------
    if intent_verdict is not None:
        intent_obj = getattr(intent_verdict, "intent", None)
        intent_value = getattr(intent_obj, "value", intent_obj)
        if intent_value == "unclear":
            reasons.append(PauseReason.INTENT_UNCLEAR)

    # --- 5. ROUTER_AMBIGUOUS ------------------------------------------------
    if router_label is not None and str(router_label).lower() == "ambiguous":
        reasons.append(PauseReason.ROUTER_AMBIGUOUS)

    # --- 6. FACTUAL_GENERAL_SWAP --------------------------------------------
    if factual_general_swap_imminent:
        reasons.append(PauseReason.FACTUAL_GENERAL_SWAP)

    return reasons


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------


_SNIPPET_MAX_CHARS = 240


def _result_to_source_dict(result: "RetrievalResult") -> dict[str, Any]:
    """Project a :class:`RetrievalResult` to the wire-friendly dict shape
    the frontend modal expects. Snapshots up to 240 chars of the chunk text."""
    chunk = result.chunk
    raw_text = chunk.text or ""
    snippet = raw_text[:_SNIPPET_MAX_CHARS]
    has_math = False
    meta = getattr(chunk, "metadata", None) or {}
    if isinstance(meta, dict):
        has_math = bool(meta.get("has_math", False))
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title or "",
        "section": chunk.section or "",
        "source_type": chunk.source_type or "document",
        "score": float(result.score) if result.score is not None else None,
        "rerank_score": (
            float(result.rerank_score)
            if result.rerank_score is not None
            else None
        ),
        "snippet": snippet,
        "has_math": has_math,
    }


def build_review_payload(
    *,
    turn_id: str,
    reasons: list[PauseReason],
    results: list["RetrievalResult"],
    descend: Optional[dict[str, Any]],
    intent_verdict: Optional["IntentVerdict"],
    router_label: Optional[str],
    retrieval_query: str,
    original_question: str,
    clue: Optional[str],
    rephrasings: list[str],
    timeout_s: float,
) -> ReviewRequired:
    """Pure builder. Truncates each chunk snippet to 240 chars."""
    sources = [_result_to_source_dict(r) for r in results]
    intent_value = "unclear"
    if intent_verdict is not None:
        intent_obj = getattr(intent_verdict, "intent", None)
        raw = getattr(intent_obj, "value", intent_obj)
        if raw is not None:
            intent_value = str(raw)
    return ReviewRequired(
        turn_id=turn_id,
        reasons=[r.value for r in reasons],
        sources=sources,
        clue=clue,
        intent=intent_value,
        router_label=router_label,
        retrieval_query=retrieval_query,
        original_question=original_question,
        rephrasings=list(rephrasings),
        taxonomy_descend=descend if isinstance(descend, dict) else None,
        timeout_s=timeout_s,
    )


# ---------------------------------------------------------------------------
# Rephrasings (best-effort, one LLM call)
# ---------------------------------------------------------------------------


_REPHRASE_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rephrase.md"

# Default prompt used when prompts/rephrase.md is missing (the prompts agent
# adds the file in parallel; this fallback keeps maybe_pause working in
# the meantime).
_DEFAULT_REPHRASE_PROMPT = (
    "Suggest 3 alternative phrasings of this question, one per line, no "
    "numbering. Keep the meaning the same but vary the vocabulary so a "
    "retriever sees different surface forms.\n\n"
    "Question: {question}\n\n"
    "Alternative phrasings:"
)

# Strip leading bullets / numbering: "- foo", "* foo", "1. foo", "2) foo"
_RE_LEADING_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _load_rephrase_template() -> str:
    """Load prompts/rephrase.md; fall back to the inline default if missing."""
    try:
        return _REPHRASE_PROMPT_PATH.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return _DEFAULT_REPHRASE_PROMPT


def generate_rephrasings(
    *,
    llm: Optional["LLMProvider"],
    question: str,
    n: int = 3,
) -> list[str]:
    """Best-effort: produce up to *n* alternative phrasings via one LLM call.

    Returns ``[]`` on any error, when *llm* is ``None``, or when *llm* lacks
    a ``.complete`` method. Never raises. Lines are stripped of leading
    bullets / numbering and the result is capped at *n* entries.
    """
    if llm is None or n <= 0:
        return []
    if not hasattr(llm, "complete"):
        return []

    template = _load_rephrase_template()
    try:
        prompt = template.format(question=question)
    except (KeyError, IndexError):
        # Template uses an unexpected placeholder — degrade to the default.
        prompt = _DEFAULT_REPHRASE_PROMPT.format(question=question)

    try:
        raw = llm.complete(prompt, temperature=0.7, max_tokens=200)
    except Exception as exc:  # noqa: BLE001 — best-effort path
        logger.debug("generate_rephrasings: LLM call failed: %s", exc)
        return []

    if not raw:
        return []

    out: list[str] = []
    for line in str(raw).splitlines():
        cleaned = _RE_LEADING_BULLET.sub("", line).strip()
        # Drop wrapping quotes if any
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
            cleaned = cleaned[1:-1].strip()
        if cleaned:
            out.append(cleaned)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# The blocking call site
# ---------------------------------------------------------------------------


def maybe_pause(
    *,
    cfg: "InteractionConfig",
    results: list["RetrievalResult"],
    descend: Optional[dict[str, Any]],
    intent_verdict: Optional["IntentVerdict"],
    router_label: Optional[str],
    factual_general_swap_imminent: bool,
    clue: Optional[str],
    question: str,
    retrieval_query: str,
    user_id: str,
    session_id: Optional[str],
    turn_id: Optional[str] = None,
    progress: Optional[Callable[[str, dict[str, Any]], None]] = None,
    store: Optional["InteractionStore"] = None,
    llm: Optional["LLMProvider"] = None,
) -> ReviewDecision:
    """Decide whether to pause and (if so) block on the user's decision.

    See module docstring. Default-off contract: when ``cfg.review_enabled``
    is False the function returns ``ReviewDecision(action="continue")``
    immediately, emits NO progress events, and does NOT touch the store.
    """
    # --- Default-off guard --------------------------------------------------
    if not _cfg_get(cfg, "review_enabled", False):
        return ReviewDecision(action="continue", reasons=[])

    reasons = should_pause(
        cfg=cfg,
        results=results,
        descend=descend,
        intent_verdict=intent_verdict,
        router_label=router_label,
        factual_general_swap_imminent=factual_general_swap_imminent,
    )

    if not reasons:
        return ReviewDecision(action="continue", reasons=[])

    # --- Allocate turn_id ---------------------------------------------------
    if not turn_id:
        turn_id = uuid.uuid4().hex

    timeout_s = float(_cfg_get(cfg, "review_timeout_s", 90.0))

    # --- Optional rephrasings ----------------------------------------------
    rephrasings: list[str] = []
    rephrasings_enabled = bool(_cfg_get(cfg, "rephrasings_enabled", False))
    triggers_for_rephrase = {PauseReason.SCORE_FLOOR, PauseReason.AMBIGUITY_DELTA}
    if rephrasings_enabled and any(r in triggers_for_rephrase for r in reasons):
        rephrasings = generate_rephrasings(llm=llm, question=question, n=3)

    # --- Build the payload --------------------------------------------------
    payload = build_review_payload(
        turn_id=turn_id,
        reasons=reasons,
        results=results,
        descend=descend,
        intent_verdict=intent_verdict,
        router_label=router_label,
        retrieval_query=retrieval_query,
        original_question=question,
        clue=clue,
        rephrasings=rephrasings,
        timeout_s=timeout_s,
    )

    # --- Emit review_required ----------------------------------------------
    if progress is not None:
        try:
            progress("review_required", payload.to_dict())
        except Exception as exc:  # noqa: BLE001 — progress is best-effort
            logger.debug("review_required progress emit failed: %s", exc)

    # --- Register + wait ----------------------------------------------------
    reasons_str = [r.value for r in reasons]

    if store is None:
        # No store configured — degrade gracefully (treat as immediate
        # timeout). The orchestrator's wiring is expected to always pass a
        # store; this branch keeps us robust against misconfiguration.
        decision = ReviewDecision(
            action="continue",
            reasons=reasons_str,
            timed_out=True,
        )
    else:
        store.create_turn(turn_id, sources_snapshot=list(payload.sources))
        raw_decision = store.wait_for_decision(turn_id, timeout_s=timeout_s)
        # Clean up the entry regardless of how we exited — the reaper would
        # eventually drop it, but eager removal keeps the store small under
        # heavy load.
        try:
            store.remove(turn_id)
        except Exception:  # noqa: BLE001
            pass

        if raw_decision is None:
            decision = ReviewDecision(
                action="continue",
                reasons=reasons_str,
                timed_out=True,
            )
        else:
            decision = ReviewDecision.from_dict(raw_decision, reasons=reasons_str)

    # --- Emit review_resolved ----------------------------------------------
    if progress is not None:
        try:
            progress(
                "review_resolved",
                {
                    "action": decision.action,
                    "timed_out": decision.timed_out,
                    "reasons": list(decision.reasons),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("review_resolved progress emit failed: %s", exc)

    return decision
