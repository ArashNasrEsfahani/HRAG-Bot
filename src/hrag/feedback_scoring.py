"""Phase 9.15 — Feedback-weighted re-ranking helpers.

Per-chunk historical feedback scores are computed from the feedback table
(joined to messages.metadata for the source chunk IDs) and applied as a
nudge on top of the cross-encoder rerank score:

    final_score = rerank_score + weight * feedback_score

All public symbols are designed to be called from the orchestrator rerank
pipeline. Heavy imports are lazy so this module can be imported without
any optional dep installed.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.db.connection import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure scoring function (no DB, no state)
# ---------------------------------------------------------------------------


def apply_feedback_to_rerank_score(
    rerank_score: float,
    feedback_score: float,
    *,
    weight: float = 0.3,
) -> float:
    """Return the feedback-adjusted rerank score.

    Parameters
    ----------
    rerank_score:
        The raw logit from the cross-encoder (or any reranker).
    feedback_score:
        EMA(thumbs_up - thumbs_down) in [-1, +1].
        0.0 = no feedback history for this chunk.
    weight:
        Multiplier on feedback_score (default 0.3 — nudge, not steamroller).

    Returns
    -------
    float — rerank_score + weight * feedback_score
    """
    return rerank_score + weight * feedback_score


# ---------------------------------------------------------------------------
# FeedbackScorer — per-turn stateful scorer (one DB query for the whole batch)
# ---------------------------------------------------------------------------


class FeedbackScorer:
    """Compute per-chunk EMA feedback scores from the thumbs-up/down table.

    Intended lifetime: one chat() turn. Instantiate → call score_many() once
    → throw away. Re-using across turns is fine but the cache is not
    invalidated automatically.

    Parameters
    ----------
    db:
        A ``hrag.db.connection.Database`` instance.
    alpha:
        EMA decay rate in [0, 1]. Higher = more weight on recent feedback.
        Default 0.3 (smooth, long memory).
    neutral_default:
        Returned for chunks with no feedback history at all. Default 0.0.
    """

    def __init__(
        self,
        db: "Database",
        *,
        alpha: float = 0.3,
        neutral_default: float = 0.0,
    ) -> None:
        self._db = db
        self._alpha = alpha
        self._neutral = neutral_default
        # Lazy-populated on first call to score_many(); cleared by reset().
        self._cache: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, chunk_id: str) -> float:
        """Return the EMA feedback score for a single chunk.

        Internally calls score_many() so it triggers the same single SQL
        query (and populates the cache for subsequent calls).
        """
        result = self.score_many([chunk_id])
        return result.get(chunk_id, self._neutral)

    def score_many(self, chunk_ids: list[str]) -> dict[str, float]:
        """Return EMA feedback scores for a batch of chunk IDs.

        Issues exactly ONE SQL query across the whole feedback table, parses
        messages.metadata in Python to find which chunk_ids appear in each
        message's sources, then accumulates EMA(+1 if up, -1 if down).

        Missing entries (no feedback history) are silently omitted from the
        returned dict; callers should use `.get(chunk_id, 0.0)`.

        Contract (Phase 8 contract 29):
        - rows where metadata IS NULL → skipped silently
        - rows where metadata is valid JSON but has no ``sources`` key → skipped
        - rows where metadata is malformed JSON → logged at DEBUG, skipped
        """
        if self._cache is None:
            self._cache = self._build_cache()

        return {
            cid: self._cache[cid]
            for cid in chunk_ids
            if cid in self._cache
        }

    def reset(self) -> None:
        """Invalidate the internal cache (for testing or multi-turn reuse)."""
        self._cache = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_cache(self) -> dict[str, float]:
        """Single SQL query + in-memory EMA aggregation."""
        try:
            rows = self._db.execute(
                """
                SELECT m.metadata, f.rating, f.created_at
                FROM feedback f
                JOIN messages m ON CAST(f.message_id AS TEXT) = CAST(m.message_id AS TEXT)
                WHERE m.metadata IS NOT NULL
                ORDER BY f.created_at ASC
                """
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("FeedbackScorer: DB query failed (%s); returning empty cache.", exc)
            return {}

        # EMA state per chunk: {chunk_id: current_ema}
        ema: dict[str, float] = {}
        alpha = self._alpha

        for row in rows:
            raw_meta = row[0] if isinstance(row, (list, tuple)) else row["metadata"]
            rating_val = row[1] if isinstance(row, (list, tuple)) else row["rating"]

            # Parse metadata JSON — honour Phase 8 contract 29.
            if raw_meta is None:
                continue
            try:
                meta = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.debug(
                    "FeedbackScorer: malformed metadata JSON (%s); skipping row.", exc
                )
                continue

            if not isinstance(meta, dict):
                continue

            sources = meta.get("sources")
            if not sources:
                # Try the phase8 nested shape: {"phase8": {..., "selected_chunk_ids": [...]}}
                phase8 = meta.get("phase8")
                if isinstance(phase8, dict):
                    sources = phase8.get("selected_chunk_ids") or phase8.get("sources")

            if not sources or not isinstance(sources, list):
                continue

            # +1 for thumbs_up, -1 for thumbs_down
            try:
                signal = float(rating_val)
            except (TypeError, ValueError):
                continue
            # Normalise to [-1, +1]: the schema stores +1 / -1 / 0.
            # We skip neutral (0) rows — they carry no signal.
            if signal == 0:
                continue
            signal = max(-1.0, min(1.0, signal))

            for cid in sources:
                if not isinstance(cid, str):
                    continue
                if cid in ema:
                    ema[cid] = alpha * signal + (1.0 - alpha) * ema[cid]
                else:
                    # First observation: seed the EMA with the signal itself.
                    ema[cid] = signal

        return ema
