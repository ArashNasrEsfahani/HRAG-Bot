"""ProfileStore: CRUD over the structured `preferences` table.

The rendered string returned by :meth:`render` is dropped verbatim into the
answer prompt via the ``{user_profile}`` placeholder declared in
``src/hrag/prompts/answer.md``. Keep the rendering deterministic and
short — it appears in every answer prompt so token cost matters.

Schema (see ``src/hrag/db/schema.sql:68``): pref_id, user_id, polarity,
topic, value, confidence, last_updated, source_session_id. Phase 3
migration adds a UNIQUE index on (user_id, topic, polarity) so the upsert
can use SQLite's INSERT ... ON CONFLICT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hrag.db.connection import Database


_POLARITIES = ("like", "dislike", "fact", "style")
_POLARITY_LABELS = {
    "fact": "Facts",
    "style": "Style preferences",
    "like": "Likes",
    "dislike": "Dislikes",
}
_RENDER_ORDER = ("fact", "style", "like", "dislike")


@dataclass
class Preference:
    pref_id: int
    user_id: str
    polarity: str
    topic: str
    value: str
    confidence: float
    last_updated: str
    source_session_id: Optional[str]


class ProfileStore:
    """Upsert and render structured per-user preferences."""

    def __init__(self, db: "Database") -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert(
        self,
        user_id: str,
        polarity: str,
        topic: str,
        value: str,
        confidence: float = 1.0,
        source_session_id: Optional[str] = None,
    ) -> int:
        """Insert or update a preference row keyed by (user_id, topic, polarity).

        Returns the pref_id of the affected row. confidence is clamped to
        [0.0, 1.0]; polarity is validated against the schema's enum.
        """
        if polarity not in _POLARITIES:
            raise ValueError(
                f"polarity must be one of {_POLARITIES}; got {polarity!r}"
            )
        if not topic.strip():
            raise ValueError("topic must be non-empty")

        confidence = max(0.0, min(1.0, float(confidence)))

        # SQLite UPSERT — the UNIQUE index on (user_id, topic, polarity) is
        # created by db/migrations.py.
        self._db.execute(
            """
            INSERT INTO preferences
                (user_id, polarity, topic, value, confidence, source_session_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, topic, polarity) DO UPDATE SET
                value = excluded.value,
                confidence = excluded.confidence,
                last_updated = datetime('now'),
                source_session_id = excluded.source_session_id
            """,
            (user_id, polarity, topic, value, confidence, source_session_id),
        )
        self._db.commit()

        cur = self._db.execute(
            "SELECT pref_id FROM preferences "
            "WHERE user_id = ? AND topic = ? AND polarity = ?",
            (user_id, topic, polarity),
        )
        row = cur.fetchone()
        return int(row["pref_id"])

    def delete(self, user_id: str, pref_id: int) -> bool:
        cur = self._db.execute(
            "DELETE FROM preferences WHERE pref_id = ? AND user_id = ?",
            (pref_id, user_id),
        )
        self._db.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def list_all(self, user_id: str) -> list[Preference]:
        cur = self._db.execute(
            "SELECT pref_id, user_id, polarity, topic, value, confidence, "
            "last_updated, source_session_id FROM preferences "
            "WHERE user_id = ? "
            "ORDER BY confidence DESC, last_updated DESC, pref_id ASC",
            (user_id,),
        )
        return [_row_to_pref(r) for r in cur.fetchall()]

    def render(
        self,
        user_id: str,
        max_items: int = 12,
        min_confidence: float = 0.5,
    ) -> str:
        """Return a short formatted profile string for the answer prompt.

        Groups by polarity (Facts → Style → Likes → Dislikes), each as one
        line with semicolon-separated ``topic: value`` entries. Drops items
        below ``min_confidence`` and truncates to ``max_items`` total. Returns
        ``"(no profile yet)"`` if nothing qualifies.
        """
        prefs = [p for p in self.list_all(user_id) if p.confidence >= min_confidence]
        prefs = prefs[:max_items]
        if not prefs:
            return "(no profile yet)"

        grouped: dict[str, list[str]] = {p: [] for p in _RENDER_ORDER}
        for p in prefs:
            if p.polarity not in grouped:
                continue
            entry = f"{p.topic}: {p.value}" if p.value else p.topic
            grouped[p.polarity].append(entry)

        lines: list[str] = []
        for polarity in _RENDER_ORDER:
            entries = grouped[polarity]
            if not entries:
                continue
            lines.append(f"{_POLARITY_LABELS[polarity]}: " + "; ".join(entries))
        return "\n".join(lines) if lines else "(no profile yet)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_pref(row) -> Preference:
    return Preference(
        pref_id=int(row["pref_id"]),
        user_id=row["user_id"],
        polarity=row["polarity"],
        topic=row["topic"],
        value=row["value"] or "",
        confidence=float(row["confidence"]),
        last_updated=row["last_updated"],
        source_session_id=row["source_session_id"],
    )
