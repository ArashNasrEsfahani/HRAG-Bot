"""SessionAutoExtractor: opt-in conversation extraction at session close.

Runs PreferenceExtractor over a closed session's messages in a daemon
thread, upserts every candidate above ``min_confidence`` to ProfileStore.
Never blocks the user's exit path; failures are logged and swallowed.

Gated by ``memory.auto_extract: true`` in config.yaml (default false).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.db.connection import Database
    from hrag.memory.extractor import PreferenceExtractor
    from hrag.memory.profile import ProfileStore


logger = logging.getLogger(__name__)


class SessionAutoExtractor:
    def __init__(
        self,
        db: "Database",
        extractor: "PreferenceExtractor",
        profile_store: "ProfileStore",
        min_confidence: float = 0.7,
    ) -> None:
        self._db = db
        self._extractor = extractor
        self._profile_store = profile_store
        self._min_confidence = float(min_confidence)

    def on_session_close(
        self,
        user_id: str,
        session_id: str,
        *,
        block: bool = False,
    ) -> threading.Thread:
        """Spawn a daemon thread that mines the session and upserts preferences.

        Returns the thread handle. ``block=True`` joins it before returning —
        only used by tests.
        """
        thread = threading.Thread(
            target=self._run,
            args=(user_id, session_id),
            name=f"auto-extract-{session_id[:8]}",
            daemon=True,
        )
        thread.start()
        if block:
            thread.join()
        return thread

    # ------------------------------------------------------------------

    def _run(self, user_id: str, session_id: str) -> None:
        try:
            conversation = self._load_messages(user_id, session_id)
            if not conversation:
                return
            candidates = self._extractor.extract(conversation)
            applied = 0
            for cand in candidates:
                if cand.confidence < self._min_confidence:
                    continue
                self._profile_store.upsert(
                    user_id=user_id,
                    polarity=cand.polarity,
                    topic=cand.topic,
                    value=cand.value,
                    confidence=cand.confidence,
                    source_session_id=session_id,
                )
                applied += 1
            logger.info(
                "SessionAutoExtractor: session=%s upserted %d/%d candidates",
                session_id, applied, len(candidates),
            )
        except Exception as exc:  # noqa: BLE001 - daemon thread, never propagate
            logger.warning(
                "SessionAutoExtractor: failed for session=%s: %s",
                session_id, exc,
            )

    def _load_messages(
        self,
        user_id: str,
        session_id: str,
    ) -> list[tuple[str, str]]:
        cur = self._db.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? AND user_id = ? "
            "ORDER BY message_id ASC",
            (session_id, user_id),
        )
        return [(row["role"], row["content"]) for row in cur.fetchall()]
