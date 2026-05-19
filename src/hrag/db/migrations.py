"""Idempotent post-schema migrations.

`schema.sql` declares tables; this module adds indexes that newer phases
need but that don't change the table shape. Each statement uses
`IF NOT EXISTS` so calling `run_migrations` on an already-current DB is
a no-op. Called from `init_db` after `init_schema`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.db.connection import Database


_MIGRATIONS: tuple[str, ...] = (
    # Phase 3: episodic memory scans by (user_id, source_type, excluded).
    "CREATE INDEX IF NOT EXISTS idx_chunks_user_source_excluded "
    "ON chunks(user_id, source_type, excluded)",

    # Phase 3: ProfileStore.upsert keys on (user_id, topic, polarity).
    # UNIQUE because the upsert uses INSERT ... ON CONFLICT(user_id, topic, polarity).
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_preferences_user_topic_polarity "
    "ON preferences(user_id, topic, polarity)",
)


def run_migrations(db: "Database") -> None:
    with db.conn:
        for stmt in _MIGRATIONS:
            db.conn.execute(stmt)
