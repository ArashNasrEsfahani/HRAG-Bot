"""Idempotent post-schema migrations.

`schema.sql` declares tables; this module adds indexes that newer phases
need but that don't change the table shape. Each statement uses
`IF NOT EXISTS` so calling `run_migrations` on an already-current DB is
a no-op. Called from `init_db` after `init_schema`.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.db.connection import Database


_MIGRATIONS: tuple[str, ...] = (
    # Phase 3: episodic memory scans by (user_id, source_type, excluded).
    "CREATE INDEX IF NOT EXISTS idx_chunks_user_source_excluded "
    "ON chunks(user_id, source_type, excluded)",

    # NOTE: the Phase-13.1 idx_chunks_doc_page index is created LATER (after the
    # ALTER TABLE that adds chunks.page) — it must not run here, or a pre-13.1
    # DB without the `page` column raises "no such column" and aborts the whole
    # migration before the column is ever added.

    # Phase 3: ProfileStore.upsert keys on (user_id, topic, polarity).
    # UNIQUE because the upsert uses INSERT ... ON CONFLICT(user_id, topic, polarity).
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_preferences_user_topic_polarity "
    "ON preferences(user_id, topic, polarity)",

    # Phase 9.12: cross-chunk triple deduplication index. Sibling table to
    # kg_triple_cache (which is keyed on chunk text); this one is keyed on
    # the canonical (subject|relation|object) triple itself. freq counts
    # supporting chunks; the per-chunk passage->phrase edges still flow
    # through kg_edges so Chroma + SQLite + NetworkX stay in sync.
    "CREATE TABLE IF NOT EXISTS kg_canonical_triples ("
    "    canonical_key       TEXT PRIMARY KEY,"
    "    subject             TEXT NOT NULL,"
    "    relation            TEXT NOT NULL,"
    "    object              TEXT NOT NULL,"
    "    first_seen_chunk_id TEXT NOT NULL,"
    "    freq                INTEGER NOT NULL DEFAULT 1,"
    "    created_at          TEXT NOT NULL DEFAULT (datetime('now'))"
    ")",
    "CREATE INDEX IF NOT EXISTS idx_kg_canonical_triples_subject "
    "ON kg_canonical_triples(subject)",

    # Phase 9.14: Contextual Retrieval (Anthropic technique). Per-chunk
    # cache for the LLM-generated "context line" that gets prepended to
    # chunk.embedding_text at ingest. Keyed by SHA-256 over
    # (chunk_text, doc_hash, model_name) so a re-ingest of the same doc
    # under the same model reuses the cache; changing the model invalidates
    # transparently. Independent of kg_triple_cache.
    "CREATE TABLE IF NOT EXISTS ingest_context_cache ("
    "    cache_key       TEXT PRIMARY KEY,"
    "    model_name      TEXT NOT NULL,"
    "    context_line    TEXT NOT NULL,"
    "    created_at      TEXT NOT NULL DEFAULT (datetime('now'))"
    ")",
    "CREATE INDEX IF NOT EXISTS idx_ingest_context_cache_model "
    "ON ingest_context_cache(model_name)",

    # Phase 9.9: rerank-fallback telemetry table. Logged once per turn where
    # the cross-encoder zero-filter trips and the unreranked top-k is used as
    # a fallback. Surfaces in `hrag feedback-stats`.
    "CREATE TABLE IF NOT EXISTS rerank_fallback_events ("
    "    event_id          TEXT PRIMARY KEY,"
    "    turn_id           TEXT NOT NULL,"
    "    session_id        TEXT,"
    "    user_id           TEXT,"
    "    query             TEXT NOT NULL,"
    "    dropped_chunk_ids TEXT,"
    "    created_at        TEXT NOT NULL DEFAULT (datetime('now'))"
    ")",
    "CREATE INDEX IF NOT EXISTS idx_rerank_fallback_session "
    "ON rerank_fallback_events(session_id)",
)


def run_migrations(db: "Database") -> None:
    with db.conn:
        for stmt in _MIGRATIONS:
            db.conn.execute(stmt)

    # Phase 8: add metadata column to messages (JSON-encoded review decision).
    # ALTER TABLE ADD COLUMN is idempotent via try/except on the duplicate-column error.
    try:
        with db.conn:
            db.conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise

    # Phase 12: per-node keywords for hybrid taxonomy routing (JSON list TEXT).
    # Additive, idempotent. Absent column on pre-Phase-12 DBs is tolerated by
    # _row_to_node (treats missing/NULL as []). The table itself may not exist
    # on pre-Phase-3 schemas — swallow that case.
    try:
        with db.conn:
            db.conn.execute("ALTER TABLE kg_taxonomy_nodes ADD COLUMN keywords TEXT")
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if "duplicate column name" not in msg and "no such table" not in msg:
            raise

    # Phase 10 fix: backfill kg_taxonomy_nodes.is_leaf for nodes wrongly
    # stored as is_leaf=0 by the pre-fix `POST /api/taxonomy/nodes` default.
    # A node with NO children is structurally a leaf, period — flip it.
    # Conversely, any node that has at least one child must be is_leaf=0.
    # The UPDATE is idempotent (setting a value to itself is a no-op) and
    # safe to run every startup. The `parent_id IS NOT NULL` guard preserves
    # the root row's existing is_leaf flag (roots are special).
    try:
        with db.conn:
            db.conn.execute(
                "UPDATE kg_taxonomy_nodes SET is_leaf = 1 "
                "WHERE parent_id IS NOT NULL "
                "AND node_id NOT IN ("
                "    SELECT DISTINCT parent_id FROM kg_taxonomy_nodes "
                "    WHERE parent_id IS NOT NULL"
                ")"
            )
            db.conn.execute(
                "UPDATE kg_taxonomy_nodes SET is_leaf = 0 "
                "WHERE node_id IN ("
                "    SELECT DISTINCT parent_id FROM kg_taxonomy_nodes "
                "    WHERE parent_id IS NOT NULL"
                ")"
            )
    except sqlite3.OperationalError:
        # Pre-Phase-3 schemas without kg_taxonomy_nodes — skip silently.
        pass

    # Phase 13.1: add page + chapter columns to chunks (additive, idempotent).
    # page INTEGER — 1-based source page, PDF only; NULL for all other formats.
    # chapter TEXT — normalised, forward-filled heading label.
    for col_ddl in (
        "ALTER TABLE chunks ADD COLUMN page INTEGER",
        "ALTER TABLE chunks ADD COLUMN chapter TEXT",
    ):
        try:
            with db.conn:
                db.conn.execute(col_ddl)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise

    # Now that chunks.page is guaranteed to exist, create its lookup index.
    # (Must come after the ALTER above — see the NOTE in _MIGRATIONS.)
    with db.conn:
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks(doc_id, page)"
        )
