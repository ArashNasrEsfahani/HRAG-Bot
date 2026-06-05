"""Migrations create the indexes Phase 3 needs and the UPSERT works."""

from __future__ import annotations


def test_run_migrations_creates_indexes(tmp_db):
    from hrag.db.migrations import run_migrations

    run_migrations(tmp_db)
    cur = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    )
    names = {row["name"] for row in cur.fetchall()}
    assert "idx_chunks_user_source_excluded" in names
    assert "idx_preferences_user_topic_polarity" in names


def test_run_migrations_is_idempotent(tmp_db):
    from hrag.db.migrations import run_migrations

    run_migrations(tmp_db)
    run_migrations(tmp_db)
    cur = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE name='idx_preferences_user_topic_polarity'"
    )
    assert cur.fetchone() is not None


def test_unique_index_blocks_duplicate_polarity(tmp_db):
    """The unique index on (user_id, topic, polarity) backs the ON CONFLICT path."""
    from hrag.db.migrations import run_migrations
    import sqlite3

    run_migrations(tmp_db)
    tmp_db.execute(
        "INSERT INTO preferences (user_id, polarity, topic, value, confidence) "
        "VALUES (?, ?, ?, ?, ?)",
        ("default", "fact", "occupation", "engineer", 0.9),
    )
    tmp_db.commit()
    try:
        tmp_db.execute(
            "INSERT INTO preferences (user_id, polarity, topic, value, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            ("default", "fact", "occupation", "scientist", 0.9),
        )
        tmp_db.commit()
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised, "duplicate (user_id, topic, polarity) should violate unique index"


# ---------------------------------------------------------------------------
# Phase 13.1 — page + chapter column migration
# ---------------------------------------------------------------------------

def test_page_chapter_columns_added_by_migration(tmp_db):
    """run_migrations must add page + chapter columns to chunks."""
    from hrag.db.migrations import run_migrations

    run_migrations(tmp_db)

    # Check via PRAGMA table_info
    cur = tmp_db.execute("PRAGMA table_info(chunks)")
    col_names = {row["name"] for row in cur.fetchall()}
    assert "page" in col_names, "page column must exist after migration"
    assert "chapter" in col_names, "chapter column must exist after migration"


def test_page_chapter_migration_idempotent(tmp_db):
    """Running migrations twice must not raise an error."""
    from hrag.db.migrations import run_migrations

    run_migrations(tmp_db)
    run_migrations(tmp_db)  # second run — must be a no-op

    cur = tmp_db.execute("PRAGMA table_info(chunks)")
    col_names = {row["name"] for row in cur.fetchall()}
    assert "page" in col_names
    assert "chapter" in col_names


def test_doc_page_index_created(tmp_db):
    """The idx_chunks_doc_page index must exist after migration."""
    from hrag.db.migrations import run_migrations

    run_migrations(tmp_db)
    cur = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_chunks_doc_page'"
    )
    assert cur.fetchone() is not None, "idx_chunks_doc_page index must be created"


def test_pre_13_1_db_upgrades_cleanly(tmp_path):
    """Regression: a pre-13.1 DB whose `chunks` table predates the page/chapter
    columns must upgrade through the FULL bootstrap (init_schema → run_migrations)
    without raising. This reproduces the real-world failure where the
    idx_chunks_doc_page index — if created during init_schema's executescript or
    before the ALTER — aborts the whole migration with "no such column: page".
    """
    import sqlite3

    from hrag.db.connection import Database
    from hrag.db.migrations import run_migrations

    db_path = tmp_path / "legacy_store.sqlite"
    # Hand-build an OLD chunks table: no `page`, no `chapter`, no doc_page index.
    raw = sqlite3.connect(str(db_path))
    raw.executescript(
        "CREATE TABLE chunks ("
        "  chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, user_id TEXT NOT NULL,"
        "  text TEXT NOT NULL, title TEXT, section TEXT, subsection TEXT,"
        "  chunk_index INTEGER NOT NULL DEFAULT 0, token_count INTEGER NOT NULL DEFAULT 0,"
        "  source_type TEXT NOT NULL DEFAULT 'document', excluded INTEGER NOT NULL DEFAULT 0,"
        "  metadata TEXT"
        ");"
        "INSERT INTO chunks (chunk_id, doc_id, user_id, text, chunk_index) "
        "VALUES ('c0', 'd0', 'default', 'legacy body', 0);"
    )
    raw.commit()
    raw.close()

    # Full bootstrap over the legacy DB — must not raise.
    db = Database(db_path)
    db.init_schema()           # executescript of the current schema.sql
    run_migrations(db)         # adds page/chapter + idx_chunks_doc_page
    db.commit()

    cols = {row["name"] for row in db.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "page" in cols and "chapter" in cols
    idx = db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_chunks_doc_page'"
    ).fetchone()
    assert idx is not None
    # The legacy row survives with NULL page/chapter (graceful upgrade).
    row = db.execute("SELECT page, chapter FROM chunks WHERE chunk_id='c0'").fetchone()
    assert row["page"] is None and row["chapter"] is None
    db.close()
