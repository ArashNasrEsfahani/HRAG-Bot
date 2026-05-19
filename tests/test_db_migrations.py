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
