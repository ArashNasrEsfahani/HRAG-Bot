"""Tests for hrag.db.connection — Database, init_db, schema, ensure_user."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import hrag.db.connection as _conn_mod
from hrag.db.connection import Database, init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_TABLES = {
    "users",
    "documents",
    "chunks",
    "sessions",
    "messages",
    "preferences",
    "kg_nodes",
    "kg_edges",
}


def _table_names(db: Database) -> set[str]:
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r["name"] for r in rows}


# ---------------------------------------------------------------------------
# Fixture: isolated DB per test (avoid singleton bleed-through)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the module-level singleton is cleared around every test."""
    _conn_mod._db_singleton = None
    yield
    _conn_mod._db_singleton = None


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def test_init_db_creates_all_tables(tmp_path: Path) -> None:
    """init_db must create every table defined in schema.sql."""
    db = init_db(tmp_path / "store.sqlite")
    tables = _table_names(db)
    assert _EXPECTED_TABLES <= tables, f"Missing tables: {_EXPECTED_TABLES - tables}"
    db.close()


def test_init_db_idempotent(tmp_path: Path) -> None:
    """Calling init_schema twice must not raise (IF NOT EXISTS guards)."""
    db = Database(tmp_path / "store.sqlite")
    db.init_schema()
    db.init_schema()   # second call — should be a no-op
    tables = _table_names(db)
    assert _EXPECTED_TABLES <= tables
    db.close()


# ---------------------------------------------------------------------------
# ensure_user idempotency
# ---------------------------------------------------------------------------

def test_ensure_user_creates_row(tmp_path: Path) -> None:
    """ensure_user must insert a row into the users table."""
    _conn_mod._db_singleton = None
    db = init_db(tmp_path / "store.sqlite", default_user_id="alice")
    row = db.execute(
        "SELECT user_id, display_name FROM users WHERE user_id = ?", ("alice",)
    ).fetchone()
    assert row is not None
    assert row["user_id"] == "alice"
    db.close()


def test_ensure_user_is_idempotent(tmp_path: Path) -> None:
    """Calling ensure_user twice with the same id must not raise or duplicate."""
    _conn_mod._db_singleton = None
    db = init_db(tmp_path / "store.sqlite", default_user_id="bob")
    db.ensure_user("bob")          # second call — INSERT OR IGNORE
    db.ensure_user("bob", "Bob")   # third call with display_name — still idempotent
    db.commit()

    rows = db.execute(
        "SELECT count(*) as cnt FROM users WHERE user_id = ?", ("bob",)
    ).fetchone()
    assert rows["cnt"] == 1
    db.close()


def test_ensure_user_display_name_default(tmp_path: Path) -> None:
    """When display_name is None, it defaults to user_id."""
    _conn_mod._db_singleton = None
    db = Database(tmp_path / "store.sqlite")
    db.init_schema()
    db.ensure_user("carol")
    db.commit()

    row = db.execute(
        "SELECT display_name FROM users WHERE user_id = ?", ("carol",)
    ).fetchone()
    assert row["display_name"] == "carol"
    db.close()


def test_ensure_multiple_distinct_users(tmp_path: Path) -> None:
    """Multiple different users can coexist in the table."""
    _conn_mod._db_singleton = None
    db = Database(tmp_path / "store.sqlite")
    db.init_schema()
    for uid in ("u1", "u2", "u3"):
        db.ensure_user(uid)
    db.commit()

    count = db.execute("SELECT count(*) as cnt FROM users").fetchone()["cnt"]
    assert count == 3
    db.close()


# ---------------------------------------------------------------------------
# PRAGMA foreign_keys
# ---------------------------------------------------------------------------

def test_foreign_keys_pragma_is_on(tmp_path: Path) -> None:
    """Database.__init__ must enable foreign key enforcement."""
    _conn_mod._db_singleton = None
    db = Database(tmp_path / "fk_test.sqlite")
    row = db.execute("PRAGMA foreign_keys").fetchone()
    # sqlite3.Row can be indexed by column name or position
    fk_value = row[0]
    assert fk_value == 1, "PRAGMA foreign_keys should be ON (1)"
    db.close()


def test_foreign_key_violation_raises(tmp_path: Path) -> None:
    """Inserting a document with a non-existent user_id must raise IntegrityError."""
    _conn_mod._db_singleton = None
    db = Database(tmp_path / "store.sqlite")
    db.init_schema()
    db.commit()

    with pytest.raises(sqlite3.IntegrityError):
        with db.conn:
            db.execute(
                """
                INSERT INTO documents (doc_id, user_id, source_path, source_type)
                VALUES ('d1', 'nonexistent_user', '/fake/path.txt', 'document')
                """
            )
    db.close()
