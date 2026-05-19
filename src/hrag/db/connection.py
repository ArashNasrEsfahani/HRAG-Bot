"""SQLite connection management and one-shot init."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """Thin wrapper around sqlite3 with schema bootstrap and a few helpers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.conn:
            self.conn.executescript(sql)

    def ensure_user(self, user_id: str, display_name: Optional[str] = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO users(user_id, display_name) VALUES (?, ?)",
                (user_id, display_name or user_id),
            )

    def execute(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def executemany(self, sql: str, params: Iterable[Iterable]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, [tuple(p) for p in params])

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
        # If we were the module-level singleton, clear it so a subsequent
        # init_db() in the same process rebuilds a live connection rather
        # than handing back this dead one.
        global _db_singleton
        if _db_singleton is self:
            _db_singleton = None


_db_singleton: Optional[Database] = None


def get_db(path: str | Path) -> Database:
    """Module-level singleton, keyed by first-seen path."""
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = Database(path)
    return _db_singleton


def init_db(path: str | Path, default_user_id: str = "default") -> Database:
    """Open the DB, run the schema, apply migrations, ensure the default user exists."""
    from hrag.db.migrations import run_migrations  # noqa: PLC0415 — avoid circular import

    db = get_db(path)
    db.init_schema()
    run_migrations(db)
    db.ensure_user(default_user_id)
    db.commit()
    return db
