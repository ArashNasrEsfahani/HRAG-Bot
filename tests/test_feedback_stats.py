"""Tests for Phase 6-B2: feedback-stats and feedback-export CLI commands.

Uses Click's CliRunner with a real (tmp) SQLite DB; no heavy deps needed.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path):
    """Return a Config wired to a tmp SQLite + chroma directory."""
    from hrag.config import (  # noqa: PLC0415
        Config,
        EmbeddingsConfig,
        LLMConfig,
        RetrievalConfig,
        StorageConfig,
    )

    cfg = Config(
        llm=LLMConfig(provider="ollama", model="test-model"),
        embeddings=EmbeddingsConfig(
            provider="sentence-transformers",
            model="sentence-transformers/all-mpnet-base-v2",
            dim=384,
        ),
        storage=StorageConfig(
            sqlite_path=str(tmp_path / "store.sqlite"),
            chroma_path=str(tmp_path / "chroma"),
            kg_path=str(tmp_path / "kg"),
            data_root=str(tmp_path / "data"),
        ),
        retrieval=RetrievalConfig(rerank_enabled=False, doc_scope_enabled=False),
    )
    cfg.project_root = tmp_path
    return cfg


def _patch_config(monkeypatch, cfg):
    """Monkeypatch hrag.cli.load_config to return cfg."""
    monkeypatch.setattr("hrag.cli.load_config", lambda *a, **kw: cfg)


def _seed_db(db, *, user_id: str = "default") -> None:
    """Ensure user exists and create the schema."""
    db.ensure_user(user_id)
    db.commit()


def _insert_session(db, user_id: str = "default") -> str:
    sid = "sess_" + uuid.uuid4().hex[:8]
    with db.conn:
        db.execute(
            "INSERT INTO sessions(session_id, user_id) VALUES (?, ?)",
            (sid, user_id),
        )
    db.commit()
    return sid


def _insert_message(db, session_id: str, role: str, content: str, user_id: str = "default") -> int:
    cur = db.execute(
        "INSERT INTO messages(session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
        (session_id, user_id, role, content),
    )
    db.commit()
    return cur.lastrowid


def _insert_feedback(db, message_id: int, session_id: str, rating: int, user_id: str = "default") -> str:
    fid = uuid.uuid4().hex
    with db.conn:
        db.execute(
            "INSERT INTO feedback(feedback_id, message_id, session_id, user_id, rating) "
            "VALUES (?, ?, ?, ?, ?)",
            (fid, str(message_id), session_id, user_id, rating),
        )
    db.commit()
    return fid


@pytest.fixture(autouse=True)
def _reset_db_singleton():
    import hrag.db.connection as _conn_mod  # noqa: PLC0415

    _conn_mod._db_singleton = None
    yield
    _conn_mod._db_singleton = None


# ---------------------------------------------------------------------------
# _feedback_summary unit tests (pure, no CLI)
# ---------------------------------------------------------------------------


def test_empty_feedback_table(tmp_path):
    """Empty feedback table → all counts are 0."""
    from hrag.db.connection import init_db  # noqa: PLC0415
    from hrag.cli import _feedback_summary  # noqa: PLC0415

    db = init_db(tmp_path / "store.sqlite", "default")
    stats = _feedback_summary(db)
    assert stats["thumbs_up"] == 0
    assert stats["thumbs_down"] == 0
    assert stats["total"] == 0
    assert stats["sessions"] == 0
    assert stats["top_negative"] == []
    db.close()


def test_counts_and_ratios(tmp_path):
    """Two thumbs-up + three thumbs-down → correct totals."""
    from hrag.db.connection import init_db  # noqa: PLC0415
    from hrag.cli import _feedback_summary  # noqa: PLC0415

    db = init_db(tmp_path / "store.sqlite", "default")
    _seed_db(db)
    sid = _insert_session(db)

    # Insert 5 user+assistant pairs
    for i in range(5):
        _insert_message(db, sid, "user", f"question {i}")
        asst_id = _insert_message(db, sid, "assistant", f"answer {i}")
        rating = 1 if i < 2 else -1  # 2 up, 3 down
        _insert_feedback(db, asst_id, sid, rating)

    stats = _feedback_summary(db)
    assert stats["thumbs_up"] == 2
    assert stats["thumbs_down"] == 3
    assert stats["total"] == 5
    assert stats["sessions"] == 1
    db.close()


def test_top_negative_finds_preceding_question(tmp_path):
    """Top-negative lookup must surface the user question before the bad reply."""
    from hrag.db.connection import init_db  # noqa: PLC0415
    from hrag.cli import _feedback_summary  # noqa: PLC0415

    db = init_db(tmp_path / "store.sqlite", "default")
    _seed_db(db)
    sid = _insert_session(db)

    question_text = "what is the loss function?"
    _insert_message(db, sid, "user", question_text)
    asst_id = _insert_message(db, sid, "assistant", "I don't know")
    _insert_feedback(db, asst_id, sid, -1)

    stats = _feedback_summary(db)
    assert len(stats["top_negative"]) == 1
    item = stats["top_negative"][0]
    assert item["question"] == question_text
    assert item["session_id"] == sid
    db.close()


# ---------------------------------------------------------------------------
# CLI: feedback-stats
# ---------------------------------------------------------------------------


def test_feedback_stats_empty(monkeypatch, tmp_path):
    """feedback-stats runs cleanly on an empty DB (no crash, shows 0)."""
    from hrag.cli import cli as cli_group  # noqa: PLC0415

    cfg = _make_config(tmp_path)
    _patch_config(monkeypatch, cfg)

    # Bootstrap the DB first.
    runner = CliRunner()
    res = runner.invoke(cli_group, ["init"])
    assert res.exit_code == 0, res.output

    res = runner.invoke(cli_group, ["feedback-stats"])
    assert res.exit_code == 0, res.output
    # Both counts should display as 0.
    assert "0" in res.output


def test_feedback_stats_with_data(monkeypatch, tmp_path):
    """feedback-stats correctly summarises seeded data."""
    from hrag.cli import cli as cli_group  # noqa: PLC0415
    from hrag.db.connection import init_db  # noqa: PLC0415

    cfg = _make_config(tmp_path)
    _patch_config(monkeypatch, cfg)

    runner = CliRunner()
    runner.invoke(cli_group, ["init"])

    # Seed: 2 up, 3 down.
    import hrag.db.connection as _conn_mod  # noqa: PLC0415

    db = init_db(tmp_path / "store.sqlite", "default")
    _seed_db(db)
    sid = _insert_session(db)
    for i in range(5):
        _insert_message(db, sid, "user", f"q{i}")
        aid = _insert_message(db, sid, "assistant", f"a{i}")
        _insert_feedback(db, aid, sid, 1 if i < 2 else -1)
    _conn_mod._db_singleton = None  # allow CLI to reopen same file

    res = runner.invoke(cli_group, ["feedback-stats"])
    assert res.exit_code == 0, res.output
    assert "2" in res.output   # thumbs up count
    assert "3" in res.output   # thumbs down count


# ---------------------------------------------------------------------------
# CLI: feedback-export
# ---------------------------------------------------------------------------


def test_feedback_export_up_only(monkeypatch, tmp_path):
    """--rating up writes only +1 rows."""
    from hrag.cli import cli as cli_group  # noqa: PLC0415
    from hrag.db.connection import init_db  # noqa: PLC0415
    import hrag.db.connection as _conn_mod  # noqa: PLC0415

    cfg = _make_config(tmp_path)
    _patch_config(monkeypatch, cfg)

    runner = CliRunner()
    runner.invoke(cli_group, ["init"])

    db = init_db(tmp_path / "store.sqlite", "default")
    _seed_db(db)
    sid = _insert_session(db)
    _insert_message(db, sid, "user", "good question")
    good_id = _insert_message(db, sid, "assistant", "great answer")
    _insert_feedback(db, good_id, sid, 1)
    _insert_message(db, sid, "user", "bad question")
    bad_id = _insert_message(db, sid, "assistant", "bad answer")
    _insert_feedback(db, bad_id, sid, -1)
    _conn_mod._db_singleton = None

    out = str(tmp_path / "up.jsonl")
    res = runner.invoke(cli_group, ["feedback-export", "--rating", "up", "--out", out])
    assert res.exit_code == 0, res.output

    lines = Path(out).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["rating"] == 1
    assert record["assistant_message"] == "great answer"


def test_feedback_export_down_with_limit(monkeypatch, tmp_path):
    """--rating down --limit 1 honours the row cap."""
    from hrag.cli import cli as cli_group  # noqa: PLC0415
    from hrag.db.connection import init_db  # noqa: PLC0415
    import hrag.db.connection as _conn_mod  # noqa: PLC0415

    cfg = _make_config(tmp_path)
    _patch_config(monkeypatch, cfg)

    runner = CliRunner()
    runner.invoke(cli_group, ["init"])

    db = init_db(tmp_path / "store.sqlite", "default")
    _seed_db(db)
    sid = _insert_session(db)
    for i in range(3):
        _insert_message(db, sid, "user", f"q{i}")
        aid = _insert_message(db, sid, "assistant", f"bad{i}")
        _insert_feedback(db, aid, sid, -1)
    _conn_mod._db_singleton = None

    out = str(tmp_path / "down_limit.jsonl")
    res = runner.invoke(
        cli_group,
        ["feedback-export", "--rating", "down", "--limit", "1", "--out", out],
    )
    assert res.exit_code == 0, res.output

    lines = Path(out).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["rating"] == -1


def test_feedback_export_roundtrip(monkeypatch, tmp_path):
    """Roundtrip: seed messages + feedback → invoke CLI → verify JSONL shape."""
    from hrag.cli import cli as cli_group  # noqa: PLC0415
    from hrag.db.connection import init_db  # noqa: PLC0415
    import hrag.db.connection as _conn_mod  # noqa: PLC0415

    cfg = _make_config(tmp_path)
    _patch_config(monkeypatch, cfg)

    runner = CliRunner()
    runner.invoke(cli_group, ["init"])

    db = init_db(tmp_path / "store.sqlite", "default")
    _seed_db(db)
    sid = _insert_session(db)
    q_text = "show me the math"
    _insert_message(db, sid, "user", q_text)
    asst_id = _insert_message(db, sid, "assistant", "Here is the math derivation…")
    _insert_feedback(db, asst_id, sid, -1)
    _conn_mod._db_singleton = None

    out = str(tmp_path / "export.jsonl")
    res = runner.invoke(cli_group, ["feedback-export", "--rating", "down", "--out", out])
    assert res.exit_code == 0, res.output
    assert "Exported 1" in res.output

    lines = Path(out).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    # Verify required keys and values.
    assert record["rating"] == -1
    assert record["user_message"] == q_text
    assert record["assistant_message"] == "Here is the math derivation…"
    assert record["session_id"] == sid
    assert "message_id" in record
    assert "created_at" in record
    assert isinstance(record["retrieved_chunks"], list)
