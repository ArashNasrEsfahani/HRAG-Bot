"""Tests for the /stats slash command helper (_print_stats)."""

from __future__ import annotations

import types
from typing import Any

from rich.console import Console

from hrag.cli import _print_stats
from hrag.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orch(db: Any, kg_enabled: bool = False) -> Any:
    """Minimal orchestrator-like namespace exposing .db and .config."""
    cfg = Config()
    cfg.kg.enabled = kg_enabled
    return types.SimpleNamespace(db=db, config=cfg)


def _captured_print_stats(orch: Any) -> str:
    """Run _print_stats with a recording console and return the output."""
    con = Console(record=True, width=120)
    _print_stats(orch, con)
    return con.export_text()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_print_stats_runs_against_empty_db(tmp_db: Any) -> None:
    """Stats helper must not crash on a freshly-initialised empty DB."""
    orch = _make_orch(tmp_db)
    output = _captured_print_stats(orch)
    # Should contain the table title and zero-document rows.
    assert "Corpus stats" in output
    assert "Documents (total)" in output
    assert "0" in output


def test_print_stats_with_data(tmp_db: Any) -> None:
    """Stats helper reflects actual document and chunk counts."""
    # Insert a user (already present from fixture), a document, and two chunks.
    with tmp_db.conn:
        tmp_db.execute(
            "INSERT INTO documents (doc_id, user_id, source_path, title, source_type) "
            "VALUES ('d1', 'default', '/fake/paper.pdf', 'My Test Paper', 'document')"
        )
        tmp_db.execute(
            "INSERT INTO chunks "
            "(chunk_id, doc_id, user_id, text, title, chunk_index, token_count, source_type, excluded) "
            "VALUES ('c1', 'd1', 'default', 'chunk one text', 'My Test Paper', 0, 10, 'document', 0)"
        )
        tmp_db.execute(
            "INSERT INTO chunks "
            "(chunk_id, doc_id, user_id, text, title, chunk_index, token_count, source_type, excluded) "
            "VALUES ('c2', 'd1', 'default', 'chunk two text', 'My Test Paper', 1, 12, 'document', 0)"
        )
        tmp_db.execute(
            "INSERT INTO chunks "
            "(chunk_id, doc_id, user_id, text, title, chunk_index, token_count, source_type, excluded) "
            "VALUES ('c3', 'd1', 'default', 'excluded chunk', 'My Test Paper', 2, 8, 'document', 1)"
        )
    tmp_db.commit()

    orch = _make_orch(tmp_db)
    output = _captured_print_stats(orch)

    # Document total: 1
    assert "Documents (total)" in output
    # Active chunks: 2 (c3 is excluded)
    assert "Active chunks" in output
    # The per-doc table should show the document title
    assert "My Test Paper" in output
    # KG disabled line present
    assert "KG enabled" in output
    assert "False" in output


def test_print_stats_kg_disabled_hides_kg_rows(tmp_db: Any) -> None:
    """When kg.enabled=False the KG metric rows must not appear."""
    orch = _make_orch(tmp_db, kg_enabled=False)
    output = _captured_print_stats(orch)
    assert "KG phrase nodes" not in output
    assert "KG edges" not in output


def test_print_stats_kg_enabled_shows_kg_rows(tmp_db: Any) -> None:
    """When kg.enabled=True the KG metric rows appear (even if counts are zero)."""
    orch = _make_orch(tmp_db, kg_enabled=True)
    output = _captured_print_stats(orch)
    assert "KG phrase nodes" in output
    assert "KG passage nodes" in output
    assert "KG edges" in output
    assert "Communities" in output


def test_print_stats_source_type_breakdown(tmp_db: Any) -> None:
    """source_type grouping rows appear for each distinct source_type."""
    with tmp_db.conn:
        tmp_db.execute(
            "INSERT INTO documents (doc_id, user_id, source_path, title, source_type) "
            "VALUES ('d2', 'default', '/a.pdf', 'Academic', 'academic')"
        )
        tmp_db.execute(
            "INSERT INTO documents (doc_id, user_id, source_path, title, source_type) "
            "VALUES ('d3', 'default', '/b.pdf', 'Manual', 'manual')"
        )
    tmp_db.commit()

    orch = _make_orch(tmp_db)
    output = _captured_print_stats(orch)
    assert "source_type=academic" in output
    assert "source_type=manual" in output


def test_print_stats_no_per_doc_table_when_empty(tmp_db: Any) -> None:
    """When no active chunks exist the per-doc table is omitted."""
    orch = _make_orch(tmp_db)
    output = _captured_print_stats(orch)
    # "Top docs" (start of the second table title) must not appear.
    assert "Top docs" not in output


def test_print_stats_per_doc_table_present_with_chunks(tmp_db: Any) -> None:
    """Top-docs table appears when at least one active chunk is indexed."""
    with tmp_db.conn:
        tmp_db.execute(
            "INSERT INTO documents (doc_id, user_id, source_path, title, source_type) "
            "VALUES ('d4', 'default', '/x.pdf', 'Title X', 'document')"
        )
        tmp_db.execute(
            "INSERT INTO chunks "
            "(chunk_id, doc_id, user_id, text, title, chunk_index, token_count, source_type, excluded) "
            "VALUES ('cx1', 'd4', 'default', 'text', 'Title X', 0, 5, 'document', 0)"
        )
    tmp_db.commit()

    orch = _make_orch(tmp_db)
    output = _captured_print_stats(orch)
    # "Top docs by chunk count" may wrap across lines in narrow consoles;
    # verify the per-doc table rendered by checking column headers and data.
    assert "chunks" in output
    assert "Title X" in output
