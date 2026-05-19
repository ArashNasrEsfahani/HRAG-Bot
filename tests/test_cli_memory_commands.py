"""CLI surface for Phase 3 memory commands.

Uses Click's CliRunner. We monkey-patch the Orchestrator factory so the
test doesn't need real Chroma / sentence-transformers / Ollama — the
heavy-dep stubs in conftest cover the import path.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner


def _patched_orchestrator(monkeypatch, tmp_path):
    """Return (cli, orch_factory) with Orchestrator mocked to a stub."""
    from hrag.cli import cli as cli_group  # noqa: PLC0415

    def _load_cfg(*a, **kw):
        from hrag.config import (  # noqa: PLC0415
            Config, EmbeddingsConfig, LLMConfig, RetrievalConfig, StorageConfig,
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
            # Skip the reranker so Orchestrator boots without the real
            # sentence_transformers.CrossEncoder import path.
            retrieval=RetrievalConfig(rerank_enabled=False, doc_scope_enabled=False),
        )
        cfg.project_root = tmp_path
        return cfg

    monkeypatch.setattr("hrag.cli.load_config", _load_cfg)
    return cli_group


@pytest.fixture(autouse=True)
def _reset_db_singleton():
    import hrag.db.connection as _conn_mod  # noqa: PLC0415

    _conn_mod._db_singleton = None
    yield
    _conn_mod._db_singleton = None


def test_remember_inline_text(monkeypatch, tmp_path):
    cli = _patched_orchestrator(monkeypatch, tmp_path)
    runner = CliRunner()
    # Init the DB so the orchestrator can boot.
    res = runner.invoke(cli, ["init"])
    assert res.exit_code == 0, res.output

    res = runner.invoke(cli, ["remember", "Postgres preferred over MySQL"])
    assert res.exit_code == 0, res.output
    assert "saved" in res.output


def test_remember_file(monkeypatch, tmp_path, sample_md_path):
    cli = _patched_orchestrator(monkeypatch, tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["init"])

    res = runner.invoke(cli, ["remember", str(sample_md_path)])
    assert res.exit_code == 0, res.output
    assert "Saved" in res.output or "saved" in res.output


def test_remember_txt_file_per_line(monkeypatch, tmp_path, sample_txt_path):
    cli = _patched_orchestrator(monkeypatch, tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["init"])

    res = runner.invoke(cli, ["remember", str(sample_txt_path)])
    assert res.exit_code == 0, res.output


def test_memory_list_empty(monkeypatch, tmp_path):
    cli = _patched_orchestrator(monkeypatch, tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["init"])

    res = runner.invoke(cli, ["memory", "list"])
    assert res.exit_code == 0, res.output
    assert "No episodic memories yet" in res.output


def test_memory_list_after_remember(monkeypatch, tmp_path):
    cli = _patched_orchestrator(monkeypatch, tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["init"])
    runner.invoke(cli, ["remember", "TypeScript over JavaScript"])
    runner.invoke(cli, ["remember", "Postgres over MySQL"])

    res = runner.invoke(cli, ["memory", "list"])
    assert res.exit_code == 0, res.output
    assert "TypeScript" in res.output or "Postgres" in res.output
