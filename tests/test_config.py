"""Tests for hrag.config — Config, load_config, env-var overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from hrag.config import load_config


# ---------------------------------------------------------------------------
# Defaults when no config file present
# ---------------------------------------------------------------------------

def test_load_config_defaults_no_file(tmp_path: Path, monkeypatch) -> None:
    """load_config() with no config.yaml must return pydantic defaults."""
    monkeypatch.chdir(tmp_path)
    # Ensure no stale HRAG_ env vars from outer environment pollute the test
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)

    cfg = load_config()

    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "gemma3:latest"
    assert cfg.embeddings.provider == "sentence-transformers"
    assert cfg.storage.sqlite_path == "data/store.sqlite"
    assert cfg.retrieval.top_k_vector == 10
    assert cfg.retrieval.reranker == "cross_encoder"
    assert cfg.retrieval.retriever == "vector"
    assert cfg.chunking.max_tokens == 400
    assert cfg.context.history_token_budget == 4000


def test_deep_read_and_ingest_phase13_1_defaults(tmp_path: Path, monkeypatch) -> None:
    """Phase 13.1 — agentic deep-read + page-metadata flags load with defaults."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)

    cfg = load_config()

    # Agentic deep-read smart-escalation defaults.
    assert cfg.deep_read.enabled is True
    assert cfg.deep_read.structural_trigger is True
    assert cfg.deep_read.escalate_on_weak_answer is True
    assert cfg.deep_read.weak_answer_floor == 0.0
    assert cfg.deep_read.plan_max_tokens == 200
    assert cfg.deep_read.structural_scan_all is False
    assert cfg.deep_read.action_repeat_guard is True

    # Ingest page/chapter metadata default-on.
    assert cfg.ingest.page_metadata_enabled is True


def test_old_yaml_without_phase13_1_keys_still_loads(tmp_path: Path, monkeypatch) -> None:
    """A pre-13.1 config.yaml (no new keys) must still load with sane defaults."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"deep_read": {"max_passes": 5}, "ingest": {"use_nougat": False}}),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.deep_read.max_passes == 5                 # explicit override honoured
    assert cfg.deep_read.structural_trigger is True      # new key defaulted
    assert cfg.ingest.page_metadata_enabled is True      # new key defaulted


def test_load_config_sets_project_root(tmp_path: Path, monkeypatch) -> None:
    """project_root should be the cwd when load_config() is called."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)

    cfg = load_config()
    assert cfg.project_root == tmp_path


# ---------------------------------------------------------------------------
# YAML override
# ---------------------------------------------------------------------------

def test_load_config_yaml_overrides_defaults(tmp_path: Path, monkeypatch) -> None:
    """Values from config.yaml must override pydantic defaults."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)

    config_data = {
        "llm": {"model": "llama3:8b", "temperature": 0.7},
        "chunking": {"max_tokens": 256},
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config_data), encoding="utf-8")

    cfg = load_config()

    assert cfg.llm.model == "llama3:8b"
    assert cfg.llm.temperature == 0.7
    assert cfg.chunking.max_tokens == 256
    # Unset keys keep their defaults
    assert cfg.llm.provider == "ollama"


def test_load_config_explicit_path(tmp_path: Path, monkeypatch) -> None:
    """load_config(path=...) reads from the given file, not cwd/config.yaml."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)

    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(yaml.dump({"llm": {"model": "custom-model"}}), encoding="utf-8")

    cfg = load_config(path=custom_yaml)
    assert cfg.llm.model == "custom-model"


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

def test_env_override_llm_model(tmp_path: Path, monkeypatch) -> None:
    """HRAG_LLM__MODEL env var must override cfg.llm.model."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("HRAG_LLM__MODEL", "foo-model")

    cfg = load_config()
    assert cfg.llm.model == "foo-model"


def test_env_override_chunking_max_tokens(tmp_path: Path, monkeypatch) -> None:
    """HRAG_CHUNKING__MAX_TOKENS overrides cfg.chunking.max_tokens (cast to int via pydantic)."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("HRAG_CHUNKING__MAX_TOKENS", "512")

    cfg = load_config()
    assert cfg.chunking.max_tokens == 512


def test_env_override_does_not_affect_unrelated_keys(tmp_path: Path, monkeypatch) -> None:
    """Setting one env override must not change other fields."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("HRAG_LLM__MODEL", "override-model")

    cfg = load_config()
    assert cfg.llm.provider == "ollama"   # unchanged default
    assert cfg.chunking.max_tokens == 400  # unchanged default


# ---------------------------------------------------------------------------
# cfg.resolve()
# ---------------------------------------------------------------------------

def test_resolve_relative_path(tmp_path: Path, monkeypatch) -> None:
    """cfg.resolve('data/store.sqlite') returns an absolute path under project_root."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)

    cfg = load_config()
    resolved = cfg.resolve("data/store.sqlite")

    assert resolved.is_absolute()
    assert resolved == tmp_path / "data" / "store.sqlite"


def test_resolve_absolute_path_passthrough(tmp_path: Path, monkeypatch) -> None:
    """cfg.resolve() with an already-absolute path returns it unchanged."""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("HRAG_"):
            monkeypatch.delenv(k)

    cfg = load_config()
    abs_path = str(tmp_path / "abs" / "file.db")
    resolved = cfg.resolve(abs_path)

    assert resolved == Path(abs_path)
