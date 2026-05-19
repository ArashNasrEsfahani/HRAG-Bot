"""Phase 2 KGConfig defaults and override behavior."""

from __future__ import annotations

from hrag.config import Config, KGConfig


def test_kg_config_defaults():
    cfg = KGConfig()
    assert cfg.enabled is False
    assert cfg.parallel_workers == 8
    assert cfg.ner == "spacy"
    assert cfg.damping == 0.5
    assert cfg.synonym_threshold == 0.8
    assert cfg.leiden_seed == 42
    assert cfg.community_levels == [0, 1, 2]


def test_config_includes_kg_section():
    cfg = Config()
    assert isinstance(cfg.kg, KGConfig)
    assert cfg.kg.enabled is False


def test_config_kg_override():
    cfg = Config(kg=KGConfig(enabled=True, parallel_workers=4, ner="llm"))
    assert cfg.kg.enabled is True
    assert cfg.kg.parallel_workers == 4
    assert cfg.kg.ner == "llm"
