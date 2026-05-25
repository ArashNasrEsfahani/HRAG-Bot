"""Phase 9.3 — per-session query embedding LRU cache tests."""

from __future__ import annotations

from typing import Sequence

import pytest

from hrag.config import EmbeddingsConfig
from hrag.providers.embeddings import EmbeddingProvider, _session_var


class _CountingEmbedder(EmbeddingProvider):
    """Concrete provider that counts embed() calls without loading a model."""

    name = "counting"

    def __init__(self, config: EmbeddingsConfig | None = None) -> None:
        super().__init__(config or EmbeddingsConfig(provider="counting"))
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    @property
    def dim(self) -> int:
        return 3


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_query_cache_defaults():
    cfg = EmbeddingsConfig()
    assert cfg.query_cache_enabled is True
    assert cfg.query_cache_size == 64


# ---------------------------------------------------------------------------
# Cache hits / misses
# ---------------------------------------------------------------------------


def test_cache_hit_returns_same_vector_with_kwarg():
    e = _CountingEmbedder()
    v1 = e.embed_one("hello", session_id="s1")
    v2 = e.embed_one("hello", session_id="s1")
    assert v1 == v2
    assert e.calls == 1


def test_default_session_id_none_uncached():
    e = _CountingEmbedder()
    e.embed_one("x")
    e.embed_one("x")
    assert e.calls == 2


def test_cache_disabled_is_passthrough():
    cfg = EmbeddingsConfig()
    cfg.query_cache_enabled = False
    e = _CountingEmbedder(cfg)
    e.embed_one("hello", session_id="s1")
    e.embed_one("hello", session_id="s1")
    assert e.calls == 2


def test_cache_invalidate_session():
    e = _CountingEmbedder()
    e.embed_one("hello", session_id="s1")
    assert e.calls == 1
    e.invalidate_session("s1")
    e.embed_one("hello", session_id="s1")
    assert e.calls == 2


def test_cache_eviction_at_capacity():
    cfg = EmbeddingsConfig()
    cfg.query_cache_size = 3
    e = _CountingEmbedder(cfg)
    for q in ("a", "b", "c"):
        e.embed_one(q, session_id="s1")
    assert e.calls == 3
    # Push a 4th: oldest ("a") should be evicted
    e.embed_one("d", session_id="s1")
    assert e.calls == 4
    # "a" should miss now (re-embed); "d" still cached
    e.embed_one("a", session_id="s1")
    assert e.calls == 5
    e.embed_one("d", session_id="s1")
    assert e.calls == 5


def test_session_isolation():
    e = _CountingEmbedder()
    e.embed_one("q", session_id="s1")
    e.embed_one("q", session_id="s2")
    assert e.calls == 2


def test_invalidate_unknown_session_no_raise():
    e = _CountingEmbedder()
    e.invalidate_session("never-existed")  # must be a no-op


# ---------------------------------------------------------------------------
# Contextvar ambient session
# ---------------------------------------------------------------------------


def test_session_contextmanager_activates_cache():
    e = _CountingEmbedder()
    with e.session("s1"):
        e.embed_one("q")
        e.embed_one("q")
    assert e.calls == 1


def test_session_contextmanager_resets_on_exit():
    e = _CountingEmbedder()
    with e.session("s1"):
        e.embed_one("q")
    # outside the block, no ambient session → cache bypassed
    e.embed_one("q")
    e.embed_one("q")
    assert e.calls == 3


def test_kwarg_overrides_contextvar():
    e = _CountingEmbedder()
    with e.session("ambient"):
        e.embed_one("q", session_id="explicit")
        e.embed_one("q", session_id="explicit")
    assert e.calls == 1
    # Different ambient session — cache miss
    with e.session("ambient2"):
        e.embed_one("q")
        assert e.calls == 2


def test_contextvar_default_is_none():
    # Even after running other tests, fresh embed_one with no kwarg/no with-block
    # should not touch any cache.
    e = _CountingEmbedder()
    assert _session_var.get() is None
    e.embed_one("q")
    assert e.calls == 1


# ---------------------------------------------------------------------------
# Backward-compatibility — abstract embed_one() signature
# ---------------------------------------------------------------------------


def test_embed_one_default_signature_unchanged():
    """embed_one(text) — single positional arg — must still work."""
    e = _CountingEmbedder()
    v = e.embed_one("hello")
    assert isinstance(v, list)
    assert len(v) == 3
