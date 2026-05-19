"""Tests for hrag.retrieval.factory — build_retriever and build_reranker.

Uses monkeypatching to avoid real model downloads.
"""

from __future__ import annotations

import sys

import pytest

from hrag.config import RetrievalConfig
from hrag.retrieval.batched_llm_reranker import BatchedLLMReranker
from hrag.retrieval.bm25 import BM25Retriever
from hrag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from hrag.retrieval.doc_scope import DocScopedRetriever
from hrag.retrieval.factory import build_retriever, build_reranker
from hrag.retrieval.hybrid import HybridRetriever
from hrag.retrieval.reranker import LLMReranker
from hrag.retrieval.vector_retriever import VectorRetriever


def _unwrap(retriever):
    """Return the inner retriever, peeling DocScopedRetriever if present.

    The factory wraps the built retriever in DocScopedRetriever by default
    (cfg.doc_scope_enabled=True). The existing tests check the *requested*
    retriever class, so we peel the wrapper before the isinstance check.
    """
    if isinstance(retriever, DocScopedRetriever):
        return retriever._wrapped
    return retriever
from tests.conftest import FakeLLM


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _FakeVectorStore:
    """Minimal stub satisfying VectorStore-like interface."""
    def query(self, *a, **kw):
        return {"ids": [[]], "distances": [[]]}


class _FakeEmbedder:
    dim = 384
    name = "fake"
    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]
    def embed_one(self, text):
        return [0.0] * self.dim


def _make_cfg(**overrides) -> RetrievalConfig:
    defaults = {
        "retriever": "vector",
        "reranker": "cross_encoder",
        "rerank_enabled": True,
        "cross_encoder_model": "stub-model",
        "rrf_k": 60,
        "rrf_weights": None,
    }
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


class _FakeCE:
    """No-op CrossEncoder replacement."""
    def __init__(self, *a, **k):
        pass


def _patch_cross_encoder(monkeypatch):
    """Inject a no-op CrossEncoder into the sentence_transformers stub module."""
    st_mod = sys.modules["sentence_transformers"]
    monkeypatch.setattr(st_mod, "CrossEncoder", _FakeCE, raising=False)


# ---------------------------------------------------------------------------
# build_retriever tests
# ---------------------------------------------------------------------------

class TestBuildRetriever:
    def test_vector_mode(self, tmp_db):
        cfg = _make_cfg(retriever="vector")
        r = build_retriever(cfg, tmp_db, _FakeVectorStore(), _FakeEmbedder())
        assert isinstance(_unwrap(r), VectorRetriever)

    def test_bm25_mode(self, tmp_db):
        cfg = _make_cfg(retriever="bm25")
        r = build_retriever(cfg, tmp_db, _FakeVectorStore(), _FakeEmbedder())
        assert isinstance(_unwrap(r), BM25Retriever)

    def test_hybrid_mode(self, tmp_db):
        cfg = _make_cfg(retriever="hybrid")
        r = build_retriever(cfg, tmp_db, _FakeVectorStore(), _FakeEmbedder())
        assert isinstance(_unwrap(r), HybridRetriever)

    def test_invalid_mode_raises_value_error(self, tmp_db):
        cfg = _make_cfg(retriever="nonexistent_mode")
        with pytest.raises(ValueError, match="nonexistent_mode"):
            build_retriever(cfg, tmp_db, _FakeVectorStore(), _FakeEmbedder())

    def test_mode_case_insensitive(self, tmp_db):
        cfg = _make_cfg(retriever="BM25")
        r = build_retriever(cfg, tmp_db, _FakeVectorStore(), _FakeEmbedder())
        assert isinstance(_unwrap(r), BM25Retriever)

    def test_mode_strips_whitespace(self, tmp_db):
        cfg = _make_cfg(retriever="  vector  ")
        r = build_retriever(cfg, tmp_db, _FakeVectorStore(), _FakeEmbedder())
        assert isinstance(_unwrap(r), VectorRetriever)

    def test_doc_scope_wrap_default(self, tmp_db):
        """By default the built retriever is wrapped in DocScopedRetriever."""
        cfg = _make_cfg(retriever="vector")
        r = build_retriever(cfg, tmp_db, _FakeVectorStore(), _FakeEmbedder())
        assert isinstance(r, DocScopedRetriever)

    def test_doc_scope_disabled_no_wrap(self, tmp_db):
        """doc_scope_enabled=False returns the raw retriever (no wrapper)."""
        cfg = _make_cfg(retriever="vector", doc_scope_enabled=False)
        r = build_retriever(cfg, tmp_db, _FakeVectorStore(), _FakeEmbedder())
        assert not isinstance(r, DocScopedRetriever)
        assert isinstance(r, VectorRetriever)


# ---------------------------------------------------------------------------
# build_reranker tests
# ---------------------------------------------------------------------------

class TestBuildReranker:
    def test_rerank_disabled_returns_none(self):
        cfg = _make_cfg(rerank_enabled=False)
        result = build_reranker(cfg, FakeLLM())
        assert result is None

    def test_cross_encoder_mode(self, monkeypatch):
        _patch_cross_encoder(monkeypatch)
        cfg = _make_cfg(reranker="cross_encoder", rerank_enabled=True)
        r = build_reranker(cfg, FakeLLM())
        assert isinstance(r, CrossEncoderReranker)

    def test_cross_encoder_alias_ce(self, monkeypatch):
        _patch_cross_encoder(monkeypatch)
        cfg = _make_cfg(reranker="ce", rerank_enabled=True)
        r = build_reranker(cfg, FakeLLM())
        assert isinstance(r, CrossEncoderReranker)

    def test_cross_encoder_alias_hyphen(self, monkeypatch):
        _patch_cross_encoder(monkeypatch)
        cfg = _make_cfg(reranker="cross-encoder", rerank_enabled=True)
        r = build_reranker(cfg, FakeLLM())
        assert isinstance(r, CrossEncoderReranker)

    def test_llm_mode(self):
        cfg = _make_cfg(reranker="llm", rerank_enabled=True)
        r = build_reranker(cfg, FakeLLM())
        assert isinstance(r, LLMReranker)

    def test_batched_llm_mode(self):
        cfg = _make_cfg(reranker="batched_llm", rerank_enabled=True)
        r = build_reranker(cfg, FakeLLM())
        assert isinstance(r, BatchedLLMReranker)

    def test_batched_llm_alias_llm_batched(self):
        cfg = _make_cfg(reranker="llm_batched", rerank_enabled=True)
        r = build_reranker(cfg, FakeLLM())
        assert isinstance(r, BatchedLLMReranker)

    def test_batched_llm_alias_batched(self):
        cfg = _make_cfg(reranker="batched", rerank_enabled=True)
        r = build_reranker(cfg, FakeLLM())
        assert isinstance(r, BatchedLLMReranker)

    def test_invalid_reranker_raises_value_error(self):
        cfg = _make_cfg(reranker="unicorn", rerank_enabled=True)
        with pytest.raises(ValueError, match="unicorn"):
            build_reranker(cfg, FakeLLM())

    def test_cross_encoder_model_name_passed(self, monkeypatch):
        """The cross_encoder_model config value is forwarded to CrossEncoderReranker."""
        captured = {}

        def tracking_init(self, model_name="default", **kw):
            captured["model_name"] = model_name

        # Inject FakeCE so the import inside CrossEncoderReranker.__init__ succeeds
        st_mod = sys.modules["sentence_transformers"]
        monkeypatch.setattr(st_mod, "CrossEncoder", _FakeCE, raising=False)
        # Intercept CrossEncoderReranker.__init__ to capture arguments
        monkeypatch.setattr(CrossEncoderReranker, "__init__", tracking_init)
        cfg = _make_cfg(reranker="cross_encoder", cross_encoder_model="my-model", rerank_enabled=True)
        build_reranker(cfg, FakeLLM())
        assert captured.get("model_name") == "my-model"
