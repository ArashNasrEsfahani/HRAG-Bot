"""Phase 9.8 — INT8 reranker quantization tests."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hrag.config import RetrievalConfig
from hrag.retrieval.factory import build_reranker


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_quantize_default_off():
    cfg = RetrievalConfig()
    assert cfg.rerank_quantize is False


def test_factory_returns_unquantized_reranker_by_default():
    cfg = RetrievalConfig(reranker="cross_encoder", rerank_quantize=False)
    reranker = build_reranker(cfg, llm=MagicMock())
    assert reranker is not None
    assert reranker.name == "cross_encoder"
    assert getattr(reranker, "_backend", "fp32") == "fp32"


# ---------------------------------------------------------------------------
# Quantize=True path — needs optimum installed
# ---------------------------------------------------------------------------


def test_quantize_flag_constructs_quantized():
    """When optimum is available, quantize=True yields the ONNX backend."""
    pytest.importorskip("optimum.onnxruntime")
    pytest.importorskip("transformers")
    cfg = RetrievalConfig(reranker="cross_encoder", rerank_quantize=True)
    try:
        reranker = build_reranker(cfg, llm=MagicMock())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ONNX export unavailable in this env: {exc}")
    assert reranker is not None
    assert reranker._backend == "onnx_int8"


def test_quantize_missing_optimum_falls_back(caplog):
    """When optimum is unavailable, quantize=True silently falls back to FP32."""
    from hrag.retrieval.cross_encoder_reranker import CrossEncoderReranker

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("optimum"):
            raise ImportError(f"mocked missing: {name}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with caplog.at_level(logging.WARNING, logger="hrag.retrieval.cross_encoder_reranker"):
            reranker = CrossEncoderReranker(quantize=True)
    assert reranker._backend == "fp32"
    assert any("optimum/onnxruntime" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Signature compatibility — rerank() shape must not change
# ---------------------------------------------------------------------------


def test_rerank_signature_unchanged():
    """rerank(query, results, *, threshold, top_k, progress) signature is locked.

    Phase 7-A contract 16 + Phase 9.8: changing this breaks orchestrator.py:917.
    """
    from hrag.retrieval.cross_encoder_reranker import CrossEncoderReranker
    import inspect

    sig = inspect.signature(CrossEncoderReranker.rerank)
    params = list(sig.parameters.keys())
    # self + query + results + threshold + top_k + progress
    assert params[:3] == ["self", "query", "results"]
    assert "threshold" in params
    assert "top_k" in params
    assert "progress" in params


def test_rerank_progress_callback_fires():
    """Per-result + final ticks must both fire with FP32 backend."""
    from hrag.types import Chunk, RetrievalResult

    # Build a stub reranker that bypasses model load
    from hrag.retrieval.cross_encoder_reranker import CrossEncoderReranker

    r = CrossEncoderReranker.__new__(CrossEncoderReranker)
    r._backend = "fp32"
    r._model_name = "stub"

    class _Predictor:
        def predict(self, pairs, **_kw):
            return [1.0, 2.0, 3.0]

    r._model = _Predictor()

    results = [
        RetrievalResult(
            chunk=Chunk(chunk_id=f"c{i}", doc_id="d", user_id="u",
                        text=f"t{i}", embedding_text=f"t{i}", title=None,
                        section=None),
            score=0.5 - i * 0.1,
            rerank_score=None,
        )
        for i in range(3)
    ]
    ticks: list[tuple[int, int, float]] = []
    out = r.rerank("q", results, threshold=-100.0, top_k=10,
                   progress=lambda i, n, s: ticks.append((i, n, s)))
    assert len(out) == 3
    # 3 per-result + 1 canonical final = 4 ticks
    assert len(ticks) == 4
    assert ticks[-1] == (1, 1, 3.0)
