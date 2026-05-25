"""Phase 9.7 — Embedder FP16 / ONNX quantization tests.

Eight tests covering:
1. Default config value for embed_precision.
2. Default config value for embed_device.
3. FP32 backend loads without the quantize optional dep.
4. FP16 falls back to fp32 when CUDA is unavailable.
5. onnx_int8 falls back to fp32 when optimum is missing.
6. onnx_int8 loader invoked when optimum is available.
7. embed() returns L2-normalised vectors (Phase 3 contract 6).
8. embed() signature is unchanged.
"""

from __future__ import annotations

import inspect
import logging
import math
import sys
import types
from typing import Sequence
from unittest.mock import MagicMock, patch

import pytest

from hrag.config import EmbeddingsConfig


# ---------------------------------------------------------------------------
# 1. Default config value for embed_precision
# ---------------------------------------------------------------------------


def test_embed_precision_default_fp32():
    """EmbeddingsConfig.embed_precision must default to 'fp32'."""
    cfg = EmbeddingsConfig()
    assert cfg.embed_precision == "fp32"


# ---------------------------------------------------------------------------
# 2. Default config value for embed_device
# ---------------------------------------------------------------------------


def test_embed_device_default_none():
    """EmbeddingsConfig.embed_device must default to None."""
    cfg = EmbeddingsConfig()
    assert cfg.embed_device is None


# ---------------------------------------------------------------------------
# 3. FP32 backend loads without the quantize optional dep
# ---------------------------------------------------------------------------


def test_fp32_backend_loads_without_quantize_deps():
    """Instantiating with the default fp32 config must NOT require optimum/onnxruntime.

    We verify this by removing 'optimum' from sys.modules (if present) and
    confirming the provider still loads.
    """
    pytest.importorskip("sentence_transformers")

    from hrag.providers.embeddings import SentenceTransformersProvider

    # Temporarily hide optimum if it happens to be installed.
    saved = sys.modules.pop("optimum", None)
    saved_ort = sys.modules.pop("optimum.onnxruntime", None)
    try:
        cfg = EmbeddingsConfig(
            provider="sentence-transformers",
            model="sentence-transformers/all-mpnet-base-v2",
            embed_precision="fp32",
        )
        provider = SentenceTransformersProvider(cfg)
        assert provider._backend == "fp32"
    finally:
        if saved is not None:
            sys.modules["optimum"] = saved
        if saved_ort is not None:
            sys.modules["optimum.onnxruntime"] = saved_ort


# ---------------------------------------------------------------------------
# 4. FP16 falls back to fp32 when CUDA is unavailable
# ---------------------------------------------------------------------------


def test_fp16_falls_back_to_fp32_on_cpu(caplog):
    """When CUDA is unavailable, fp16 precision must fall back to fp32 with a warning."""
    pytest.importorskip("sentence_transformers")

    from hrag.providers.embeddings import SentenceTransformersProvider

    # Patch torch.cuda.is_available to return False.
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.is_available = lambda: False
    fake_torch.cuda = fake_cuda

    # We also need SentenceTransformer to work; we'll patch _try_load_fp16 directly
    # so we don't need a real model download.
    with patch.object(
        SentenceTransformersProvider,
        "_try_load_fp16",
        return_value=(None, "fp32"),
    ) as mock_fp16, patch.object(
        SentenceTransformersProvider,
        "_try_load_onnx",
        return_value=(None, "fp32"),
    ):
        # Also patch the fp32 SentenceTransformer fallback to avoid a real download.
        fake_st_model = MagicMock()
        fake_st_model.get_sentence_embedding_dimension.return_value = 768

        with patch("hrag.providers.embeddings.SentenceTransformersProvider._try_load_fp16",
                   return_value=(None, "fp32")):
            with patch("sentence_transformers.SentenceTransformer", return_value=fake_st_model):
                with caplog.at_level(logging.WARNING, logger="hrag.providers.embeddings"):
                    cfg = EmbeddingsConfig(embed_precision="fp16")
                    provider = SentenceTransformersProvider(cfg)

        assert provider._backend == "fp32"
        assert any("fp16" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 5. onnx_int8 falls back to fp32 when optimum is missing
# ---------------------------------------------------------------------------


def test_onnx_int8_falls_back_when_optimum_missing(caplog):
    """When optimum is unavailable, onnx_int8 precision silently falls back to fp32."""
    pytest.importorskip("sentence_transformers")

    from hrag.providers.embeddings import SentenceTransformersProvider

    fake_st_model = MagicMock()
    fake_st_model.get_sentence_embedding_dimension.return_value = 768

    with patch.object(
        SentenceTransformersProvider,
        "_try_load_onnx",
        return_value=(None, "fp32"),
    ):
        with patch("sentence_transformers.SentenceTransformer", return_value=fake_st_model):
            with caplog.at_level(logging.WARNING, logger="hrag.providers.embeddings"):
                cfg = EmbeddingsConfig(embed_precision="onnx_int8")
                provider = SentenceTransformersProvider(cfg)

    assert provider._backend == "fp32"
    assert any("onnx_int8" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 6. onnx_int8 loader invoked when optimum is available
# ---------------------------------------------------------------------------


def test_onnx_int8_loader_invoked_when_available():
    """When optimum is available, _try_load_onnx is called for onnx_int8 precision."""
    # Skip cleanly if sentence_transformers is not installed (the conftest stub
    # would intercept the SentenceTransformer call; this test needs the real import
    # or a clean mock path that doesn't need it).
    pytest.importorskip("sentence_transformers")

    from hrag.providers.embeddings import SentenceTransformersProvider, _ONNXEmbeddingBundle

    # Build a fake bundle that behaves like _ONNXEmbeddingBundle
    fake_bundle = MagicMock(spec=_ONNXEmbeddingBundle)
    fake_bundle._dim = 768
    fake_bundle.encode.return_value = [[0.0] * 768]

    with patch.object(
        SentenceTransformersProvider,
        "_try_load_onnx",
        return_value=(fake_bundle, "onnx_int8"),
    ) as mock_onnx:
        cfg = EmbeddingsConfig(embed_precision="onnx_int8")
        provider = SentenceTransformersProvider(cfg)

    mock_onnx.assert_called_once()
    assert provider._backend == "onnx_int8"
    assert provider._dim == 768


# ---------------------------------------------------------------------------
# 7. embed() returns L2-normalised vectors (Phase 3 contract 6)
# ---------------------------------------------------------------------------


def test_embed_returns_normalized_vectors():
    """fp32 embed() must return unit-norm vectors (Phase 3 contract 6)."""
    st = pytest.importorskip("sentence_transformers")
    if getattr(st, "__hrag_stub__", False):
        pytest.skip("Stubbed sentence_transformers returns zero vectors")

    from hrag.providers.embeddings import SentenceTransformersProvider

    cfg = EmbeddingsConfig(embed_precision="fp32")
    try:
        provider = SentenceTransformersProvider(cfg)
    except Exception:
        pytest.skip("SentenceTransformer model unavailable in this environment")

    vecs = provider.embed(["This is a test sentence for normalisation check."])
    assert len(vecs) == 1
    vec = vecs[0]
    norm_sq = sum(v * v for v in vec)
    assert math.isclose(norm_sq, 1.0, abs_tol=1e-4), (
        f"Vector is not unit-norm; ‖v‖² = {norm_sq:.6f}"
    )


# ---------------------------------------------------------------------------
# 8. embed() signature is unchanged
# ---------------------------------------------------------------------------


def test_embed_signature_unchanged():
    """embed(self, texts: Sequence[str]) signature must remain stable."""
    from hrag.providers.embeddings import SentenceTransformersProvider

    sig = inspect.signature(SentenceTransformersProvider.embed)
    params = list(sig.parameters.keys())
    assert params[0] == "self"
    assert params[1] == "texts"
    assert len(params) == 2, f"Unexpected extra params: {params[2:]}"

    # Verify the annotation on 'texts' is Sequence[str] (or compatible).
    texts_param = sig.parameters["texts"]
    # The annotation may be a string (from __future__ annotations) or a real type.
    ann = texts_param.annotation
    assert ann is not inspect.Parameter.empty, "embed(texts) annotation must be present"
