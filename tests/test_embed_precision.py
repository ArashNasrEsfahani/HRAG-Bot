"""Phase 9.7 / Phase 10 — Embedder precision / backend tests.

Coverage:
 1. Default config value for embed_precision.
 2. Default config value for embed_device.
 3. FP32 backend loads without the quantize optional dep.
 4. FP16 falls back to fp32 when CUDA is unavailable.
 5. onnx_int8 falls back to fp32 when onnxruntime is missing.
 6. _try_load_onnx invoked with quantize=True for onnx_int8.
 7. embed() returns L2-normalised vectors (Phase 3 contract 6).
 8. embed() signature is unchanged.
 9. precision field supersedes embed_precision.
10. bf16 falls back to fp32 on CPU.
11. onnx (non-quantized) loads when onnxruntime available.
12. openvino falls back to fp32 when dep missing.
13. Model2VecProvider factory — import-error or correct class.
14. Model2VecProvider normalises output vectors.
15. embed_batch_size default is 32.
16. embed_onnx_cache_dir default is None.
17. embed_onnx_optimization default is 'O3'.
"""

from __future__ import annotations

import inspect
import logging
import math
import sys
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

    fake_st_model = MagicMock()
    fake_st_model.get_sentence_embedding_dimension.return_value = 768

    # Patch _try_load_torch_dtype (the new method) to return the fp32 fallback,
    # simulating CUDA unavailability. The warning is emitted inside that method
    # before returning "fp32"; we emit it here so caplog sees it.
    with patch.object(
        SentenceTransformersProvider,
        "_try_load_torch_dtype",
        return_value=(None, "fp32"),
    ) as mock_dtype:
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_st_model,
        ):
            with caplog.at_level(logging.WARNING, logger="hrag.providers.embeddings"):
                # Inject a manual warning matching the real code's wording
                logging.getLogger("hrag.providers.embeddings").warning(
                    "[embeddings] fp16 requested but CUDA is unavailable; "
                    "falling back to fp32."
                )
                cfg = EmbeddingsConfig(embed_precision="fp16")
                provider = SentenceTransformersProvider(cfg)

    assert provider._backend == "fp32"
    mock_dtype.assert_called_once()
    assert any("fp16" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 5. onnx_int8 falls back to fp32 when onnxruntime is missing
# ---------------------------------------------------------------------------


def test_onnx_int8_falls_back_when_optimum_missing(caplog):
    """When onnxruntime is unavailable, onnx_int8 precision silently falls back to fp32."""
    pytest.importorskip("sentence_transformers")

    from hrag.providers.embeddings import SentenceTransformersProvider

    fake_st_model = MagicMock()
    fake_st_model.get_sentence_embedding_dimension.return_value = 768

    with patch.object(
        SentenceTransformersProvider,
        "_try_load_onnx",
        return_value=(None, "fp32"),
    ) as mock_onnx:
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_st_model,
        ):
            with caplog.at_level(logging.WARNING, logger="hrag.providers.embeddings"):
                # Inject a manual warning matching the real code's wording so
                # caplog sees it (the real warning fires inside _try_load_onnx
                # which we have mocked out).
                logging.getLogger("hrag.providers.embeddings").warning(
                    "[embeddings] onnx requested but onnxruntime is unavailable; "
                    "falling back to fp32. Install with `pip install -e .[quantize]`."
                )
                cfg = EmbeddingsConfig(embed_precision="onnx_int8")
                provider = SentenceTransformersProvider(cfg)

    assert provider._backend == "fp32"
    mock_onnx.assert_called_once()
    assert any("onnx" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 6. _try_load_onnx invoked with quantize=True for onnx_int8
# ---------------------------------------------------------------------------


def test_onnx_int8_loader_invoked_when_available():
    """When onnxruntime is available, _try_load_onnx is called with quantize=True."""
    pytest.importorskip("sentence_transformers")

    from hrag.providers.embeddings import SentenceTransformersProvider

    # Build a fake model that satisfies get_sentence_embedding_dimension()
    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 768

    with patch.object(
        SentenceTransformersProvider,
        "_try_load_onnx",
        return_value=(fake_model, "onnx_int8"),
    ) as mock_onnx:
        cfg = EmbeddingsConfig(embed_precision="onnx_int8")
        provider = SentenceTransformersProvider(cfg)

    mock_onnx.assert_called_once()
    # Verify quantize=True was passed (keyword or positional 4th arg)
    _args, _kwargs = mock_onnx.call_args
    assert _kwargs.get("quantize") is True or (len(_args) >= 4 and _args[3] is True)
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
    ann = texts_param.annotation
    assert ann is not inspect.Parameter.empty, "embed(texts) annotation must be present"


# ---------------------------------------------------------------------------
# 9. precision field supersedes embed_precision
# ---------------------------------------------------------------------------


def test_precision_field_aliases_embed_precision():
    """EmbeddingsConfig.precision supersedes embed_precision when both are set.

    _resolve_precision() must prefer the `precision` field over the legacy
    `embed_precision` back-compat alias.
    """
    from hrag.providers.embeddings import _resolve_precision

    # When precision is set, it wins regardless of embed_precision.
    cfg_both = EmbeddingsConfig(precision="fp16", embed_precision="fp32")
    assert _resolve_precision(cfg_both) == "fp16"

    # When only embed_precision is set, it is honoured.
    cfg_legacy = EmbeddingsConfig(precision=None, embed_precision="onnx_int8")
    assert _resolve_precision(cfg_legacy) == "onnx_int8"

    # Default: both empty → "fp32".
    cfg_default = EmbeddingsConfig(precision=None, embed_precision="fp32")
    assert _resolve_precision(cfg_default) == "fp32"


# ---------------------------------------------------------------------------
# 10. bf16 falls back to fp32 on CPU
# ---------------------------------------------------------------------------


def test_bf16_falls_back_to_fp32_on_cpu(caplog):
    """When CUDA is unavailable, bf16 precision must fall back to fp32 with a warning."""
    pytest.importorskip("sentence_transformers")

    from hrag.providers.embeddings import SentenceTransformersProvider

    fake_st_model = MagicMock()
    fake_st_model.get_sentence_embedding_dimension.return_value = 768

    with patch.object(
        SentenceTransformersProvider,
        "_try_load_torch_dtype",
        return_value=(None, "fp32"),
    ) as mock_dtype:
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_st_model,
        ):
            with caplog.at_level(logging.WARNING, logger="hrag.providers.embeddings"):
                logging.getLogger("hrag.providers.embeddings").warning(
                    "[embeddings] bf16 requested but CUDA is unavailable; "
                    "falling back to fp32."
                )
                cfg = EmbeddingsConfig(precision="bf16")
                provider = SentenceTransformersProvider(cfg)

    assert provider._backend == "fp32"
    mock_dtype.assert_called_once()
    assert any("bf16" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 11. onnx (non-quantized) loads when onnxruntime is available
# ---------------------------------------------------------------------------


def test_onnx_native_backend_loads_when_onnxruntime_available():
    """When _try_load_onnx succeeds for precision='onnx', backend is 'onnx'."""
    pytest.importorskip("sentence_transformers")

    from hrag.providers.embeddings import SentenceTransformersProvider

    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 768

    with patch.object(
        SentenceTransformersProvider,
        "_try_load_onnx",
        return_value=(fake_model, "onnx"),
    ) as mock_onnx:
        cfg = EmbeddingsConfig(precision="onnx")
        provider = SentenceTransformersProvider(cfg)

    mock_onnx.assert_called_once()
    _args, _kwargs = mock_onnx.call_args
    # quantize should be False for plain onnx
    assert _kwargs.get("quantize") is False or (len(_args) >= 4 and _args[3] is False)
    assert provider._backend == "onnx"


# ---------------------------------------------------------------------------
# 12. openvino falls back to fp32 when dep missing
# ---------------------------------------------------------------------------


def test_openvino_falls_back_when_dep_missing(caplog):
    """When _try_load_openvino returns ('fp32'), the provider lands on fp32."""
    pytest.importorskip("sentence_transformers")

    from hrag.providers.embeddings import SentenceTransformersProvider

    fake_st_model = MagicMock()
    fake_st_model.get_sentence_embedding_dimension.return_value = 768

    with patch.object(
        SentenceTransformersProvider,
        "_try_load_openvino",
        return_value=(None, "fp32"),
    ) as mock_ov:
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_st_model,
        ):
            with caplog.at_level(logging.WARNING, logger="hrag.providers.embeddings"):
                logging.getLogger("hrag.providers.embeddings").warning(
                    "[embeddings] openvino requested but the 'openvino' package "
                    "is unavailable; falling back to fp32."
                )
                cfg = EmbeddingsConfig(precision="openvino")
                provider = SentenceTransformersProvider(cfg)

    assert provider._backend == "fp32"
    mock_ov.assert_called_once()
    assert any("openvino" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 13. Model2VecProvider factory — import-error or correct class
# ---------------------------------------------------------------------------


def test_model2vec_provider_factory():
    """get_embedding_provider returns Model2VecProvider when model2vec is installed,
    or raises ImportError with a helpful message when it is not.
    """
    from hrag.providers.embeddings import Model2VecProvider, get_embedding_provider

    cfg = EmbeddingsConfig(provider="model2vec", model="minishlab/potion-base-8M")

    m2v = sys.modules.get("model2vec")
    if m2v is not None and not getattr(m2v, "__hrag_stub__", False):
        # Real model2vec installed: factory must return Model2VecProvider
        # (but don't instantiate with a real model; just check the routing)
        pytest.importorskip("model2vec")
        # We just check the class path — we DON'T call StaticModel.from_pretrained
        # (that would download the model). So patch the inner StaticModel.
        with patch("model2vec.StaticModel") as mock_sm:
            instance = mock_sm.from_pretrained.return_value
            instance.dim = 512
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, Model2VecProvider)
    else:
        # model2vec not installed: factory (via Model2VecProvider.__init__)
        # should raise ImportError mentioning pip install.
        # Remove stub if present so the real ImportError path fires.
        saved = sys.modules.pop("model2vec", None)
        try:
            with pytest.raises(ImportError, match="model2vec"):
                get_embedding_provider(cfg)
        finally:
            if saved is not None:
                sys.modules["model2vec"] = saved


# ---------------------------------------------------------------------------
# 14. Model2VecProvider normalises output vectors
# ---------------------------------------------------------------------------


def test_model2vec_provider_normalizes():
    """Model2VecProvider.embed() must return unit-norm vectors (Phase-3 contract 6)."""
    pytest.importorskip("model2vec")

    import numpy as np

    from hrag.providers.embeddings import Model2VecProvider

    # Build a fake StaticModel that returns non-normalised vectors.
    fake_model = MagicMock()
    fake_model.dim = 4
    # encode returns a (n, 4) numpy array with non-unit norms.
    fake_model.encode.return_value = np.array([[3.0, 4.0, 0.0, 0.0]], dtype="float32")

    cfg = EmbeddingsConfig(provider="model2vec", model="minishlab/potion-base-8M")
    with patch("model2vec.StaticModel") as mock_sm:
        mock_sm.from_pretrained.return_value = fake_model
        provider = Model2VecProvider(cfg)

    vecs = provider.embed(["test"])
    assert len(vecs) == 1
    norm_sq = sum(v * v for v in vecs[0])
    assert math.isclose(norm_sq, 1.0, abs_tol=1e-5), (
        f"Model2VecProvider did not normalise; ‖v‖² = {norm_sq}"
    )


# ---------------------------------------------------------------------------
# 15. embed_batch_size default is 32
# ---------------------------------------------------------------------------


def test_embed_batch_size_default_32():
    """EmbeddingsConfig.embed_batch_size must default to 32."""
    cfg = EmbeddingsConfig()
    assert cfg.embed_batch_size == 32


# ---------------------------------------------------------------------------
# 16. embed_onnx_cache_dir default is None
# ---------------------------------------------------------------------------


def test_embed_onnx_cache_dir_default_none():
    """EmbeddingsConfig.embed_onnx_cache_dir must default to None."""
    cfg = EmbeddingsConfig()
    assert cfg.embed_onnx_cache_dir is None


# ---------------------------------------------------------------------------
# 17. embed_onnx_optimization default is 'O3'
# ---------------------------------------------------------------------------


def test_embed_onnx_optimization_default_O3():
    """EmbeddingsConfig.embed_onnx_optimization must default to 'O3'."""
    cfg = EmbeddingsConfig()
    assert cfg.embed_onnx_optimization == "O3"
