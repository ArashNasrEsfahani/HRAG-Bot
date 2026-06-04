"""EmbeddingProvider interface and concrete implementations.

Phase 9.3: providers carry an optional per-session LRU cache for query
embeddings. ``embed_one(text, *, session_id=None)`` consults the cache
when ``session_id`` is set AND ``config.query_cache_enabled``. Existing
call sites that omit ``session_id`` can opt in implicitly by entering an
``embedder.session(sid)`` context; ``embed_one`` falls back to the
contextvar when its kwarg is None. Default behaviour (no context, no
kwarg) bypasses the cache, preserving Phase-3 contract #4 + Phase-7-A
contract 16.

Phase 9.7 / Phase 10 Track A: SentenceTransformersProvider dispatches on
``cfg.precision`` (preferred) or ``cfg.embed_precision`` (back-compat
alias) across six inference backends:
  "fp32"      — default, byte-identical to previous behaviour.
  "fp16"      — half-precision via ``torch_dtype=float16`` (CUDA only;
                falls back to fp32 with a warning on CPU-only hosts).
  "bf16"      — bfloat16 via ``torch_dtype=bfloat16`` (CUDA only;
                falls back to fp32 with a warning on CPU-only hosts).
  "onnx"      — native sentence-transformers ONNX backend
                (``SentenceTransformer(..., backend="onnx")``).
                Falls back to fp32 when ``onnxruntime`` is missing.
  "onnx_int8" — ``onnx`` + ``export_dynamic_quantized_onnx_model`` (AVX-512
                VNNI dynamic quantization). Falls back through onnx to
                fp32 on any failure.
  "openvino"  — native sentence-transformers OpenVINO backend
                (``SentenceTransformer(..., backend="openvino")``).
                Falls back to fp32 when the openvino dep is missing.

Phase 10 Track C: ``Model2VecProvider`` exposes the
`StaticModel`-backed model2vec family for CPU-friendly distilled
embeddings (e.g. ``minishlab/potion-base-8M``). Selectable via
``config.provider = "model2vec"``.

All heavy deps (sentence_transformers, torch, optimum, onnxruntime,
openvino, model2vec, openai, numpy) are imported lazily inside the
method/branch that needs them so ``import hrag`` stays light.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Iterator, Optional, Sequence

from hrag.config import EmbeddingsConfig

logger = logging.getLogger(__name__)


# Phase 9.3 — ambient session id picked up by embed_one when its kwarg is None.
# Lets the orchestrator wrap the retrieval block in `with embedder.session(sid):`
# and have every retriever's internal embed_one call hit the cache without
# widening 9 retriever signatures.
_session_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hrag_embedder_session", default=None,
)


class EmbeddingProvider(ABC):
    name: str = "abstract"

    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        # Phase 9.3 — session_id -> OrderedDict[query, vector]. Bounded per
        # session by config.query_cache_size; oldest entries evicted first.
        self._query_cache: dict[str, OrderedDict[str, list[float]]] = {}

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text."""

    def embed_one(self, text: str, *, session_id: Optional[str] = None) -> list[float]:
        """Embed a single text. Caches when a session is active and caching is on.

        Resolution order for the session key:
          1. ``session_id`` kwarg (explicit).
          2. The ambient ``_session_var`` set by ``embedder.session(sid)``.
          3. None — cache bypassed.
        """
        sid = session_id if session_id is not None else _session_var.get()
        if sid is None or not getattr(self.config, "query_cache_enabled", True):
            return self.embed([text])[0]

        sess = self._query_cache.get(sid)
        if sess is None:
            sess = OrderedDict()
            self._query_cache[sid] = sess
        cached = sess.get(text)
        if cached is not None:
            sess.move_to_end(text)
            return cached

        vec = self.embed([text])[0]
        sess[text] = vec
        sess.move_to_end(text)
        cap = int(getattr(self.config, "query_cache_size", 64) or 64)
        while len(sess) > cap:
            sess.popitem(last=False)
        return vec

    def invalidate_session(self, session_id: str) -> None:
        """Drop the per-session cache for ``session_id``. No-op if absent."""
        self._query_cache.pop(session_id, None)

    @contextlib.contextmanager
    def session(self, session_id: Optional[str]) -> Iterator[None]:
        """Set an ambient session id for the duration of the block.

        ``embed_one`` calls inside this block (even from deeply-nested
        retriever code) automatically share the per-session cache without
        needing to thread ``session_id`` through every signature.
        """
        token = _session_var.set(session_id)
        try:
            yield
        finally:
            _session_var.reset(token)

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


def get_embedding_provider(config: EmbeddingsConfig) -> EmbeddingProvider:
    name = config.provider.lower().strip()
    if name in ("sentence-transformers", "st", "sbert"):
        return SentenceTransformersProvider(config)
    if name == "model2vec":
        return Model2VecProvider(config)
    if name == "openai":
        return OpenAIEmbeddingProvider(config)
    raise ValueError(f"Unknown embedding provider: {config.provider!r}")


def _resolve_precision(config: EmbeddingsConfig) -> str:
    """Phase 9.7 back-compat: prefer ``precision`` when set, else fall back
    to the older ``embed_precision`` alias. Returns a lower-case string;
    defaults to ``"fp32"`` when neither is provided.
    """
    precision = getattr(config, "precision", None)
    if precision is None or precision == "":
        precision = getattr(config, "embed_precision", None)
    if precision is None or precision == "":
        precision = "fp32"
    return str(precision).lower().strip()


class SentenceTransformersProvider(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)
        precision = _resolve_precision(config)
        device = getattr(config, "embed_device", None)
        onnx_cache_dir = getattr(config, "embed_onnx_cache_dir", None)

        self._backend: str = "fp32"
        self._model: Any = None

        logger.info("[embeddings] loading %s as %s", config.model, precision)

        if precision == "fp16":
            self._model, self._backend = self._try_load_torch_dtype(
                config.model, device, "float16",
            )
        elif precision == "bf16":
            self._model, self._backend = self._try_load_torch_dtype(
                config.model, device, "bfloat16",
            )
        elif precision == "onnx":
            self._model, self._backend = self._try_load_onnx(
                config.model, device, onnx_cache_dir, quantize=False,
            )
        elif precision == "onnx_int8":
            self._model, self._backend = self._try_load_onnx(
                config.model, device, onnx_cache_dir, quantize=True,
            )
        elif precision == "openvino":
            self._model, self._backend = self._try_load_openvino(
                config.model, device,
            )
        # precision == "fp32" (and any unrecognised value) — fall through.

        if self._backend == "fp32":
            # Default path — identical to pre-Phase-9.7 behaviour.
            self._model = self._load_fp32(config.model, device)

        self._dim: int = int(self._model.get_sentence_embedding_dimension())
        logger.info("[embeddings] backend=%s dim=%d", self._backend, self._dim)

    # ------------------------------------------------------------------
    # Precision-backend loaders. Each returns (model_or_None, backend_tag).
    # ``backend_tag == "fp32"`` signals "fallback requested" and the
    # caller will re-load via ``_load_fp32``.
    # ------------------------------------------------------------------

    @staticmethod
    def _load_fp32(model_id: str, device: Optional[str]) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "The 'sentence-transformers' package is required."
            ) from e
        kw: dict[str, Any] = {}
        if device is not None:
            kw["device"] = device
        return SentenceTransformer(model_id, **kw)

    @staticmethod
    def _try_load_torch_dtype(
        model_id: str, device: Optional[str], dtype_str: str,
    ) -> tuple[Any, str]:
        """Load via ``model_kwargs={"torch_dtype": dtype_str}``.

        ``dtype_str`` is one of ``"float16"`` / ``"bfloat16"``. Only honoured
        when CUDA is available; CPU-only hosts silently fall back to fp32
        with a warning (half-precision on CPU often produces NaN outputs).
        """
        backend_tag = "fp16" if dtype_str == "float16" else "bf16"
        try:
            import torch  # noqa: PLC0415
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "[embeddings] %s requested but torch/sentence_transformers "
                "missing; falling back to fp32.", backend_tag,
            )
            return None, "fp32"

        try:
            cuda_available = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            cuda_available = False
        if not cuda_available:
            logger.warning(
                "[embeddings] %s requested but CUDA is unavailable; "
                "falling back to fp32.", backend_tag,
            )
            return None, "fp32"

        resolved_device = device if device is not None else "cuda"
        try:
            model = SentenceTransformer(
                model_id,
                device=resolved_device,
                model_kwargs={"torch_dtype": dtype_str},
            )
            return model, backend_tag
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[embeddings] %s load failed (%s); falling back to fp32.",
                backend_tag, exc,
            )
            return None, "fp32"

    @staticmethod
    def _try_load_onnx(
        model_id: str,
        device: Optional[str],
        cache_dir: Optional[str],
        *,
        quantize: bool,
    ) -> tuple[Any, str]:
        """Native sentence-transformers ONNX backend; optionally quantized.

        Falls back through onnx → fp32 on any error. The quantize path uses
        ``export_dynamic_quantized_onnx_model`` with the ``avx512_vnni``
        preset; the resulting ONNX file is cached inside the ST model
        directory so subsequent loads skip the export step.
        """
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "[embeddings] onnx requested but sentence_transformers is "
                "missing; falling back to fp32.",
            )
            return None, "fp32"
        try:
            import onnxruntime  # noqa: F401, PLC0415
        except ImportError:
            logger.warning(
                "[embeddings] onnx requested but onnxruntime is unavailable; "
                "falling back to fp32. Install with `pip install -e .[quantize]`.",
            )
            return None, "fp32"

        kw: dict[str, Any] = {"backend": "onnx"}
        if device is not None:
            kw["device"] = device
        if cache_dir:
            kw["cache_folder"] = cache_dir

        try:
            model = SentenceTransformer(model_id, **kw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[embeddings] onnx load failed (%s); falling back to fp32.", exc,
            )
            return None, "fp32"

        if not quantize:
            return model, "onnx"

        # Quantize: best-effort. If the export helper or the quantization
        # backend is missing we keep the un-quantized ONNX model rather than
        # falling all the way back to fp32 — the user opted into ONNX and
        # got it, just not the int8 variant.
        try:
            from sentence_transformers.backend import (  # noqa: PLC0415
                export_dynamic_quantized_onnx_model,
            )
        except ImportError:
            logger.warning(
                "[embeddings] onnx_int8 requested but "
                "export_dynamic_quantized_onnx_model is unavailable in this "
                "sentence_transformers version; serving plain onnx.",
            )
            return model, "onnx"

        try:
            export_dynamic_quantized_onnx_model(model, "avx512_vnni", model_id)
            return model, "onnx_int8"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[embeddings] onnx_int8 quantization failed (%s); serving "
                "plain onnx.", exc,
            )
            return model, "onnx"

    @staticmethod
    def _try_load_openvino(
        model_id: str, device: Optional[str],
    ) -> tuple[Any, str]:
        """Native sentence-transformers OpenVINO backend."""
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "[embeddings] openvino requested but sentence_transformers "
                "is missing; falling back to fp32.",
            )
            return None, "fp32"
        try:
            import openvino  # noqa: F401, PLC0415
        except ImportError:
            logger.warning(
                "[embeddings] openvino requested but the 'openvino' package "
                "is unavailable; falling back to fp32.",
            )
            return None, "fp32"

        kw: dict[str, Any] = {"backend": "openvino"}
        if device is not None:
            kw["device"] = device

        try:
            model = SentenceTransformer(model_id, **kw)
            return model, "openvino"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[embeddings] openvino load failed (%s); falling back to fp32.",
                exc,
            )
            return None, "fp32"

    # ------------------------------------------------------------------
    # embed() — Phase 3 contract 6: returned vectors must be L2-normalised.
    # All ST backends (fp32 / fp16 / bf16 / onnx / onnx_int8 / openvino)
    # share the same encode() interface.
    # ------------------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    @property
    def dim(self) -> int:
        return int(self._dim)


def dimension_for_model(model_id: str) -> int | None:
    """Look up the embedding dimension of *model_id* in the curated suggestions.

    Returns the ``dim`` integer when the model appears in
    ``EmbeddingsConfig.suggested_models``, or ``None`` when it is not on the
    list (the provider will still load it; we just cannot cheaply determine
    the dimension without instantiating the model, which is expensive).

    Pure function — no model loading, no network calls.
    """
    from hrag.config import EmbeddingsConfig  # noqa: PLC0415

    defaults = EmbeddingsConfig().suggested_models
    for entry in defaults:
        if entry.get("model") == model_id:
            return entry.get("dim")
    return None


class Model2VecProvider(EmbeddingProvider):
    """Phase 10 Track C — model2vec ``StaticModel`` family.

    Distilled static embeddings (e.g. ``minishlab/potion-base-8M``) that run
    on CPU at hundreds of MB/s. Model2vec's ``encode()`` does NOT unit-norm
    the output, so we apply our own L2-normalization to honour Phase-3
    contract 6.
    """

    name = "model2vec"

    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)
        try:
            from model2vec import StaticModel  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "The 'model2vec' package is required for the model2vec "
                "embedding provider. Install with: pip install -e .[model2vec]"
            ) from e
        self._model = StaticModel.from_pretrained(config.model)
        self._dim: int = int(self._model.dim)
        logger.info(
            "[embeddings] backend=model2vec model=%s dim=%d",
            config.model, self._dim,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        import numpy as np  # noqa: PLC0415

        vecs = np.asarray(self._model.encode(list(texts)), dtype=np.float32)
        # Phase-3 contract 6: every embedding L2-normalised. model2vec does
        # not normalize by default, and the README's `model.encode([...])`
        # example exposes no `normalize=` kwarg, so we always do it here.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return (vecs / norms).tolist()

    @property
    def dim(self) -> int:
        return int(self._dim)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for OpenAI embeddings.")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("The 'openai' package is required.") from e
        self._client = OpenAI(api_key=api_key)
        # text-embedding-3-small=1536, text-embedding-3-large=3072
        self._dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.config.model, input=list(texts))
        return [d.embedding for d in resp.data]

    @property
    def dim(self) -> int:
        return self._dim_map.get(self.config.model, self.config.dim)
