"""EmbeddingProvider interface and concrete implementations.

Phase 9.3: providers carry an optional per-session LRU cache for query
embeddings. ``embed_one(text, *, session_id=None)`` consults the cache
when ``session_id`` is set AND ``config.query_cache_enabled``. Existing
call sites that omit ``session_id`` can opt in implicitly by entering an
``embedder.session(sid)`` context; ``embed_one`` falls back to the
contextvar when its kwarg is None. Default behaviour (no context, no
kwarg) bypasses the cache, preserving Phase-3 contract #4 + Phase-7-A
contract 16.

Phase 9.7: SentenceTransformersProvider supports three precision backends:
  "fp32"      — default, byte-identical to previous behaviour.
  "fp16"      — half-precision on GPU; auto-falls-back to fp32 on CPU-only.
  "onnx_int8" — INT8 ONNX export via optimum/onnxruntime; best CPU throughput.
                Falls back to fp32 when the optional dep group is missing.
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
    if name == "openai":
        return OpenAIEmbeddingProvider(config)
    raise ValueError(f"Unknown embedding provider: {config.provider!r}")


class SentenceTransformersProvider(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)
        precision = getattr(config, "embed_precision", "fp32") or "fp32"
        device = getattr(config, "embed_device", None)

        self._backend: str = "fp32"
        self._model: Any

        logger.info("[embeddings] loading %s as %s", config.model, precision)

        if precision == "onnx_int8":
            self._model, self._backend = self._try_load_onnx(config.model, device)
            if self._backend == "fp32":
                logger.warning(
                    "[embeddings] onnx_int8 requested but optimum/onnxruntime is "
                    "unavailable; falling back to fp32. "
                    "Install with `pip install -e .[quantize]`."
                )
        elif precision == "fp16":
            self._model, self._backend = self._try_load_fp16(config.model, device)
            if self._backend == "fp32":
                logger.warning(
                    "[embeddings] fp16 requested but CUDA is unavailable; "
                    "falling back to fp32."
                )

        if self._backend == "fp32":
            # Default path — identical to pre-Phase-9.7 behaviour.
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as e:
                raise ImportError(
                    "The 'sentence-transformers' package is required."
                ) from e
            kw: dict[str, Any] = {}
            if device is not None:
                kw["device"] = device
            self._model = SentenceTransformer(config.model, **kw)
            self._dim: int = self._model.get_sentence_embedding_dimension()
        elif self._backend == "onnx_int8":
            # dim stored by _try_load_onnx via a hidden attr; read it back.
            self._dim = self._model._dim  # type: ignore[attr-defined]
        else:
            # fp16 — model is a real SentenceTransformer, dim is readable normally.
            self._dim = self._model.get_sentence_embedding_dimension()

        logger.info("[embeddings] backend=%s", self._backend)

    # ------------------------------------------------------------------
    # Precision-backend loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _try_load_fp16(model_id: str, device: Optional[str]) -> tuple[Any, str]:
        """Best-effort FP16 load. Returns (model, backend_tag).

        FP16 on CPU is generally unsupported; we only apply .half() when CUDA
        is available so we don't silently produce NaN outputs.
        """
        try:
            import torch  # noqa: PLC0415
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            return None, "fp32"

        try:
            cuda_available = torch.cuda.is_available()
            resolved_device = device if device is not None else ("cuda" if cuda_available else "cpu")
            model = SentenceTransformer(model_id, device=resolved_device)
            if cuda_available:
                # Call .half() on the underlying transformer model.
                model._first_module().auto_model.half()
                return model, "fp16"
            else:
                return model, "fp32"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[embeddings] FP16 load failed (%s); using fp32.", exc)
            return None, "fp32"

    @staticmethod
    def _try_load_onnx(model_id: str, device: Optional[str]) -> tuple[Any, str]:
        """Best-effort INT8 ONNX load. Returns (bundle_or_none, backend_tag)."""
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction  # noqa: PLC0415
            from transformers import AutoConfig, AutoTokenizer  # noqa: PLC0415
        except ImportError:
            return None, "fp32"

        try:
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                model_id,
                export=True,
                provider="CPUExecutionProvider",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            hf_config = AutoConfig.from_pretrained(model_id)
            dim = hf_config.hidden_size
            bundle = _ONNXEmbeddingBundle(
                model=ort_model,
                tokenizer=tokenizer,
                dim=dim,
            )
            return bundle, "onnx_int8"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[embeddings] ONNX quantized load failed (%s); using fp32.", exc)
            return None, "fp32"

    # ------------------------------------------------------------------
    # embed() — Phase 3 contract 6: returned vectors must be L2-normalised.
    # ------------------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._backend == "onnx_int8":
            return self._model.encode(list(texts), normalize_embeddings=True)
        # fp32 or fp16 — both use the SentenceTransformer interface.
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


class _ONNXEmbeddingBundle:
    """Wraps an ORT feature-extraction model + tokenizer to expose a
    SentenceTransformer.encode-shaped API.

    Mean-pools token embeddings (weighted by attention mask) then L2-normalises
    the result — matching the default pooling strategy of all-mpnet-base-v2.
    Phase 3 contract 6: returned vectors are always unit-norm.
    """

    def __init__(self, model: Any, tokenizer: Any, dim: int) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._dim = dim

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> list[list[float]]:
        import numpy as np  # noqa: PLC0415

        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = self._model(**enc)
            # outputs.last_hidden_state: (batch, seq, hidden)
            hidden = outputs.last_hidden_state  # torch tensor or np array
            # Convert to numpy if necessary.
            if hasattr(hidden, "detach"):
                hidden = hidden.detach().cpu().numpy()
            else:
                hidden = np.asarray(hidden)

            # Attention-mask weighted mean pooling.
            mask = enc["attention_mask"]
            if hasattr(mask, "numpy"):
                mask = mask.numpy()
            else:
                mask = np.asarray(mask)
            mask_expanded = mask[:, :, None].astype(np.float32)
            sum_hidden = (hidden * mask_expanded).sum(axis=1)
            count = mask_expanded.sum(axis=1).clip(min=1e-9)
            pooled = sum_hidden / count  # (batch, hidden)

            if normalize_embeddings:
                norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
                pooled = pooled / norms

            all_vecs.extend(pooled.tolist())
        return all_vecs


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
