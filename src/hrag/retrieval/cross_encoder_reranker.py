"""Cross-encoder local reranker — drop-in replacement for LLMReranker."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from hrag.types import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder local reranker. Drop-in replacement for LLMReranker.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, 22M params).
    A single forward pass scores all candidates at once — typically <300ms
    on CPU for 10 chunks.

    Scores are raw logits (unbounded, but typically in the range -12..+12
    for ms-marco-trained models). Higher = more relevant.

    Phase 9.8: pass ``quantize=True`` to load the INT8 ONNX export instead
    of the FP32 sentence-transformers model. Requires the optional dep
    group ``quantize``. On import failure (optimum / onnxruntime missing),
    a single warning is logged and the FP32 backend is used.
    """

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        quantize: bool = False,
    ) -> None:
        self._model_name = model_name
        self._backend: str = "fp32"
        self._model: Any

        if quantize:
            self._model, self._backend = self._try_load_quantized(model_name, max_length)
            if self._backend == "fp32":
                logger.warning(
                    "CrossEncoderReranker: quantize=True requested but optimum/onnxruntime "
                    "is unavailable; falling back to FP32. Install with `pip install -e .[quantize]`."
                )

        if self._backend == "fp32":
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "The 'sentence-transformers' package is required for CrossEncoderReranker. "
                    "Install it with: pip install sentence-transformers"
                ) from e

            self._model = CrossEncoder(
                model_name,
                device=device,
                max_length=max_length,
            )

    @staticmethod
    def _try_load_quantized(model_name: str, max_length: int) -> tuple[Any, str]:
        """Best-effort INT8 ONNX load. Returns (model_or_none, backend_tag)."""
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification  # noqa: PLC0415
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError:
            return None, "fp32"

        try:
            model = ORTModelForSequenceClassification.from_pretrained(
                model_name,
                export=True,
                provider="CPUExecutionProvider",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
            return _ONNXBundle(model=model, tokenizer=tokenizer, max_length=max_length), "onnx_int8"
        except Exception as exc:  # noqa: BLE001
            logger.warning("ONNX quantized load failed (%s); using FP32.", exc)
            return None, "fp32"

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        threshold: float = 0.0,
        top_k: Optional[int] = None,
        progress: Optional[Callable[[int, int, float], None]] = None,
    ) -> list[RetrievalResult]:
        """Score, filter, and sort *results* using the cross-encoder.

        Parameters
        ----------
        query:      The user's original query string.
        results:    Candidate retrieval results to be scored.
        threshold:  Minimum logit score (inclusive) to keep a result.
                    Default 0.0 is a rough "neutral" cutoff for ms-marco models.
        top_k:      If given, truncate the final ranked list to this length.
        progress:   Optional callback emitted once after the batched forward
                    pass with (1, 1, max_score) for UX parity with LLMReranker.
                    Also emitted per-result as (i, total, score) so that CLI
                    progress bars can tick naturally.

        Returns
        -------
        A new list of RetrievalResult with .rerank_score populated,
        filtered (>= threshold) and sorted by score descending, truncated
        to top_k if given.
        """
        if not results:
            return []

        # Build (query, passage) pairs — use chunk.text (not embedding_text).
        pairs = [[query, result.chunk.text] for result in results]

        # Single batched forward pass — this is the key advantage over LLMReranker.
        if self._backend == "onnx_int8":
            scores = self._model.predict(pairs, batch_size=32)
        else:
            scores = self._model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # Mutate rerank_score on each result and apply threshold filter.
        scored: list[RetrievalResult] = []
        total = len(results)
        max_score = float(max(scores))

        for idx, (result, raw_score) in enumerate(zip(results, scores), start=1):
            score = float(raw_score)
            result.rerank_score = score
            if score >= threshold:
                scored.append(result)
            # Emit per-result ticks so the CLI progress bar advances naturally.
            if progress is not None:
                progress(idx, total, score)

        # Emit the canonical single-batch callback after all per-result ticks.
        if progress is not None:
            progress(1, 1, max_score)

        # Stable sort: primary = rerank_score desc, secondary = original score desc.
        scored.sort(
            key=lambda r: (r.rerank_score, r.score),  # type: ignore[operator]
            reverse=True,
        )

        if top_k is not None:
            scored = scored[:top_k]

        return scored


class _ONNXBundle:
    """Wraps an ORT model + tokenizer to expose a CrossEncoder.predict-shaped API."""

    def __init__(self, model, tokenizer, max_length: int = 512) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = max_length

    def predict(self, pairs, batch_size: int = 32):
        import numpy as np  # noqa: PLC0415

        out: list[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            texts_a = [p[0] for p in batch]
            texts_b = [p[1] for p in batch]
            enc = self._tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            logits = self._model(**enc).logits.detach().cpu().numpy()
            out.extend(float(x) for x in logits.reshape(-1))
        return np.asarray(out, dtype=np.float32)
