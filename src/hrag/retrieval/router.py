"""QueryRouter: classify a query and dispatch to the appropriate Retriever(s).

The router is the keystone of the cross-document fix in Phase 2: it inspects
the question with a tiny LLM classifier, then routes to vector / KG-PPR /
community retrievers as appropriate. For comparison questions like
"How do HippoRAG and RAGate differ?" the router fans out to all three and
fuses the result lists with Reciprocal Rank Fusion (RRF).

Routes
------
- entity          -> kg_ppr + bm25 + vector, fused via RRF
                     (when short_circuit=True: kg_ppr only, or first available)
- global          -> community (fall back to vector)
                     (when short_circuit=True: community only, or first available)
- cross_document  -> kg_ppr + bm25 + community + vector, fused via RRF
- ambiguous       -> kg_ppr + bm25 + vector, fused via RRF

Sub-retrievers are all optional. If a route requires a retriever that wasn't
injected, the router silently degrades to whatever IS available (e.g.
``cross_document`` with only vector + community returns those two fused;
``entity`` with no kg_ppr or bm25 falls back to vector).

Classifications are cached in-memory per query string. Wave 5 wiring note: the
LLM call is small (~20 tokens, temperature=0.0) but adds one round-trip per
unique question. To make it optional, factor a ``classify=`` argument into
build_retriever / orchestrator that swaps in a NoopRouter (always returns
"ambiguous") for benchmarking.

Phase 9.11 — Speculative short-circuit
---------------------------------------
When ``short_circuit=True`` (passed by the factory when
``cfg.retrieval.router_short_circuit`` is true, default ON), the router
skips the multi-retriever RRF fusion for ``entity`` and ``global`` routes:

  entity  -> kg_ppr only (falls back to bm25, then vector if absent)
  global  -> community only (falls back to vector if absent)

``cross_document`` and ``ambiguous`` always use the full RRF fan-out — those
routes exist precisely to hedge across multiple sources.

A ``router_short_circuit`` progress event is emitted (via the optional
``progress`` callback) when short-circuiting fires:
  payload: {"label": str, "retriever": str}

NOTE: logprob-based confidence is not available from local Ollama models.
The short-circuit fires unconditionally on the label for entity/global; the
RRF fan-out was always a hedge for cross_document/ambiguous, not for these
clearly-targeted routes. OpenAI/Anthropic providers that expose logprobs could
plug in a confidence check in a future phase, but it is not needed here.

Observable events emitted by router.py
---------------------------------------
  ``router_short_circuit`` — payload: {"label": str, "retriever": str}.
    Fires when short_circuit=True AND the label is "entity" or "global".
    Only emitted when a ``progress`` callback was passed at construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional

from hrag.providers.llm import LLMProvider
from hrag.retrieval.base import Retriever
from hrag.types import RetrievalResult

if TYPE_CHECKING:
    from hrag.intent import Intent


logger = logging.getLogger(__name__)


ClassLabel = Literal["entity", "global", "cross_document", "ambiguous"]
_VALID_LABELS: tuple[ClassLabel, ...] = ("entity", "global", "cross_document", "ambiguous")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_llm_output(raw: str) -> str:
    """Strip common wrappers an LLM may add around a single-token answer.

    Mirrored from ``query_rewriter._clean_llm_output`` to keep parsing
    consistent across LLM-driven micro-modules.
    """
    text = raw.strip().strip("`").strip()
    for _ in range(3):
        prev = text
        for prefix in ("answer:", "label:", "category:", "a:"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        if text == prev:
            break
    # Take only the first line / first whitespace-delimited token-ish segment.
    text = text.splitlines()[0].strip() if text else text
    return text


# Mirrored from hybrid.py to avoid cross-module private import. The router
# composes RRF fusion over an arbitrary list of result-lists; HybridRetriever
# uses an equivalent algorithm internally over its sub-retrievers.
def _rrf_fuse(
    per_retriever: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int = 30,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion over multiple result lists.

    score(c) = Σ over retriever r:  1 / (k + rank_r(c))

    where rank is 1-indexed. Chunks absent from a list contribute 0.
    Returns up to ``top_k`` ranked results, tagged ``retriever="router"``.
    """
    if not per_retriever:
        return []

    fused: dict[str, dict] = {}

    for results in per_retriever:
        for rank, result in enumerate(results, start=1):
            cid = result.chunk.chunk_id
            contribution = 1.0 / (k + rank)
            if cid not in fused:
                fused[cid] = {"chunk": result.chunk, "rrf_score": 0.0}
            fused[cid]["rrf_score"] += contribution

    ranked = sorted(fused.values(), key=lambda d: d["rrf_score"], reverse=True)
    ranked = ranked[:top_k]

    return [
        RetrievalResult(
            chunk=entry["chunk"],
            score=entry["rrf_score"],
            retriever="router",
            rerank_score=None,
        )
        for entry in ranked
    ]


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


class QueryRouter(Retriever):
    """Classifies a query and dispatches to the right Retriever(s).

    All sub-retrievers are optional; routes degrade gracefully when something
    isn't wired in. Classification results are cached in-memory per query.
    """

    name = "router"

    def __init__(
        self,
        llm: LLMProvider,
        vector_retriever: Optional[Retriever] = None,
        kg_ppr_retriever: Optional[Retriever] = None,
        community_retriever: Optional[Retriever] = None,
        bm25_retriever: Optional[Retriever] = None,
        rrf_k: int = 60,
        short_circuit: bool = False,
        progress: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self._llm = llm
        self._vector = vector_retriever
        self._kg_ppr = kg_ppr_retriever
        self._community = community_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k
        # Phase 9.11: when True, skip multi-retriever RRF fusion for entity/global
        # routes and call only the primary retriever for that label. Default False
        # so direct construction in tests preserves the existing fusion behaviour;
        # the factory passes cfg.retrieval.router_short_circuit (default True).
        self._short_circuit = short_circuit
        # Optional progress callback. Called with (event_name, payload_dict) when
        # short-circuiting fires. Matches the orchestrator's progress signature.
        self._progress = progress

        prompt_path = Path(__file__).parent.parent / "prompts" / "router.md"
        self._template = prompt_path.read_text(encoding="utf-8")

        self._cache: dict[str, ClassLabel] = {}

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, query: str) -> ClassLabel:
        """Run the LLM classifier; cache; return one of the four labels.

        On any LLM failure or unrecognized output, default to ``"entity"``
        — precision-favoring fallback that pins retrieval to the KG/vector
        path with sparse anchors rather than fanning out blindly.
        """
        if not query or not query.strip():
            return "entity"

        cached = self._cache.get(query)
        if cached is not None:
            return cached

        try:
            prompt = self._template.format(query=query)
            raw = self._llm.complete(prompt, temperature=0.0, max_tokens=20)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Router LLM call failed (%s); defaulting to entity.", exc)
            label: ClassLabel = "entity"
            self._cache[query] = label
            return label

        cleaned = _clean_llm_output(raw or "").lower()
        label = self._match_label(cleaned)
        self._cache[query] = label
        return label

    @staticmethod
    def _match_label(text: str) -> ClassLabel:
        """Substring-match against the four labels.

        Order matters: ``cross_document`` is checked first so it isn't shadowed
        by a substring match against ``entity`` or ``ambiguous``. ``global``
        and ``ambiguous`` come before ``entity`` for the same reason.
        Unknown / empty text → ``"entity"`` (precision-favoring fallback).
        """
        if not text:
            return "entity"
        if "cross_document" in text or "cross-document" in text or "cross document" in text:
            return "cross_document"
        if "global" in text:
            return "global"
        if "ambiguous" in text:
            return "ambiguous"
        if "entity" in text:
            return "entity"
        return "entity"

    # ------------------------------------------------------------------
    # Retriever interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 30,
        source_types: Optional[list[str]] = None,
        intent_hint: Optional["Intent"] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []

        # Short-circuit: when the orchestrator has already classified the intent
        # we skip the LLM classify call entirely. FACTUAL queries (the only ones
        # that reach retrieval) are routed to cross_document — the RRF fusion
        # over kg_ppr + bm25 + community + vector is the safest default for any
        # factual question that may draw on multiple documents.
        if intent_hint is not None:
            label: ClassLabel = "cross_document"
        else:
            label = self.classify(query)

        if label == "entity":
            # Phase 9.11 — speculative short-circuit.
            # When enabled, skip the multi-retriever RRF fusion and call ONLY
            # the primary entity-route retriever (kg_ppr, or first available
            # fallback). This is safe because entity queries are tightly scoped;
            # the RRF fan-out was a hedge for cross_document/ambiguous, not for
            # clearly-targeted entity lookups.
            if self._short_circuit:
                chosen = self._call_first_available(
                    [self._kg_ppr, self._bm25, self._vector],
                    query, user_id, top_k, source_types, intent_hint, where,
                )
                chosen_name = self._first_available_name(
                    [self._kg_ppr, self._bm25, self._vector]
                )
                if self._progress is not None:
                    self._progress(
                        "router_short_circuit",
                        {"label": label, "retriever": chosen_name},
                    )
                return chosen

            # Full fusion path (short_circuit=False or cross_document/ambiguous).
            # updated for iter-3: entity route RRF-fuses kg_ppr+bm25+vector
            # so exact technical phrases ("synonymy threshold", "0.05") are caught
            # by BM25 even when KG-PPR seeds weakly on multi-token terms.
            per_retriever = self._call_all(
                [self._kg_ppr, self._bm25, self._vector],
                query, user_id, top_k, source_types, intent_hint, where,
            )
            if not per_retriever:
                logger.warning("Router: no retrievers available for entity route.")
                return []
            if len(per_retriever) == 1:
                # Only one retriever wired; passthrough (no need to fuse).
                return per_retriever[0][:top_k]
            return _rrf_fuse(per_retriever, k=self._rrf_k, top_k=top_k)

        if label == "global":
            # Phase 9.11 — speculative short-circuit.
            # global queries target the community summary store; calling vector
            # as a fallback is already in _call_first_available. No fusion needed.
            if self._short_circuit:
                chosen = self._call_first_available(
                    [self._community, self._vector],
                    query, user_id, top_k, source_types, intent_hint, where,
                )
                chosen_name = self._first_available_name(
                    [self._community, self._vector]
                )
                if self._progress is not None:
                    self._progress(
                        "router_short_circuit",
                        {"label": label, "retriever": chosen_name},
                    )
                return chosen

            # Full path (short_circuit=False): same _call_first_available, but
            # with no fusion anyway (global was always a single-retriever path).
            return self._call_first_available(
                [self._community, self._vector],
                query, user_id, top_k, source_types, intent_hint, where,
            )

        if label == "cross_document":
            per_retriever = self._call_all(
                [self._kg_ppr, self._bm25, self._community, self._vector],
                query, user_id, top_k, source_types, intent_hint, where,
            )
            if not per_retriever:
                logger.warning("Router: no retrievers available for cross_document route.")
                return []
            if len(per_retriever) == 1:
                # Only one retriever wired; passthrough (no need to fuse).
                return per_retriever[0][:top_k]
            return _rrf_fuse(per_retriever, k=self._rrf_k, top_k=top_k)

        # ambiguous (default)
        per_retriever = self._call_all(
            [self._kg_ppr, self._bm25, self._vector],
            query, user_id, top_k, source_types, intent_hint, where,
        )
        if not per_retriever:
            logger.warning("Router: no retrievers available for ambiguous route.")
            return []
        if len(per_retriever) == 1:
            return per_retriever[0][:top_k]
        return _rrf_fuse(per_retriever, k=self._rrf_k, top_k=top_k)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_available_name(retrievers: list[Optional[Retriever]]) -> str:
        """Return the ``name`` attribute of the first non-None retriever.

        Used by the short-circuit path to include the chosen retriever's name
        in the ``router_short_circuit`` progress event payload.
        """
        for r in retrievers:
            if r is not None:
                return getattr(r, "name", r.__class__.__name__)
        return "none"

    def _call_first_available(
        self,
        retrievers: list[Optional[Retriever]],
        query: str,
        user_id: str,
        top_k: int,
        source_types: Optional[list[str]],
        intent_hint: Optional["Intent"] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Call the first non-None retriever that returns at least one result.

        Falls through if a retriever returns []. If a retriever raises, log
        and continue to the next.
        """
        for retriever in retrievers:
            if retriever is None:
                continue
            results = self._safe_retrieve(
                retriever, query, user_id, top_k, source_types, intent_hint, where,
            )
            if results:
                return results
        return []

    def _call_all(
        self,
        retrievers: list[Optional[Retriever]],
        query: str,
        user_id: str,
        top_k: int,
        source_types: Optional[list[str]],
        intent_hint: Optional["Intent"] = None,
        where: Optional[dict] = None,
    ) -> list[list[RetrievalResult]]:
        """Run every non-None retriever; return per-retriever result lists.

        Each retriever returns top_k candidates. Failed retrievers are
        dropped (with a warning), not propagated.
        """
        oversample = top_k
        outputs: list[list[RetrievalResult]] = []
        for retriever in retrievers:
            if retriever is None:
                continue
            results = self._safe_retrieve(
                retriever, query, user_id, oversample, source_types, intent_hint, where,
            )
            outputs.append(results)
        return outputs

    @staticmethod
    def _safe_retrieve(
        retriever: Retriever,
        query: str,
        user_id: str,
        top_k: int,
        source_types: Optional[list[str]],
        intent_hint: Optional["Intent"] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Call retriever.retrieve; on exception, log and return []."""
        try:
            return retriever.retrieve(
                query=query,
                user_id=user_id,
                top_k=top_k,
                source_types=source_types,
                intent_hint=intent_hint,
                where=where,
            )
        except TypeError:
            # Older retriever stubs may not accept `where=`. Retry without it
            # to stay backward-compatible with downstream wrappers/tests.
            try:
                return retriever.retrieve(
                    query=query,
                    user_id=user_id,
                    top_k=top_k,
                    source_types=source_types,
                    intent_hint=intent_hint,
                )
            except Exception as exc:  # noqa: BLE001
                name = getattr(retriever, "name", retriever.__class__.__name__)
                logger.warning("Router: retriever %s raised %s; skipping.", name, exc)
                return []
        except Exception as exc:  # noqa: BLE001
            name = getattr(retriever, "name", retriever.__class__.__name__)
            logger.warning("Router: retriever %s raised %s; skipping.", name, exc)
            return []
