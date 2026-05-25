"""Factories that build a Retriever and a Reranker from config.

These keep `orchestrator.py` simple: one call each, behaviour selected by config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

from hrag.config import RetrievalConfig
from hrag.db.connection import Database
from hrag.providers.embeddings import EmbeddingProvider
from hrag.providers.llm import LLMProvider
from hrag.retrieval.base import Retriever
from hrag.retrieval.batched_llm_reranker import BatchedLLMReranker
from hrag.retrieval.bm25 import BM25Retriever
from hrag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from hrag.retrieval.hybrid import HybridRetriever
from hrag.retrieval.query_rewriter import (
    HeuristicRewriter,
    LLMRewriter,
    NoopRewriter,
    QueryRewriter,
)
from hrag.retrieval.reranker import LLMReranker
from hrag.retrieval.vector import VectorStore
from hrag.retrieval.vector_retriever import VectorRetriever

if TYPE_CHECKING:
    from hrag.config import KGConfig, TaxonomyConfig
    from hrag.kg.communities import CommunityStore
    from hrag.kg.store import KGStore
    from hrag.retrieval.mst import MSTOrganizer
    from hrag.taxonomy.store import TaxonomyStore


class _RerankerLike(Protocol):
    name: str

    def rerank(self, query, results, threshold, top_k=None, progress=None): ...


def build_retriever(
    cfg: RetrievalConfig,
    db: Database,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    *,
    llm: "LLMProvider | None" = None,
    kg_store: "KGStore | None" = None,
    community_store: "CommunityStore | None" = None,
    kg_cfg: "KGConfig | None" = None,
    taxonomy_store: "TaxonomyStore | None" = None,
    taxonomy_cfg: "TaxonomyConfig | None" = None,
) -> Retriever:
    """Construct a Retriever from config.

    Modes:
      "vector"         -> VectorRetriever (dense only)
      "bm25"           -> BM25Retriever (sparse only)
      "hybrid"         -> HybridRetriever([VectorRetriever, BM25Retriever]) using RRF
      "kg_ppr"         -> KGPPRRetriever (HippoRAG-style PPR over KG)
      "community"      -> CommunityRetriever (GraphRAG community summaries)
      "router"         -> QueryRouter (LLM-classified dispatch over multiple retrievers)
      "taxonomy"       -> TaxonomyRetriever (beam-search a category tree, then chunk-fetch)

    When ``cfg.doc_scope_enabled`` is true (default), the built retriever is
    wrapped with ``DocScopedRetriever`` before being returned — adds a
    title-aware hard-filter + coarse-doc-rank fallback. Falls through
    gracefully when neither regime matches.
    """
    inner = _build_inner_retriever(
        cfg, db, vector_store, embedder,
        llm=llm, kg_store=kg_store,
        community_store=community_store, kg_cfg=kg_cfg,
        taxonomy_store=taxonomy_store, taxonomy_cfg=taxonomy_cfg,
    )
    if getattr(cfg, "doc_scope_enabled", True):
        from hrag.retrieval.doc_scope import DocScopedRetriever  # noqa: PLC0415

        return DocScopedRetriever(inner, db, embedder)
    return inner


def _build_inner_retriever(
    cfg: RetrievalConfig,
    db: Database,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    *,
    llm: "LLMProvider | None" = None,
    kg_store: "KGStore | None" = None,
    community_store: "CommunityStore | None" = None,
    kg_cfg: "KGConfig | None" = None,
    taxonomy_store: "TaxonomyStore | None" = None,
    taxonomy_cfg: "TaxonomyConfig | None" = None,
) -> Retriever:
    """Build the requested Retriever, without the doc-scope wrapper."""
    mode = cfg.retriever.lower().strip()

    if mode == "vector":
        return VectorRetriever(db, vector_store, embedder)

    if mode == "bm25":
        return BM25Retriever(db)

    if mode == "hybrid":
        vr = VectorRetriever(db, vector_store, embedder)
        br = BM25Retriever(db)
        retrievers: list[Retriever] = [vr, br]

        # Lightweight-default KG fusion: when the KG is populated, fan hybrid
        # retrieval over vector + BM25 + KG-PPR. No LLM router call needed —
        # all three retrievers run unconditionally, results fused via RRF.
        if (
            kg_store is not None
            and kg_cfg is not None
            and kg_cfg.enabled
            and llm is not None
        ):
            from hrag.kg.ner import build_ner  # noqa: PLC0415
            from hrag.retrieval.kg_ppr import KGPPRRetriever  # noqa: PLC0415

            ner = build_ner(kg_cfg, llm)
            retrievers.append(
                KGPPRRetriever(db, kg_store, ner, damping=kg_cfg.damping, seed_top_k=kg_cfg.seed_top_k, passage_node_alpha=kg_cfg.passage_node_alpha)
            )

        return HybridRetriever(
            retrievers,
            weights=cfg.rrf_weights,
            rrf_k=cfg.rrf_k,
        )

    if mode == "kg_ppr":
        if kg_store is None or kg_cfg is None or llm is None:
            raise ValueError(
                "retriever='kg_ppr' requires kg_store, kg_cfg, and llm"
            )
        from hrag.kg.ner import build_ner  # noqa: PLC0415
        from hrag.retrieval.kg_ppr import KGPPRRetriever  # noqa: PLC0415

        ner = build_ner(kg_cfg, llm)
        return KGPPRRetriever(db, kg_store, ner, damping=kg_cfg.damping, seed_top_k=kg_cfg.seed_top_k, passage_node_alpha=kg_cfg.passage_node_alpha)

    if mode == "community":
        if community_store is None:
            raise ValueError(
                "retriever='community' requires community_store"
            )
        if kg_cfg is not None and not kg_cfg.use_communities:
            raise ValueError(
                "retriever='community' requires kg.use_communities=true; "
                "rebuild the KG with that flag to populate the community store."
            )
        from hrag.retrieval.community import CommunityRetriever  # noqa: PLC0415

        return CommunityRetriever(community_store, embedder)

    if mode == "router":
        if llm is None:
            raise ValueError(
                "retriever='router' requires llm"
            )
        from hrag.retrieval.router import QueryRouter  # noqa: PLC0415

        # Always build the vector retriever.
        vector_r = VectorRetriever(db, vector_store, embedder)

        # Always build the BM25 retriever — sparse keyword match anchors
        # the cross_document and ambiguous RRF mixes for exact-string queries
        # (e.g. "RAGate-PEFT", "synonymy threshold") that dense retrieval
        # alone misses.
        bm25_r = BM25Retriever(db)

        # Build KG-PPR retriever if deps are present.
        kg_ppr_r = None
        if kg_store is not None and kg_cfg is not None:
            from hrag.kg.ner import build_ner  # noqa: PLC0415
            from hrag.retrieval.kg_ppr import KGPPRRetriever  # noqa: PLC0415

            ner = build_ner(kg_cfg, llm)
            kg_ppr_r = KGPPRRetriever(db, kg_store, ner, damping=kg_cfg.damping, seed_top_k=kg_cfg.seed_top_k, passage_node_alpha=kg_cfg.passage_node_alpha)

        # Build community retriever if deps are present.
        community_r = None
        if community_store is not None:
            from hrag.retrieval.community import CommunityRetriever  # noqa: PLC0415

            community_r = CommunityRetriever(community_store, embedder)

        return QueryRouter(
            llm,
            vector_retriever=vector_r,
            kg_ppr_retriever=kg_ppr_r,
            community_retriever=community_r,
            bm25_retriever=bm25_r,
            rrf_k=cfg.rrf_k,
            short_circuit=getattr(cfg, "router_short_circuit", True),
        )

    if mode == "taxonomy":
        if taxonomy_store is None or taxonomy_cfg is None:
            raise ValueError(
                "retriever='taxonomy' requires taxonomy_store and taxonomy_cfg"
            )
        from hrag.retrieval.taxonomy import TaxonomyRetriever  # noqa: PLC0415

        # Always have a fallback so empty-tree queries still work.
        fallback_mode = (taxonomy_cfg.fallback_retriever or "vector").lower()
        if fallback_mode == "vector":
            fallback = VectorRetriever(db, vector_store, embedder)
        elif fallback_mode == "bm25":
            fallback = BM25Retriever(db)
        elif fallback_mode == "hybrid":
            fallback = HybridRetriever(
                [VectorRetriever(db, vector_store, embedder), BM25Retriever(db)],
                weights=cfg.rrf_weights,
                rrf_k=cfg.rrf_k,
            )
        else:
            fallback = VectorRetriever(db, vector_store, embedder)
        return TaxonomyRetriever(
            db, vector_store, embedder, taxonomy_store, taxonomy_cfg, fallback,
        )

    raise ValueError(
        f"Unknown retriever mode: {cfg.retriever!r}. "
        f"Expected one of: vector, bm25, hybrid, kg_ppr, community, router, taxonomy."
    )


def build_mst_organizer(
    kg_cfg: "KGConfig",
    kg_store: "KGStore | None",
) -> "MSTOrganizer | None":
    """Build the MST organizer when KG is enabled and the store is non-None.

    Returns None when KG is disabled or store is missing — orchestrator
    treats None as 'skip MST step'.
    """
    if not kg_cfg or not kg_cfg.enabled or kg_store is None:
        return None
    from hrag.retrieval.mst import MSTOrganizer  # noqa: PLC0415

    return MSTOrganizer(kg_store)


def build_reranker(
    cfg: RetrievalConfig,
    llm: LLMProvider,
) -> Optional[_RerankerLike]:
    """Construct a Reranker from config, or None if reranking is disabled.

    Modes:
      "cross_encoder" -> CrossEncoderReranker (default; local, fast)
      "llm"           -> LLMReranker (per-chunk LLM scoring; slow but accurate)
      "batched_llm"   -> BatchedLLMReranker (one LLM call for all chunks)
    """
    if not cfg.rerank_enabled:
        return None

    mode = cfg.reranker.lower().strip()

    if mode in ("cross_encoder", "ce", "cross-encoder"):
        return CrossEncoderReranker(
            model_name=cfg.cross_encoder_model,
            quantize=cfg.rerank_quantize,
        )

    if mode == "llm":
        return LLMReranker(llm)

    if mode in ("batched_llm", "llm_batched", "batched"):
        return BatchedLLMReranker(llm)

    raise ValueError(
        f"Unknown reranker mode: {cfg.reranker!r}. "
        f"Expected one of: cross_encoder, llm, batched_llm."
    )


def build_query_rewriter(
    cfg: RetrievalConfig,
    llm: LLMProvider,
) -> QueryRewriter:
    """Construct a QueryRewriter from config.

    Modes:
      "heuristic" -> HeuristicRewriter (default; no LLM call)
      "llm"       -> LLMRewriter (one extra LLM call per turn)
      "none"      -> NoopRewriter (passthrough)
    """
    mode = (cfg.query_rewrite or "heuristic").lower().strip()

    if mode == "heuristic":
        return HeuristicRewriter()
    if mode == "llm":
        return LLMRewriter(llm)
    if mode in ("none", "noop", "off"):
        return NoopRewriter()

    raise ValueError(
        f"Unknown query_rewrite mode: {cfg.query_rewrite!r}. "
        f"Expected one of: heuristic, llm, none."
    )
