"""HybridRetriever: fuses multiple retrievers via Reciprocal Rank Fusion (RRF).

RRF score for chunk c:
    Σ over retriever r:  weight_r / (k + rank_r(c))

where rank_r(c) is the 1-indexed rank of c in retriever r's result list
(or absent, meaning it contributes 0 to the sum). k is a smoothing constant
(default 60) that prevents very high scores for top-ranked chunks dominating
the fusion completely.

Typical usage
-------------
    bm25 = BM25Retriever(db)
    vector = VectorRetriever(db, vector_store, embedder)
    hybrid = HybridRetriever([vector, bm25], weights=[1.0, 1.0])
    results = hybrid.retrieve(query, user_id, top_k=10)

Each sub-retriever is called with top_k * 2 (oversampling) so that RRF has
meaningful signal even for chunks that appear only in one list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from hrag.retrieval.base import Retriever
from hrag.types import RetrievalResult

if TYPE_CHECKING:
    from hrag.intent import Intent


class HybridRetriever(Retriever):
    """Fuses two or more retrievers via Reciprocal Rank Fusion (RRF).

    RRF score for chunk c:  Σ over retriever r:  weight_r / (k + rank_r(c))
    where rank_r(c) is the 1-indexed rank of c in retriever r's results
    (or infinity if not retrieved). k is a smoothing constant; default 60.

    This is a near-instant alternative to LLM reranking — both retrievers
    run independently then their rankings are merged.
    """

    name = "hybrid"

    def __init__(
        self,
        retrievers: list[Retriever],
        weights: Optional[list[float]] = None,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one retriever.")

        if weights is None:
            weights = [1.0] * len(retrievers)

        if len(weights) != len(retrievers):
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(retrievers)={len(retrievers)}."
            )

        self._retrievers = retrievers
        self._weights = weights
        self._rrf_k = rrf_k

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
        """Run each sub-retriever (oversampled), fuse with RRF, return top_k.

        Each returned RetrievalResult:
        - Takes its Chunk from whichever sub-retriever ranked it first.
        - Sets ``retriever="hybrid"``.
        - Sets ``score`` to the fused RRF score.

        ``where`` is forwarded to every sub-retriever. Sub-retrievers that
        don't honour metadata filters simply ignore it.
        """
        oversample = top_k * 2

        # chunk_id → {"chunk": Chunk, "rrf_score": float}
        fused: dict[str, dict] = {}

        for retriever, weight in zip(self._retrievers, self._weights):
            results = retriever.retrieve(
                query=query,
                user_id=user_id,
                top_k=oversample,
                source_types=source_types,
                intent_hint=intent_hint,
                where=where,
            )

            for rank, result in enumerate(results, start=1):
                cid = result.chunk.chunk_id
                contribution = weight / (self._rrf_k + rank)

                if cid not in fused:
                    fused[cid] = {
                        "chunk": result.chunk,
                        "rrf_score": 0.0,
                    }
                fused[cid]["rrf_score"] += contribution

        # Sort by fused RRF score descending; take top_k
        ranked = sorted(fused.values(), key=lambda d: d["rrf_score"], reverse=True)
        ranked = ranked[:top_k]

        return [
            RetrievalResult(
                chunk=entry["chunk"],
                score=entry["rrf_score"],
                retriever=self.name,
                rerank_score=None,
            )
            for entry in ranked
        ]
