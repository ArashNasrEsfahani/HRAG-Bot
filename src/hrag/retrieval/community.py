"""CommunityRetriever: Retriever over GraphRAG community summaries.

Phase 2 retriever that embeds the query, searches the 'hrag_community_summaries'
ChromaDB collection via CommunityStore, hydrates summaries from SQLite, and
returns each community as a RetrievalResult whose Chunk.text is the community
summary.  Community results are drop-in compatible with the existing answer
prompt — no special-casing needed downstream.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from hrag.retrieval.base import Retriever
from hrag.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from hrag.intent import Intent
    from hrag.kg.communities import CommunityStore
    from hrag.providers.embeddings import EmbeddingProvider


logger = logging.getLogger(__name__)


class CommunityRetriever(Retriever):
    """Retriever over GraphRAG community summaries.

    Embeds the query, searches the 'hrag_community_summaries' Chroma collection,
    hydrates summaries from SQLite, and returns each as a RetrievalResult whose
    Chunk.text is the community summary. This makes community results
    drop-in compatible with the existing answer prompt — no special-casing
    needed downstream.
    """

    name = "community"

    def __init__(
        self,
        community_store: "CommunityStore",
        embedder: "EmbeddingProvider",
        levels: Optional[list[int]] = None,
    ) -> None:
        """Initialise the retriever.

        Parameters
        ----------
        community_store:
            Populated ``CommunityStore`` instance (Phase 2).
        embedder:
            Embedding provider used to embed the query.
        levels:
            Restrict retrieval to specific Leiden resolution levels.
            ``None`` means all levels.
        """
        self._community_store = community_store
        self._embedder = embedder
        self._levels = list(levels) if levels is not None else None

    # ------------------------------------------------------------------
    # Retriever interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        source_types: Optional[list[str]] = None,
        intent_hint: Optional["Intent"] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Return up to *top_k* community summaries ranked by embedding similarity.

        Parameters
        ----------
        query:
            The user query to embed and search with.
        user_id:
            Scopes retrieval to this user's community index.
        top_k:
            Maximum number of results to return.
        source_types:
            If provided and does not contain ``"community"``, returns ``[]``
            immediately — this retriever only produces community results.

        Note: this retriever does not honour ``where=`` metadata filters; pass
        to a vector-based retriever instead.
        """
        # `where` is accepted for Retriever Protocol compatibility but ignored.
        del where  # explicitly ignored
        # source_types filter: community results only make sense when
        # the caller hasn't explicitly excluded them.
        if source_types is not None and "community" not in source_types:
            return []

        # Guard against empty / whitespace-only queries.
        if not query or not query.strip():
            return []

        # Embed the query once.
        query_embedding: list[float] = self._embedder.embed_one(query)

        # Search the community collection — returns (community_id, score) pairs.
        pairs = self._community_store.query(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            levels=self._levels,
        )

        if not pairs:
            return []

        # Hydrate each community from SQLite and build RetrievalResult objects.
        results: list[RetrievalResult] = []
        _warned_missing: bool = False

        for community_id, similarity_score in pairs:
            community = self._community_store.get_community(community_id)
            if community is None:
                if not _warned_missing:
                    logger.warning(
                        "CommunityRetriever: community_id %r not found in SQLite; "
                        "skipping (further missing-community warnings suppressed "
                        "for this call).",
                        community_id,
                    )
                    _warned_missing = True
                continue

            level: int = community["level"]
            summary: str = community["summary"]
            member_chunks: list[str] = community.get("member_chunks") or []

            chunk = Chunk(
                chunk_id=f"community::{community_id}",
                doc_id=f"community_level_{level}",
                user_id=user_id,
                text=summary,
                embedding_text=summary,
                title=f"Community {community_id} (level {level})",
                section=f"Members: {len(member_chunks)} chunks",
                subsection="",
                chunk_index=0,
                token_count=0,
                source_type="community",
                metadata={
                    "community_id": community_id,
                    "level": level,
                    "member_chunk_ids": member_chunks,
                },
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=similarity_score,
                    retriever="community",
                    rerank_score=None,
                )
            )

        return results
