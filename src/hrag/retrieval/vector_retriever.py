"""VectorRetriever: Retriever implementation backed by VectorStore + SQLite hydration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from hrag.db.connection import Database
from hrag.providers.embeddings import EmbeddingProvider
from hrag.retrieval.base import Retriever
from hrag.retrieval.vector import VectorStore
from hrag.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from hrag.intent import Intent


class VectorRetriever(Retriever):
    """Embeds a query, searches the VectorStore, and hydrates results from SQLite.

    Retrieval flow
    --------------
    1. Embed query with the injected EmbeddingProvider.
    2. Query VectorStore → list[(chunk_id, score)].
    3. Batch-hydrate all chunk_ids from the `chunks` SQLite table (single IN query).
    4. Skip rows where excluded=1 (double-safety; Chroma already filters them,
       but the SQLite record is the authoritative tombstone).
    5. Reconstruct Chunk dataclasses and wrap in RetrievalResult.
    """

    name = "vector"

    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._embedder = embedder

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
        """Embed *query*, search Chroma, hydrate from SQLite, return results.

        ``where`` is an optional Chroma-style metadata filter (Phase 7-A);
        threaded into :meth:`VectorStore.query` and AND-combined with the
        existing user/excluded/source_type scoping.
        """

        # 1. Embed the query.
        query_embedding: list[float] = self._embedder.embed_one(query)

        # 2. Vector search → (chunk_id, score) pairs.
        pairs = self._vector_store.query(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            source_types=source_types,
            where=where,
        )

        if not pairs:
            return []

        # 3. Hydrate from SQLite in a single batch query.
        chunk_ids = [cid for cid, _ in pairs]
        score_map: dict[str, float] = {cid: score for cid, score in pairs}
        chunks_by_id = self._hydrate(chunk_ids)

        # 4 & 5. Build results, skipping tombstoned/missing chunks.
        results: list[RetrievalResult] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                # Chunk missing from SQLite — skip (deleted after index was built).
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score_map[chunk_id],
                    retriever=self.name,
                    rerank_score=None,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hydrate(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """SELECT chunks matching *chunk_ids* from SQLite; skip excluded rows.

        Returns a dict keyed by chunk_id.
        """
        if not chunk_ids:
            return {}

        placeholders = ",".join(["?"] * len(chunk_ids))
        sql = f"""
            SELECT
                chunk_id, doc_id, user_id,
                text, title, section, subsection,
                chunk_index, token_count, source_type,
                excluded, metadata, page, chapter
            FROM chunks
            WHERE chunk_id IN ({placeholders})
        """
        cursor = self._db.execute(sql, chunk_ids)
        rows = cursor.fetchall()

        result: dict[str, Chunk] = {}
        for row in rows:
            # Skip tombstoned chunks (excluded=1).
            if row["excluded"] == 1:
                continue

            raw_meta = row["metadata"]
            meta: dict = {}
            if raw_meta:
                try:
                    meta = json.loads(raw_meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            # Phase 13.1: rehydrate page + chapter defensively (old rows may
            # lack these columns; row.keys() guard handles that gracefully).
            row_keys = row.keys() if hasattr(row, "keys") else []
            page_val: Optional[int] = None
            if "page" in row_keys:
                try:
                    page_val = row["page"]  # may be None / NULL
                except (IndexError, KeyError):
                    pass

            if "chapter" in row_keys:
                try:
                    chapter_val: str = row["chapter"] or ""
                except (IndexError, KeyError):
                    chapter_val = ""
            else:
                chapter_val = ""

            # Merge chapter into metadata so consumers that inspect
            # chunk.metadata["chapter"] still find it.
            if chapter_val and "chapter" not in meta:
                meta["chapter"] = chapter_val

            chunk = Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                user_id=row["user_id"],
                text=row["text"],
                # embedding_text is not stored in SQLite; use text as a fallback.
                embedding_text=row["text"],
                title=row["title"] or "",
                section=row["section"] or "",
                subsection=row["subsection"] or "",
                chunk_index=row["chunk_index"],
                token_count=row["token_count"],
                source_type=row["source_type"],
                page=page_val,
                metadata=meta,
            )
            result[row["chunk_id"]] = chunk

        return result
