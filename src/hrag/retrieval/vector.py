"""Vector-store facade for the HRAG retrieval subsystem.

VectorStore is the public class call-sites already use. It delegates to a
:class:`VectorBackend` (default: :class:`ChromaBackend`); the backend can be
swapped via ``RetrievalConfig.vector_backend`` without touching call-sites.

Metadata layout per record
--------------------------
user_id      : str   — owner scope
doc_id       : str   — parent document
source_type  : str   — "document" | "episodic"
excluded     : int   — 0 = active, 1 = tombstoned (/forget)
chunk_index  : int   — positional index inside the document
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hrag.retrieval.backends.base import VectorBackend
from hrag.retrieval.backends.chroma import ChromaBackend
from hrag.types import Chunk


class VectorStore:
    """Pluggable vector store. Talks to a :class:`VectorBackend` under the hood.

    The public API (``add_chunks``, ``delete_doc``, ``query``) is unchanged.
    """

    def __init__(
        self,
        persist_path: str | Path,
        embed_dim: int,
        *,
        backend: VectorBackend | None = None,
    ) -> None:
        self._persist_path = Path(persist_path)
        self._embed_dim = embed_dim
        # Default to Chroma for byte-for-byte compatibility with the previous
        # behaviour. Callers that want sqlite-vec (or a mock) pass `backend=`.
        self._backend: VectorBackend = backend or ChromaBackend(self._persist_path)

    # ------------------------------------------------------------------
    # Backward-compatibility shim
    # ------------------------------------------------------------------

    @property
    def _collection(self):
        """Expose the underlying Chroma collection when present.

        :func:`hrag.memory.store._chroma_tombstone` reaches in here to mirror
        ``excluded=1`` metadata. Keeping the attribute available preserves
        that best-effort tombstone path without forcing every caller through
        a new abstraction. Non-Chroma backends simply don't expose it (and
        ``_chroma_tombstone``'s ``getattr(..., None)`` fallback no-ops).
        """
        return getattr(self._backend, "_collection", None)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        user_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks with their pre-computed embeddings."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length."
            )

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        embs: list[list[float]] = []

        for chunk, emb in zip(chunks, embeddings):
            ids.append(chunk.chunk_id)
            docs.append(chunk.text)
            metas.append(
                {
                    "user_id": chunk.user_id,
                    "doc_id": chunk.doc_id,
                    "source_type": chunk.source_type,
                    "excluded": 0,
                    "chunk_index": chunk.chunk_index,
                }
            )
            embs.append(emb)

        self._backend.upsert(
            ids=ids,
            embeddings=embs,
            documents=docs,
            metadatas=metas,
        )

    def delete_doc(self, user_id: str, doc_id: str) -> None:
        """Remove all chunks belonging to *doc_id* for *user_id*."""
        self._backend.delete_where(
            {"$and": [{"user_id": {"$eq": user_id}}, {"doc_id": {"$eq": doc_id}}]}
        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def query(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 30,
        source_types: Optional[list[str]] = None,
        doc_ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, similarity_score) pairs, highest similarity first.

        Backends return cosine *distance* in [0, 2]; we convert:
            similarity = 1 - distance   (clamped to [0, 1])

        The where clause always scopes by user_id and excluded=0.
        ``source_types`` adds a $in filter on source_type. ``doc_ids`` adds a
        $in filter on doc_id — used by :class:`TaxonomyRetriever` so the scan
        only touches chunks belonging to the leaf-picked documents.
        ``where`` is an optional extension dict (Phase 7-A): each key/value
        pair is AND-combined with the user/excluded/source_type/doc_id
        scoping as ``{key: {"$eq": value}}``.
        """
        where = _build_where(
            user_id=user_id,
            source_types=source_types,
            doc_ids=doc_ids,
            extra_where=where,
        )

        chunk_ids, distances = self._backend.query_one(
            embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        pairs: list[tuple[str, float]] = []
        for cid, dist in zip(chunk_ids, distances):
            score = max(0.0, min(1.0, 1.0 - dist))
            pairs.append((cid, score))

        # Backend already returns ascending distance; reverse gives descending score.
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_where(
    user_id: str,
    source_types: Optional[list[str]],
    doc_ids: Optional[list[str]] = None,
    extra_where: Optional[dict] = None,
) -> dict:
    """Build a Chroma-style `where` dict that always filters user_id and excluded.

    Chroma's `$and` requires at least two conditions; build the list
    dynamically so we only use `$and` when there are multiple conditions.

    ``extra_where`` (Phase 7-A) is an optional dict of additional equality
    constraints (e.g. ``{"has_math": True}``); each pair is AND-combined
    into the conditions list as ``{key: {"$eq": value}}``.
    """
    conditions: list[dict] = [
        {"user_id": {"$eq": user_id}},
        {"excluded": {"$eq": 0}},
    ]

    if source_types:
        if len(source_types) == 1:
            conditions.append({"source_type": {"$eq": source_types[0]}})
        else:
            conditions.append({"source_type": {"$in": source_types}})

    if doc_ids:
        if len(doc_ids) == 1:
            conditions.append({"doc_id": {"$eq": doc_ids[0]}})
        else:
            conditions.append({"doc_id": {"$in": doc_ids}})

    if extra_where:
        for key, value in extra_where.items():
            # If the caller already wrapped the value in a chroma operator
            # (e.g. {"$eq": True} or {"$in": [...]}), pass it through;
            # otherwise default to equality.
            if isinstance(value, dict) and any(
                isinstance(k, str) and k.startswith("$") for k in value.keys()
            ):
                conditions.append({key: value})
            else:
                conditions.append({key: {"$eq": value}})

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}
