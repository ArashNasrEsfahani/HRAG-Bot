"""Vector-backend Protocol.

The :class:`VectorBackend` Protocol is the seam between
:class:`hrag.retrieval.vector.VectorStore` and the concrete vector index.
Anything that satisfies the protocol can be plugged in via
``RetrievalConfig.vector_backend`` (currently ``"chroma"`` or
``"sqlite_vec"``).

Design rules:
- Methods are intentionally low-level — they mirror the few Chroma operations
  VectorStore actually needs. They are NOT the public surface that retrievers
  or the ingest pipeline call; those still go through :class:`VectorStore`.
- ``query_one`` returns ``(ids, distances)`` so VectorStore owns the
  distance→similarity conversion. Keeps the protocol provider-neutral.
- ``update_metadata`` is best-effort. The SQLite ``excluded`` column is
  authoritative for tombstones; the metadata mirror is only an optimization so
  ``VectorBackend.query_one`` can pre-filter excluded rows. Backends that
  cannot update metadata should raise NotImplementedError and the caller
  silently falls back to the SQLite filter.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """Minimal contract VectorStore depends on.

    Implementations: :class:`ChromaBackend` (default),
    :class:`SqliteVecBackend` (stub).
    """

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Insert or replace vectors. All four lists must have equal length."""
        ...

    def delete_where(self, where: dict) -> None:
        """Delete every record matching the Chroma-style where clause."""
        ...

    def query_one(
        self,
        embedding: list[float],
        top_k: int,
        where: Optional[dict],
    ) -> tuple[list[str], list[float]]:
        """Nearest-neighbour search for a single query vector.

        Returns ``(ids, distances)`` in ascending-distance order. Distances
        are cosine distance in ``[0, 2]``; VectorStore converts to similarity.
        """
        ...

    def update_metadata(self, ids: list[str], metadatas: list[dict]) -> None:
        """Update metadata for existing records (best-effort tombstone mirror)."""
        ...

    def count(self) -> int:
        """Return total record count (for diagnostics / status pages)."""
        ...

    def dim(self) -> Optional[int]:
        """Return the embedding dimensionality stored in the index, or None if empty.

        Used by :meth:`Orchestrator._check_embedding_dim_match` at startup to
        detect a model-swap-without-reingest situation before any retrieval occurs.
        Backends that cannot introspect the dim cheaply should return ``None``
        (the check is silently skipped).
        """
        ...
