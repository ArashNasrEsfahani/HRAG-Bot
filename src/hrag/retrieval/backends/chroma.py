"""ChromaDB backend — current default, lifted from VectorStore.__init__."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_COLLECTION_NAME = "hrag_chunks"


class ChromaBackend:
    """Wraps a Chroma PersistentClient + one collection.

    Behaviour is byte-for-byte identical to the pre-refactor VectorStore: same
    collection name, same cosine space, same anonymized_telemetry=False, same
    upsert/delete/query semantics. The heavy ``chromadb`` import is deferred
    into ``__init__`` so module-level import of
    ``hrag.retrieval.backends.chroma`` stays cheap.
    """

    name = "chroma"

    def __init__(self, persist_path: str | Path) -> None:
        # Deferred import — keeps module load fast even when chromadb is heavy
        # or temporarily unavailable. VectorStore is constructed eagerly in
        # Orchestrator.__init__, so the import cost still happens there; the
        # win is that mere `import hrag.retrieval.backends` does not pay it.
        import chromadb  # noqa: PLC0415
        from chromadb.config import Settings  # noqa: PLC0415

        self._persist_path = Path(persist_path)
        self._persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_path),
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # VectorBackend protocol
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_where(self, where: dict) -> None:
        self._collection.delete(where=where)

    def query_one(
        self,
        embedding: list[float],
        top_k: int,
        where: Optional[dict],
    ) -> tuple[list[str], list[float]]:
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["distances"],
        )
        ids: list[str] = results["ids"][0]
        distances: list[float] = results["distances"][0]
        return ids, distances

    def update_metadata(self, ids: list[str], metadatas: list[dict]) -> None:
        # Chroma's update is best-effort; callers should treat failures as
        # non-fatal because SQLite is the authoritative tombstone record.
        self._collection.update(ids=list(ids), metadatas=list(metadatas))

    def count(self) -> int:
        return int(self._collection.count())
