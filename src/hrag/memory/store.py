"""EpisodicMemoryStore: per-user episodic memory built on top of IngestPipeline.

One memory = one Document with source_type='episodic' = (usually) one chunk.
Memories share the same chunks/Chroma index as documents and compete with them
in retrieval by default. /forget tombstones via the existing chunks.excluded
column; the Chroma where-clause already filters excluded=1 out at query time
(see src/hrag/retrieval/vector.py:_build_where).

Why not a separate table or collection? Because re-using the document path
lets memories ride the same embedding model, the same reranker, the same
query rewriter, and the same RRF fusion. The router/KG layer is skipped for
episodic ingests (pipeline.py guards on source_type != 'episodic') to keep
/remember write latency under 100 ms.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from hrag.types import Chunk, Document

if TYPE_CHECKING:
    from hrag.db.connection import Database
    from hrag.ingest.pipeline import IngestPipeline


class EpisodicMemoryStore:
    """Add, list, and tombstone per-user episodic memories.

    All writes flow through ``IngestPipeline.ingest_document`` so the chunker,
    embedder, SQLite mirror and Chroma upsert all get the same treatment they
    do for regular documents. The KG triple-extraction step is suppressed
    inside the pipeline when ``doc.source_type == 'episodic'``.
    """

    def __init__(self, db: "Database", ingest_pipeline: "IngestPipeline") -> None:
        self._db = db
        self._ingest = ingest_pipeline

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add(
        self,
        user_id: str,
        text: str,
        *,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        source: str = "user",
    ) -> str:
        """Ingest *text* as one episodic memory and return the memory id (= doc_id).

        Title defaults to the first 60 characters of *text*. Tags and the
        source_session_id go into the document metadata JSON so they survive
        round-trips through the schema.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("EpisodicMemoryStore.add: text must be non-empty")

        memory_id = f"episodic:{user_id}:{uuid.uuid4().hex}"
        derived_title = title or _derive_title(text)

        doc = Document(
            doc_id=memory_id,
            user_id=user_id,
            source_path=f"memory://{memory_id}",
            title=derived_title,
            text=text,
            source_type="episodic",
            metadata={
                "tags": list(tags) if tags else [],
                "session_id": session_id,
                "source": source,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            },
        )
        self._ingest.ingest_document(doc)
        return memory_id

    def add_batch(
        self,
        user_id: str,
        items: list[dict],
        *,
        session_id: Optional[str] = None,
        source: str = "bulk",
    ) -> list[str]:
        """Add many memories sequentially; return list of memory ids in order.

        Each item is a dict with required key ``text`` and optional keys
        ``title``, ``tags``. Reuses ``add`` per item — the speed-up comes from
        the chunker batching embeddings inside ``ingest_document`` when a
        single memory produces multiple chunks. For short single-chunk notes
        this is one embed call per memory (~10–50 ms each on CPU mpnet);
        thousands of notes per minute is realistic.
        """
        ids: list[str] = []
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            ids.append(
                self.add(
                    user_id,
                    text,
                    title=item.get("title"),
                    tags=item.get("tags"),
                    session_id=session_id,
                    source=source,
                )
            )
        return ids

    # ------------------------------------------------------------------
    # Tombstone (/forget) path
    # ------------------------------------------------------------------

    def forget(self, user_id: str, chunk_id: str) -> bool:
        """Set excluded=1 for one chunk; mirror to Chroma metadata.

        Returns True if the row existed and was flipped (or was already
        excluded), False if no matching row was found.
        """
        cur = self._db.execute(
            "SELECT chunk_id, doc_id FROM chunks "
            "WHERE chunk_id = ? AND user_id = ?",
            (chunk_id, user_id),
        )
        row = cur.fetchone()
        if row is None:
            return False

        self._db.execute(
            "UPDATE chunks SET excluded = 1 WHERE chunk_id = ? AND user_id = ?",
            (chunk_id, user_id),
        )
        self._db.commit()
        _chroma_tombstone(self._ingest.vector_store, [chunk_id])
        return True

    def forget_memory(self, user_id: str, memory_id: str) -> int:
        """Tombstone every chunk that belongs to *memory_id*. Returns the count flipped."""
        cur = self._db.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ? AND user_id = ? AND excluded = 0",
            (memory_id, user_id),
        )
        chunk_ids = [r["chunk_id"] for r in cur.fetchall()]
        if not chunk_ids:
            return 0
        self._db.execute(
            "UPDATE chunks SET excluded = 1 WHERE doc_id = ? AND user_id = ?",
            (memory_id, user_id),
        )
        self._db.commit()
        _chroma_tombstone(self._ingest.vector_store, chunk_ids)
        return len(chunk_ids)

    def forget_by_query(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        retriever=None,
    ) -> list[str]:
        """Semantic-match *query* against episodic memories; return matched chunk_ids.

        The CLI layer is responsible for confirming with the user before
        calling :meth:`forget` on each id. Pass ``retriever`` (any
        ``Retriever`` instance) to use a different retriever; defaults to a
        fresh VectorRetriever built on the in-memory pipeline.
        """
        if retriever is None:
            from hrag.retrieval.vector_retriever import VectorRetriever  # noqa: PLC0415

            retriever = VectorRetriever(
                self._db,
                self._ingest.vector_store,
                self._ingest.embedder,
            )
        results = retriever.retrieve(
            query,
            user_id,
            top_k=top_k,
            source_types=["episodic"],
        )
        return [r.chunk.chunk_id for r in results]

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def list_recent(self, user_id: str, limit: int = 50) -> list[Chunk]:
        """Return episodic chunks for *user_id*, most-recently-ingested first.

        Ordering uses the parent ``documents.ingested_at`` for stability —
        chunk-level timestamps don't exist in the schema.
        """
        cur = self._db.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.user_id, c.text, c.title,
                   c.section, c.subsection, c.chunk_index, c.token_count,
                   c.source_type, c.metadata
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.user_id = ?
              AND c.source_type = 'episodic'
              AND c.excluded = 0
            ORDER BY d.ingested_at DESC, c.chunk_index ASC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return [_row_to_chunk(row) for row in cur.fetchall()]

    def count(self, user_id: str) -> int:
        cur = self._db.execute(
            "SELECT COUNT(*) AS n FROM chunks "
            "WHERE user_id = ? AND source_type = 'episodic' AND excluded = 0",
            (user_id,),
        )
        return int(cur.fetchone()["n"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_title(text: str, max_chars: int = 60) -> str:
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 1] + "…"


def _row_to_chunk(row) -> Chunk:
    raw_meta = row["metadata"]
    meta: dict = {}
    if raw_meta:
        try:
            meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        user_id=row["user_id"],
        text=row["text"],
        embedding_text=row["text"],
        title=row["title"] or "",
        section=row["section"] or "",
        subsection=row["subsection"] or "",
        chunk_index=row["chunk_index"],
        token_count=row["token_count"],
        source_type=row["source_type"],
        metadata=meta,
    )


def _chroma_tombstone(vector_store, chunk_ids: list[str]) -> None:
    """Best-effort mirror of excluded=1 into Chroma metadata.

    Falls back silently when the underlying store doesn't expose an `update`
    method — the SQLite tombstone is authoritative; the Chroma where-clause
    re-checks excluded at every query.
    """
    if not chunk_ids:
        return
    collection = getattr(vector_store, "_collection", None)
    if collection is None or not hasattr(collection, "update"):
        return
    try:
        collection.update(
            ids=list(chunk_ids),
            metadatas=[{"excluded": 1} for _ in chunk_ids],
        )
    except Exception:
        # Chroma's authoritative copy is SQLite — losing the metadata mirror
        # is non-fatal because VectorStore.query also re-checks via SQL hydrate.
        pass
