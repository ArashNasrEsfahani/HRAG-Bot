"""IngestPipeline: orchestrates loading, chunking, embedding, and storing documents."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hrag.config import Config
from hrag.db.connection import Database
from hrag.providers.embeddings import EmbeddingProvider
from hrag.types import Chunk, Document
from hrag.ingest.loaders import load_document
from hrag.ingest.chunker import chunk_document
from hrag.ingest.quality import filter_chunks

if TYPE_CHECKING:  # pragma: no cover
    from hrag.kg.store import KGStore
    from hrag.providers.llm import LLMProvider
    from hrag.taxonomy.store import TaxonomyStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VectorStore Protocol (not implemented here; import from hrag.retrieval.vector)
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorStore(Protocol):
    def add_chunks(
        self, user_id: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None: ...

    def delete_doc(self, user_id: str, doc_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Supported extensions for directory ingestion
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
_EMBED_BATCH_SIZE = 32


def _mean_embedding(embeddings: list[list[float]]) -> list[float] | None:
    """Component-wise mean, L2-renormalized. Used to derive a doc-level centroid."""
    if not embeddings:
        return None
    dim = len(embeddings[0])
    acc = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            acc[i] += v
    n = float(len(embeddings))
    acc = [v / n for v in acc]
    norm = sum(v * v for v in acc) ** 0.5
    if norm > 0.0:
        acc = [v / norm for v in acc]
    return acc


# ---------------------------------------------------------------------------
# IngestPipeline
# ---------------------------------------------------------------------------

class IngestPipeline:
    """Orchestrates the full ingest flow: load -> chunk -> embed -> store."""

    def __init__(
        self,
        config: Config,
        db: Database,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        *,
        llm: "LLMProvider | None" = None,
        kg_store: "KGStore | None" = None,
        taxonomy_store: "TaxonomyStore | None" = None,
    ) -> None:
        self.config = config
        self.db = db
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.kg_store = kg_store
        self.taxonomy_store = taxonomy_store

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def ingest_path(self, path: str | Path, user_id: str) -> Document:
        """Load a file from disk, then run the full ingest pipeline."""
        doc = load_document(path, user_id)
        return self.ingest_document(doc)

    def ingest_document(self, doc: Document) -> Document:
        """Run the full ingest pipeline on an already-loaded Document.

        Steps:
          1. Upsert row in `documents` table.
          2. Chunk the document.
          3. Upsert chunks in `chunks` table.
          4. Embed chunk.embedding_text in batches of 32.
          5. Remove old vectors then add new ones.
          6. Commit and return.
        """
        # Stamp ingestion time
        doc.ingested_at = datetime.now(tz=timezone.utc)

        # 1. Upsert document row
        self._upsert_document(doc)

        # 2. Chunk
        chunks = chunk_document(doc, self.config.chunking)

        # 2b. Quality filter (optional, document-only)
        # Episodic memories are typically short ("Postgres > MySQL" = 3 tokens)
        # and would be dropped by the min_tokens/min_chars guards. Skip the
        # filter entirely for episodic ingests — the user explicitly chose to
        # remember each one.
        quality_cfg = self.config.chunking.quality
        if quality_cfg.enabled and doc.source_type != "episodic":
            chunks, dropped = filter_chunks(chunks, quality_cfg)
            breakdown: dict[str, int] = {}
            for _chunk, reason in dropped:
                # Normalise reason to a short key (everything before first space/paren)
                key = re.split(r"[ (]", reason)[0] if reason else "unknown"
                breakdown[key] = breakdown.get(key, 0) + 1
            print(
                f"[ingest] {doc.title}: {len(chunks)} chunks kept "
                f"({len(dropped)} dropped: {breakdown})"
            )
        else:
            print(f"[ingest] {doc.title}: {len(chunks)} chunks")

        # 3. Upsert chunk rows
        self._upsert_chunks(chunks)

        # 4. Embed in batches
        embeddings = self._embed_chunks(chunks)

        # 5. Update vector store
        self.vector_store.delete_doc(doc.user_id, doc.doc_id)
        if chunks:
            self.vector_store.add_chunks(doc.user_id, chunks, embeddings)

        # 5b. KG triple extraction + upsert (only when config.kg.enabled,
        # and only for document-type ingests — episodic memories skip the KG
        # to keep /remember write latency under 100 ms).
        if self.config.kg.enabled and doc.source_type != "episodic":
            if self.llm is None or self.kg_store is None:
                logger.warning(
                    "[ingest] kg.enabled=True but llm or kg_store is None — "
                    "skipping KG triple extraction for %r",
                    doc.title,
                )
            else:
                from hrag.kg.builder import TripleExtractor  # noqa: PLC0415 — lazy import
                extractor = TripleExtractor(
                    self.llm,
                    max_workers=self.config.kg.parallel_workers,
                    db=self.db,
                )
                triples = extractor.extract_batch(chunks)
                chunk_id_to_doc_id = {c.chunk_id: c.doc_id for c in chunks}
                self.kg_store.upsert_triples(doc.user_id, doc.doc_id, triples, chunk_id_to_doc_id)
                print(f"[ingest] {doc.title}: {len(triples)} triples extracted")

        # 5c. Taxonomy auto-assignment. Both documents and episodic memories
        # are filed (when taxonomy.include_episodic is true) — the whole point
        # of a personal tree is that memories live in it. Skipped when the
        # tree is empty — DocAssigner returns None and the doc remains
        # unfiled until the next `hrag taxonomy build`.
        include_episodic = getattr(
            self.config.taxonomy, "include_episodic", True
        )
        eligible_source = (
            doc.source_type != "episodic" or include_episodic
        )
        if (
            self.config.taxonomy.enabled
            and self.config.taxonomy.auto_assign_on_ingest
            and eligible_source
            and self.taxonomy_store is not None
            and self.llm is not None
            and chunks
        ):
            try:
                from hrag.taxonomy.assigner import DocAssigner  # noqa: PLC0415

                # Reuse the embeddings already computed above — DocAssigner
                # would otherwise re-embed every chunk just to derive the
                # doc centroid.
                doc_centroid = _mean_embedding(embeddings)
                assigner = DocAssigner(
                    self.db, self.llm, self.embedder, self.taxonomy_store,
                    self.config.taxonomy,
                )
                node_id = assigner.assign(
                    doc.user_id, doc.doc_id, centroid=doc_centroid,
                )
                if node_id:
                    print(f"[ingest] {doc.title}: filed under {node_id}")
                else:
                    print(f"[ingest] {doc.title}: taxonomy empty — unfiled")
            except Exception as exc:  # noqa: BLE001
                logger.warning("[ingest] taxonomy auto-assign failed for %r: %s", doc.title, exc)

        # 6. Commit
        self.db.commit()

        return doc

    def ingest_directory(
        self,
        dir_path: str | Path,
        user_id: str,
        recursive: bool = True,
    ) -> list[Document]:
        """Walk a directory and ingest all supported files.

        Skips files larger than 50 MB with a warning.
        Returns list of successfully ingested Documents.
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        pattern = "**/*" if recursive else "*"
        ingested: list[Document] = []

        for file_path in sorted(dir_path.glob(pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue

            file_size = file_path.stat().st_size
            if file_size > _MAX_FILE_BYTES:
                print(
                    f"[ingest] SKIP {file_path.name} "
                    f"({file_size / 1024 / 1024:.1f} MB > 50 MB limit)"
                )
                continue

            try:
                doc = self.ingest_path(file_path, user_id)
                ingested.append(doc)
            except Exception as exc:
                print(f"[ingest] ERROR {file_path.name}: {exc}")

        return ingested

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upsert_document(self, doc: Document) -> None:
        ingested_at_str = (
            doc.ingested_at.isoformat() if doc.ingested_at else datetime.now(tz=timezone.utc).isoformat()
        )
        self.db.execute(
            """
            INSERT OR REPLACE INTO documents
                (doc_id, user_id, source_path, title, source_type, metadata, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.doc_id,
                doc.user_id,
                doc.source_path,
                doc.title,
                doc.source_type,
                json.dumps(doc.metadata),
                ingested_at_str,
            ),
        )

    def _upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.user_id,
                chunk.text,
                chunk.title,
                chunk.section,
                chunk.subsection,
                chunk.chunk_index,
                chunk.token_count,
                chunk.source_type,
                json.dumps(chunk.metadata),
            )
            for chunk in chunks
        ]
        self.db.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, user_id, text, title, section, subsection,
                 chunk_index, token_count, source_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed chunk.embedding_text in batches; return all embeddings."""
        if not chunks:
            return []

        all_embeddings: list[list[float]] = []
        texts = [c.embedding_text for c in chunks]

        for batch_start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[batch_start : batch_start + _EMBED_BATCH_SIZE]
            batch_embeddings = self.embedder.embed(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
