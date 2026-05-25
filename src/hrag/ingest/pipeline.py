"""IngestPipeline: orchestrates loading, chunking, embedding, and storing documents."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

from hrag.config import Config
from hrag.db.connection import Database
from hrag.providers.embeddings import EmbeddingProvider
from hrag.types import Chunk, Document
from hrag.ingest.loaders import load_document
from hrag.ingest.chunker import chunk_document
from hrag.ingest.quality import filter_chunks


class CancelledError(Exception):
    """Raised inside a progress_cb to abort an in-flight ingest cleanly.

    The web layer's worker thread sets a cancel ``threading.Event`` and the
    callback raises this to unwind the pipeline. Caught at the call site
    (``_run_ingest_job``) so the rest of the queue continues unaffected.
    """

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
_EMBED_BATCH_SIZE = 8  # Smaller batches → more frequent progress events (was 32)


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

    def ingest_path(
        self,
        path: str | Path,
        user_id: str,
        *,
        skip_taxonomy: bool = False,
        progress_cb: Optional[Callable[[str, dict], None]] = None,
    ) -> Document:
        """Load a file from disk, then run the full ingest pipeline.

        ``skip_taxonomy=True`` short-circuits the "5c" taxonomy auto-assign
        hook so the doc lands unfiled — caller is expected to either drag
        it onto a node manually (taxonomy GUI) or POST
        ``/api/taxonomy/auto-assign/{doc_id}`` later.

        ``progress_cb(stage, payload)`` — see ``ingest_document`` for the
        full event contract; this wrapper additionally emits the ``load``
        stage around ``load_document``.
        """
        path_obj = Path(path)

        def _emit(stage: str, **kwargs) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(stage, {"stage": stage, "ts": time.time(), **kwargs})
            except CancelledError:
                raise
            except Exception:
                logger.exception("progress_cb raised; continuing")

        try:
            file_bytes = path_obj.stat().st_size if path_obj.exists() else None
        except Exception:  # noqa: BLE001
            file_bytes = None

        _emit(
            "load",
            message=f"Reading {path_obj.name}",
            n_done=0,
            n_total=1,
            bytes=file_bytes,
        )
        doc = load_document(path, user_id, cfg=self.config)
        _emit(
            "load",
            message="Document loaded",
            n_done=1,
            n_total=1,
            bytes=file_bytes,
            n_pages=(
                doc.metadata.get("n_pages")
                if isinstance(doc.metadata, dict)
                else None
            ),
        )
        return self.ingest_document(
            doc, skip_taxonomy=skip_taxonomy, progress_cb=progress_cb
        )

    def ingest_document(
        self,
        doc: Document,
        *,
        skip_taxonomy: bool = False,
        progress_cb: Optional[Callable[[str, dict], None]] = None,
    ) -> Document:
        """Run the full ingest pipeline on an already-loaded Document.

        Steps:
          1. Upsert row in `documents` table.
          2. Chunk the document.
          3. Upsert chunks in `chunks` table.
          4. Embed chunk.embedding_text in batches of 32.
          5. Remove old vectors then add new ones.
          5b. KG triple extraction (when ``config.kg.enabled``).
          5c. Taxonomy auto-assignment (skipped when ``skip_taxonomy=True``).
          6. Commit and return.

        ``progress_cb(stage, payload)`` is called at every stage transition
        AND after every embed batch. Stages emitted (in order):

          ``chunk`` → ``filter`` → ``embed`` (one event per batch + start/end)
          → ``index`` → ``assign`` (omitted when skip_taxonomy or tree empty)
          → ``done``.

        ``ingest_path`` additionally emits ``load`` before invoking this.

        Every payload contains ``stage``, ``ts``, ``message``, ``n_done``,
        ``n_total``. Stage-specific extras: ``n_chunks`` (chunk end),
        ``n_kept`` / ``n_dropped`` / ``dropped_breakdown`` (filter end),
        ``batch_index`` / ``batch_size`` (embed batches),
        ``node_id`` / ``node_label`` (assign end), ``doc_id`` / ``n_chunks``
        (done).

        The callback may raise ``CancelledError`` to abort. It bubbles up
        unchanged so the caller can clean up.
        """
        def _emit(stage: str, **kwargs) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(stage, {"stage": stage, "ts": time.time(), **kwargs})
            except CancelledError:
                raise
            except Exception:
                logger.exception("progress_cb raised; continuing")

        # Stamp ingestion time
        doc.ingested_at = datetime.now(tz=timezone.utc)

        # 1. Upsert document row
        self._upsert_document(doc)

        # 2. Chunk
        _emit("chunk", message="Chunking", n_done=0, n_total=1)
        chunks = chunk_document(doc, self.config.chunking)
        _emit(
            "chunk",
            message=f"Made {len(chunks)} chunks",
            n_done=1,
            n_total=1,
            n_chunks=len(chunks),
        )

        # 2b. Quality filter (optional, document-only)
        # Episodic memories are typically short ("Postgres > MySQL" = 3 tokens)
        # and would be dropped by the min_tokens/min_chars guards. Skip the
        # filter entirely for episodic ingests — the user explicitly chose to
        # remember each one.
        quality_cfg = self.config.chunking.quality
        n_pre_filter = len(chunks)
        if quality_cfg.enabled and doc.source_type != "episodic":
            _emit(
                "filter",
                message="Quality-filtering",
                n_done=0,
                n_total=n_pre_filter or 1,
            )
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
            _emit(
                "filter",
                message=f"Kept {len(chunks)} of {n_pre_filter}",
                n_done=n_pre_filter or 1,
                n_total=n_pre_filter or 1,
                n_kept=len(chunks),
                n_dropped=len(dropped),
                dropped_breakdown=breakdown,
            )
        else:
            print(f"[ingest] {doc.title}: {len(chunks)} chunks")
            _emit(
                "filter",
                message="Filter skipped",
                n_done=1,
                n_total=1,
                n_kept=len(chunks),
                n_dropped=0,
            )

        # 2c. Phase 9.14 — Contextual Retrieval (Anthropic technique).
        # Augment each chunk's embedding_text with an LLM-generated 50-100
        # token "context line" describing its role in the parent document.
        # Mutates chunk.embedding_text in place; chunk.text is left alone so
        # downstream KG / taxonomy paths still see the original body
        # (Phase-3 contracts 1 + 2). Episodic memories skip — they have no
        # parent document to situate within.
        if (
            self.config.ingest.contextual_retrieval_enabled
            and self.llm is not None
            and doc.source_type != "episodic"
            and chunks
        ):
            try:
                from hrag.ingest.contextual import ContextualAugmenter  # noqa: PLC0415
                augmenter = ContextualAugmenter(
                    self.llm,
                    self.db,
                    max_doc_chars=self.config.ingest.contextual_retrieval_max_doc_chars,
                    max_workers=self.config.ingest.contextual_retrieval_max_workers,
                    max_context_tokens=self.config.ingest.contextual_retrieval_max_context_tokens,
                    progress_cb=progress_cb,
                )
                chunks = augmenter.augment_chunks(chunks, doc.text)
            except Exception as exc:  # noqa: BLE001 — never break ingest
                logger.warning(
                    "[ingest] contextual augmentation failed for %r (%s); "
                    "continuing without contextual prefix",
                    doc.title, exc,
                )

        # 3. Upsert chunk rows
        self._upsert_chunks(chunks)

        # 4. Embed in batches (per-batch progress events emitted from helper)
        embeddings = self._embed_chunks(chunks, progress_cb=progress_cb)

        # 5. Update vector store
        n_kept = len(chunks)
        _emit(
            "index",
            message="Indexing in vector store",
            n_done=0,
            n_total=n_kept or 1,
        )
        self.vector_store.delete_doc(doc.user_id, doc.doc_id)
        if chunks:
            self.vector_store.add_chunks(doc.user_id, chunks, embeddings)
        _emit(
            "index",
            message=f"Indexed {n_kept}",
            n_done=n_kept or 1,
            n_total=n_kept or 1,
            n_chunks=n_kept,
        )

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
                    dedup_enabled=getattr(self.config.kg, "dedup_enabled", True),
                    progress_cb=progress_cb,
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
            not skip_taxonomy
            and self.config.taxonomy.enabled
            and self.config.taxonomy.auto_assign_on_ingest
            and eligible_source
            and self.taxonomy_store is not None
            and self.llm is not None
            and chunks
        ):
            _emit(
                "assign",
                message="Assigning taxonomy node",
                n_done=0,
                n_total=1,
            )
            assigned_node_id: Optional[str] = None
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
                assigned_node_id = assigner.assign(
                    doc.user_id, doc.doc_id, centroid=doc_centroid,
                )
                if assigned_node_id:
                    print(f"[ingest] {doc.title}: filed under {assigned_node_id}")
                else:
                    print(f"[ingest] {doc.title}: taxonomy empty — unfiled")
            except Exception as exc:  # noqa: BLE001
                logger.warning("[ingest] taxonomy auto-assign failed for %r: %s", doc.title, exc)
            _emit(
                "assign",
                message=(
                    f"Filed under {assigned_node_id}"
                    if assigned_node_id
                    else "Taxonomy empty — unfiled"
                ),
                n_done=1,
                n_total=1,
                node_id=assigned_node_id,
                node_label=assigned_node_id,
            )

        # 6. Commit
        self.db.commit()

        _emit(
            "done",
            message="Complete",
            n_done=1,
            n_total=1,
            doc_id=doc.doc_id,
            n_chunks=len(chunks),
        )

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

    def _embed_chunks(
        self,
        chunks: list[Chunk],
        *,
        progress_cb: Optional[Callable[[str, dict], None]] = None,
    ) -> list[list[float]]:
        """Embed chunk.embedding_text in batches; return all embeddings.

        Emits an ``embed`` progress event at start, after every batch, and at
        end. The mid-batch events are the SOTA progress UI's primary
        throughput signal — without them the bar would freeze for ~90 seconds
        on a 30-page PDF.
        """
        def _emit(stage: str, **kwargs) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(stage, {"stage": stage, "ts": time.time(), **kwargs})
            except CancelledError:
                raise
            except Exception:
                logger.exception("progress_cb raised; continuing")

        n_total = len(chunks)
        if not chunks:
            # Still emit start+end so the UI sees the stage transition cleanly.
            _emit("embed", message="No chunks to embed", n_done=0, n_total=1)
            _emit("embed", message="No chunks to embed", n_done=1, n_total=1)
            return []

        all_embeddings: list[list[float]] = []
        texts = [c.embedding_text for c in chunks]
        _emit("embed", message="Embedding", n_done=0, n_total=n_total)

        embedded = 0
        for batch_index, batch_start in enumerate(
            range(0, len(texts), _EMBED_BATCH_SIZE)
        ):
            batch = texts[batch_start : batch_start + _EMBED_BATCH_SIZE]
            batch_embeddings = self.embedder.embed(batch)
            all_embeddings.extend(batch_embeddings)
            embedded += len(batch)
            _emit(
                "embed",
                message=f"Embedded {embedded}/{n_total}",
                n_done=embedded,
                n_total=n_total,
                batch_index=batch_index,
                batch_size=len(batch),
            )

        return all_embeddings
