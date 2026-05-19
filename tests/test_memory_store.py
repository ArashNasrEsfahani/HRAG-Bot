"""EpisodicMemoryStore: add, list_recent, count, forget, forget_memory.

Uses a minimal FakeIngestPipeline that mirrors the contract
EpisodicMemoryStore depends on (``ingest_document`` + ``vector_store`` +
``embedder``) without exercising the real chunker / quality filter / KG.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hrag.types import Chunk


class _FakeVectorStore:
    """Records add/delete/update calls so the test can assert behavior."""

    def __init__(self) -> None:
        self.added: list[Chunk] = []
        self.deleted_docs: list[tuple[str, str]] = []
        self.tombstoned: list[list[str]] = []

    # Mirror VectorStore.add_chunks
    def add_chunks(self, user_id, chunks, embeddings):
        self.added.extend(chunks)

    def delete_doc(self, user_id, doc_id):
        self.deleted_docs.append((user_id, doc_id))

    # The _chroma_tombstone helper looks for ._collection.update
    @property
    def _collection(self):
        return self

    def update(self, ids, metadatas):
        self.tombstoned.append(list(ids))


class _FakeIngestPipeline:
    """Minimal pipeline that records one chunk per Document straight into SQLite."""

    def __init__(self, db, embedder, vector_store) -> None:
        self.db = db
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest_document(self, doc):
        # Single-chunk synthesis — matches what the real pipeline does for
        # short episodic notes.
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}::0",
            doc_id=doc.doc_id,
            user_id=doc.user_id,
            text=doc.text,
            embedding_text=doc.text,
            title=doc.title,
            section="",
            subsection="",
            chunk_index=0,
            token_count=max(1, len(doc.text.split())),
            source_type=doc.source_type,
            metadata=doc.metadata,
        )
        doc.ingested_at = datetime.now(tz=timezone.utc)
        self.db.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, user_id, source_path, title, source_type, metadata, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                doc.doc_id,
                doc.user_id,
                doc.source_path,
                doc.title,
                doc.source_type,
                json.dumps(doc.metadata),
                doc.ingested_at.isoformat(),
            ),
        )
        self.db.execute(
            "INSERT OR REPLACE INTO chunks "
            "(chunk_id, doc_id, user_id, text, title, section, subsection, "
            " chunk_index, token_count, source_type, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )
        self.db.commit()
        self.vector_store.add_chunks(doc.user_id, [chunk], self.embedder.embed([chunk.text]))
        return doc


@pytest.fixture()
def store(tmp_db, fake_embedder):
    from hrag.db.migrations import run_migrations
    from hrag.memory.store import EpisodicMemoryStore

    run_migrations(tmp_db)
    vs = _FakeVectorStore()
    pipeline = _FakeIngestPipeline(tmp_db, fake_embedder, vs)
    return EpisodicMemoryStore(tmp_db, pipeline), vs


def test_add_persists_episodic_chunk(store):
    memory_store, vs = store
    mid = memory_store.add("default", "Postgres preferred over MySQL")
    assert mid.startswith("episodic:default:")
    chunks = memory_store.list_recent("default")
    assert len(chunks) == 1
    assert chunks[0].source_type == "episodic"
    assert chunks[0].text == "Postgres preferred over MySQL"
    # Vector store was called.
    assert len(vs.added) == 1


def test_add_rejects_empty_text(store):
    memory_store, _ = store
    with pytest.raises(ValueError):
        memory_store.add("default", "   ")


def test_count_matches_added(store):
    memory_store, _ = store
    assert memory_store.count("default") == 0
    for i in range(3):
        memory_store.add("default", f"memory {i}")
    assert memory_store.count("default") == 3


def test_add_batch_persists_each_item(store):
    memory_store, _ = store
    ids = memory_store.add_batch(
        "default",
        [
            {"text": "TypeScript over JavaScript"},
            {"text": "OKRs due the 15th", "title": "okr-deadline"},
            {"text": "  "},  # empty: skipped
        ],
    )
    assert len(ids) == 2
    assert memory_store.count("default") == 2


def test_forget_flips_excluded_in_sql(tmp_db, store):
    memory_store, vs = store
    memory_store.add("default", "remember me")
    chunks = memory_store.list_recent("default")
    chunk_id = chunks[0].chunk_id

    ok = memory_store.forget("default", chunk_id)
    assert ok is True

    # SQL says excluded=1
    row = tmp_db.execute(
        "SELECT excluded FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    assert row["excluded"] == 1
    # Chroma tombstone mirror was called.
    assert vs.tombstoned == [[chunk_id]]
    # list_recent now hides the chunk.
    assert memory_store.count("default") == 0


def test_forget_unknown_returns_false(store):
    memory_store, _ = store
    assert memory_store.forget("default", "no-such-chunk") is False


def test_forget_memory_tombstones_all_chunks(tmp_db, store):
    memory_store, _ = store
    mid = memory_store.add("default", "one note")
    # Inject a second chunk manually to simulate a multi-chunk memory.
    tmp_db.execute(
        "INSERT INTO chunks (chunk_id, doc_id, user_id, text, source_type) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"{mid}::1", mid, "default", "extra fragment", "episodic"),
    )
    tmp_db.commit()
    assert memory_store.count("default") == 2

    n = memory_store.forget_memory("default", mid)
    assert n == 2
    assert memory_store.count("default") == 0


def test_list_recent_orders_by_ingested_at_desc(store):
    memory_store, _ = store
    first = memory_store.add("default", "first")
    memory_store.add("default", "second")
    third = memory_store.add("default", "third")
    chunks = memory_store.list_recent("default", limit=10)
    # Most recent first.
    assert chunks[0].doc_id == third
    assert chunks[-1].doc_id == first


def test_episodic_does_not_appear_for_other_users(tmp_db, store):
    memory_store, _ = store
    tmp_db.ensure_user("alice")
    tmp_db.ensure_user("bob")
    memory_store.add("alice", "alice's note")
    memory_store.add("bob", "bob's note")
    assert memory_store.count("alice") == 1
    assert memory_store.count("bob") == 1
    assert memory_store.count("default") == 0
