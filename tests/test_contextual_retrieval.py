"""Tests for Phase 9.14 — Contextual Retrieval (Anthropic technique).

Covers the ``ContextualAugmenter`` module + its IngestPipeline integration.

Each test exercises one of the contract points:
1. Default-off: flag stays False unless explicitly opted in.
2. Cache hit: same chunk + doc + model → LLM called once.
3. Cache invalidation by chunk-text change.
4. Cache invalidation by document change.
5. embedding_text mutated; text untouched (HeteRAG dual-text invariant).
6. Progress events fire with documented payload shape.
7. Oversized doc is truncated with a ``[...]`` marker.
8. ThreadPoolExecutor parallelism (max_workers respected).
9. Pipeline skips ContextualAugmenter when flag=False.
10. Pipeline integration: embedding_text gets the augmentation prefix.
11. KG path still sees the original chunk.text (Phase-3 contract 1).
12. Migration creates the ingest_context_cache table on a fresh DB.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from hrag.config import IngestConfig
from hrag.types import Chunk


# ---------------------------------------------------------------------------
# Fixture: tmp_db + migrations (the shared ``tmp_db`` stops at init_schema,
# so the Phase-9.14 ``ingest_context_cache`` table isn't created without
# this extra step.)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db_migrated(tmp_db):
    """``tmp_db`` plus the migrations that production ``init_db`` runs.

    Required for any test that reads or writes ``ingest_context_cache``.
    """
    from hrag.db.migrations import run_migrations
    run_migrations(tmp_db)
    return tmp_db


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _CountingLLM:
    """LLM that returns a deterministic context line, counts calls."""

    name = "counting_llm"
    model_name = "stub-model"

    def __init__(self, output: str = "This chunk discusses the Methods section.") -> None:
        self._output = output
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return self._output

    def generate(self, *_a, **_kw):  # pragma: no cover
        raise NotImplementedError


def _make_chunk(chunk_id: str, text: str, embedding_text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        user_id="default",
        text=text,
        embedding_text=embedding_text if embedding_text is not None else f"[Title] {text}",
        title="Title",
        section="Methods",
        chunk_index=0,
        token_count=len(text) // 4,
    )


# ---------------------------------------------------------------------------
# 1. Default OFF
# ---------------------------------------------------------------------------


def test_contextual_default_off():
    cfg = IngestConfig()
    assert cfg.contextual_retrieval_enabled is False
    # Other knobs have sensible defaults
    assert cfg.contextual_retrieval_max_workers >= 1
    assert cfg.contextual_retrieval_max_doc_chars >= 10_000
    assert cfg.contextual_retrieval_max_context_tokens >= 50


# ---------------------------------------------------------------------------
# 2. Cache hit on identical (chunk, doc, model)
# ---------------------------------------------------------------------------


def test_augmenter_cache_hit_skips_llm(tmp_db_migrated):
    from hrag.ingest.contextual import ContextualAugmenter

    llm = _CountingLLM()
    aug = ContextualAugmenter(llm, tmp_db_migrated, max_workers=1)

    chunks = [_make_chunk("c1", "Methods text body.")]
    doc_text = "Document with intro and methods and results."

    # First pass: LLM called once
    aug.augment_chunks(chunks, doc_text)
    assert len(llm.calls) == 1
    # Commit so the second instance sees the cache row (the augmenter
    # doesn't commit on its own — the production pipeline's terminal
    # ``db.commit()`` handles that).
    tmp_db_migrated.commit()

    # Reset embedding_text for the second pass so we can detect re-augmentation
    chunks2 = [_make_chunk("c1", "Methods text body.")]
    aug2 = ContextualAugmenter(llm, tmp_db_migrated, max_workers=1)
    aug2.augment_chunks(chunks2, doc_text)
    # Still exactly 1 LLM call → cache hit
    assert len(llm.calls) == 1
    # And the prefix was applied from cache
    assert chunks2[0].embedding_text.startswith("This chunk discusses the Methods section.")


# ---------------------------------------------------------------------------
# 3. Cache invalidates when chunk.text changes
# ---------------------------------------------------------------------------


def test_augmenter_cache_invalidates_on_chunk_text_change(tmp_db_migrated):
    from hrag.ingest.contextual import ContextualAugmenter

    llm = _CountingLLM()
    aug = ContextualAugmenter(llm, tmp_db_migrated, max_workers=1)
    doc_text = "shared document"

    aug.augment_chunks([_make_chunk("c1", "original body")], doc_text)
    assert len(llm.calls) == 1
    tmp_db_migrated.commit()

    # Same chunk_id, same doc, but DIFFERENT chunk text → cache miss
    aug.augment_chunks([_make_chunk("c1", "completely different body")], doc_text)
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# 4. Cache invalidates when the parent document changes
# ---------------------------------------------------------------------------


def test_augmenter_cache_invalidates_on_doc_change(tmp_db_migrated):
    from hrag.ingest.contextual import ContextualAugmenter

    llm = _CountingLLM()
    aug = ContextualAugmenter(llm, tmp_db_migrated, max_workers=1)

    aug.augment_chunks([_make_chunk("c1", "shared chunk body")], "document A")
    assert len(llm.calls) == 1
    tmp_db_migrated.commit()

    # Same chunk text, but a different parent document → cache miss
    aug.augment_chunks([_make_chunk("c1", "shared chunk body")], "document B (different)")
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# 5. chunk.text MUST NOT be mutated; only embedding_text gets the prefix
# ---------------------------------------------------------------------------


def test_augmenter_does_not_mutate_chunk_text(tmp_db):
    from hrag.ingest.contextual import ContextualAugmenter

    llm = _CountingLLM(output="Situating context.")
    aug = ContextualAugmenter(llm, tmp_db, max_workers=1)

    original_text = "the raw chunk body the LLM will see at answer time"
    original_embedding_text = "[Title] " + original_text
    chunk = _make_chunk("c1", original_text, embedding_text=original_embedding_text)

    aug.augment_chunks([chunk], "Some document text.")

    # chunk.text is preserved verbatim — KG / taxonomy / answer prompt must
    # see the same body as before augmentation.
    assert chunk.text == original_text
    # embedding_text has the situating prefix
    assert chunk.embedding_text.startswith("Situating context.")
    # ... followed by the ORIGINAL embedding_text (HeteRAG metadata fusion preserved)
    assert original_embedding_text in chunk.embedding_text


# ---------------------------------------------------------------------------
# 6. Progress events fire with the documented payload shape
# ---------------------------------------------------------------------------


def test_augmenter_progress_events(tmp_db):
    from hrag.ingest.contextual import ContextualAugmenter

    llm = _CountingLLM()
    events: list[tuple[str, dict]] = []

    def cb(event: str, payload: dict) -> None:
        events.append((event, payload))

    aug = ContextualAugmenter(llm, tmp_db, max_workers=1, progress_cb=cb)
    chunks = [
        _make_chunk("c1", "body one"),
        _make_chunk("c2", "body two"),
        _make_chunk("c3", "body three"),
    ]
    aug.augment_chunks(chunks, "the parent doc text")

    names = [e[0] for e in events]
    assert names[0] == "contextual_augment_start"
    assert names[-1] == "contextual_augment_done"
    # One per-chunk event for each chunk
    chunk_events = [e for e in events if e[0] == "contextual_augment_chunk"]
    assert len(chunk_events) == 3

    # Start payload shape
    start_payload = events[0][1]
    assert "n_chunks" in start_payload
    assert "doc_title" in start_payload
    assert "cached_hits" in start_payload
    assert start_payload["n_chunks"] == 3

    # Per-chunk payload shape
    for _, payload in chunk_events:
        for key in ("i", "n", "chunk_id", "from_cache", "ok"):
            assert key in payload, f"missing key {key!r} in chunk payload: {payload!r}"

    # Done payload shape
    done_payload = events[-1][1]
    for key in ("n_chunks", "n_cached", "n_called", "total_s"):
        assert key in done_payload, f"missing key {key!r} in done payload: {done_payload!r}"


# ---------------------------------------------------------------------------
# 7. Oversized doc truncates with [...] marker
# ---------------------------------------------------------------------------


def test_augmenter_handles_oversized_doc_via_truncation(tmp_db):
    from hrag.ingest.contextual import ContextualAugmenter

    llm = _CountingLLM()
    # Tight max_doc_chars so a 100k-char doc is forced to truncate
    aug = ContextualAugmenter(llm, tmp_db, max_workers=1, max_doc_chars=10_000)

    big_doc = "A" * 60_000 + "MIDDLE" + "B" * 40_000   # 100,006 chars
    chunks = [_make_chunk("c1", "chunk body")]
    aug.augment_chunks(chunks, big_doc)

    # LLM was called and its prompt contains the truncation marker
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["prompt"]
    assert "[...]" in prompt
    # The prompt's document-block must be smaller than the original 100k doc
    # (allow some headroom for the prompt scaffolding text).
    assert len(prompt) < 60_000


# ---------------------------------------------------------------------------
# 8. ThreadPoolExecutor parallelism — max_workers respected
# ---------------------------------------------------------------------------


def test_augmenter_parallel_workers(tmp_db):
    """Ensure ThreadPoolExecutor is used with the configured max_workers."""
    from hrag.ingest import contextual as cmod

    llm = _CountingLLM()
    aug = cmod.ContextualAugmenter(llm, tmp_db, max_workers=4)

    # Patch ThreadPoolExecutor inside the contextual module so we observe
    # the max_workers it was constructed with — without altering its
    # actual behaviour (we still want the work to complete).
    seen_max_workers: list[int] = []
    original_cls = cmod.ThreadPoolExecutor

    def _spy_pool(max_workers=None, **kwargs):
        seen_max_workers.append(max_workers)
        return original_cls(max_workers=max_workers, **kwargs)

    chunks = [_make_chunk(f"c{i}", f"body {i}") for i in range(8)]
    with patch.object(cmod, "ThreadPoolExecutor", side_effect=_spy_pool):
        aug.augment_chunks(chunks, "parent doc text")

    assert seen_max_workers, "ThreadPoolExecutor was never constructed"
    assert seen_max_workers[0] == 4, f"expected max_workers=4, got {seen_max_workers[0]!r}"
    # All 8 chunks were processed (all got the LLM-generated prefix)
    for c in chunks:
        assert c.embedding_text.startswith("This chunk discusses the Methods section.")


# ---------------------------------------------------------------------------
# 9. Pipeline with flag=False → ContextualAugmenter NEVER instantiated
# ---------------------------------------------------------------------------


def test_pipeline_default_off_skips_augment(
    tmp_db, fake_embedder, sample_config, sample_md_path
):
    """Default-off contract (Phase 9.14 Q1): zero LLM calls, zero new SQL."""
    from hrag.ingest.pipeline import IngestPipeline
    from hrag.ingest import contextual as cmod

    sample_config.chunking.quality.enabled = False
    # contextual_retrieval_enabled defaults to False; do not flip it
    assert sample_config.ingest.contextual_retrieval_enabled is False

    llm = _CountingLLM()

    class _FakeVS:
        def __init__(self):
            self.added = []

        def add_chunks(self, *a, **kw):
            self.added.append((a, kw))

        def delete_doc(self, *a, **kw):
            pass

    pipeline = IngestPipeline(sample_config, tmp_db, fake_embedder, _FakeVS(), llm=llm)

    # Spy on the augmenter constructor
    with patch.object(cmod, "ContextualAugmenter") as mock_aug:
        pipeline.ingest_path(sample_md_path, "default")
        mock_aug.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Pipeline with flag=True → embedding_text gets the augmentation prefix
# ---------------------------------------------------------------------------


def test_pipeline_with_flag_on_augments_then_embeds(
    tmp_db, fake_embedder, sample_config, sample_md_path
):
    from hrag.ingest.pipeline import IngestPipeline

    sample_config.chunking.quality.enabled = False
    sample_config.ingest.contextual_retrieval_enabled = True
    sample_config.ingest.contextual_retrieval_max_workers = 1

    llm = _CountingLLM(output="Section overview: introduces background.")

    captured: dict[str, Any] = {"embedded_texts": []}

    class _CapturingVS:
        def add_chunks(self, user_id, chunks, embeddings):
            # The chunks the vector store receives must already have the
            # augmented embedding_text.
            captured["embedded_texts"] = [c.embedding_text for c in chunks]

        def delete_doc(self, *a, **kw):
            pass

    pipeline = IngestPipeline(sample_config, tmp_db, fake_embedder, _CapturingVS(), llm=llm)
    pipeline.ingest_path(sample_md_path, "default")

    assert captured["embedded_texts"], "vector store received no chunks"
    # Every embedded chunk got the LLM-generated prefix
    for et in captured["embedded_texts"]:
        assert et.startswith("Section overview: introduces background."), (
            f"embedding_text missing contextual prefix: {et[:100]!r}"
        )
    # And the LLM was actually called (>=1 per chunk; could be cached on later
    # passes but this is a fresh DB so all are first-time)
    assert len(llm.calls) >= len(captured["embedded_texts"])


# ---------------------------------------------------------------------------
# 11. KG extractor still sees chunk.text unchanged (Phase-3 contract 1)
# ---------------------------------------------------------------------------


def test_pipeline_kg_path_unaffected(
    tmp_db, fake_embedder, sample_config, sample_md_path, tmp_path
):
    """Even with contextual augmentation enabled, the KG triple extractor must
    see chunk.text (the raw body), NOT the augmented embedding_text. The
    contextual prefix should never leak into extracted triples."""
    pytest.importorskip("networkx")
    pytest.importorskip("numpy")
    from hrag.kg.store import KGStore
    from hrag.ingest.pipeline import IngestPipeline

    tmp_db.ensure_user("default")
    tmp_db.commit()

    sample_config.chunking.quality.enabled = False
    sample_config.ingest.contextual_retrieval_enabled = True
    sample_config.kg.enabled = True

    seen_triple_prompts: list[str] = []

    PREFIX_MARKER = "AUGMENT_PREFIX_DO_NOT_LEAK"

    class _DualLLM:
        """Returns a context line for contextual prompts, a triple list for
        triple-extraction prompts. Records every triple prompt for inspection."""

        name = "dual_stub"
        model_name = "stub-model"

        def complete(self, prompt: str, **kwargs):
            # Triple-extraction prompts contain the canonical JSON markers.
            # If we ever see the contextual prefix marker INSIDE a triple
            # prompt, that's a regression.
            if "<chunk>" in prompt and "<document>" in prompt:
                # Contextual prompt
                return PREFIX_MARKER + " — chunk role: methods overview"
            # Triple-extraction prompt
            seen_triple_prompts.append(prompt)
            return '[{"head":"A","relation":"describes","tail":"B"}]'

        def generate(self, *_a, **_kw):  # pragma: no cover
            raise NotImplementedError

    llm = _DualLLM()
    kg_store = KGStore(
        db=tmp_db,
        embedder=fake_embedder,
        kg_path=tmp_path / "kg",
        synonym_threshold=0.99,
    )

    class _FakeVS:
        def add_chunks(self, *a, **kw):
            pass

        def delete_doc(self, *a, **kw):
            pass

    pipeline = IngestPipeline(
        sample_config, tmp_db, fake_embedder, _FakeVS(),
        llm=llm, kg_store=kg_store,
    )
    pipeline.ingest_path(sample_md_path, "default")

    # KG was actually exercised
    assert seen_triple_prompts, "Expected at least one triple-extraction prompt"
    # NONE of the triple-extraction prompts should contain the contextual prefix
    for tp in seen_triple_prompts:
        assert PREFIX_MARKER not in tp, (
            "Contextual augmentation prefix leaked into KG triple extraction. "
            "chunk.text was mutated; only embedding_text should change."
        )


# ---------------------------------------------------------------------------
# 12. Migration creates the ingest_context_cache table
# ---------------------------------------------------------------------------


def test_migration_creates_ingest_context_cache_table(tmp_path):
    """Run init_schema + run_migrations on a fresh DB; table must exist."""
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None
    from hrag.db.connection import Database
    from hrag.db.migrations import run_migrations

    db = Database(tmp_path / "fresh.sqlite")
    try:
        db.init_schema()
        run_migrations(db)

        # Table exists
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='ingest_context_cache'"
        ).fetchone()
        assert row is not None, "ingest_context_cache table not created"

        # Columns match the documented contract
        cols = {r["name"] for r in db.execute(
            "PRAGMA table_info(ingest_context_cache)"
        ).fetchall()}
        assert {"cache_key", "model_name", "context_line", "created_at"} <= cols

        # Idempotent: running migrations again is a no-op (no exception)
        run_migrations(db)
    finally:
        db.close()
        _conn_mod._db_singleton = None
