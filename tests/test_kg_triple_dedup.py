"""Tests for Phase 9.12 — cross-chunk triple deduplication.

The dedup layer has two complementary halves:

1. ``kg_triple_cache`` — keyed on SHA-256(chunk_text + model_name). Existing
   from Phase 2; identical chunk text short-circuits the LLM call entirely.
2. ``kg_canonical_triples`` — keyed on SHA-256(canonical subject|relation|object).
   Each unique triple is inserted once and gets a ``freq`` counter incremented
   on every subsequent sighting.

These tests cover the new canonical-key layer plus the ``kg_dedup_hit``
progress event surfaced by the extractor.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from hrag.db.migrations import run_migrations
from hrag.kg.builder import (
    Triple,
    TripleExtractor,
    _canon_field,
    canonical_triple_key,
)
from hrag.types import Chunk


@pytest.fixture()
def tmp_db_migrated(tmp_db):
    """``tmp_db`` plus the migrations the production ``init_db`` path runs.

    The shared ``tmp_db`` fixture stops at ``init_schema()`` so the Phase-9.12
    ``kg_canonical_triples`` table is missing. Wrap it so tests that rely on
    the dedup table get a fully-migrated DB.
    """
    run_migrations(tmp_db)
    return tmp_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str = "c1", text: str = "Some text.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        embedding_text=text,
        doc_id="d1",
        user_id="u1",
    )


class _CountingLLM:
    """Minimal LLMProvider stand-in that returns a fixed JSON list."""

    def __init__(
        self,
        output: str = '[{"head": "A", "relation": "describes", "tail": "B"}]',
        model_name: str = "stub-model",
    ) -> None:
        self._output = output
        self.calls: list[str] = []
        self.model_name = model_name

    def complete(self, prompt: str, **_kwargs: Any) -> str:
        self.calls.append(prompt)
        return self._output


def _freq(db, key: str) -> int:
    row = db.execute(
        "SELECT freq FROM kg_canonical_triples WHERE canonical_key=?",
        (key,),
    ).fetchone()
    return int(row["freq"]) if row is not None else 0


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def test_canonical_key_stable() -> None:
    """Equivalent surface forms produce the same canonical key."""
    a = canonical_triple_key("Marie Curie", "discovered", "polonium")
    b = canonical_triple_key("  marie curie  ", "DISCOVERED", "Polonium")
    c = canonical_triple_key("MARIE CURIE", "discovered", "polonium ")
    assert a == b == c

    # Different triples produce different keys.
    d = canonical_triple_key("Marie Curie", "discovered", "radium")
    assert a != d


def test_canon_field_lowercases_and_strips() -> None:
    assert _canon_field("  Foo BAR  ") == "foo bar"
    assert _canon_field("\t\nBaz") == "baz"
    assert _canon_field("") == ""


# ---------------------------------------------------------------------------
# Cross-chunk dedup: same triple across two chunks
# ---------------------------------------------------------------------------


def test_cross_chunk_dedup_increments_freq(tmp_db_migrated) -> None:
    """Same canonical triple extracted from two different chunks: one row, freq=2."""
    llm = _CountingLLM()
    extractor = TripleExtractor(llm, db=tmp_db_migrated)

    # Two distinct chunk texts but the LLM returns the same triple for each.
    extractor.extract_one(_make_chunk(chunk_id="c1", text="text one"))
    extractor.extract_one(_make_chunk(chunk_id="c2", text="text two"))

    # LLM was called twice (chunk texts differ, so the kg_triple_cache cannot
    # short-circuit) — but the canonical_triples row exists once with freq=2.
    assert len(llm.calls) == 2

    key = canonical_triple_key("A", "describes", "B")
    assert _freq(tmp_db_migrated, key) == 2

    # And only one row exists for this canonical triple.
    count = tmp_db_migrated.execute(
        "SELECT COUNT(*) FROM kg_canonical_triples WHERE canonical_key=?",
        (key,),
    ).fetchone()[0]
    assert count == 1


def test_first_seen_chunk_id_recorded(tmp_db_migrated) -> None:
    llm = _CountingLLM()
    extractor = TripleExtractor(llm, db=tmp_db_migrated)
    extractor.extract_one(_make_chunk(chunk_id="first_chunk", text="alpha"))
    extractor.extract_one(_make_chunk(chunk_id="second_chunk", text="beta"))

    key = canonical_triple_key("A", "describes", "B")
    row = tmp_db_migrated.execute(
        "SELECT first_seen_chunk_id FROM kg_canonical_triples WHERE canonical_key=?",
        (key,),
    ).fetchone()
    assert row["first_seen_chunk_id"] == "first_chunk"


# ---------------------------------------------------------------------------
# Per-chunk hash short-circuit: identical chunk text skips the LLM call
# ---------------------------------------------------------------------------


def test_identical_chunk_text_reuses_triples(tmp_db_migrated) -> None:
    """Re-ingest a chunk with identical text: the LLM is called zero extra times."""
    llm = _CountingLLM()
    extractor = TripleExtractor(llm, db=tmp_db_migrated)

    chunk_a = _make_chunk(chunk_id="c1", text="identical body")
    chunk_b = _make_chunk(chunk_id="c2", text="identical body")  # same text, new id

    extractor.extract_one(chunk_a)
    assert len(llm.calls) == 1

    extractor.extract_one(chunk_b)
    # Second call must NOT have invoked the LLM — kg_triple_cache hit.
    assert len(llm.calls) == 1

    # And the canonical-triple freq counter still tracks both sightings.
    key = canonical_triple_key("A", "describes", "B")
    assert _freq(tmp_db_migrated, key) == 2


# ---------------------------------------------------------------------------
# Concurrency: parallel inserts on the same key
# ---------------------------------------------------------------------------


def test_concurrent_extraction_no_duplicate_inserts(tmp_db_migrated) -> None:
    """Fire two threads on the same canonical key: only one row results."""
    llm = _CountingLLM()
    extractor = TripleExtractor(llm, db=tmp_db_migrated, max_workers=2)

    barrier = threading.Barrier(2)

    def _worker(chunk_id: str, text: str) -> None:
        barrier.wait()
        extractor.extract_one(_make_chunk(chunk_id=chunk_id, text=text))

    t1 = threading.Thread(target=_worker, args=("c_par_1", "thread one text"))
    t2 = threading.Thread(target=_worker, args=("c_par_2", "thread two text"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive() and not t2.is_alive()

    key = canonical_triple_key("A", "describes", "B")
    count = tmp_db_migrated.execute(
        "SELECT COUNT(*) FROM kg_canonical_triples WHERE canonical_key=?",
        (key,),
    ).fetchone()[0]
    assert count == 1
    # freq should be 2 (one per thread) under correct upsert semantics.
    assert _freq(tmp_db_migrated, key) == 2


# ---------------------------------------------------------------------------
# Progress event
# ---------------------------------------------------------------------------


def test_progress_event_fires_on_chunk_hash_hit(tmp_db_migrated) -> None:
    """The kg_dedup_hit event fires when an identical chunk text is reused."""
    events: list[tuple[str, dict]] = []

    def _cb(name: str, payload: dict) -> None:
        events.append((name, payload))

    llm = _CountingLLM()
    extractor = TripleExtractor(llm, db=tmp_db_migrated, progress_cb=_cb)

    extractor.extract_one(_make_chunk(chunk_id="c1", text="repeatable"))
    extractor.extract_one(_make_chunk(chunk_id="c2", text="repeatable"))

    dedup_events = [p for name, p in events if name == "kg_dedup_hit"]
    assert len(dedup_events) >= 1
    # The second call's payload should declare the LLM was short-circuited.
    last = dedup_events[-1]
    assert last["hashed_skipped_llm"] is True
    assert last["chunk_id"] == "c2"
    assert last["triples_reused"] == 1


def test_progress_event_fires_on_canonical_dedup_only(tmp_db_migrated) -> None:
    """Different chunk texts producing the same triple still fire kg_dedup_hit."""
    events: list[tuple[str, dict]] = []
    llm = _CountingLLM()
    extractor = TripleExtractor(
        llm, db=tmp_db_migrated, progress_cb=lambda n, p: events.append((n, p))
    )

    extractor.extract_one(_make_chunk(chunk_id="c1", text="first body"))
    extractor.extract_one(_make_chunk(chunk_id="c2", text="second body"))

    dedup_events = [p for name, p in events if name == "kg_dedup_hit"]
    # First chunk: no prior triple, no event. Second chunk: triple already
    # seen, event fires.
    assert len(dedup_events) == 1
    assert dedup_events[0]["hashed_skipped_llm"] is False
    assert dedup_events[0]["canonical_seen"] == 1


# ---------------------------------------------------------------------------
# Disable switch
# ---------------------------------------------------------------------------


def test_dedup_disabled_skips_canonical_writes(tmp_db_migrated) -> None:
    """When dedup_enabled=False, no rows are written to kg_canonical_triples."""
    llm = _CountingLLM()
    extractor = TripleExtractor(llm, db=tmp_db_migrated, dedup_enabled=False)
    extractor.extract_one(_make_chunk(chunk_id="c1", text="one"))
    extractor.extract_one(_make_chunk(chunk_id="c2", text="two"))
    count = tmp_db_migrated.execute(
        "SELECT COUNT(*) FROM kg_canonical_triples"
    ).fetchone()[0]
    assert count == 0


# ---------------------------------------------------------------------------
# Migration idempotency
# ---------------------------------------------------------------------------


def test_idempotent_migration(tmp_db) -> None:
    """Running run_migrations twice must not corrupt or duplicate the schema."""
    # tmp_db's init_schema already ran the schema. Now run migrations twice.
    run_migrations(tmp_db)
    run_migrations(tmp_db)

    # Table exists exactly once.
    rows = tmp_db.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='kg_canonical_triples'"
    ).fetchall()
    assert len(rows) == 1

    # Column set matches what we declared.
    cols = {
        r["name"]
        for r in tmp_db.execute(
            "PRAGMA table_info(kg_canonical_triples)"
        ).fetchall()
    }
    expected = {
        "canonical_key", "subject", "relation", "object",
        "first_seen_chunk_id", "freq", "created_at",
    }
    assert expected.issubset(cols)

    # Existing rows survive a second migration.
    tmp_db.execute(
        "INSERT INTO kg_canonical_triples"
        "(canonical_key, subject, relation, object, first_seen_chunk_id, freq) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("k1", "a", "r", "b", "c1", 5),
    )
    tmp_db.commit()
    run_migrations(tmp_db)
    row = tmp_db.execute(
        "SELECT freq FROM kg_canonical_triples WHERE canonical_key=?",
        ("k1",),
    ).fetchone()
    assert row["freq"] == 5


# ---------------------------------------------------------------------------
# Graceful degradation when the dedup table is missing (legacy DB)
# ---------------------------------------------------------------------------


def test_legacy_db_without_dedup_table(tmp_db_migrated) -> None:
    """Extraction must keep working when kg_canonical_triples is missing."""
    tmp_db_migrated.execute("DROP TABLE IF EXISTS kg_canonical_triples")
    tmp_db_migrated.commit()
    llm = _CountingLLM()
    extractor = TripleExtractor(llm, db=tmp_db_migrated)
    triples = extractor.extract_one(_make_chunk(chunk_id="c1", text="xyz"))
    assert len(triples) == 1


# ---------------------------------------------------------------------------
# Canonical surface forms collapse to one row
# ---------------------------------------------------------------------------


def test_surface_variants_collapse_to_one_row(tmp_db_migrated) -> None:
    llm = _CountingLLM(
        output='[{"head": "Marie Curie", "relation": "Discovered", "tail": "Polonium"}]'
    )
    extractor = TripleExtractor(llm, db=tmp_db_migrated)
    extractor.extract_one(_make_chunk(chunk_id="c1", text="passage A"))

    # New LLM returns the same triple with different casing/whitespace.
    llm._output = (
        '[{"head": "  marie curie  ", "relation": "discovered", "tail": "POLONIUM"}]'
    )
    extractor.extract_one(_make_chunk(chunk_id="c2", text="passage B"))

    count = tmp_db_migrated.execute(
        "SELECT COUNT(*) FROM kg_canonical_triples"
    ).fetchone()[0]
    assert count == 1
    key = canonical_triple_key("marie curie", "discovered", "polonium")
    assert _freq(tmp_db_migrated, key) == 2
