"""Tests for hrag.kg.builder — TripleExtractor and Triple dataclass."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from hrag.kg.builder import Triple, TripleExtractor
from hrag.types import Chunk


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str = "c1", text: str = "Some text.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        embedding_text=text,
        doc_id="d1",
        user_id="u1",
    )


class _StubLLM:
    """Minimal LLMProvider stand-in — no network required."""

    def __init__(
        self,
        output: str = "",
        raise_exc: Exception | None = None,
        model_name: str = "stub-model",
    ) -> None:
        self._output = output
        self._raise = raise_exc
        self.calls: list[str] = []
        self.model_name = model_name

    def complete(self, prompt: str, **_kwargs: Any) -> str:  # noqa: D401
        self.calls.append(prompt)
        if self._raise is not None:
            raise self._raise
        return self._output


# Clean JSON returned by the LLM
_CLEAN_JSON = (
    '[{"head": "Marie Curie", "relation": "born in", "tail": "Warsaw"},'
    ' {"head": "Marie Curie", "relation": "discovered", "tail": "polonium"}]'
)

# Same JSON wrapped in markdown fences
_FENCED_JSON = f"```json\n{_CLEAN_JSON}\n```"

# Prose before and after the JSON list
_PROSE_WRAPPED = f"Here are the triples:\n{_CLEAN_JSON}\nThat is all."

# Completely invalid JSON
_BAD_JSON = "I cannot extract any triples from this passage."

# JSON array with one valid item and one missing the "tail" key
_PARTIAL_JSON = (
    '[{"head": "BERT", "relation": "based on", "tail": "transformer"},'
    ' {"head": "GPT", "relation": "based on"}]'
)


# ---------------------------------------------------------------------------
# Triple dataclass
# ---------------------------------------------------------------------------


def test_triple_canonical_defaults_to_surface() -> None:
    # Direct construction — __post_init__ copies surface form to canonicals.
    t = Triple(head="Foo", relation="likes", tail="Bar", source_chunk_id="c1")
    assert t.head_canonical == "Foo"
    assert t.tail_canonical == "Bar"
    # Explicit canonical overrides the default.
    t2 = Triple(
        head="Foo", relation="likes", tail="Bar",
        source_chunk_id="c1", head_canonical="foo", tail_canonical="bar",
    )
    assert t2.head_canonical == "foo"
    assert t2.tail_canonical == "bar"


# ---------------------------------------------------------------------------
# extract_one — happy paths
# ---------------------------------------------------------------------------


def test_extract_one_clean_json() -> None:
    llm = _StubLLM(output=_CLEAN_JSON)
    extractor = TripleExtractor(llm)
    chunk = _make_chunk()
    triples = extractor.extract_one(chunk)

    assert len(triples) == 2
    assert triples[0].head == "marie curie"
    assert triples[0].relation == "born in"
    assert triples[0].tail == "warsaw"
    assert triples[0].source_chunk_id == "c1"
    assert triples[1].relation == "discovered"
    assert triples[1].tail == "polonium"


def test_extract_one_markdown_fences_stripped() -> None:
    llm = _StubLLM(output=_FENCED_JSON)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert len(triples) == 2
    assert triples[0].head == "marie curie"


def test_extract_one_prose_around_json_tolerated() -> None:
    llm = _StubLLM(output=_PROSE_WRAPPED)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert len(triples) == 2


def test_extract_one_fields_are_lowercased_and_stripped() -> None:
    raw = '[{"head": "  Marie Curie  ", "relation": " Born In ", "tail": "  WARSAW  "}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert triples[0].head == "marie curie"
    assert triples[0].relation == "born in"
    assert triples[0].tail == "warsaw"


# ---------------------------------------------------------------------------
# extract_one — error / robustness paths
# ---------------------------------------------------------------------------


def test_extract_one_malformed_json_returns_empty_and_warns() -> None:
    llm = _StubLLM(output=_BAD_JSON)
    extractor = TripleExtractor(llm)
    with pytest.warns(UserWarning, match="could not parse JSON"):
        result = extractor.extract_one(_make_chunk())
    assert result == []


def test_extract_one_missing_keys_skipped_silently() -> None:
    llm = _StubLLM(output=_PARTIAL_JSON)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    # Only the first item has all three keys
    assert len(triples) == 1
    assert triples[0].head == "bert"


def test_extract_one_non_dict_items_skipped() -> None:
    raw = '["not a dict", {"head": "X", "relation": "r", "tail": "Y"}, 42]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert len(triples) == 1
    assert triples[0].head == "x"


def test_extract_one_non_string_field_values_skipped() -> None:
    raw = '[{"head": 1, "relation": "r", "tail": "Y"}, {"head": "A", "relation": "r", "tail": "B"}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert len(triples) == 1
    assert triples[0].head == "a"


def test_extract_one_prompt_contains_chunk_text() -> None:
    llm = _StubLLM(output="[]")
    extractor = TripleExtractor(llm)
    chunk = _make_chunk(text="Special passage text.")
    extractor.extract_one(chunk)
    assert "Special passage text." in llm.calls[0]


# ---------------------------------------------------------------------------
# extract_batch
# ---------------------------------------------------------------------------


def test_extract_batch_five_chunks_union() -> None:
    llm = _StubLLM(output=_CLEAN_JSON)
    extractor = TripleExtractor(llm)
    chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(5)]
    triples = extractor.extract_batch(chunks)
    # 2 triples per chunk × 5 chunks
    assert len(triples) == 10
    chunk_ids = {t.source_chunk_id for t in triples}
    assert chunk_ids == {"c0", "c1", "c2", "c3", "c4"}


def test_extract_batch_empty_list_returns_empty_no_llm_call() -> None:
    llm = _StubLLM(output=_CLEAN_JSON)
    extractor = TripleExtractor(llm)
    result = extractor.extract_batch([])
    assert result == []
    assert llm.calls == []


def test_extract_batch_exception_in_one_chunk_does_not_kill_others() -> None:
    """Stub raises on chunk_id=='c2', succeeds for all others."""

    call_count = 0

    class _RaisingStub:
        calls: list[str] = []

        def complete(self, prompt: str, **_kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            # Raise for the third call (chunk c2)
            if call_count == 3:
                raise RuntimeError("deliberate failure")
            return _CLEAN_JSON

    extractor = TripleExtractor(_RaisingStub(), max_workers=1)  # type: ignore[arg-type]
    chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(5)]

    with pytest.warns(UserWarning, match="unhandled exception"):
        triples = extractor.extract_batch(chunks)

    # 4 successful chunks × 2 triples each = 8
    assert len(triples) == 8
    failed_ids = {t.source_chunk_id for t in triples}
    assert "c2" not in failed_ids


def test_extract_batch_preserves_source_chunk_ids() -> None:
    llm = _StubLLM(output='[{"head": "A", "relation": "r", "tail": "B"}]')
    extractor = TripleExtractor(llm)
    chunks = [_make_chunk(chunk_id="alpha"), _make_chunk(chunk_id="beta")]
    triples = extractor.extract_batch(chunks)
    ids = [t.source_chunk_id for t in triples]
    assert ids == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Triple validation — empty / whitespace / non-alphanumeric / over-length
# ---------------------------------------------------------------------------


def test_extract_one_empty_head_dropped_others_kept() -> None:
    raw = (
        '[{"head": "", "relation": "r", "tail": "B"},'
        ' {"head": "A", "relation": "r", "tail": "B"}]'
    )
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert len(triples) == 1
    assert triples[0].head == "a"


def test_extract_one_whitespace_only_fields_dropped() -> None:
    raw = (
        '[{"head": "   ", "relation": "r", "tail": "B"},'
        ' {"head": "A", "relation": "  ", "tail": "B"},'
        ' {"head": "A", "relation": "r", "tail": "\\t\\n"}]'
    )
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert triples == []


def test_extract_one_non_alnum_relation_dropped() -> None:
    raw = '[{"head": "A", "relation": ".", "tail": "B"}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert triples == []


def test_extract_one_non_alnum_head_dropped() -> None:
    raw = '[{"head": "---", "relation": "r", "tail": "B"}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert triples == []


def test_extract_one_overlong_field_dropped() -> None:
    long_head = "x" * 250
    raw = f'[{{"head": "{long_head}", "relation": "r", "tail": "B"}}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    triples = extractor.extract_one(_make_chunk())
    assert triples == []


def test_extract_one_drop_counts_accumulated() -> None:
    raw = (
        '[{"head": "", "relation": "r", "tail": "B"},'
        ' {"head": "A", "relation": ".", "tail": "B"},'
        f' {{"head": "{"x" * 250}", "relation": "r", "tail": "B"}},'
        ' {"head": "A", "relation": "r", "tail": "B"}]'
    )
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    counts: dict[str, int] = {}
    triples = extractor.extract_one(_make_chunk(), drop_counts=counts)

    assert len(triples) == 1
    assert counts.get("empty_head") == 1
    assert counts.get("non_alnum") == 1
    assert counts.get("too_long") == 1


def test_extract_batch_emits_one_summary_warning() -> None:
    raw = (
        '[{"head": "", "relation": "r", "tail": "B"},'
        ' {"head": "A", "relation": "r", "tail": "B"}]'
    )
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm, max_workers=1)
    chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(3)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        triples = extractor.extract_batch(chunks)

    # 3 chunks × 1 valid triple each = 3 valid triples
    assert len(triples) == 3
    # Exactly one summary warning across the whole batch.
    summary = [w for w in caught if "dropped" in str(w.message) and "invalid triples" in str(w.message)]
    assert len(summary) == 1
    msg = str(summary[0].message)
    # 3 chunks × 1 invalid head per chunk = 3
    assert "empty_head=3" in msg


def test_extract_batch_no_warning_when_all_valid() -> None:
    llm = _StubLLM(output=_CLEAN_JSON)
    extractor = TripleExtractor(llm, max_workers=1)
    chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(2)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extractor.extract_batch(chunks)

    summary = [w for w in caught if "dropped" in str(w.message) and "invalid triples" in str(w.message)]
    assert summary == []


# ---------------------------------------------------------------------------
# Triple validation — stop-word relations and self-loops
# ---------------------------------------------------------------------------


def test_drop_stop_relation_to() -> None:
    raw = '[{"head": "Orchestrator", "relation": "to", "tail": "Backend"}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    counts: dict[str, int] = {}
    triples = extractor.extract_one(_make_chunk(), drop_counts=counts)
    assert triples == []
    assert counts.get("stop_relation") == 1


def test_drop_stop_relation_is() -> None:
    raw = '[{"head": "RAGate", "relation": "is", "tail": "framework"}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    counts: dict[str, int] = {}
    triples = extractor.extract_one(_make_chunk(), drop_counts=counts)
    assert triples == []
    assert counts.get("stop_relation") == 1


def test_drop_stop_relation_caseinsensitive() -> None:
    # The extractor lowercases before checking, so "To" must also be rejected.
    raw = '[{"head": "Orchestrator", "relation": "To", "tail": "Backend"}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    counts: dict[str, int] = {}
    triples = extractor.extract_one(_make_chunk(), drop_counts=counts)
    assert triples == []
    assert counts.get("stop_relation") == 1


def test_drop_self_loop() -> None:
    raw = '[{"head": "hipporag", "relation": "mentions", "tail": "hipporag"}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    counts: dict[str, int] = {}
    triples = extractor.extract_one(_make_chunk(), drop_counts=counts)
    assert triples == []
    assert counts.get("self_loop") == 1


def test_keep_compound_relation_starting_with_stop_word() -> None:
    # "is part of" is a meaningful compound — NOT an exact stop-relation match.
    raw = '[{"head": "ChromaDB", "relation": "is part of", "tail": "retrieval stack"}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    counts: dict[str, int] = {}
    triples = extractor.extract_one(_make_chunk(), drop_counts=counts)
    assert len(triples) == 1
    assert triples[0].relation == "is part of"
    assert counts.get("stop_relation", 0) == 0


def test_keep_compound_relation_starting_with_to() -> None:
    # "to which" is not in the stop set — only exact bare "to" is rejected.
    raw = '[{"head": "query", "relation": "to which", "tail": "index"}]'
    llm = _StubLLM(output=raw)
    extractor = TripleExtractor(llm)
    counts: dict[str, int] = {}
    triples = extractor.extract_one(_make_chunk(), drop_counts=counts)
    assert len(triples) == 1
    assert triples[0].relation == "to which"
    assert counts.get("stop_relation", 0) == 0


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


def test_cache_hit_skips_llm_call(tmp_db) -> None:
    """Second extract_one with identical chunk text reuses cached triples."""
    stub = _StubLLM(output='[{"head":"A","relation":"describes","tail":"B"}]')
    extractor = TripleExtractor(stub, db=tmp_db)
    chunk = Chunk(
        chunk_id="c1", doc_id="d1", user_id="u1",
        text="some text", embedding_text="some text",
    )
    first = extractor.extract_one(chunk)
    assert len(first) == 1
    assert len(stub.calls) == 1

    # Re-extract — should hit cache, no new LLM call.
    second = extractor.extract_one(chunk)
    assert len(second) == 1
    assert len(stub.calls) == 1   # unchanged


def test_cache_isolated_by_model(tmp_db) -> None:
    """Same chunk text with a different model_name misses cache."""
    output = '[{"head":"A","relation":"describes","tail":"B"}]'
    stub_a = _StubLLM(output=output, model_name="model-alpha")
    stub_b = _StubLLM(output=output, model_name="model-beta")

    extractor_a = TripleExtractor(stub_a, db=tmp_db)
    extractor_b = TripleExtractor(stub_b, db=tmp_db)

    chunk = Chunk(
        chunk_id="c1", doc_id="d1", user_id="u1",
        text="shared text", embedding_text="shared text",
    )

    # First extractor populates cache for model-alpha.
    extractor_a.extract_one(chunk)
    assert len(stub_a.calls) == 1

    # Second extractor with a different model_name gets a cache miss.
    extractor_b.extract_one(chunk)
    assert len(stub_b.calls) == 1   # had to call LLM — different key

    # Now re-run extractor_a — should hit its own cache entry.
    extractor_a.extract_one(chunk)
    assert len(stub_a.calls) == 1   # still 1


def test_cache_miss_when_db_is_none(tmp_db) -> None:
    """Caching is disabled when no db is passed."""
    stub = _StubLLM(output='[{"head":"A","relation":"describes","tail":"B"}]')
    extractor = TripleExtractor(stub, db=None)
    chunk = Chunk(
        chunk_id="c1", doc_id="d1", user_id="u1",
        text="x", embedding_text="x",
    )
    extractor.extract_one(chunk)
    extractor.extract_one(chunk)
    assert len(stub.calls) == 2   # both calls went to LLM


def test_cache_handles_missing_table_gracefully(tmp_db) -> None:
    """If kg_triple_cache table is missing (legacy DB), extraction still works."""
    tmp_db.execute("DROP TABLE IF EXISTS kg_triple_cache")
    tmp_db.commit()
    stub = _StubLLM(output='[{"head":"A","relation":"describes","tail":"B"}]')
    extractor = TripleExtractor(stub, db=tmp_db)
    chunk = Chunk(
        chunk_id="c1", doc_id="d1", user_id="u1",
        text="x", embedding_text="x",
    )
    triples = extractor.extract_one(chunk)
    assert len(triples) == 1   # extraction succeeds despite missing cache table


def test_cache_parse_failure_not_cached(tmp_db) -> None:
    """A transient LLM parse failure must NOT be written to cache."""
    stub = _StubLLM(output="I cannot parse this.")
    extractor = TripleExtractor(stub, db=tmp_db)
    chunk = Chunk(
        chunk_id="c1", doc_id="d1", user_id="u1",
        text="retry me", embedding_text="retry me",
    )
    with pytest.warns(UserWarning, match="could not parse JSON"):
        extractor.extract_one(chunk)
    assert len(stub.calls) == 1

    # Switch to a valid-output stub using the SAME extractor (same cache key).
    stub._output = '[{"head":"A","relation":"r","tail":"B"}]'
    stub._raise = None
    triples = extractor.extract_one(chunk)
    # Second call must hit LLM again (parse failure was not cached).
    assert len(stub.calls) == 2
    assert len(triples) == 1


def test_cache_empty_result_is_cached(tmp_db) -> None:
    """A genuinely empty triple list (valid JSON []) is cached to avoid re-calling LLM."""
    stub = _StubLLM(output="[]")
    extractor = TripleExtractor(stub, db=tmp_db)
    chunk = Chunk(
        chunk_id="c1", doc_id="d1", user_id="u1",
        text="no triples here", embedding_text="no triples here",
    )
    first = extractor.extract_one(chunk)
    assert first == []
    assert len(stub.calls) == 1

    second = extractor.extract_one(chunk)
    assert second == []
    assert len(stub.calls) == 1   # served from cache
