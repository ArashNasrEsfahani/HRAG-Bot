"""Tests for progress callbacks on TaxonomyBuilder.build_for_user and
DocAssigner.assign_all.

All tests run without heavy deps (chromadb, sentence-transformers, ollama)
using in-memory stubs — same pattern as test_taxonomy_store.py.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import pytest

from hrag.config import TaxonomyConfig
from hrag.db.connection import Database
from hrag.taxonomy.assigner import DocAssigner
from hrag.taxonomy.builder import TaxonomyBuilder
from hrag.taxonomy.store import TaxonomyStore


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _DictEmbedder:
    """Deterministic 8-dim embedder."""

    name = "dict"
    _DIM = 8

    def __init__(self, mapping: dict | None = None) -> None:
        self._map = dict(mapping or {})

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        if text in self._map:
            return list(self._map[text])
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b / 255.0) * 2.0 - 1.0 for b in digest[: self._DIM]]

    @property
    def dim(self) -> int:
        return self._DIM


_TREE_JSON = """{
  "tree": {
    "label": "root",
    "children": [
      {
        "label": "Science",
        "description": "Science papers",
        "children": null,
        "doc_ids": ["__DOC_IDS__"]
      }
    ]
  }
}"""


class _StubLLM:
    """Synchronous LLM stub: returns a fixed tree JSON or one-line summary."""

    name = "stub_llm"

    def __init__(self, doc_ids: list[str]) -> None:
        self._doc_ids = doc_ids

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Proposal prompts contain "doc_id" (from the template body) and ask
        # for a taxonomy JSON.  Summary / tiebreak prompts do not.
        if "doc_id" in prompt and ("leaf" in prompt.lower() or "categor" in prompt.lower()):
            ids_str = '", "'.join(self._doc_ids)
            tree = _TREE_JSON.replace('"__DOC_IDS__"', f'"{ids_str}"')
            return tree
        # Summary prompt → return a short one-liner.
        return "A short document summary."

    def generate_stream(self, prompt: str, **kwargs):
        yield self.complete(prompt)


class _ErrorOnSecondCallCallback:
    """Callable that raises RuntimeError on the second invocation."""

    def __init__(self) -> None:
        self._calls = 0

    def __call__(self, stage: str, payload: dict) -> None:
        self._calls += 1
        if self._calls == 2:
            raise RuntimeError("intentional error on second call")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.sqlite"
    db = Database(path)
    db.init_schema()
    db.ensure_user("u1")
    yield db
    db.close()


@pytest.fixture
def cfg():
    return TaxonomyConfig(
        enabled=True,
        use_llm_doc_summaries=False,  # fast; avoids LLM call for summaries
        propose_sample_size=100,
        max_children_per_node=5,
        max_depth=4,
        parallel_workers=2,
    )


def _insert_docs(db: Database, user_id: str, n: int) -> list[str]:
    """Insert n fake document rows and one chunk each. Return doc_ids."""
    doc_ids = []
    for i in range(n):
        did = f"doc_{i:03d}"
        doc_ids.append(did)
        db.execute(
            "INSERT OR IGNORE INTO documents(doc_id, user_id, source_path, title) "
            "VALUES (?, ?, ?, ?)",
            (did, user_id, f"/tmp/paper_{i}.pdf", f"Paper {i}"),
        )
        db.execute(
            "INSERT OR IGNORE INTO chunks"
            "(chunk_id, doc_id, user_id, chunk_index, text, token_count, excluded) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (
                f"chunk_{i:03d}",
                did,
                user_id,
                0,
                f"Content of paper {i}.",
                10,
            ),
        )
    db.commit()
    return doc_ids


def _make_builder(db: Database, cfg: TaxonomyConfig, doc_ids: list[str]):
    embedder = _DictEmbedder()
    store = TaxonomyStore(db, embedder)
    llm = _StubLLM(doc_ids)
    return TaxonomyBuilder(db=db, llm=llm, embedder=embedder, store=store, cfg=cfg)


def _make_assigner(db: Database, cfg: TaxonomyConfig, doc_ids: list[str]):
    embedder = _DictEmbedder()
    store = TaxonomyStore(db, embedder)
    llm = _StubLLM(doc_ids)
    return DocAssigner(db=db, llm=llm, embedder=embedder, store=store, cfg=cfg), store


# ---------------------------------------------------------------------------
# TaxonomyBuilder.build_for_user progress tests
# ---------------------------------------------------------------------------


def test_builder_start_stage_fires(db, cfg):
    """build_for_user with a progress callback fires at least one 'start' event."""
    doc_ids = _insert_docs(db, "u1", 3)
    builder = _make_builder(db, cfg, doc_ids)

    events: list[tuple[str, dict]] = []
    builder.build_for_user("u1", progress=lambda s, p: events.append((s, p)))

    stages = [e[0] for e in events]
    assert "start" in stages, f"'start' not in {stages}"


def test_builder_stage_order_start_to_done(db, cfg):
    """build_for_user emits 'start' before 'done' and in overall correct order."""
    doc_ids = _insert_docs(db, "u1", 2)
    builder = _make_builder(db, cfg, doc_ids)

    events: list[tuple[str, dict]] = []
    builder.build_for_user("u1", progress=lambda s, p: events.append((s, p)))

    stages = [e[0] for e in events]
    assert stages[0] == "start", f"First stage should be 'start', got {stages}"
    assert stages[-1] == "done", f"Last stage should be 'done', got {stages}"

    # Structural ordering invariants.
    assert stages.index("start") < stages.index("summaries_done")
    assert stages.index("summaries_done") < stages.index("propose_tree_start")
    assert stages.index("propose_tree_start") < stages.index("propose_tree_done")
    assert stages.index("propose_tree_done") < stages.index("materialize_start")
    assert stages.index("materialize_start") < stages.index("materialize_done")
    assert stages.index("materialize_done") < stages.index("done")


def test_builder_summaries_done_count(db, cfg):
    """'summaries_done' payload's n_summarized matches the number of docs."""
    n = 4
    doc_ids = _insert_docs(db, "u1", n)
    builder = _make_builder(db, cfg, doc_ids)

    payloads: dict[str, dict] = {}
    def cb(stage, payload):
        payloads[stage] = payload

    builder.build_for_user("u1", progress=cb)

    assert "summaries_done" in payloads, "summaries_done not emitted"
    assert payloads["summaries_done"]["n_summarized"] == n


def test_builder_doc_summary_emitted_per_doc(db, cfg):
    """'doc_summary' is emitted exactly n_docs times."""
    n = 3
    doc_ids = _insert_docs(db, "u1", n)
    builder = _make_builder(db, cfg, doc_ids)

    events: list[tuple[str, dict]] = []
    builder.build_for_user("u1", progress=lambda s, p: events.append((s, p)))

    doc_summary_events = [e for e in events if e[0] == "doc_summary"]
    assert len(doc_summary_events) == n, (
        f"Expected {n} 'doc_summary' events, got {len(doc_summary_events)}"
    )
    # Each event must carry i, n, doc_id, title.
    for stage, payload in doc_summary_events:
        assert "doc_id" in payload
        assert "title" in payload
        assert payload["n"] == n


def test_builder_callback_exception_does_not_crash(db, cfg):
    """A callback that raises on the second call must not abort the build."""
    doc_ids = _insert_docs(db, "u1", 3)
    builder = _make_builder(db, cfg, doc_ids)

    cb = _ErrorOnSecondCallCallback()
    # Should complete without raising.
    result = builder.build_for_user("u1", progress=cb)

    # Build completed — structural sanity check on result.
    assert "docs_processed" in result
    assert result["docs_processed"] == 3
    # Subsequent stages after the error still fired.
    assert cb._calls > 2, "Expected more than 2 callback invocations"


def test_builder_no_progress_is_noop(db, cfg):
    """Calling build_for_user without progress=... completes without error."""
    doc_ids = _insert_docs(db, "u1", 2)
    builder = _make_builder(db, cfg, doc_ids)

    # Must not raise.
    result = builder.build_for_user("u1")
    assert "docs_processed" in result


# ---------------------------------------------------------------------------
# DocAssigner.assign_all progress tests
# ---------------------------------------------------------------------------


def _build_minimal_tree(db: Database, cfg: TaxonomyConfig, doc_ids: list[str]) -> TaxonomyStore:
    """Build a tree so assign_all has something to descend into."""
    import struct

    embedder = _DictEmbedder()
    store = TaxonomyStore(db, embedder)
    root = store.ensure_root("u1")
    leaf = store.add_node("u1", root.node_id, "All Docs", "All docs leaf", is_leaf=True)
    # Give the leaf a centroid directly via SQL (no set_node_centroid helper).
    centroid = embedder.embed_one("All Docs")
    blob = struct.pack(f"<{len(centroid)}f", *centroid)
    db.execute(
        "UPDATE kg_taxonomy_nodes SET centroid = ?, centroid_dim = ? WHERE node_id = ?",
        (blob, len(centroid), leaf.node_id),
    )
    db.commit()
    return store


def test_assigner_assign_all_start_and_done(db, cfg):
    """assign_all fires 'start' and 'done' with correct shapes."""
    doc_ids = _insert_docs(db, "u1", 3)
    store = _build_minimal_tree(db, cfg, doc_ids)
    embedder = _DictEmbedder()
    llm = _StubLLM(doc_ids)
    assigner = DocAssigner(db=db, llm=llm, embedder=embedder, store=store, cfg=cfg)

    events: list[tuple[str, dict]] = []
    assigner.assign_all("u1", progress=lambda s, p: events.append((s, p)))

    stages = [e[0] for e in events]
    assert "start" in stages, f"'start' not fired; got {stages}"
    assert "done" in stages, f"'done' not fired; got {stages}"
    assert stages[0] == "start"
    assert stages[-1] == "done"

    start_payload = dict(next(p for s, p in events if s == "start"))
    assert "n_docs" in start_payload
    assert start_payload["n_docs"] == 3

    done_payload = dict(next(p for s, p in events if s == "done"))
    assert "n_assigned" in done_payload
    assert "duration_s" in done_payload


def test_assigner_assign_all_per_item_events(db, cfg):
    """assign_all fires one 'assign' event per doc with score field."""
    n = 3
    doc_ids = _insert_docs(db, "u1", n)
    store = _build_minimal_tree(db, cfg, doc_ids)
    embedder = _DictEmbedder()
    llm = _StubLLM(doc_ids)
    assigner = DocAssigner(db=db, llm=llm, embedder=embedder, store=store, cfg=cfg)

    events: list[tuple[str, dict]] = []
    assigner.assign_all("u1", progress=lambda s, p: events.append((s, p)))

    assign_events = [e for e in events if e[0] == "assign"]
    assert len(assign_events) == n, (
        f"Expected {n} 'assign' events, got {len(assign_events)}"
    )
    for stage, payload in assign_events:
        assert "i" in payload
        assert "n" in payload
        assert "doc_id" in payload
        assert "score" in payload
        assert payload["n"] == n
