"""Tests for hrag.retrieval.doc_scope — DocScopedRetriever wrapper.

Covers:
  - Regime A (explicit-title): "HippoRAG" matches HippoRAG only, not HippoRAG 2.
  - Regime A precedence: "HippoRAG 2" matches HippoRAG 2 only.
  - Graceful fallthrough when both regimes yield zero docs (empty corpus).
  - Post-hoc filter excludes off-topic docs.
"""

from __future__ import annotations

from typing import Optional

from hrag.retrieval.doc_scope import DocScopedRetriever
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubRetriever:
    """Returns canned results; counts retrieve() calls and remembers top_k."""

    def __init__(self, name: str, results: list[RetrievalResult]) -> None:
        self.name = name
        self._results = results
        self.calls = 0
        self.last_top_k: int | None = None

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 30,
        source_types: Optional[list[str]] = None,
        intent_hint=None,  # ignored; required for Retriever contract compat
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        self.calls += 1
        self.last_top_k = top_k
        return list(self._results)


class _StubEmbedder:
    """Doc-scope only uses the embedder for regime B in the abstract; concrete
    coarse-rank uses the wrapped retriever, so this is a placeholder."""

    name = "stub-embedder"
    dim = 8

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]

    def embed_one(self, text):
        return [0.0] * self.dim


def _make_chunk(chunk_id: str, doc_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        user_id="default",
        text=f"text-{chunk_id}",
        embedding_text=f"text-{chunk_id}",
    )


def _make_result(
    chunk_id: str, doc_id: str, score: float = 1.0
) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(chunk_id, doc_id),
        score=score,
        retriever="vector",
    )


def _seed_documents(db, rows: list[tuple[str, str]]) -> None:
    """Insert (doc_id, title) rows into the documents table."""
    for doc_id, title in rows:
        db.execute(
            "INSERT INTO documents(doc_id, user_id, source_path, title, "
            "source_type) VALUES (?, ?, ?, ?, 'document')",
            (doc_id, "default", f"/tmp/{doc_id}.pdf", title),
        )
    db.commit()


# ---------------------------------------------------------------------------
# Regime A: explicit-title match
# ---------------------------------------------------------------------------


def test_title_match_hipporag_excludes_hipporag2(tmp_db) -> None:
    """'What does HippoRAG do?' must match HIPPORAG only, not HippoRAG 2."""
    _seed_documents(
        tmp_db,
        [
            ("docA", "HIPPORAG"),
            ("docB", "2502.14802v2"),    # HippoRAG 2
            ("docC", "2025.findings-naacl.30"),  # RAGate
        ],
    )
    inner = _StubRetriever("inner", [])
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    matched = wrapper._title_match("What does HippoRAG do?", "default")
    assert matched == {"docA"}


def test_title_match_hipporag2_excludes_base(tmp_db) -> None:
    """'HippoRAG 2 ablations' must match HippoRAG 2 only, NOT base HippoRAG."""
    _seed_documents(
        tmp_db,
        [
            ("docA", "HIPPORAG"),
            ("docB", "2502.14802v2"),
            ("docC", "2025.findings-naacl.30"),
        ],
    )
    inner = _StubRetriever("inner", [])
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    matched = wrapper._title_match("HippoRAG 2 ablations", "default")
    assert matched == {"docB"}


def test_title_match_ragate(tmp_db) -> None:
    _seed_documents(
        tmp_db,
        [
            ("docA", "HIPPORAG"),
            ("docB", "2502.14802v2"),
            ("docC", "2025.findings-naacl.30"),
        ],
    )
    inner = _StubRetriever("inner", [])
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    matched = wrapper._title_match("What is RAGate?", "default")
    assert matched == {"docC"}


def test_title_match_compare_two_papers(tmp_db) -> None:
    """Multi-title query restricts to all matched docs."""
    _seed_documents(
        tmp_db,
        [
            ("docA", "HIPPORAG"),
            ("docB", "2502.14802v2"),
            ("docC", "2025.findings-naacl.30"),
        ],
    )
    inner = _StubRetriever("inner", [])
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    matched = wrapper._title_match("How do HippoRAG and RAGate differ?", "default")
    assert matched == {"docA", "docC"}


def test_title_match_no_alias_in_query(tmp_db) -> None:
    _seed_documents(tmp_db, [("docA", "HIPPORAG")])
    inner = _StubRetriever("inner", [])
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    matched = wrapper._title_match("Tell me about something else", "default")
    assert matched == set()


def test_title_match_empty_query(tmp_db) -> None:
    _seed_documents(tmp_db, [("docA", "HIPPORAG")])
    inner = _StubRetriever("inner", [])
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    assert wrapper._title_match("", "default") == set()
    assert wrapper._title_match("   ", "default") == set()


def test_title_match_word_boundary_protects_base_alias(tmp_db) -> None:
    """The bare token 'hipporag2' (no space) should NOT match the base
    HippoRAG alias because of the \\b word boundary."""
    _seed_documents(
        tmp_db,
        [
            ("docA", "HIPPORAG"),
            ("docB", "2502.14802v2"),
        ],
    )
    inner = _StubRetriever("inner", [])
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    matched = wrapper._title_match("compare hipporag2 ablations", "default")
    # Should match HippoRAG 2 only — not the base HIPPORAG paper.
    assert matched == {"docB"}


# ---------------------------------------------------------------------------
# Regime B / fallthrough behaviour (full retrieve())
# ---------------------------------------------------------------------------


def test_fallthrough_when_corpus_empty(tmp_db) -> None:
    """No title match + no chunks coming back from coarse-rank => fall
    through unchanged: the FINAL wrapped retrieve is called with the
    original top_k (not the oversampled count)."""
    # No documents inserted.
    inner = _StubRetriever("inner", [])
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    out = wrapper.retrieve("a totally unknown query", "default", top_k=5)
    assert out == []
    # Inner is called once for coarse-rank attempt (returns empty), then
    # once more for the graceful fallthrough call. Either way, the LAST
    # call must use the original top_k=5 (no oversampling, since no
    # regime matched).
    assert inner.calls >= 1
    assert inner.last_top_k == 5


def test_post_hoc_filter_excludes_off_topic_docs(tmp_db) -> None:
    """When regime A matches, only chunks from allowed docs survive the filter."""
    _seed_documents(
        tmp_db,
        [
            ("docA", "HIPPORAG"),
            ("docB", "2502.14802v2"),
            ("docC", "2025.findings-naacl.30"),
        ],
    )
    # Wrapped retriever returns a mix of all three docs.
    results = [
        _make_result("c1", "docA", score=0.9),
        _make_result("c2", "docB", score=0.8),
        _make_result("c3", "docA", score=0.7),
        _make_result("c4", "docC", score=0.6),
        _make_result("c5", "docB", score=0.5),
    ]
    inner = _StubRetriever("inner", results)
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    # Regime A picks docA only.
    out = wrapper.retrieve("Explain HippoRAG.", "default", top_k=10)
    chunk_ids = [r.chunk.chunk_id for r in out]
    assert chunk_ids == ["c1", "c3"]


def test_post_hoc_filter_truncates_to_top_k(tmp_db) -> None:
    _seed_documents(tmp_db, [("docA", "HIPPORAG")])
    results = [_make_result(f"c{i}", "docA", score=1.0 - 0.01 * i) for i in range(20)]
    inner = _StubRetriever("inner", results)
    wrapper = DocScopedRetriever(inner, tmp_db, _StubEmbedder())

    out = wrapper.retrieve("HippoRAG please", "default", top_k=5)
    assert len(out) == 5


def test_oversamples_when_regime_matches(tmp_db) -> None:
    """When a regime matches, the wrapper oversamples top_k * factor from inner."""
    _seed_documents(tmp_db, [("docA", "HIPPORAG")])
    inner = _StubRetriever("inner", [])
    wrapper = DocScopedRetriever(
        inner, tmp_db, _StubEmbedder(), oversample_factor=4
    )

    wrapper.retrieve("HippoRAG something", "default", top_k=5)
    assert inner.last_top_k == 20  # 5 * 4


def test_coarse_doc_rank_picks_top_k_docs(tmp_db) -> None:
    """No title match -> coarse rank aggregates scores by doc_id and
    restricts retrieval to the top-k. Even chunks from off-topic docs in
    the wrapped retriever's oversampled set are filtered out."""
    _seed_documents(
        tmp_db,
        [
            ("docA", "Some Paper A"),
            ("docB", "Some Paper B"),
            ("docC", "Some Paper C"),
            ("docD", "Some Paper D"),
        ],
    )
    # docA gets two high-scoring hits; docB one mid; docC and docD weak.
    results = [
        _make_result("a1", "docA", score=0.9),
        _make_result("a2", "docA", score=0.8),
        _make_result("b1", "docB", score=0.5),
        _make_result("c1", "docC", score=0.1),
        _make_result("d1", "docD", score=0.05),
    ]
    inner = _StubRetriever("inner", results)
    wrapper = DocScopedRetriever(
        inner, tmp_db, _StubEmbedder(), coarse_top_k=2
    )

    out = wrapper.retrieve("query without any paper alias", "default", top_k=10)
    # docA + docB win the coarse rank; docC/docD chunks must be filtered.
    surviving = {r.chunk.doc_id for r in out}
    assert surviving == {"docA", "docB"}
