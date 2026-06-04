"""Phase 13 — deep-read pure helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hrag.deepread import (
    DeepReadState,
    build_parts,
    is_broad_query,
    pick_target_doc,
)


# ---------------------------------------------------------------------------
# is_broad_query
# ---------------------------------------------------------------------------


def test_broad_query_positives() -> None:
    for q in [
        "tell me about the red book",
        "tell me everything about carl jung",
        "explain the red book to me",
        "give me an overview of jung's work",
        "i want to learn about carl jung",
        "walk me through the document",
        "summarize the red book",
        "what is the red book about",
        "can you break this down for me",
    ]:
        assert is_broad_query(q), q


def test_broad_query_negatives() -> None:
    for q in [
        "what is my name",
        "hi",
        "where do i work",
        "define recursion",
        "who wrote it",
        "",
        "thanks",
    ]:
        assert not is_broad_query(q), q


# ---------------------------------------------------------------------------
# pick_target_doc
# ---------------------------------------------------------------------------


@dataclass
class _Ch:
    doc_id: str
    title: str = ""
    source_type: str = "document"
    chunk_id: str = "c"


@dataclass
class _Res:
    chunk: _Ch
    score: float = 0.0
    rerank_score: Optional[float] = None


def test_pick_target_doc_by_aggregate_score() -> None:
    results = [
        _Res(_Ch("docA", "Doc A"), score=0.2),
        _Res(_Ch("docB", "Doc B"), score=0.9),
        _Res(_Ch("docB", "Doc B"), score=0.8),
        _Res(_Ch("docA", "Doc A"), score=0.1),
    ]
    picked = pick_target_doc(results)
    assert picked == ("docB", "Doc B")


def test_pick_target_doc_ignores_episodic_and_empty() -> None:
    assert pick_target_doc([]) is None
    only_ep = [_Res(_Ch("epdoc", "mem", source_type="episodic"), score=5.0)]
    assert pick_target_doc(only_ep) is None


# ---------------------------------------------------------------------------
# build_parts + DeepReadState
# ---------------------------------------------------------------------------


def test_build_parts_bounded_and_ordered() -> None:
    # i<10 → clean heading; i>=10 → digit-led date fragment (rejected).
    rows = [(i, "INTRODUCTION" if i < 10 else f"{i}. JAN 1748") for i in range(0, 100)]
    parts = build_parts(rows, n_parts=10)
    assert len(parts) == 10
    # contiguous, non-overlapping, ascending
    for a, b in zip(parts, parts[1:]):
        assert a.hi < b.lo
        assert a.idx + 1 == b.idx
    # a clean heading is used as a label where one exists
    assert parts[0].label == "INTRODUCTION"
    # noisy digit-led headings fall back to "Part N"
    assert any(p.label.startswith("Part ") for p in parts[1:])


def test_build_parts_collapses_huge_section_counts() -> None:
    # 569-section-style doc → still a bounded map.
    rows = [(i, f"§ heading {i}") for i in range(0, 600)]
    parts = build_parts(rows, n_parts=10)
    assert len(parts) == 10


def test_state_opens_part_for_chunk_index() -> None:
    rows = [(i, "INTRODUCTION" if i < 50 else "CONCLUSION") for i in range(0, 100)]
    parts = build_parts(rows, n_parts=10)
    state = DeepReadState(doc_id="d", doc_title="Doc", parts=parts)
    assert state.remaining() == 10
    p, newly = state.open_for_index(5)     # early chunk → first part
    assert newly and p.idx == 0 and p.status == "read" and p.quotes == 1
    # opening another chunk in the same part is not "newly", bumps quotes
    p2, newly2 = state.open_for_index(7)
    assert not newly2 and p2.idx == 0 and p2.quotes == 2
    # a late chunk opens a later part
    p3, newly3 = state.open_for_index(95)
    assert newly3 and p3.idx == len(parts) - 1
    assert state.remaining() == 8
