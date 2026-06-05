"""Phase 13 — deep-read pure helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hrag.deepread import (
    DeepReadState,
    DocPart,
    build_parts,
    distinct_chapter_labels,
    find_toc_chunk,
    is_broad_query,
    is_structural_query,
    is_weak_answer,
    parse_action,
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


# ---------------------------------------------------------------------------
# is_structural_query
# ---------------------------------------------------------------------------


def test_structural_query_positives() -> None:
    for q in [
        "How many chapters does this book have?",
        "what are the sections",
        "list the chapters",
        "table of contents",
        "contents page",
        "structure of the book",
        "overall structure of this document",
        "how is it organized",
        "how is the book structured",
        "how long is the book",
        "number of pages",
        "چند فصل دارد",
        "فهرست مطالب",
        "ساختار کتاب چیست",
    ]:
        assert is_structural_query(q), q


def test_structural_query_negatives() -> None:
    for q in [
        "how many people died",
        "tell me about jung",
        "what is my name",
        "",
    ]:
        assert not is_structural_query(q), q


# ---------------------------------------------------------------------------
# is_weak_answer
# ---------------------------------------------------------------------------


def test_weak_answer_positive_and_negative() -> None:
    assert is_weak_answer("The passages don't specify the total number.")
    assert is_weak_answer("I couldn't find that in the document.")
    assert is_weak_answer("اطلاعاتی ندارم")
    assert not is_weak_answer("The book has 7 chapters.")
    assert not is_weak_answer("")


# ---------------------------------------------------------------------------
# parse_action
# ---------------------------------------------------------------------------


def _state_with_parts() -> DeepReadState:
    parts = [
        DocPart(idx=0, label="A", lo=0, hi=9, status="read"),
        DocPart(idx=1, label="B", lo=10, hi=19, status="unread"),
        DocPart(idx=2, label="C", lo=20, hi=29, status="unread"),
    ]
    return DeepReadState(doc_id="d", doc_title="Doc", parts=parts)


def test_parse_action_read_part_in_range() -> None:
    state = _state_with_parts()
    act = parse_action({"action": "read_part", "part_idx": 1, "note": "n"}, state)
    assert act.kind == "read_part"
    assert act.part_idx == 1
    assert act.lo == 10 and act.hi == 19
    assert act.note == "n"


def test_parse_action_read_part_already_read_redirects() -> None:
    state = _state_with_parts()  # part 0 is read
    act = parse_action({"action": "read_part", "part_idx": 0}, state)
    assert act.kind == "read_part"
    assert act.part_idx != 0          # redirected away from the read part
    assert act.part_idx == 1          # first unread
    assert act.lo == 10 and act.hi == 19


def test_parse_action_read_part_out_of_range_clamped() -> None:
    state = _state_with_parts()
    act = parse_action({"action": "read_part", "part_idx": 99}, state)
    assert act.kind == "read_part"
    assert act.part_idx == len(state.parts) - 1


def test_parse_action_read_part_non_int_does_not_raise() -> None:
    state = _state_with_parts()
    act = parse_action({"action": "read_part", "part_idx": "abc"}, state)
    # falls back to first unread part (idx 1), never raises
    assert act.kind == "read_part"
    assert act.part_idx == 1


def test_parse_action_all_parts_read_downgrades_to_answer() -> None:
    parts = [DocPart(idx=0, label="A", lo=0, hi=9, status="read")]
    state = DeepReadState(doc_id="d", doc_title="Doc", parts=parts)
    act = parse_action({"action": "read_part", "part_idx": 0}, state)
    assert act.kind == "answer"


def test_parse_action_read_part_no_parts_downgrades_to_search() -> None:
    state = DeepReadState(doc_id="d", doc_title="Doc", parts=[])
    act = parse_action({"action": "read_part", "part_idx": 0}, state)
    assert act.kind == "search"
    assert act.query == ""


def test_parse_action_garbage_and_unknown() -> None:
    state = _state_with_parts()
    for data in [{}, {"action": "nonsense"}, {"action": ""}]:
        act = parse_action(data, state)
        assert act.kind == "search"
        assert act.query == ""


def test_parse_action_read_page_disabled_downgrades_to_search() -> None:
    state = _state_with_parts()
    act = parse_action(
        {"action": "read_page", "from": 3, "to": 5, "query": "q"},
        state,
        pages_available=False,
    )
    assert act.kind == "search"
    assert act.query == "q"


def test_parse_action_read_page_enabled() -> None:
    state = _state_with_parts()
    act = parse_action(
        {"action": "read_page", "from": 8, "to": 3},
        state,
        pages_available=True,
    )
    assert act.kind == "read_page"
    assert act.lo == 3 and act.hi == 8


def test_parse_action_read_page_missing_bounds_downgrades() -> None:
    state = _state_with_parts()
    act = parse_action(
        {"action": "read_page", "from": 8},
        state,
        pages_available=True,
    )
    assert act.kind == "search"


def test_parse_action_answer() -> None:
    state = _state_with_parts()
    act = parse_action({"action": "answer", "note": "done"}, state)
    assert act.kind == "answer"
    assert act.note == "done"


def test_parse_action_search_with_query() -> None:
    state = _state_with_parts()
    act = parse_action({"action": "search", "query": "carl jung"}, state)
    assert act.kind == "search"
    assert act.query == "carl jung"


# ---------------------------------------------------------------------------
# distinct_chapter_labels
# ---------------------------------------------------------------------------


def test_distinct_chapter_labels_cleans_and_dedupes() -> None:
    rows = [
        {"section": "Introduction"},
        {"section": "Introduction"},      # duplicate → dropped
        {"section": "12. JAN 1748"},       # digit-led → junk
        {"section": "The Red Book"},
        {"section": "����"},                # mojibake → junk
        {"section": ""},                   # empty → junk
        {"section": "Conclusion"},
    ]
    labels = distinct_chapter_labels(rows)
    assert labels == ["Introduction", "The Red Book", "Conclusion"]


# ---------------------------------------------------------------------------
# find_toc_chunk
# ---------------------------------------------------------------------------


def test_find_toc_chunk_detects_toc_shape() -> None:
    toc_text = (
        "Table of Contents\n"
        "Chapter 1 .... 3\n"
        "Chapter 2 .... 31\n"
        "Chapter 3 .... 64\n"
        "Chapter 4 .... 98\n"
    )
    rows = [{"chunk_index": 0, "text": "Title page"}, {"chunk_index": 1, "text": toc_text}]
    # pad with non-TOC chunks so the early window still includes the TOC
    rows += [{"chunk_index": i, "text": "body paragraph text"} for i in range(2, 40)]
    found = find_toc_chunk(rows)
    assert found is not None
    assert found[0] == 1
    assert "Table of Contents" in found[1]


def test_find_toc_chunk_none_when_absent() -> None:
    rows = [
        {"chunk_index": i, "text": "ordinary prose with no toc shape at all"}
        for i in range(0, 40)
    ]
    assert find_toc_chunk(rows) is None
