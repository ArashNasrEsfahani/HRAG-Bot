"""Phase 13 — agentic iterative "deep read".

Pure, side-effect-free helpers for the deep-read loop: detecting a broad query,
picking the document to read, and modelling the document as a bounded set of
ordered *parts* that "open" as the reader visits them. Real documents carry
hundreds of noisy auto-extracted section headings, so we segment by chunk order
into a tidy, bounded map instead of listing every heading.

The loop itself (LLM calls, retrieval, SSE events) lives in
``Orchestrator._run_deep_read`` — everything here is deterministic and unit
testable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

# Exploratory phrasings that warrant a multi-pass read rather than a single
# retrieve→answer. Kept reasonably precise: a false positive triggers a slow
# multi-pass job, so we require an explicit "explore / explain / overview" cue.
_BROAD_RE = re.compile(
    r"\b(?:"
    r"tell me (?:more |everything |all )?about|"
    r"explain(?:\s+to me)?\b|"
    r"give me (?:an? )?(?:overview|summary|rundown|breakdown|tour)|"
    r"(?:an?\s+)?overview of|"
    r"walk me through|"
    r"teach me(?:\s+about)?|"
    r"i(?:'?d like| want| wanna| wish) to (?:learn|know|understand|read|hear) (?:more )?about|"
    r"deep[\s-]?dive|dive into|go deep(?:er)? (?:on|into)|"
    r"everything (?:about|on|in)|"
    r"break (?:it |this |that )?down|"
    r"summari[sz]e|"
    r"what(?:'?s| is| are|'?re) .{0,40}\b(?:about|cover|discuss)"
    r")\b",
    re.IGNORECASE,
)


def is_broad_query(question: str) -> bool:
    """True for open-ended/exploratory questions worth a deep read. Pure."""
    if not question:
        return False
    if len(question.split()) < 3:
        return False
    return bool(_BROAD_RE.search(question))


# Structural / meta questions about how a document is organised (chapter/section
# counts, table of contents, overall structure). Short by nature, so — unlike
# ``_BROAD_RE`` — there is deliberately NO ≥3-word guard.
_STRUCTURAL_RE = re.compile(
    r"(?:"
    r"how many (?:chapters|sections|parts|pages|volumes)|"
    r"number of (?:chapters|sections|pages|parts|volumes)|"
    r"table of contents|contents page|\btoc\b|"
    r"what (?:are|were) the (?:chapters|sections|parts)|"
    r"list (?:the |all )?(?:chapters|sections|parts)|"
    r"how (?:is|are) (?:it|the book|the document|the paper|the text|this) "
    r"(?:organi[sz]ed|structured|divided|laid out)|"
    r"(?:overall )?structure of (?:the|this) (?:book|document|paper|text)|"
    r"how long is (?:the|this) (?:book|document)"
    r")",
    re.IGNORECASE,
)

# Farsi structural cues (matched as plain substrings — no word boundaries in FA).
_STRUCTURAL_FA = (
    "چند فصل",
    "چند بخش",
    "چند صفحه",
    "چند قسمت",
    "فهرست مطالب",
    "فهرست",
    "ساختار کتاب",
    "ساختار سند",
    "تعداد فصل",
    "تعداد صفحات",
)


def is_structural_query(q: str) -> bool:
    """True for questions about a document's *organisation* (chapter/section/page
    counts, table of contents, overall structure). Pure, bilingual EN + FA.

    Tight by design: ``"how many people died"`` and ``"tell me about jung"`` are
    both False.
    """
    return bool(q) and (
        bool(_STRUCTURAL_RE.search(q)) or any(p in q for p in _STRUCTURAL_FA)
    )


# Phrasings that signal an answer failed to find the requested fact. Used by the
# tests + a deferred re-read path; kept pure.
_WEAK_ANSWER_RE = re.compile(
    r"(?:"
    r"could ?n[o']?t find|could not find|"
    r"does ?n[o']?t specify|do(?:es)? ?n[o']?t specify|"
    r"does ?n[o']?t mention|do(?:es)? ?n[o']?t mention|"
    r"do ?n[o']?t have (?:enough )?information|"
    r"no information|not specified|"
    r"the passages (?:do not|do ?n[o']?t) (?:specify|mention|say)"
    r")",
    re.IGNORECASE,
)

_WEAK_ANSWER_FA = (
    "نمی‌دانم",
    "مشخص نشده",
    "اطلاعاتی ندارم",
)


def is_weak_answer(text: str) -> bool:
    """True when a generated answer signals failure-to-find. Pure."""
    return bool(text) and (
        bool(_WEAK_ANSWER_RE.search(text)) or any(p in text for p in _WEAK_ANSWER_FA)
    )


def pick_target_doc(results) -> Optional[tuple[str, str]]:
    """Pick the single most-relevant document from seed retrieval results.

    Aggregates per-``doc_id`` score (rerank score when present, else raw score)
    and returns ``(doc_id, title)`` for the strongest document, or ``None`` when
    there are no document-backed results. Episodic chunks are ignored.
    """
    agg: dict[str, list] = {}  # doc_id -> [score_sum, count, title]
    for r in results:
        ch = getattr(r, "chunk", None)
        if ch is None or getattr(ch, "source_type", "") == "episodic":
            continue
        doc_id = getattr(ch, "doc_id", None)
        if not doc_id:
            continue
        score = getattr(r, "rerank_score", None)
        if score is None:
            score = getattr(r, "score", 0.0) or 0.0
        entry = agg.setdefault(doc_id, [0.0, 0, getattr(ch, "title", "") or ""])
        entry[0] += float(score)
        entry[1] += 1
        if not entry[2]:
            entry[2] = getattr(ch, "title", "") or ""
    if not agg:
        return None
    doc_id, entry = max(agg.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return doc_id, (entry[2] or doc_id)


def _clean_heading(s: str) -> Optional[str]:
    """A short, title-like heading is usable as a part label; dates, fragments,
    and mojibake are not."""
    s = (s or "").strip()
    if not s or len(s) > 48 or "�" in s:
        return None
    if s[0].isdigit():
        return None
    if not any(c.isalpha() for c in s):
        return None
    return s


@dataclass
class DocPart:
    idx: int            # 0-based part number, in document order
    label: str          # display label (clean heading, else "Part N")
    lo: int             # chunk_index lower bound (inclusive)
    hi: int             # chunk_index upper bound (inclusive)
    status: str = "unread"   # unread | read
    quotes: int = 0          # chunks read from this part

    def public(self) -> dict:
        d = asdict(self)
        return d


def build_parts(rows: list[tuple[int, str]], n_parts: int = 10) -> list[DocPart]:
    """Segment a document into up to ``n_parts`` ordered parts.

    ``rows`` is ``(chunk_index, section)`` for every chunk of the document, in
    any order. Parts are contiguous chunk-index ranges; each is labelled by the
    first clean heading found inside it, falling back to "Part N".
    """
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r[0])
    idxs = [r[0] for r in rows]
    lo_idx, hi_idx = idxs[0], idxs[-1]
    span = max(1, hi_idx - lo_idx + 1)
    n_parts = max(1, min(n_parts, span))
    size = span / n_parts
    parts: list[DocPart] = []
    for k in range(n_parts):
        lo = lo_idx + int(round(k * size))
        hi = (lo_idx + int(round((k + 1) * size)) - 1) if k < n_parts - 1 else hi_idx
        if hi < lo:
            hi = lo
        heading = None
        for ci, sec in rows:
            if lo <= ci <= hi:
                heading = _clean_heading(sec)
                if heading:
                    break
        parts.append(DocPart(idx=k, label=heading or f"Part {k + 1}", lo=lo, hi=hi))
    return parts


@dataclass
class DeepReadState:
    """Mutable bookkeeping for one deep-read run."""

    doc_id: str
    doc_title: str
    parts: list[DocPart]
    seen_chunk_ids: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    passes: int = 0

    def part_for_index(self, chunk_index: int) -> Optional[DocPart]:
        for p in self.parts:
            if p.lo <= chunk_index <= p.hi:
                return p
        return self.parts[-1] if self.parts else None

    def open_for_index(self, chunk_index: int) -> tuple[Optional[DocPart], bool]:
        """Record a chunk read at ``chunk_index``; return (part, newly_opened)."""
        p = self.part_for_index(chunk_index)
        if p is None:
            return None, False
        newly = p.status != "read"
        p.status = "read"
        p.quotes += 1
        return p, newly

    def remaining(self) -> int:
        return sum(1 for p in self.parts if p.status != "read")


@dataclass
class PlannerAction:
    """A single, hardened navigation action chosen by the planner LLM.

    The planner picks one action per pass from a closed menu; ``parse_action``
    coerces its raw JSON into one of these so a weak model can never loop or
    crash the loop.
    """

    kind: str                        # "read_part" | "search" | "answer" | "read_page"
    note: str = ""
    part_idx: Optional[int] = None   # for read_part (resolved, in-range)
    query: str = ""                  # for search
    lo: Optional[int] = None         # chunk_index/page lower bound (read_part/read_page)
    hi: Optional[int] = None         # chunk_index/page upper bound


def _coerce_int(value) -> Optional[int]:
    """Best-effort int coercion tolerating str/float/None. Never raises."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


def _first_unread_idx(state: DeepReadState) -> Optional[int]:
    for p in state.parts:
        if p.status != "read":
            return p.idx
    return None


def parse_action(
    data: dict,
    state: DeepReadState,
    *,
    pages_available: bool = False,
) -> PlannerAction:
    """Harden a raw parsed-JSON planner dict into a safe, executable action.

    Pure and defensive: the weak planner model must never be able to loop on an
    already-read part or crash on malformed input. Unknown/empty actions fall
    back to a (possibly empty) ``search``.
    """
    if not isinstance(data, dict):
        data = {}
    note = str(data.get("note", "")).strip()
    action = str(data.get("action", "")).strip().lower()

    if action == "answer":
        return PlannerAction("answer", note=note)

    if action == "read_part":
        if not state.parts:
            return PlannerAction("search", note=note, query="")
        idx = _coerce_int(data.get("part_idx"))
        if idx is None:
            fallback = _first_unread_idx(state)
            idx = fallback if fallback is not None else 0
        # Clamp into range.
        idx = max(0, min(idx, len(state.parts) - 1))
        # Action-repeat guard: never re-read a part already marked read.
        if state.parts[idx].status == "read":
            nxt = _first_unread_idx(state)
            if nxt is None:
                return PlannerAction("answer", note=note)
            idx = nxt
        part = state.parts[idx]
        return PlannerAction("read_part", note=note, part_idx=idx, lo=part.lo, hi=part.hi)

    if action == "read_page":
        if not pages_available:
            return PlannerAction(
                "search", note=note, query=str(data.get("query", "")).strip()
            )
        a = _coerce_int(data.get("from", data.get("lo")))
        b = _coerce_int(data.get("to", data.get("hi")))
        if a is None or b is None:
            return PlannerAction(
                "search", note=note, query=str(data.get("query", "")).strip()
            )
        return PlannerAction("read_page", note=note, lo=min(a, b), hi=max(a, b))

    if action == "search":
        return PlannerAction(
            "search", note=note, query=str(data.get("query", "")).strip()
        )

    # Unknown / empty / missing action → safe default.
    return PlannerAction("search", note=note, query="")


def _row_field(row, key: str):
    """Flexible accessor for sqlite3.Row / dict / object / tuple-like rows."""
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(row, key, None)


def distinct_chapter_labels(rows: list) -> list[str]:
    """Clean, dedupe (first-seen order) the ``section`` field of each row.

    ``rows`` may be sqlite3.Row, dicts, or objects carrying a ``section`` field.
    Junk headings (long/numeric/mojibake) are dropped via ``_clean_heading``.
    """
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        label = _clean_heading(_row_field(row, "section") or "")
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


_TOC_CUE_RE = re.compile(r"table of contents|contents|فهرست مطالب|فهرست", re.IGNORECASE)
_TOC_CHAPTER_RE = re.compile(r"chapter|فصل|بخش", re.IGNORECASE)
_TOC_LINE_END_RE = re.compile(r"\d+$")


def find_toc_chunk(rows: list) -> Optional[tuple[int, str]]:
    """Find an early chunk that looks like a table of contents.

    ``rows`` carry ``chunk_index:int`` and ``text:str`` (flexible accessor).
    Only the earliest chunks are considered; the best-scoring one is returned as
    ``(chunk_index, text)`` if it clears a small threshold, else ``None``.
    Conservative by design — a false ``None`` is fine (the caller falls back to
    ``distinct_chapter_labels``).
    """
    parsed: list[tuple[int, str]] = []
    for row in rows:
        ci = _row_field(row, "chunk_index")
        txt = _row_field(row, "text")
        if ci is None:
            continue
        try:
            ci_int = int(ci)
        except (ValueError, TypeError):
            continue
        parsed.append((ci_int, str(txt or "")))
    if not parsed:
        return None
    parsed.sort(key=lambda r: r[0])
    n_early = max(15, len(parsed) // 12)
    early = parsed[:n_early]

    best: Optional[tuple[int, str]] = None
    best_score = 0.0
    for ci, text in early:
        low = text.lower()
        score = 0.0
        if _TOC_CUE_RE.search(low):
            score += 3
        score += min(5, len(_TOC_CHAPTER_RE.findall(low)))
        toc_lines = 0
        for line in text.split("\n"):
            stripped = line.strip()
            if len(stripped) < 60 and _TOC_LINE_END_RE.search(stripped):
                toc_lines += 1
        score += min(5, toc_lines)
        if score > best_score:
            best_score = score
            best = (ci, text)
    if best is not None and best_score >= 4:
        return best
    return None
