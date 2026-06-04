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
