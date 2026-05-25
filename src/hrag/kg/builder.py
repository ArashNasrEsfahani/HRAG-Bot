"""Triple extractor for the KG layer (Phase 2).

Extracts (head, relation, tail) triples from document chunks via an LLM using
the HippoRAG-2-style OpenIE prompt in prompts/triple_extraction.md.

Concurrency model: ThreadPoolExecutor over chunks.  Ollama serialises
internally; OpenAI/Anthropic get true concurrency.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:
    from hrag.db.connection import Database
    from hrag.providers.llm import LLMProvider
    from hrag.types import Chunk


# ---------------------------------------------------------------------------
# Canonicalisation (Phase 9.12)
# ---------------------------------------------------------------------------


def _canon_field(text: str) -> str:
    """Canonical surface form for a single triple field.

    Mirrors :func:`hrag.kg.store._canon` (lowercase + strip): the same canonical
    form that the graph store's synonym-merging step uses. Keeping the two in
    lockstep means a triple deduped here will also collapse onto the same
    phrase node downstream. Heavier normalisations (embedding-based synonym
    merge) still happen in :class:`KGStore`; this function is the cheap
    deterministic floor.
    """
    return text.strip().lower()


def canonical_triple_key(subject: str, relation: str, obj: str) -> str:
    """Stable SHA-256 key over the canonical (subject, relation, object).

    The pipe separator is reserved: it never appears inside a canonicalised
    field (we strip leading/trailing whitespace; pipes inside a phrase are
    unusual but would not collide because the order is fixed and each field
    has been pre-canonicalised).
    """
    payload = f"{_canon_field(subject)}|{_canon_field(relation)}|{_canon_field(obj)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Triple:
    """A knowledge-graph triple extracted from a single chunk.

    ``head_canonical`` and ``tail_canonical`` are filled in later by the graph
    store's synonym-merging step; they default to the surface form so callers
    never see ``None`` or empty strings.
    """

    head: str
    relation: str
    tail: str
    source_chunk_id: str
    head_canonical: str = field(default="")
    tail_canonical: str = field(default="")

    def __post_init__(self) -> None:
        if not self.head_canonical:
            self.head_canonical = self.head
        if not self.tail_canonical:
            self.tail_canonical = self.tail


# ---------------------------------------------------------------------------
# Prompt loading helpers
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "triple_extraction.md"


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _render_prompt(template: str, passage: str) -> str:
    """Render the prompt template safely.

    The template contains JSON example objects with ``{...}`` braces that would
    break ``str.format()``.  We therefore do a targeted replacement of the
    single ``{passage}`` placeholder only.
    """
    return template.replace("{passage}", passage)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_FENCE_PREFIXES = ("```json", "```")


def _strip_fences(text: str) -> str:
    """Remove leading/trailing Markdown code fences if present."""
    stripped = text.strip()
    for prefix in _FENCE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def _extract_json_list(raw: str) -> list | None:
    """Locate the outermost ``[...]`` in *raw* and parse it.

    Returns ``None`` on any failure so the caller can warn and return ``[]``.
    """
    text = _strip_fences(raw)
    start = text.find("[")
    if start == -1:
        return None
    # Walk forward to find the matching closing bracket.
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Triple validation
# ---------------------------------------------------------------------------

# Maximum length for any single field (head/relation/tail). qwen sometimes
# emits run-on phrases; cap defensively so downstream graph nodes stay sane.
_MAX_FIELD_LEN = 200

# A field must contain at least one alphanumeric character to count as
# meaningful content. Rejects ".", "-", ",", "...", etc.
_RE_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")

# Placeholder strings the LLM occasionally emits literally instead of leaving a
# field blank. Values are lowercased. Anything matching exactly is rejected.
_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {
        "<empty>", "<none>", "<null>", "<n/a>", "<unknown>", "<>",
        "none", "null", "n/a", "na", "unknown", "unspecified",
        "tbd", "todo", "placeholder", "...",
    }
)

# Relations that are pure connective/preposition with no semantic content.
# Compared after lowercase + strip. If a triple's relation matches exactly,
# the triple is dropped.
_STOP_RELATIONS: frozenset[str] = frozenset({
    "to", "of", "in", "on", "at", "by", "for", "from", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an",
    "and", "or", "but",
    "as", "this", "that", "these", "those",
    "it", "they",
    "has", "have", "had",
})


def _drop_reason(head: str, relation: str, tail: str) -> str | None:
    """Return a reason name if the (head, relation, tail) should be dropped, else None.

    Reasons (in priority order): ``empty_head``, ``empty_relation``,
    ``empty_tail``, ``non_alnum``, ``placeholder``, ``stop_relation``,
    ``self_loop``, ``too_long``.
    """
    if not head:
        return "empty_head"
    if not relation:
        return "empty_relation"
    if not tail:
        return "empty_tail"
    if not (_RE_HAS_ALNUM.search(head) and _RE_HAS_ALNUM.search(relation) and _RE_HAS_ALNUM.search(tail)):
        return "non_alnum"
    if head in _PLACEHOLDER_VALUES or relation in _PLACEHOLDER_VALUES or tail in _PLACEHOLDER_VALUES:
        return "placeholder"
    if relation in _STOP_RELATIONS:
        return "stop_relation"
    if head == tail:
        return "self_loop"
    if len(head) > _MAX_FIELD_LEN or len(relation) > _MAX_FIELD_LEN or len(tail) > _MAX_FIELD_LEN:
        return "too_long"
    return None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class TripleExtractor:
    """Extracts (head, relation, tail) triples per chunk via the LLM.

    Loads ``prompts/triple_extraction.md`` once at ``__init__`` time.
    Concurrent over chunks via ``ThreadPoolExecutor`` — Ollama serialises
    internally but we still save request overhead; OpenAI/Anthropic get true
    concurrency.
    """

    name = "triple_extractor"

    def __init__(
        self,
        llm: LLMProvider,
        max_workers: int = 8,
        db: Database | None = None,           # NEW: cache backend
        dedup_enabled: bool = True,           # Phase 9.12: cross-chunk dedup
        progress_cb: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self._llm = llm
        self._max_workers = max_workers
        self._prompt_template = _load_prompt_template()
        self._db = db
        self._dedup_enabled = bool(dedup_enabled)
        self._progress_cb = progress_cb
        self._model_name: str = (
            getattr(llm, "model_name", None)
            or getattr(getattr(llm, "config", None), "model", None)
            or "unknown"
        )

    def _emit(self, event: str, payload: dict) -> None:
        """Safe progress emit — never lets a callback bug break extraction."""
        if self._progress_cb is None:
            return
        try:
            self._progress_cb(event, payload)
        except Exception:  # noqa: BLE001 — progress is best-effort
            pass

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, chunk_text: str) -> str:
        """SHA-256 over chunk text + model name. Stable across runs."""
        import hashlib
        h = hashlib.sha256()
        h.update(chunk_text.encode("utf-8"))
        h.update(b"\x00")
        h.update(self._model_name.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> list[Triple] | None:
        if self._db is None:
            return None
        try:
            row = self._db.execute(
                "SELECT triples_json FROM kg_triple_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
        except Exception:
            return None  # cache table missing or transient DB error → cache miss
        if row is None:
            return None
        try:
            raw = json.loads(row[0] if not hasattr(row, "keys") else row["triples_json"])
        except (json.JSONDecodeError, TypeError):
            return None
        triples: list[Triple] = []
        for d in raw:
            if not isinstance(d, dict):
                continue
            triples.append(
                Triple(
                    head=d.get("head", ""),
                    relation=d.get("relation", ""),
                    tail=d.get("tail", ""),
                    source_chunk_id=d.get("source_chunk_id", ""),
                )
            )
        return triples

    def _cache_put(self, key: str, triples: list[Triple]) -> None:
        if self._db is None:
            return
        payload = json.dumps(
            [
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "source_chunk_id": t.source_chunk_id,
                }
                for t in triples
            ]
        )
        try:
            with self._db.conn:
                self._db.execute(
                    "INSERT OR REPLACE INTO kg_triple_cache"
                    "(cache_key, model_name, triples_json) VALUES (?, ?, ?)",
                    (key, self._model_name, payload),
                )
        except Exception:
            pass  # cache write failure must not break extraction

    # ------------------------------------------------------------------
    # Cross-chunk dedup (Phase 9.12)
    # ------------------------------------------------------------------

    def _record_canonical_triples(
        self, triples: list[Triple]
    ) -> tuple[int, int]:
        """Insert each canonical triple into ``kg_canonical_triples`` once.

        Returns ``(n_new, n_seen)``: how many canonical keys were inserted
        for the first time vs how many already existed. Concurrent inserts
        on the same key are safe — SQLite serialises writes at the file
        level and the ``ON CONFLICT(canonical_key) DO UPDATE`` atomically
        increments ``freq``.

        Disabled when ``dedup_enabled=False`` or no ``db`` is wired up.
        Cache-table-missing (legacy DB) is tolerated silently — extraction
        must keep working without the dedup index.
        """
        if not self._dedup_enabled or self._db is None or not triples:
            return (0, 0)
        n_new = 0
        n_seen = 0
        for t in triples:
            key = canonical_triple_key(t.head, t.relation, t.tail)
            try:
                with self._db.conn:
                    cur = self._db.execute(
                        "INSERT INTO kg_canonical_triples"
                        "(canonical_key, subject, relation, object,"
                        " first_seen_chunk_id, freq)"
                        " VALUES (?, ?, ?, ?, ?, 1)"
                        " ON CONFLICT(canonical_key) DO UPDATE"
                        " SET freq = freq + 1",
                        (
                            key,
                            _canon_field(t.head),
                            _canon_field(t.relation),
                            _canon_field(t.tail),
                            t.source_chunk_id,
                        ),
                    )
                # rowcount == 1 on both INSERT and UPSERT in SQLite, so we
                # disambiguate via a follow-up SELECT on freq.
                row = self._db.execute(
                    "SELECT freq FROM kg_canonical_triples WHERE canonical_key=?",
                    (key,),
                ).fetchone()
                freq = row["freq"] if row is not None and hasattr(row, "keys") else (row[0] if row else 1)
                if freq == 1:
                    n_new += 1
                else:
                    n_seen += 1
                _ = cur  # keep symmetry with _cache_put pattern
            except Exception:
                # Table missing or transient DB error — silent fallback.
                # Counts stay zero for this triple; caller treats as no-op.
                continue
        return (n_new, n_seen)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_one(
        self,
        chunk: Chunk,
        drop_counts: dict[str, int] | None = None,
    ) -> list[Triple]:
        """Extract triples from a single *chunk*.

        Returns an empty list (and emits a ``warnings.warn``) on any parse
        failure — never raises. Invalid triples (empty / non-alphanumeric /
        over-length head/relation/tail) are silently dropped here; if
        *drop_counts* is provided, drop reasons are accumulated into it for
        a batch-level summary warning.
        """
        key = self._cache_key(chunk.text)
        cached = self._cache_get(key)
        if cached is not None:
            # Re-stamp source_chunk_id in case the cached entry came from a
            # different chunk that happened to share text (rare but possible).
            restamped = [
                Triple(
                    head=t.head,
                    relation=t.relation,
                    tail=t.tail,
                    source_chunk_id=chunk.chunk_id,
                )
                for t in cached
            ]
            # Cross-chunk dedup bookkeeping still runs so freq counters track
            # every chunk that contains this triple, even when the LLM call
            # was short-circuited by an identical chunk text.
            n_new, n_seen = self._record_canonical_triples(restamped)
            self._emit(
                "kg_dedup_hit",
                {
                    "chunk_id": chunk.chunk_id,
                    "hashed_skipped_llm": True,
                    "triples_reused": len(restamped),
                    "canonical_new": n_new,
                    "canonical_seen": n_seen,
                },
            )
            print(
                f"[kg] chunk {chunk.chunk_id}: dedup hit, "
                f"skipped LLM ({len(restamped)} triples reused)",
                flush=True,
            )
            return restamped

        prompt = _render_prompt(self._prompt_template, chunk.text)
        raw = self._llm.complete(prompt, temperature=0.0)

        parsed = _extract_json_list(raw)
        if parsed is None:
            warnings.warn(
                f"TripleExtractor: could not parse JSON from LLM output for chunk "
                f"{chunk.chunk_id!r}. Raw output (first 200 chars): {raw[:200]!r}",
                stacklevel=2,
            )
            # Parse failure is transient — do NOT cache so we retry next run.
            return []

        triples: list[Triple] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            head = item.get("head")
            relation = item.get("relation")
            tail = item.get("tail")
            if not (isinstance(head, str) and isinstance(relation, str) and isinstance(tail, str)):
                continue

            head_norm = head.strip().lower()
            relation_norm = relation.strip().lower()
            tail_norm = tail.strip().lower()

            reason = _drop_reason(head_norm, relation_norm, tail_norm)
            if reason is not None:
                if drop_counts is not None:
                    drop_counts[reason] = drop_counts.get(reason, 0) + 1
                continue

            triples.append(
                Triple(
                    head=head_norm,
                    relation=relation_norm,
                    tail=tail_norm,
                    source_chunk_id=chunk.chunk_id,
                )
            )
        # Cache successful result (including known-empty []) so we don't re-call
        # the LLM for chunks that genuinely yield no triples.
        self._cache_put(key, triples)
        # Cross-chunk dedup: count newly-seen vs already-seen triples and
        # emit a progress event when any sighting was a duplicate. The
        # passage edge for *this* chunk is still added downstream by
        # KGStore.upsert_triples so mirror correctness is preserved.
        n_new, n_seen = self._record_canonical_triples(triples)
        if n_seen > 0:
            self._emit(
                "kg_dedup_hit",
                {
                    "chunk_id": chunk.chunk_id,
                    "hashed_skipped_llm": False,
                    "triples_reused": n_seen,
                    "canonical_new": n_new,
                    "canonical_seen": n_seen,
                },
            )
        return triples

    def extract_batch(self, chunks: Sequence[Chunk]) -> list[Triple]:
        """Extract triples from multiple *chunks* concurrently.

        One bad chunk never kills the batch — per-chunk exceptions are caught,
        warned about, and skipped.  Results are returned in the same order as
        the input chunks.

        Per-reason drop counts (``empty_head``, ``empty_relation``,
        ``empty_tail``, ``non_alnum``, ``too_long``) are accumulated across
        all chunks in this batch and emitted as a single summary warning when
        non-zero, rather than one warning per dropped triple.
        """
        if not chunks:
            return []

        # Shared across worker threads. Mutation here is fine: each worker only
        # increments its own keys via ``+= 1`` and the GIL serialises dict
        # writes; the final summary is read after threads have joined.
        drop_counts: dict[str, int] = {}

        all_triples: list[Triple] = []
        chunks_list = list(chunks)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit in order; zip preserves correspondence.
            futures = [
                (executor.submit(self.extract_one, chunk, drop_counts), chunk)
                for chunk in chunks_list
            ]
            for fut, chunk in futures:
                try:
                    all_triples.extend(fut.result())
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"TripleExtractor: unhandled exception for chunk "
                        f"{chunk.chunk_id!r}: {exc}",
                        stacklevel=2,
                    )

        if drop_counts:
            total = sum(drop_counts.values())
            details = ", ".join(
                f"{reason}={count}" for reason, count in sorted(drop_counts.items())
            )
            warnings.warn(
                f"TripleExtractor: dropped {total} invalid triples in this batch ({details}).",
                stacklevel=2,
            )

        return all_triples
