"""Contextual Retrieval (Anthropic technique) — Phase 9.14.

Before embedding each chunk, prepend an LLM-generated 50-100-token "context
line" to its ``embedding_text``. The context line describes the chunk's
role inside its parent document so the embedder produces vectors that
encode global as well as local meaning. Anthropic reports ~35% reduction
in retrieval failures from this step alone (~67% in combination with
BM25 + reranking).

Pipeline integration: ``IngestPipeline.ingest_document`` calls
``ContextualAugmenter.augment_chunks`` between the quality-filter step
and the embed step, but only when ``cfg.ingest.contextual_retrieval_enabled``
is True. The augmenter mutates ``chunk.embedding_text`` in place; it
NEVER touches ``chunk.text`` (the LLM-visible body) — so downstream
consumers like the KG triple extractor and taxonomy assigner see the
original text unchanged.

Cache: results are persisted to ``ingest_context_cache`` (one row per
``sha256(chunk_text || doc_hash || model_name)``). Identical re-ingests
incur zero LLM calls. Changing the LLM model invalidates transparently.

Concurrency: ``ThreadPoolExecutor`` over chunks. Ollama serialises
internally; OpenAI / Anthropic get true parallelism. Same pattern as
``hrag.kg.builder.TripleExtractor.extract_batch``.

Failure mode: a single chunk that raises during augmentation is logged
and skipped — its ``embedding_text`` stays as the original HeteRAG
prefix. A summary warning fires at end-of-doc if any chunk failed.
"""

from __future__ import annotations

import hashlib
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from hrag.db.connection import Database
    from hrag.providers.llm import LLMProvider
    from hrag.types import Chunk


logger = logging.getLogger(__name__)


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "contextual_chunk.md"


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_hash(text: str) -> str:
    """Short SHA-256 over the parent document body. Used as a cache key
    component so the same chunk text sitting in two different docs gets two
    different cache entries (the LLM's situating sentence depends on the
    surrounding document)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _truncate_document(text: str, max_chars: int) -> str:
    """Cap a document body at ``max_chars`` characters.

    When over-budget, keep the front 70% and the tail 30%, joining with a
    ``[...]`` marker. Front-heavy because abstracts / introductions usually
    carry the most situating signal. Tail-included so concluding sections
    aren't entirely invisible to the augmenter.
    """
    if len(text) <= max_chars:
        return text
    front = int(max_chars * 0.7)
    tail = max_chars - front - 8  # 8 chars for "\n[...]\n"
    if tail < 0:
        tail = 0
    return text[:front] + "\n[...]\n" + (text[-tail:] if tail > 0 else "")


def _render_prompt(template: str, chunk_text: str, document_text: str) -> str:
    """Render the contextual prompt safely.

    Like ``hrag.kg.builder._render_prompt``, we avoid ``.format()`` because
    the prompt template may contain literal ``{`` / ``}`` characters in the
    XML-style ``<chunk>`` / ``<document>`` framing or, in future revisions,
    JSON examples. Targeted string replacement on the two placeholders is
    enough.
    """
    return template.replace("{chunk_text}", chunk_text).replace(
        "{document_text}", document_text
    )


# ---------------------------------------------------------------------------
# Augmenter
# ---------------------------------------------------------------------------


class ContextualAugmenter:
    """Generate per-chunk situating context via the LLM, with a SQLite cache.

    Construction is cheap: prompt template loaded once, no LLM calls.
    ``augment_chunks`` is the only public entry point.
    """

    name = "contextual_augmenter"

    def __init__(
        self,
        llm: "LLMProvider",
        db: "Database | None" = None,
        *,
        max_doc_chars: int = 60_000,
        max_workers: int = 4,
        max_context_tokens: int = 100,
        progress_cb: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self._llm = llm
        self._db = db
        self._max_doc_chars = max_doc_chars
        self._max_workers = max(1, int(max_workers))
        self._max_context_tokens = max_context_tokens
        self._progress_cb = progress_cb
        self._prompt_template = _load_prompt_template()
        # max_tokens for the LLM call. The Anthropic technique targets
        # 50-100 tokens; we budget ~80% headroom (max_tokens=180 by default
        # to be safe against verbose models that emit preamble despite the
        # prompt instruction).
        self._llm_max_tokens = max(max_context_tokens + 80, 180)
        self._model_name: str = (
            getattr(llm, "model_name", None)
            or getattr(getattr(llm, "config", None), "model", None)
            or "unknown"
        )

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def _emit(self, event: str, payload: dict) -> None:
        """Best-effort progress emit; a buggy callback never breaks augment."""
        if self._progress_cb is None:
            return
        try:
            self._progress_cb(event, payload)
        except Exception:  # noqa: BLE001 — progress is best-effort
            pass

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, chunk_text: str, doc_hash: str, model_name: str) -> str:
        """SHA-256 over (chunk_text, doc_hash, model_name).

        Stable across runs. NUL-byte separators prevent boundary collisions
        between fields (e.g. a chunk ending in the model name string).
        """
        h = hashlib.sha256()
        h.update(chunk_text.encode("utf-8"))
        h.update(b"\x00")
        h.update(doc_hash.encode("utf-8"))
        h.update(b"\x00")
        h.update(model_name.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if self._db is None:
            return None
        try:
            row = self._db.execute(
                "SELECT context_line FROM ingest_context_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
        except Exception:
            return None  # cache table missing or transient DB error -> miss
        if row is None:
            return None
        try:
            return row[0] if not hasattr(row, "keys") else row["context_line"]
        except (KeyError, IndexError, TypeError):
            return None

    def _cache_put(self, key: str, context_line: str) -> None:
        if self._db is None:
            return
        try:
            # Note: we intentionally do NOT wrap this in ``with self._db.conn:``.
            # The connection may already have an open implicit transaction from
            # ``IngestPipeline._upsert_document`` (which writes the documents
            # row and relies on the final ``db.commit()`` to flush, with
            # foreign-key checks deferred until commit). A nested
            # ``with conn:`` would commit prematurely between the doc INSERT
            # and the chunks INSERT and, on some sqlite3 builds, observably
            # interact with the parent transaction. The pipeline's terminal
            # ``self.db.commit()`` handles persistence for us; the cache row
            # is captured by the same commit.
            self._db.execute(
                "INSERT OR REPLACE INTO ingest_context_cache"
                "(cache_key, model_name, context_line) VALUES (?, ?, ?)",
                (key, self._model_name, context_line),
            )
        except Exception:
            # Cache write failure must not break augmentation.
            pass

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, chunk_text: str, doc_text: str) -> str:
        prompt = _render_prompt(self._prompt_template, chunk_text, doc_text)
        raw = self._llm.complete(
            prompt,
            temperature=0.0,
            max_tokens=self._llm_max_tokens,
        )
        # Strip whitespace + the most common stray wrappers (quotes, code
        # fences) that LLMs occasionally emit despite the prompt instruction.
        line = (raw or "").strip()
        for fence in ("```", "```text", "```md"):
            if line.startswith(fence):
                line = line[len(fence):].strip()
        if line.endswith("```"):
            line = line[: -3].strip()
        if len(line) >= 2 and line[0] == line[-1] and line[0] in ("'", '"'):
            line = line[1:-1].strip()
        return line

    def _augment_one(
        self,
        chunk: "Chunk",
        doc_hash: str,
        truncated_doc: str,
    ) -> tuple[str, bool, bool]:
        """Return ``(context_line, from_cache, ok)``.

        ``ok=False`` means the LLM call raised — caller skips augmentation
        for this chunk and increments a failure counter.
        """
        key = self._cache_key(chunk.text, doc_hash, self._model_name)
        cached = self._cache_get(key)
        if cached is not None:
            return (cached, True, True)
        try:
            line = self._call_llm(chunk.text, truncated_doc)
        except Exception as exc:  # noqa: BLE001 — never let one chunk kill the doc
            logger.warning(
                "[contextual] LLM call failed for chunk %s: %s",
                chunk.chunk_id,
                exc,
            )
            return ("", False, False)
        if not line:
            # Empty / whitespace-only response: don't cache so a retry
            # might succeed, and don't augment.
            return ("", False, False)
        self._cache_put(key, line)
        return (line, False, True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment_chunks(
        self,
        chunks: list["Chunk"],
        full_document_text: str,
    ) -> list["Chunk"]:
        """Augment every chunk's ``embedding_text`` in place; return the list.

        Each chunk's ``embedding_text`` is rewritten as::

            <context_line>\\n\\n<original_embedding_text>

        ``chunk.text`` (LLM-visible body) is NEVER mutated. Preserves
        Phase-3 contract 1 (KG / taxonomy still consume the original text).

        Cache hits skip the LLM. On per-chunk LLM failure, the chunk's
        ``embedding_text`` is left untouched and a counter is incremented;
        a single summary warning fires at end-of-doc if any failed.
        """
        n = len(chunks)
        if n == 0:
            self._emit(
                "contextual_augment_start",
                {"n_chunks": 0, "doc_title": "", "cached_hits": 0},
            )
            self._emit(
                "contextual_augment_done",
                {"n_chunks": 0, "n_cached": 0, "n_called": 0, "total_s": 0.0},
            )
            return chunks

        doc_hash = _doc_hash(full_document_text)
        truncated_doc = _truncate_document(full_document_text, self._max_doc_chars)
        doc_title = chunks[0].title if chunks else ""

        # Pre-scan cache to surface a "cached hits" count before any LLM
        # calls — useful for the progress bar to show how much work
        # is going to actually happen.
        pre_cached = 0
        cache_lines: dict[str, Optional[str]] = {}
        for chunk in chunks:
            key = self._cache_key(chunk.text, doc_hash, self._model_name)
            line = self._cache_get(key)
            cache_lines[chunk.chunk_id] = line
            if line is not None:
                pre_cached += 1

        self._emit(
            "contextual_augment_start",
            {"n_chunks": n, "doc_title": doc_title, "cached_hits": pre_cached},
        )
        print(
            f"[contextual] start: {n} chunks (cached={pre_cached}, "
            f"to_call={n - pre_cached}) for {doc_title!r}",
            flush=True,
        )

        t0 = time.time()
        n_failures = 0
        n_called = 0
        n_cached = 0

        # Worker fn that operates on a single chunk; reads cache_lines first
        # to avoid recomputing the cache lookup. Cache puts inside _augment_one
        # happen at most once per chunk so the loop is safe to parallelise.
        def _worker(idx: int) -> tuple[int, "Chunk", str, bool, bool]:
            chunk = chunks[idx]
            cached_line = cache_lines.get(chunk.chunk_id)
            if cached_line is not None:
                return (idx, chunk, cached_line, True, True)
            line, from_cache, ok = self._augment_one(chunk, doc_hash, truncated_doc)
            return (idx, chunk, line, from_cache, ok)

        results: list[tuple[int, "Chunk", str, bool, bool]] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(_worker, i) for i in range(n)]
            for completed_i, fut in enumerate(futures):
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    # ThreadPool itself raised (shouldn't happen given the
                    # try/except inside _augment_one, but defend anyway).
                    logger.warning(
                        "[contextual] worker raised for chunk index %d: %s",
                        completed_i,
                        exc,
                    )
                    res = (completed_i, chunks[completed_i], "", False, False)
                results.append(res)
                idx, chunk, line, from_cache, ok = res
                # Per-item visibility (CLAUDE.md visibility rule).
                print(
                    f"[contextual] [{completed_i + 1}/{n}] "
                    f"{'cache' if from_cache else 'llm '} "
                    f"{chunk.chunk_id}{' (failed)' if not ok else ''}",
                    flush=True,
                )
                self._emit(
                    "contextual_augment_chunk",
                    {
                        "i": completed_i + 1,
                        "n": n,
                        # Mirror i/n as the canonical n_done/n_total keys the
                        # web job-state writer reads; without these the UI
                        # progress bar can't advance during this phase.
                        "n_done": completed_i + 1,
                        "n_total": n,
                        "message": f"Augmented {completed_i + 1}/{n} chunks",
                        "chunk_id": chunk.chunk_id,
                        "from_cache": from_cache,
                        "ok": ok,
                    },
                )

        # Apply results in original order. Mutate embedding_text in place.
        results.sort(key=lambda r: r[0])
        for _idx, chunk, line, from_cache, ok in results:
            if not ok or not line:
                n_failures += 1
                continue
            if from_cache:
                n_cached += 1
            else:
                n_called += 1
            # Prepend context line + blank-line separator. Don't touch chunk.text.
            chunk.embedding_text = f"{line}\n\n{chunk.embedding_text}"

        total_s = time.time() - t0

        if n_failures:
            warnings.warn(
                f"ContextualAugmenter: {n_failures}/{n} chunks failed to augment "
                f"for document {doc_title!r}. Their embedding_text was left "
                f"unchanged (still ingestible — search will work, just without "
                f"the contextual situating).",
                stacklevel=2,
            )

        print(
            f"[contextual] done: cached={n_cached} llm_calls={n_called} "
            f"failures={n_failures} in {total_s:.1f}s",
            flush=True,
        )
        self._emit(
            "contextual_augment_done",
            {
                "n_chunks": n,
                "n_cached": n_cached,
                "n_called": n_called,
                "n_failures": n_failures,
                "total_s": total_s,
            },
        )

        return chunks
