"""TaxonomyBuilder: propose + materialise a hierarchical document taxonomy.

The builder runs in three phases:

1. **Per-doc summaries + centroids** — for every document belonging to a user,
   compute a one-line LLM summary (or title-truncation, if
   ``cfg.use_llm_doc_summaries`` is false) and a mean-of-chunk-embeddings
   centroid. Both are cached on ``kg_taxonomy_doc_meta`` via the
   :class:`TaxonomyStore` so subsequent rebuilds are nearly free.
2. **LLM tree proposal** — sample up to ``cfg.propose_sample_size`` docs and
   ask the LLM to propose a thematic taxonomy as JSON. Parse + retry once on
   malformed JSON.
3. **Materialise + assign overflow** — wipe the user's existing tree, replay
   the proposed JSON via ``store.add_node`` / ``store.assign_doc``. For docs
   not in the LLM sample, file them in via :class:`DocAssigner`.

Progress is surfaced via a Rich progress bar (per-item ticks) AND an optional
``progress(stage, payload)`` callback for programmatic listeners. Long phases
(summarisation, overflow assignment) run on a ``ThreadPoolExecutor``; the
parallelism style mirrors ``hrag.kg.builder.TripleExtractor``.
"""

from __future__ import annotations

import json
import logging
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hrag.config import TaxonomyConfig
    from hrag.db.connection import Database
    from hrag.providers.embeddings import EmbeddingProvider
    from hrag.providers.llm import LLMProvider
    from hrag.taxonomy.store import TaxonomyStore


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DOC_SUMMARY_PATH = _PROMPTS_DIR / "taxonomy_doc_summary.md"
_PROPOSE_PATH = _PROMPTS_DIR / "taxonomy_propose.md"


def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON cleaning helpers (mirrors hrag.kg.builder)
# ---------------------------------------------------------------------------

_FENCE_PREFIXES = ("```json", "```")


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    for prefix in _FENCE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def _extract_json_object(raw: str) -> Optional[dict]:
    """Locate the outermost ``{...}`` in *raw* and parse it.

    Returns ``None`` on any failure so the caller can retry or warn.
    """
    text = _strip_fences(raw)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        result = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def _salvage_truncated_json(raw: str) -> Optional[dict]:
    """Best-effort recovery from a truncated JSON response.

    Walks the string char-by-char tracking brace/bracket depth and the longest
    valid prefix that parses. Then closes any open arrays/objects synthetically
    and retries. Yields useful trees out of LLM responses that ran past
    ``max_tokens``.
    """
    text = _strip_fences(raw)
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    # Walk and remember the last position where braces+brackets are balanced.
    depth_obj = 0
    depth_arr = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1

    # Synthesize the missing closers.
    closers = "]" * max(0, depth_arr) + "}" * max(0, depth_obj)
    if not closers:
        return None
    # Strip any dangling partial-token tail (after the last comma is safer than
    # parsing a half-typed key).
    body = text
    last_safe = max(body.rfind(","), body.rfind("}"), body.rfind("]"))
    if last_safe > 0:
        body = body[: last_safe + 1].rstrip(",")
    candidate = body + closers
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


_QUOTE_CHARS = "\"'`“”‘’"


def _clean_summary(text: str, max_chars: int) -> str:
    """Strip whitespace + surrounding quotes and clamp length."""
    s = (text or "").strip()
    # Collapse multi-line output into the first non-empty line.
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break
    # Strip a single layer of matching quotes if present.
    if len(s) >= 2 and s[0] in _QUOTE_CHARS and s[-1] in _QUOTE_CHARS:
        s = s[1:-1].strip()
    if max_chars and len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s


def _mean_normalize(vectors: list[list[float]]) -> Optional[list[float]]:
    """Component-wise mean of *vectors*, then L2-normalise. None if empty."""
    if not vectors:
        return None
    dim = len(vectors[0])
    sums = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            continue
        for i, x in enumerate(v):
            sums[i] += float(x)
    n = float(len(vectors))
    mean = [s / n for s in sums]
    norm = sum(x * x for x in mean) ** 0.5
    if norm <= 0.0:
        return mean
    return [x / norm for x in mean]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TaxonomyBuilder:
    """Build (or rebuild) a hierarchical document taxonomy for a user.

    Stage diagram:

        documents -> [centroid+summary cache] -> sample -> propose JSON
                                                |
                                                +-> overflow: DocAssigner
                                                |
                                                +-> materialise nodes + assign
    """

    name = "taxonomy_builder"

    def __init__(
        self,
        db: "Database",
        llm: "LLMProvider",
        embedder: "EmbeddingProvider",
        store: "TaxonomyStore",
        cfg: "TaxonomyConfig",
    ) -> None:
        self._db = db
        self._llm = llm
        self._embedder = embedder
        self._store = store
        self._cfg = cfg
        self._summary_template = _load(_DOC_SUMMARY_PATH)
        self._propose_template = _load(_PROPOSE_PATH)
        self._max_workers = int(getattr(cfg, "parallel_workers", 8) or 8)

    # ------------------------------------------------------------------
    # Public: full build
    # ------------------------------------------------------------------

    def build_for_user(
        self,
        user_id: str,
        *,
        force: bool = False,
        progress: Optional[Callable[[str, dict], None]] = None,
    ) -> dict:
        """End-to-end build/rebuild of the taxonomy tree for ``user_id``."""
        t0 = time.monotonic()

        def _p(stage: str, **payload: object) -> None:
            """Fire progress callback, swallowing any caller exception."""
            if progress is not None:
                try:
                    progress(stage, payload)
                except Exception:  # noqa: BLE001
                    pass

        docs = self._list_docs(user_id)
        if not docs:
            logger.info("TaxonomyBuilder: user %s has no documents", user_id)
            return {
                "docs_processed": 0,
                "nodes_created": 0,
                "leaves": 0,
                "duration_s": round(time.monotonic() - t0, 3),
            }

        _p("start", user_id=user_id, n_docs=len(docs))

        # ------------------------------------------------------------------
        # Phase 1: ensure each doc has a cached summary + centroid.
        # ------------------------------------------------------------------
        t_sum = time.monotonic()
        doc_meta = self._materialize_doc_meta(
            user_id, docs, force=force, progress=progress
        )
        _p("summaries_done",
           n_summarized=len(doc_meta),
           duration_s=round(time.monotonic() - t_sum, 3))

        # ------------------------------------------------------------------
        # Phase 2: sample + propose tree.
        # ------------------------------------------------------------------
        sample_size = max(1, int(self._cfg.propose_sample_size))
        doc_ids = list(doc_meta.keys())
        if len(doc_ids) > sample_size:
            sampled_ids = random.sample(doc_ids, sample_size)
        else:
            sampled_ids = list(doc_ids)
        sampled_set = set(sampled_ids)

        _p("propose_tree_start",
           sample_size=len(sampled_ids),
           max_children=int(self._cfg.max_children_per_node))

        t_prop = time.monotonic()
        sampled_pairs = [
            (did, doc_meta[did]["title"], doc_meta[did]["summary"])
            for did in sampled_ids
        ]
        tree_json = self._propose_tree(sampled_pairs)
        if tree_json is None:
            warnings.warn(
                "TaxonomyBuilder: LLM tree proposal failed after retry; "
                "no taxonomy was built.",
                stacklevel=2,
            )
            return {
                "docs_processed": len(doc_meta),
                "nodes_created": 0,
                "leaves": 0,
                "duration_s": round(time.monotonic() - t0, 3),
            }

        _p("propose_tree_done",
           n_nodes=len(sampled_ids),
           duration_s=round(time.monotonic() - t_prop, 3))

        # ------------------------------------------------------------------
        # Phase 3: route overflow docs (those NOT in the sample) via DocAssigner.
        #
        # We need the materialised tree first, so we route AFTER step 4 below.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Phase 4: wipe existing tree, materialise proposed JSON.
        # ------------------------------------------------------------------
        self._store.clear(user_id)

        _p("materialize_start", n_nodes=len(sampled_ids))
        t_mat = time.monotonic()
        materialize_stats = self._materialize_tree(user_id, tree_json, doc_meta)
        _p("materialize_done",
           n_nodes=materialize_stats["nodes_created"],
           duration_s=round(time.monotonic() - t_mat, 3))

        # Overflow assignment (only if we sampled).
        overflow_ids = [d for d in doc_ids if d not in sampled_set]
        n_overflow = len(overflow_ids)
        _p("assign_docs_start", n_docs=n_overflow)
        t_assign = time.monotonic()
        if overflow_ids:
            self._assign_overflow(user_id, overflow_ids, doc_meta, progress=progress)
        _p("assign_docs_done",
           n_assigned=n_overflow,
           duration_s=round(time.monotonic() - t_assign, 3))

        # ------------------------------------------------------------------
        # Phase 5: recompute centroids for all internal nodes.
        # ------------------------------------------------------------------
        try:
            self._store.recompute_all_centroids(user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("recompute_all_centroids failed: %s", exc)

        duration = round(time.monotonic() - t0, 3)
        _p("done",
           total_duration_s=duration,
           n_nodes=materialize_stats["nodes_created"],
           n_docs_assigned=n_overflow)

        return {
            "docs_processed": len(doc_meta),
            "nodes_created": materialize_stats["nodes_created"],
            "leaves": materialize_stats["leaves"],
            "duration_s": duration,
        }

    # ------------------------------------------------------------------
    # Public: per-doc helpers (also used by DocAssigner)
    # ------------------------------------------------------------------

    def build_doc_centroid(
        self,
        user_id: str,
        doc_id: str,
        *,
        summary: Optional[str] = None,
    ) -> Optional[list[float]]:
        """Embedding of a short "topic vector" for the doc.

        Uses ``title || summary`` as the topic text and produces ONE embedding
        — orders of magnitude faster than averaging every chunk's embedding,
        with effectively the same downstream cosine signal at tree-navigation
        scale. Falls back to title+first-chunk excerpt when ``summary`` is None.
        """
        if summary:
            title, _ = self._doc_title_and_excerpt(user_id, doc_id, max_chars=0)
            topic = f"{title or ''}\n{summary}".strip()
            if not topic:
                return None
            return self._embedder.embed_one(topic)

        # Fallback: build a small topic vector from title + the first 1KB of
        # the doc. Still one embed call, not per-chunk.
        title, excerpt = self._doc_title_and_excerpt(user_id, doc_id, max_chars=1000)
        topic = f"{title or ''}\n{excerpt}".strip()
        if not topic:
            return None
        return self._embedder.embed_one(topic)

    def summarize_doc(self, user_id: str, doc_id: str) -> Optional[str]:
        """LLM-summarise the doc to one line, or fall back to title."""
        title, excerpt = self._doc_title_and_excerpt(user_id, doc_id)
        if title is None and not excerpt:
            return None

        max_chars = max(1, int(self._cfg.summary_max_chars))

        if not self._cfg.use_llm_doc_summaries:
            return _clean_summary(title or "", max_chars)

        prompt = self._summary_template.format(
            title=title or "(untitled)",
            excerpt=excerpt or "(no text available)",
        )
        try:
            raw = self._llm.complete(prompt, temperature=0.0, max_tokens=150)
        except Exception as exc:  # noqa: BLE001
            logger.warning("summarize_doc(%s) LLM error: %s", doc_id, exc)
            return _clean_summary(title or "", max_chars)
        return _clean_summary(raw, max_chars)

    # ------------------------------------------------------------------
    # Internal: doc fetching
    # ------------------------------------------------------------------

    def _list_docs(self, user_id: str) -> list[dict]:
        """List a user's documents, including episodic memories when the
        config flag ``include_episodic`` is set."""
        if getattr(self._cfg, "include_episodic", True):
            rows = self._db.execute(
                "SELECT doc_id, title, source_path, source_type FROM documents "
                "WHERE user_id = ? ORDER BY ingested_at ASC",
                (user_id,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT doc_id, title, source_path, source_type FROM documents "
                "WHERE user_id = ? AND source_type = 'document' "
                "ORDER BY ingested_at ASC",
                (user_id,),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            out.append(
                {
                    "doc_id": r["doc_id"],
                    "title": (r["title"] or "") or Path(r["source_path"] or "").stem,
                    "source_path": r["source_path"] or "",
                }
            )
        return out

    def _doc_title_and_excerpt(
        self, user_id: str, doc_id: str, *, max_chars: int = 2000
    ) -> tuple[Optional[str], str]:
        """Pull title + first ~2000 chars (concat of first chunks by index)."""
        title_row = self._db.execute(
            "SELECT title, source_path FROM documents WHERE doc_id = ? AND user_id = ?",
            (doc_id, user_id),
        ).fetchone()
        if title_row is None:
            return None, ""
        title = title_row["title"] or Path(title_row["source_path"] or "").stem or None

        chunk_rows = self._db.execute(
            "SELECT text FROM chunks "
            "WHERE doc_id = ? AND user_id = ? AND excluded = 0 "
            "ORDER BY chunk_index ASC LIMIT 8",
            (doc_id, user_id),
        ).fetchall()
        parts: list[str] = []
        total = 0
        for cr in chunk_rows:
            t = cr["text"] or ""
            if not t:
                continue
            parts.append(t)
            total += len(t)
            if total >= max_chars:
                break
        excerpt = "\n\n".join(parts)[:max_chars]
        return title, excerpt

    # ------------------------------------------------------------------
    # Internal: phase 1 - summaries + centroids in parallel
    # ------------------------------------------------------------------

    def _materialize_doc_meta(
        self,
        user_id: str,
        docs: list[dict],
        *,
        force: bool,
        progress: Optional[Callable[[str, dict], None]],
    ) -> dict[str, dict]:
        """Return ``{doc_id: {title, summary, centroid}}``, computing + caching
        missing fields concurrently."""
        # Pre-load existing cache rows.
        cache: dict[str, dict] = {}
        if not force:
            for d in docs:
                try:
                    meta = self._store.get_doc_meta(user_id, d["doc_id"])
                except Exception:  # noqa: BLE001 - store may not have row
                    meta = None
                if meta is None:
                    continue
                summary = meta.get("summary") if isinstance(meta, dict) else getattr(meta, "summary", None)
                centroid = meta.get("centroid") if isinstance(meta, dict) else getattr(meta, "centroid", None)
                if summary and centroid:
                    cache[d["doc_id"]] = {"summary": summary, "centroid": list(centroid)}

        # Lazy-import Rich for the progress bar.
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        out: dict[str, dict] = {}
        to_compute: list[dict] = []
        for d in docs:
            if d["doc_id"] in cache:
                out[d["doc_id"]] = {
                    "title": d["title"],
                    "summary": cache[d["doc_id"]]["summary"],
                    "centroid": cache[d["doc_id"]]["centroid"],
                }
            else:
                to_compute.append(d)

        total = len(docs)
        completed = len(out)

        if not to_compute:
            return out

        console = Console()
        bar = Progress(
            TextColumn("[bold]taxonomy[/bold] summarise+centroid"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("• ETA"),
            TimeRemainingColumn(),
            console=console,
        )

        # Three-phase approach:
        #  Phase A — pull title + excerpt for every doc (serial, fast SQLite).
        #  Phase B — LLM summaries in parallel (network-bound).
        #  Phase C — batch-embed all "title + summary" topic strings in ONE
        #            call; ~22 vectors per call is much faster than averaging
        #            chunk embeddings per doc.
        with bar:
            task = bar.add_task("docs", total=total, completed=completed)
            done_count = completed

            # Phase A — serial DB reads (fast, no model load on hot path).
            doc_inputs: dict[str, dict] = {}
            for d in to_compute:
                did = d["doc_id"]
                try:
                    title, excerpt = self._doc_title_and_excerpt(user_id, did)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("title/excerpt failed for %s: %s", did, exc)
                    title, excerpt = d["title"], ""
                doc_inputs[did] = {
                    "title": title or d["title"],
                    "excerpt": excerpt,
                }

            # Phase B — parallel LLM summary (pure-network, no shared mutable state).
            def _summarize_only(did: str, title: str, excerpt: str) -> tuple[str, Optional[str]]:
                if not self._cfg.use_llm_doc_summaries:
                    return did, _clean_summary(title or "", self._cfg.summary_max_chars)
                if not (title or excerpt):
                    return did, None
                try:
                    prompt = self._summary_template.format(
                        title=title or "(untitled)",
                        excerpt=excerpt or "(no text available)",
                    )
                    raw = self._llm.complete(prompt, temperature=0.0, max_tokens=512)
                    return did, _clean_summary(raw, self._cfg.summary_max_chars)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("summary failed for %s: %s", did, exc)
                    return did, None

            def _p_inner(stage: str, **payload: object) -> None:
                """Fire progress callback from within _materialize_doc_meta."""
                if progress is not None:
                    try:
                        progress(stage, payload)
                    except Exception:  # noqa: BLE001
                        pass

            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {
                    pool.submit(_summarize_only, did, di["title"], di["excerpt"]): did
                    for did, di in doc_inputs.items()
                }
                for fut in as_completed(futures):
                    did = futures[fut]
                    try:
                        _, summary = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("summarize future failed for %s: %s", did, exc)
                        summary = None
                    di = doc_inputs[did]
                    final_summary = (
                        summary
                        or di["title"][: self._cfg.summary_max_chars]
                        or did
                    )
                    di["summary"] = final_summary
                    done_count += 1
                    bar.advance(task)
                    _p_inner(
                        "doc_summary",
                        i=done_count,
                        n=total,
                        doc_id=did,
                        title=di.get("title") or did,
                    )

            # Phase C — batch-embed all (title + summary) topic strings in ONE
            # embedder call. Sentence-transformers handles its own batching.
            dids = list(doc_inputs.keys())
            topic_strings: list[str] = []
            for did in dids:
                di = doc_inputs[did]
                topic = f"{di.get('title') or ''}\n{di.get('summary') or ''}".strip()
                topic_strings.append(topic or did)
            _p_inner("embed_centroids_start", n_inputs=len(topic_strings))
            t_embed = time.monotonic()
            try:
                topic_vecs = self._embedder.embed(topic_strings)
            except Exception as exc:  # noqa: BLE001
                logger.warning("batch centroid embed failed: %s", exc)
                topic_vecs = [[0.0] * self._embedder.dim for _ in topic_strings]
            _p_inner("embed_centroids_done",
                     duration_s=round(time.monotonic() - t_embed, 3))

            for did, centroid in zip(dids, topic_vecs):
                di = doc_inputs[did]
                final_summary = di.get("summary") or di["title"][: self._cfg.summary_max_chars] or did
                out[did] = {
                    "title": di["title"],
                    "summary": final_summary,
                    "centroid": list(centroid),
                }
                try:
                    self._store.upsert_doc_meta(
                        user_id, did, final_summary, list(centroid),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("upsert_doc_meta(%s) failed: %s", did, exc)

        return out

    # ------------------------------------------------------------------
    # Internal: phase 2 - propose tree from sample
    # ------------------------------------------------------------------

    def _propose_tree(
        self, sampled_pairs: list[tuple[str, str, str]]
    ) -> Optional[dict]:
        """Call the propose prompt, retry once on parse failure."""
        if not sampled_pairs:
            return None
        doc_list = "\n".join(
            f"{did}\t{(title or '').replace(chr(9), ' ')}\t{(summary or '').replace(chr(9), ' ')}"
            for did, title, summary in sampled_pairs
        )
        prompt = self._propose_template.format(
            max_children=int(self._cfg.max_children_per_node),
            max_depth=int(self._cfg.max_depth),
            doc_list=doc_list,
        )

        for attempt in range(2):
            try:
                # Ollama treats max_tokens=None as a small default that
                # silently truncates long JSON. 8192 covers a ~150-doc tree.
                raw = self._llm.complete(
                    prompt, temperature=0.0, max_tokens=8192,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("propose LLM error (attempt %d): %s", attempt + 1, exc)
                continue
            parsed = _extract_json_object(raw)
            if parsed is not None and "tree" in parsed:
                return parsed
            # Salvage attempt: trim to the last balanced brace.
            salvaged = _salvage_truncated_json(raw)
            if salvaged is not None and "tree" in salvaged:
                logger.warning(
                    "propose JSON was truncated at attempt %d; salvaged partial tree.",
                    attempt + 1,
                )
                return salvaged
            warnings.warn(
                f"TaxonomyBuilder: propose JSON parse failed (attempt {attempt + 1}). "
                f"Raw (first 200 chars): {raw[:200]!r}",
                stacklevel=2,
            )
            prompt = (
                prompt
                + "\n\nYour previous response was invalid JSON or truncated. "
                "Return ONLY a valid, COMPACT JSON object that matches the "
                "schema. Keep descriptions short (one short sentence) and "
                "limit each leaf's doc_ids to the most relevant doc per leaf. "
                "Do not include markdown fences."
            )
        return None

    # ------------------------------------------------------------------
    # Internal: phase 3 - overflow assignment
    # ------------------------------------------------------------------

    def _assign_overflow(
        self,
        user_id: str,
        overflow_ids: list[str],
        doc_meta: dict[str, dict],
        *,
        progress: Optional[Callable[[str, dict], None]],
    ) -> None:
        """Route each overflow doc through DocAssigner (parallel)."""
        # Local import to avoid a cycle; DocAssigner lives next to us.
        from hrag.taxonomy.assigner import DocAssigner

        assigner = DocAssigner(
            db=self._db,
            llm=self._llm,
            embedder=self._embedder,
            store=self._store,
            cfg=self._cfg,
        )

        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        console = Console()
        bar = Progress(
            TextColumn("[bold]taxonomy[/bold] route overflow"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("• ETA"),
            TimeRemainingColumn(),
            console=console,
        )

        def _work(doc_id: str) -> str:
            meta = doc_meta.get(doc_id, {})
            try:
                assigner.assign(
                    user_id,
                    doc_id,
                    summary=meta.get("summary"),
                    centroid=meta.get("centroid"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("overflow assign(%s) failed: %s", doc_id, exc)
            return doc_id

        with bar:
            task = bar.add_task("overflow", total=len(overflow_ids))
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {pool.submit(_work, did): did for did in overflow_ids}
                done = 0
                for fut in as_completed(futures):
                    try:
                        did = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("overflow worker error: %s", exc)
                        did = futures[fut]
                    done += 1
                    bar.advance(task)
                    if progress is not None:
                        try:
                            progress(
                                "overflow_assign",
                                {"i": done, "n": len(overflow_ids), "doc_id": did},
                            )
                        except Exception:  # noqa: BLE001
                            pass

    # ------------------------------------------------------------------
    # Internal: phase 4 - materialise JSON tree into the store
    # ------------------------------------------------------------------

    def _materialize_tree(
        self,
        user_id: str,
        tree_json: dict,
        doc_meta: dict[str, dict],
    ) -> dict[str, int]:
        """Walk ``tree_json`` top-down, calling store.add_node / assign_doc.

        Returns ``{"nodes_created": N, "leaves": K}``.
        """
        root_node = self._store.ensure_root(user_id)
        root_id = root_node.node_id
        stats = {"nodes_created": 0, "leaves": 0}

        tree = tree_json.get("tree") if isinstance(tree_json, dict) else None
        if not isinstance(tree, dict):
            return stats
        # The top-level "root" node already exists via ensure_root; descend
        # into its children directly.
        children = tree.get("children") or []
        for child in children:
            if isinstance(child, dict):
                self._walk_node(user_id, root_id, child, doc_meta, stats, depth=1)
        return stats

    def _walk_node(
        self,
        user_id: str,
        parent_id: str,
        node_json: dict,
        doc_meta: dict[str, dict],
        stats: dict[str, int],
        depth: int,
    ) -> None:
        label = str(node_json.get("label") or "Unnamed").strip() or "Unnamed"
        description = str(node_json.get("description") or "").strip()
        children = node_json.get("children")
        doc_ids = node_json.get("doc_ids")

        # Phase 12 — per-node keywords. Prefer the LLM-proposed list (free,
        # emitted by the keyword-aware propose prompt); fall back to a local
        # extraction over the node's signal text (label + description + member
        # doc summaries for a leaf) so no node is left un-keyworded.
        kw_raw = node_json.get("keywords")
        node_keywords: list[str] = []
        if isinstance(kw_raw, list):
            node_keywords = [str(k).strip() for k in kw_raw if str(k).strip()]
        if not node_keywords:
            node_keywords = self._local_keywords(label, description, doc_ids, doc_meta)

        # Disambiguate leaf vs internal. The schema requires exactly one of the
        # two but we defend against minor LLM drift.
        is_leaf = bool(doc_ids) and not children
        if not children and not doc_ids:
            # A leaf with no docs is forbidden — skip.
            return
        # Respect max_depth: anything deeper than max_depth becomes a leaf
        # holding whatever doc_ids we can pull from its subtree.
        max_depth = max(1, int(self._cfg.max_depth))
        if depth >= max_depth:
            is_leaf = True
            # Collect all doc_ids in the subtree (in case there are children).
            collected: list[str] = list(doc_ids or [])
            if children:
                _collect_doc_ids(children, collected)
            doc_ids = collected
            children = None

        new_node = self._store.add_node(
            user_id,
            parent_id,
            label,
            description,
            is_leaf=is_leaf,
            keywords=node_keywords,
        )
        node_id = new_node.node_id
        stats["nodes_created"] += 1

        if is_leaf:
            stats["leaves"] += 1
            for did in (doc_ids or []):
                if not isinstance(did, str):
                    continue
                if did not in doc_meta:
                    # LLM hallucinated a doc id — skip.
                    continue
                try:
                    self._store.assign_doc(
                        user_id,
                        did,
                        node_id,
                        score=1.0,
                        is_primary=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "assign_doc(%s -> %s) failed: %s", did, node_id, exc
                    )
            return

        for child in (children or []):
            if isinstance(child, dict):
                self._walk_node(user_id, node_id, child, doc_meta, stats, depth + 1)


    def _local_keywords(
        self,
        label: str,
        description: str,
        doc_ids,
        doc_meta: dict[str, dict],
    ) -> list[str]:
        """Extract keywords locally (no LLM) from a node's signal text.

        Uses the label + description plus, for a leaf, its member doc summaries.
        Pure/cheap — the fallback when the LLM omitted ``keywords``.
        """
        from hrag.taxonomy.keywords import extract_keywords  # local import

        texts: list[str] = []
        if label:
            texts.append(label)
        if description:
            texts.append(description)
        if isinstance(doc_ids, list):
            for did in doc_ids:
                meta = doc_meta.get(did) if isinstance(did, str) else None
                if meta:
                    summ = meta.get("summary") or meta.get("title")
                    if summ:
                        texts.append(str(summ))
        top_k = int(getattr(self._cfg, "keywords_per_node", 8) or 8)
        return extract_keywords(texts, top_k=top_k)


def _collect_doc_ids(children: list, into: list[str]) -> None:
    """Recursively gather all leaf doc_ids from a subtree of children."""
    for c in children or []:
        if not isinstance(c, dict):
            continue
        dids = c.get("doc_ids")
        if isinstance(dids, list):
            for d in dids:
                if isinstance(d, str):
                    into.append(d)
        sub = c.get("children")
        if isinstance(sub, list):
            _collect_doc_ids(sub, into)
