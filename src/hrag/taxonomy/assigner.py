"""DocAssigner: file a single document into an existing taxonomy tree.

Algorithm (per doc):

1. Compute (or reuse caller-provided) a one-line summary and a centroid for the
   doc. Both are cached on ``kg_taxonomy_doc_meta`` via the
   :class:`TaxonomyStore`.
2. Greedy beam-of-1 descent from the user's root:
   - At each level, score the doc's centroid against every child's centroid
     by cosine similarity.
   - If the top-2 scores are within ``_TIEBREAK_EPSILON`` OR if both top-2
     scores are >= ``_TIEBREAK_BOTH_HIGH``, ask the LLM tiebreak prompt to
     pick one. Otherwise take the argmax greedily.
3. Once we reach a leaf, ``store.assign_doc(..., is_primary=True)``. If
   ``cfg.allow_secondary_assignment`` is True AND a runner-up leaf's score is
   within ``_SECONDARY_GAP`` of the primary, also assign as secondary.

Tree topology is read once at the start via ``store.list_nodes`` and held in
memory for the duration of the call; for ``assign_all`` we refresh per-doc only
if the caller hints that the tree is being mutated mid-run (default: don't —
the topology is stable).
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hrag.config import TaxonomyConfig
    from hrag.db.connection import Database
    from hrag.providers.embeddings import EmbeddingProvider
    from hrag.providers.llm import LLMProvider
    from hrag.taxonomy.store import TaxonomyStore
    from hrag.taxonomy.types import TaxonomyNode


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt + tunables
# ---------------------------------------------------------------------------

_TIEBREAK_PROMPT_PATH = (
    Path(__file__).parent.parent / "prompts" / "taxonomy_route_tiebreak.md"
)

# Top-2 scores within this absolute gap -> LLM tiebreak.
_TIEBREAK_EPSILON = 0.05
# Or: top-2 BOTH >= this threshold -> LLM tiebreak (high confidence either way).
_TIEBREAK_BOTH_HIGH = 0.5
# Runner-up leaf within this gap of the primary -> also file as secondary.
_SECONDARY_GAP = 0.1
# Hard safety bound on descent depth.
_MAX_DESCENT_DEPTH = 32


# ---------------------------------------------------------------------------
# Cosine helper
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Both vectors assumed numeric; returns 0.0 on mismatch."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 0.0:
        return 0.0
    return dot / denom


# ---------------------------------------------------------------------------
# Assigner
# ---------------------------------------------------------------------------


class DocAssigner:
    """File a single doc into the user's existing taxonomy tree."""

    name = "doc_assigner"

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
        self._tiebreak_template = _TIEBREAK_PROMPT_PATH.read_text(encoding="utf-8")
        self._max_workers = int(getattr(cfg, "parallel_workers", 8) or 8)
        # Lazy-loaded tree cache per call (NOT shared across assigns intentionally
        # except inside assign_all, which loads once and reuses).
        self._tree_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public: single-doc assignment
    # ------------------------------------------------------------------

    def assign(
        self,
        user_id: str,
        doc_id: str,
        *,
        summary: Optional[str] = None,
        centroid: Optional[list[float]] = None,
        refresh_centroids: bool = True,
    ) -> Optional[str]:
        """Route the doc to a leaf and persist the assignment.

        Returns the chosen leaf's ``node_id``, or ``None`` if the tree is
        empty or no centroid could be computed.

        When ``refresh_centroids`` is True (default), the chosen leaf's
        centroid (and every ancestor up to root) is recomputed so that
        future ``beam_descend`` queries can route to the new doc. Pass
        False from bulk paths (``assign_all``) that finish with a single
        ``recompute_all_centroids`` to avoid concurrent ancestor-write
        races between worker threads.
        """
        tree = self._get_tree(user_id)
        if not tree["root"] or not tree["children_of"].get(tree["root"]):
            return None  # Empty tree.

        # Lazily compute centroid + summary if not provided.
        if centroid is None or summary is None:
            built_centroid, built_summary = self._build_meta_if_missing(
                user_id, doc_id, summary, centroid
            )
            centroid = centroid if centroid is not None else built_centroid
            summary = summary if summary is not None else built_summary

        if centroid is None:
            logger.info(
                "DocAssigner: %s has no centroid (no chunks?); skipping assign", doc_id
            )
            return None

        # Cache summary + centroid on the doc-meta table.
        try:
            self._store.upsert_doc_meta(user_id, doc_id, summary or "", centroid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("upsert_doc_meta(%s) failed: %s", doc_id, exc)

        # Beam descent.
        primary_leaf, primary_score, runner_up = self._descend(
            tree, centroid, query_text=summary or ""
        )
        if primary_leaf is None:
            return None

        primary_ok = False
        try:
            self._store.assign_doc(
                user_id,
                doc_id,
                primary_leaf.node_id,
                score=float(primary_score),
                is_primary=True,
            )
            primary_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "assign_doc(%s -> %s) failed: %s", doc_id, primary_leaf.node_id, exc
            )

        # Optional secondary assignment for multi-topic docs.
        secondary_leaf_id: Optional[str] = None
        if getattr(self._cfg, "allow_secondary_assignment", False) and runner_up is not None:
            runner_node, runner_score = runner_up
            if runner_score >= primary_score - _SECONDARY_GAP:
                try:
                    self._store.assign_doc(
                        user_id,
                        doc_id,
                        runner_node.node_id,
                        score=float(runner_score),
                        is_primary=False,
                    )
                    secondary_leaf_id = runner_node.node_id
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "secondary assign_doc(%s -> %s) failed: %s",
                        doc_id,
                        runner_node.node_id,
                        exc,
                    )

        # Refresh centroids on the affected leaf-to-root path(s). Without
        # this, the leaf's ``centroid`` column stays NULL after on-ingest
        # auto-assign, so ``TaxonomyStore.beam_descend`` scores the leaf
        # at ``min_score - 1.0`` and prunes its whole branch on every
        # future query — i.e. the doc is silently unreachable via taxonomy
        # retrieval. ``recompute_node_centroid`` also refreshes the stale
        # ``doc_count`` column the GUI shows.
        if refresh_centroids and primary_ok:
            self._refresh_path_to_root(user_id, primary_leaf.node_id)
            if secondary_leaf_id is not None:
                self._refresh_path_to_root(user_id, secondary_leaf_id)

        return primary_leaf.node_id

    # ------------------------------------------------------------------
    # Public: assign every doc for a user
    # ------------------------------------------------------------------

    def assign_all(
        self,
        user_id: str,
        *,
        progress: Optional[Callable[[str, dict], None]] = None,
    ) -> dict:
        """Assign every document for ``user_id``. Returns counts."""

        def _p(stage: str, **payload: object) -> None:
            """Fire progress callback, swallowing any caller exception."""
            if progress is not None:
                try:
                    progress(stage, payload)
                except Exception:  # noqa: BLE001
                    pass

        if getattr(self._cfg, "include_episodic", True):
            rows = self._db.execute(
                "SELECT doc_id FROM documents WHERE user_id = ? "
                "ORDER BY ingested_at ASC",
                (user_id,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT doc_id FROM documents "
                "WHERE user_id = ? AND source_type = 'document' "
                "ORDER BY ingested_at ASC",
                (user_id,),
            ).fetchall()
        doc_ids = [r["doc_id"] for r in rows]
        if not doc_ids:
            return {"assigned": 0, "skipped": 0}

        _p("start", n_docs=len(doc_ids))

        # Load tree topology once.
        self._tree_cache = None  # force reload
        tree = self._get_tree(user_id)
        if not tree["root"] or not tree["children_of"].get(tree["root"]):
            return {"assigned": 0, "skipped": len(doc_ids)}

        # Rich progress bar.
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
            TextColumn("[bold]taxonomy[/bold] assign"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("• ETA"),
            TimeRemainingColumn(),
            console=console,
        )

        assigned = 0
        skipped = 0
        t0 = time.monotonic()

        def _work(doc_id: str) -> tuple[str, Optional[str], float]:
            t_w = time.monotonic()
            try:
                # Skip per-doc centroid refresh — concurrent worker threads
                # can race on shared ancestors. We pay one bulk recompute
                # after the pool drains (below).
                node_id = self.assign(user_id, doc_id, refresh_centroids=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("assign(%s) failed: %s", doc_id, exc)
                return doc_id, None, 0.0
            score = round(time.monotonic() - t_w, 4)
            return doc_id, node_id, score

        with bar:
            task = bar.add_task("docs", total=len(doc_ids))
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {pool.submit(_work, did): did for did in doc_ids}
                done = 0
                for fut in as_completed(futures):
                    try:
                        did, node_id, score = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("assign_all worker error: %s", exc)
                        did = futures[fut]
                        node_id = None
                        score = 0.0
                    if node_id is None:
                        skipped += 1
                    else:
                        assigned += 1
                    done += 1
                    bar.advance(task)
                    _p(
                        "assign",
                        i=done,
                        n=len(doc_ids),
                        doc_id=did,
                        node_id=node_id,
                        score=score,
                    )

        # Single bulk centroid refresh after every worker is done. The
        # per-doc path is gated off in `_work` to avoid concurrent ancestor
        # writes; without this the leaves remain centroid-less and queries
        # can't route to anything just assigned.
        if assigned > 0:
            try:
                self._store.recompute_all_centroids(user_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("recompute_all_centroids after assign_all failed: %s", exc)

        _p("done",
           n_assigned=assigned,
           duration_s=round(time.monotonic() - t0, 3))

        return {"assigned": assigned, "skipped": skipped}

    # ------------------------------------------------------------------
    # Internal: tree topology cache
    # ------------------------------------------------------------------

    def _get_tree(self, user_id: str) -> dict:
        """Load + cache the user's tree topology.

        Returns ``{"root": node_id, "by_id": {nid: TaxonomyNode},
        "children_of": {nid: [TaxonomyNode...]}}``.
        """
        if self._tree_cache is not None and self._tree_cache.get("user_id") == user_id:
            return self._tree_cache
        nodes: list[TaxonomyNode] = list(self._store.list_nodes(user_id))
        by_id: dict[str, TaxonomyNode] = {n.node_id: n for n in nodes}
        children_of: dict[str, list[TaxonomyNode]] = {}
        root: Optional[str] = None
        for n in nodes:
            if n.parent_id is None:
                root = n.node_id
            else:
                children_of.setdefault(n.parent_id, []).append(n)
        self._tree_cache = {
            "user_id": user_id,
            "root": root,
            "by_id": by_id,
            "children_of": children_of,
        }
        return self._tree_cache

    # ------------------------------------------------------------------
    # Internal: beam descent
    # ------------------------------------------------------------------

    def _descend(
        self,
        tree: dict,
        centroid: list[float],
        query_text: str,
    ) -> tuple[Optional["TaxonomyNode"], float, Optional[tuple["TaxonomyNode", float]]]:
        """Beam-of-1 descent. Returns (leaf, score, runner_up_leaf_or_None)."""
        current = tree["by_id"].get(tree["root"]) if tree["root"] else None
        if current is None:
            return None, 0.0, None

        # Track the best runner-up *leaf* seen at any sibling-level branch.
        best_runner_up: Optional[tuple["TaxonomyNode", float]] = None
        current_score = 1.0  # root cosine to itself is trivially 1; we ignore.

        for _depth in range(_MAX_DESCENT_DEPTH):
            children = tree["children_of"].get(current.node_id, [])
            if not children:
                # We are at a leaf.
                return current, current_score, best_runner_up

            scored = self._score_children(children, centroid)
            if not scored:
                # No children scoreable (all centroids missing). Fall back to
                # the first child to keep moving — better than aborting.
                current = children[0]
                current_score = 0.0
                continue

            # Pick top-2.
            scored.sort(key=lambda t: t[1], reverse=True)
            top_node, top_score = scored[0]
            second: Optional[tuple["TaxonomyNode", float]] = (
                scored[1] if len(scored) > 1 else None
            )

            need_tiebreak = False
            if second is not None:
                gap = top_score - second[1]
                if gap < _TIEBREAK_EPSILON:
                    need_tiebreak = True
                elif top_score >= _TIEBREAK_BOTH_HIGH and second[1] >= _TIEBREAK_BOTH_HIGH:
                    need_tiebreak = True

            if need_tiebreak and second is not None:
                # Phase 12 — try the cheap keyword signal first. When the doc's
                # keywords clearly favour one candidate, skip the LLM tiebreak
                # entirely (speed + cost win, esp. on the gemma-only/8 GB setup).
                kw_pick = self._keyword_tiebreak(query_text, top_node, second[0])
                if kw_pick is not None:
                    pick = kw_pick
                else:
                    pick = self._llm_tiebreak(query_text, [top_node, second[0]])
                if pick == 1:
                    chosen, chosen_score = second
                    runner = (top_node, top_score)
                else:
                    chosen, chosen_score = top_node, top_score
                    runner = second
            else:
                chosen, chosen_score = top_node, top_score
                runner = second

            # Track best leaf runner-up so multi-topic docs can find a 2nd
            # home if allow_secondary_assignment is on. We pick a runner-up
            # only if it's already a leaf — internal nodes can't be assigned to.
            if runner is not None and runner[0].is_leaf:
                if best_runner_up is None or runner[1] > best_runner_up[1]:
                    best_runner_up = runner

            current = chosen
            current_score = chosen_score

        # Safety: exceeded max depth. Treat current as a leaf.
        return current, current_score, best_runner_up

    def _score_children(
        self,
        children: list["TaxonomyNode"],
        centroid: list[float],
    ) -> list[tuple["TaxonomyNode", float]]:
        """Cosine-score each child whose centroid is populated."""
        out: list[tuple["TaxonomyNode", float]] = []
        for c in children:
            if c.centroid is None:
                # Untrained internal node — treat as 0 so other branches win.
                out.append((c, 0.0))
                continue
            out.append((c, _cosine(centroid, list(c.centroid))))
        return out

    def _keyword_tiebreak(
        self,
        query_text: str,
        top_node: "TaxonomyNode",
        second_node: "TaxonomyNode",
    ) -> Optional[int]:
        """Resolve a tiebreak via keyword overlap, or None if inconclusive.

        Returns 0 (keep top), 1 (prefer second), or None when the keyword
        signal can't separate them (caller then falls back to the LLM). Pure +
        cheap. No-op unless ``keyword_tiebreak_skip`` and keyword routing are on
        and both candidates carry keywords.
        """
        if not getattr(self._cfg, "keyword_tiebreak_skip", False):
            return None
        if not getattr(self._cfg, "keyword_routing_enabled", False):
            return None
        kw_top = list(getattr(top_node, "keywords", []) or [])
        kw_second = list(getattr(second_node, "keywords", []) or [])
        if not kw_top and not kw_second:
            return None
        from hrag.taxonomy.keywords import keyword_overlap, tokenize  # local import

        q_kw = tokenize(query_text)
        if not q_kw:
            return None
        ov_top = keyword_overlap(q_kw, kw_top)
        ov_second = keyword_overlap(q_kw, kw_second)
        # Require a clear margin so we only short-circuit the LLM when the
        # keyword signal is genuinely decisive.
        margin = 0.15
        if abs(ov_top - ov_second) < margin:
            return None
        return 0 if ov_top >= ov_second else 1

    def _llm_tiebreak(
        self,
        query_text: str,
        candidates: list["TaxonomyNode"],
    ) -> int:
        """Ask the LLM to pick a 0-based index among ``candidates``.

        Returns 0 on parse failure / LLM error so the greedy winner is kept.
        """
        if not candidates:
            return 0
        numbered = "\n".join(
            f"{i}. {(c.label or 'unnamed')} — {(c.description or '').strip()}"
            for i, c in enumerate(candidates)
        )
        prompt = self._tiebreak_template.format(
            query=(query_text or "").strip() or "(no query text)",
            candidates=numbered,
        )
        try:
            raw = self._llm.complete(prompt, temperature=0.0, max_tokens=8)
        except Exception as exc:  # noqa: BLE001
            logger.warning("tiebreak LLM error: %s", exc)
            return 0
        digits = "".join(ch for ch in (raw or "") if ch.isdigit())
        if not digits:
            return 0
        try:
            idx = int(digits[:3])  # first up to 3 digits is plenty
        except ValueError:
            return 0
        if idx < 0 or idx >= len(candidates):
            return 0
        return idx

    # ------------------------------------------------------------------
    # Internal: refresh centroids along the chosen leaf-to-root path
    # ------------------------------------------------------------------

    def _refresh_path_to_root(self, user_id: str, leaf_node_id: str) -> None:
        """Recompute the leaf's centroid, then walk up to root.

        Each call updates `centroid`/`centroid_dim`/`doc_count` on the row
        so beam_descend can score against it. Walks ancestors via
        ``store.get_node`` because the local tree cache stores
        immutable snapshots (its centroids would be stale after the
        first hop).
        """
        try:
            self._store.recompute_node_centroid(user_id, leaf_node_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("recompute_node_centroid(%s) failed: %s", leaf_node_id, exc)
            return

        # Invalidate the cached tree topology — its centroids are now stale.
        self._tree_cache = None

        node = self._store.get_node(leaf_node_id)
        if node is None:
            return
        cursor_parent = node.parent_id
        # Hard bound on tree depth to avoid runaway loops on malformed data.
        for _ in range(_MAX_DESCENT_DEPTH):
            if cursor_parent is None:
                break
            try:
                self._store.recompute_node_centroid(user_id, cursor_parent)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "recompute_node_centroid(%s) failed: %s", cursor_parent, exc
                )
                break
            parent_node = self._store.get_node(cursor_parent)
            if parent_node is None:
                break
            cursor_parent = parent_node.parent_id

    # ------------------------------------------------------------------
    # Internal: build summary + centroid for a doc on demand
    # ------------------------------------------------------------------

    def _build_meta_if_missing(
        self,
        user_id: str,
        doc_id: str,
        summary: Optional[str],
        centroid: Optional[list[float]],
    ) -> tuple[Optional[list[float]], Optional[str]]:
        """Fill in whichever of (centroid, summary) is None.

        Tries the cache first via ``store.get_doc_meta``; only delegates to
        TaxonomyBuilder when truly missing — avoids paying the LLM cost twice.
        """
        cached_summary: Optional[str] = None
        cached_centroid: Optional[list[float]] = None
        try:
            meta = self._store.get_doc_meta(user_id, doc_id)
        except Exception:  # noqa: BLE001
            meta = None
        if meta is not None:
            cached_summary = (
                meta.get("summary") if isinstance(meta, dict) else getattr(meta, "summary", None)
            )
            raw_centroid = (
                meta.get("centroid") if isinstance(meta, dict) else getattr(meta, "centroid", None)
            )
            if raw_centroid:
                cached_centroid = list(raw_centroid)

        if centroid is None:
            centroid = cached_centroid
        if summary is None:
            summary = cached_summary

        if centroid is None or summary is None:
            # Delegate the truly missing bits to a TaxonomyBuilder helper.
            from hrag.taxonomy.builder import TaxonomyBuilder

            helper = TaxonomyBuilder(
                db=self._db,
                llm=self._llm,
                embedder=self._embedder,
                store=self._store,
                cfg=self._cfg,
            )
            if centroid is None:
                try:
                    centroid = helper.build_doc_centroid(user_id, doc_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("centroid build failed for %s: %s", doc_id, exc)
                    centroid = None
            if summary is None:
                try:
                    summary = helper.summarize_doc(user_id, doc_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("summary build failed for %s: %s", doc_id, exc)
                    summary = None

        return centroid, summary
