"""GraphRAG-style community detection + summarization for HRAG.

Phase 2 layer that runs Leiden clustering on the phrase subgraph at multiple
resolution levels, summarises each cluster with the LLM, and persists summaries
in both ChromaDB (for retrieval) and SQLite (for hydration).

Heavy deps (``leidenalg``, ``igraph``, ``networkx``, ``chromadb``) are
lazy-imported inside the methods that need them so this module stays
importable when those packages are absent.

Filtering thresholds
--------------------
Two thresholds gate which clusters survive into the summarisation stage:

* ``min_phrase_members`` (default ``3``) — a cluster must have at least this
  many phrase nodes.
* ``min_chunk_members`` (default ``2``) — a cluster's phrase nodes must
  collectively reference at least this many distinct supporting chunks.
  Singleton- and 2-member-chunk clusters dominate raw Leiden output and
  account for the vast majority of low-signal communities; filtering them
  here cuts LLM summarisation calls (and Chroma index size) sharply.

``min_chunk_members`` is currently only configurable via the
``CommunityDetector`` constructor; ``KGConfig`` does not yet expose it.
Edit the default below or pass the constructor arg directly to override.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hrag.config import KGConfig
    from hrag.db.connection import Database
    from hrag.kg.store import KGStore
    from hrag.providers.embeddings import EmbeddingProvider
    from hrag.providers.llm import LLMProvider


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "community_summary.md"


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Resolution mapping
# ---------------------------------------------------------------------------

# Coarse -> fine resolution. Higher resolution = more, smaller communities.
_LEVEL_TO_RESOLUTION: dict[int, float] = {
    0: 0.5,
    1: 1.0,
    2: 2.0,
}


# Communities with fewer than this many phrase members are dropped — too small
# to summarise meaningfully. Used as the default for
# ``CommunityDetector(min_phrase_members=...)``.
_MIN_CLUSTER_SIZE = 3

# Communities with fewer supporting chunks than this are dropped: 1- or
# 2-chunk communities are usually noise (a single quirky passage that pulled
# in a handful of phrase nodes). Default for
# ``CommunityDetector(min_chunk_members=...)``.
_MIN_CHUNK_MEMBERS = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Community:
    """A detected community at a given Leiden resolution level."""

    community_id: str
    level: int
    member_phrase_node_ids: list[str] = field(default_factory=list)
    member_chunk_ids: list[str] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Helper: phrase-only subgraph extraction
# ---------------------------------------------------------------------------


def _phrase_subgraph(kg_store: "KGStore"):
    """Return a NetworkX undirected graph containing ONLY phrase nodes and
    phrase->phrase edges (i.e. drop passage nodes and 'contains' edges).

    Multi-edges between the same pair of phrase nodes collapse into a single
    edge whose ``weight`` is the sum of constituent weights.
    """
    import networkx as nx  # noqa: PLC0415

    graph = kg_store._graph  # underscore is a soft hint; inside hrag.kg this is fine

    phrase_ids = [
        n
        for n, data in graph.nodes(data=True)
        if data.get("node_type") == "phrase"
    ]
    phrase_set = set(phrase_ids)

    sub = nx.Graph()
    for pid in phrase_ids:
        sub.add_node(pid, **graph.nodes[pid])

    # Collapse multi-edges to summed weights, undirected.
    accum: dict[tuple[str, str], float] = {}
    for u, v, edata in graph.edges(data=True):
        if edata.get("relation") == "contains":
            continue
        if u not in phrase_set or v not in phrase_set:
            continue
        if u == v:
            continue
        key = tuple(sorted((u, v)))
        weight = float(edata.get("weight", 1.0))
        accum[key] = accum.get(key, 0.0) + weight

    for (u, v), w in accum.items():
        sub.add_edge(u, v, weight=w)

    return sub


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CommunityDetector:
    """Run Leiden on the phrase subgraph at multiple resolution levels.

    Levels map to resolutions: 0 -> 0.5 (coarse), 1 -> 1.0 (medium),
    2 -> 2.0 (fine). Tunable; v1 keeps the mapping fixed.
    """

    name = "leiden_community_detector"

    def __init__(
        self,
        kg_store: "KGStore",
        leiden_seed: int = 42,
        levels: Optional[list[int]] = None,
        min_phrase_members: int = _MIN_CLUSTER_SIZE,
        min_chunk_members: int = _MIN_CHUNK_MEMBERS,
    ) -> None:
        self._kg_store = kg_store
        self._leiden_seed = int(leiden_seed)
        self._levels = list(levels) if levels is not None else [0, 1, 2]
        self._min_phrase_members = max(1, int(min_phrase_members))
        self._min_chunk_members = max(1, int(min_chunk_members))

    # ------------------------------------------------------------------

    def detect(self, user_id: str) -> list[Community]:
        """Build the phrase subgraph from the KGStore, run Leiden at each
        configured level, and return a flat list of Community objects across
        all levels.

        Communities with fewer than ``min_phrase_members`` phrase nodes are
        skipped, as are communities whose phrase nodes collectively reference
        fewer than ``min_chunk_members`` supporting chunks.
        """
        # Lazy imports — keep the project importable when these are missing.
        try:
            import igraph as ig  # noqa: PLC0415
            import leidenalg as la  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import path
            raise ImportError(
                "install leidenalg + python-igraph: pip install '.[kg]'"
            ) from exc

        nx_sub = _phrase_subgraph(self._kg_store)

        if nx_sub.number_of_nodes() < self._min_phrase_members:
            return []

        # Build igraph analogue.
        nx_node_ids: list[str] = list(nx_sub.nodes())
        nx_id_to_idx = {nid: i for i, nid in enumerate(nx_node_ids)}

        ig_edges: list[tuple[int, int]] = []
        ig_weights: list[float] = []
        for u, v, edata in nx_sub.edges(data=True):
            ig_edges.append((nx_id_to_idx[u], nx_id_to_idx[v]))
            ig_weights.append(float(edata.get("weight", 1.0)))

        g = ig.Graph(n=len(nx_node_ids), edges=ig_edges, directed=False)
        if ig_weights:
            g.es["weight"] = ig_weights

        communities: list[Community] = []
        for level in self._levels:
            resolution = _LEVEL_TO_RESOLUTION.get(level, 1.0)
            try:
                partition = la.find_partition(
                    g,
                    la.RBConfigurationVertexPartition,
                    resolution_parameter=resolution,
                    weights="weight" if ig_weights else None,
                    seed=self._leiden_seed,
                )
            except TypeError:
                # Older leidenalg versions don't accept `seed` kwarg.
                partition = la.find_partition(
                    g,
                    la.RBConfigurationVertexPartition,
                    resolution_parameter=resolution,
                    weights="weight" if ig_weights else None,
                )

            for cluster_idx, vertex_indices in enumerate(partition):
                if len(vertex_indices) < self._min_phrase_members:
                    continue
                phrase_ids = [nx_node_ids[i] for i in vertex_indices]
                chunk_ids = sorted(
                    self._kg_store.passage_nodes_for(phrase_ids)
                )
                if len(chunk_ids) < self._min_chunk_members:
                    # Skip — too few supporting passages to summarise
                    # meaningfully (also covers the empty case).
                    continue
                communities.append(
                    Community(
                        community_id=f"level{level}_c{cluster_idx}",
                        level=level,
                        member_phrase_node_ids=phrase_ids,
                        member_chunk_ids=chunk_ids,
                    )
                )

        return communities


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class CommunitySummarizer:
    """Generate a summary for each community via concurrent LLM calls."""

    name = "leiden_community_summarizer"

    _PLACEHOLDER = "<summary unavailable>"

    def __init__(
        self,
        llm: "LLMProvider",
        db: "Database",
        max_workers: int = 8,
        max_passages_per_community: int = 12,
    ) -> None:
        self._llm = llm
        self._db = db
        self._max_workers = max(1, int(max_workers))
        self._max_passages = max(1, int(max_passages_per_community))
        self._template = _load_prompt_template()
        # SQLite connection is shared across worker threads; serialise access.
        self._db_lock = threading.Lock()

    # ------------------------------------------------------------------

    def summarize_all(self, communities: list[Community]) -> list[Community]:
        """Concurrent ThreadPoolExecutor over communities.

        Failures are logged and the affected community gets a placeholder so
        nothing downstream breaks.
        """
        if not communities:
            return communities

        by_id: dict[str, Community] = {c.community_id: c for c in communities}

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._summarize_one, c): c.community_id
                for c in communities
            }
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    summary = fut.result()
                except Exception as exc:  # noqa: BLE001 - log and continue
                    logger.warning(
                        "Community %s summarization failed: %s", cid, exc
                    )
                    by_id[cid].summary = self._PLACEHOLDER
                else:
                    by_id[cid].summary = summary

        return communities

    # ------------------------------------------------------------------

    def _summarize_one(self, community: Community) -> str:
        passages = self._fetch_passages(community.member_chunk_ids)
        if not passages:
            return self._PLACEHOLDER

        # Sort by length ascending so smaller, more focused chunks come first.
        passages.sort(key=lambda row: len(row["text"] or ""))
        passages = passages[: self._max_passages]

        rendered_passages = _format_member_passages(passages)
        community_label = community.community_id

        prompt = self._template.format(
            community_label=community_label,
            member_passages=rendered_passages,
        )

        try:
            text = self._llm.complete(prompt, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM error summarising %s: %s", community.community_id, exc
            )
            return self._PLACEHOLDER

        text = (text or "").strip()
        return text or self._PLACEHOLDER

    def _fetch_passages(self, chunk_ids: list[str]) -> list[dict]:
        """Hydrate chunk rows for *chunk_ids* via a single IN query."""
        if not chunk_ids:
            return []
        placeholders = ",".join(["?"] * len(chunk_ids))
        sql = (
            f"SELECT chunk_id, title, section, text FROM chunks "
            f"WHERE chunk_id IN ({placeholders})"
        )
        with self._db_lock:
            cur = self._db.execute(sql, chunk_ids)
            rows = cur.fetchall()
        out: list[dict] = []
        for row in rows:
            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "title": row["title"] or "",
                    "section": row["section"] or "",
                    "text": row["text"] or "",
                }
            )
        return out


def _format_member_passages(passages: list[dict]) -> str:
    """Render a list of passage rows as a numbered, trimmed prompt block."""
    parts: list[str] = []
    for i, row in enumerate(passages, start=1):
        title = row.get("title", "") or ""
        section = row.get("section", "") or ""
        text = (row.get("text", "") or "")[:800]
        header = f"[{i}]"
        if title or section:
            header = f"[{i}] {title} / {section}".rstrip(" /")
        parts.append(f"{header}: {text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Store: ChromaDB collection + SQLite mirror
# ---------------------------------------------------------------------------


class CommunityStore:
    """ChromaDB-backed collection for community summaries plus a SQLite
    metadata mirror in ``kg_communities``."""

    name = "community_store"

    _COLLECTION_NAME = "hrag_community_summaries"

    def __init__(
        self,
        db: "Database",
        embedder: "EmbeddingProvider",
        chroma_path: str | Path,
    ) -> None:
        import chromadb  # noqa: PLC0415
        from chromadb.config import Settings  # noqa: PLC0415

        self._db = db
        self._embedder = embedder
        self._chroma_path = Path(chroma_path)
        self._chroma_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert(self, user_id: str, communities: list[Community]) -> None:
        """Embed each community's summary, upsert into the Chroma collection,
        and INSERT OR REPLACE into the SQLite mirror."""
        if not communities:
            return

        summaries = [c.summary or "" for c in communities]
        embeddings = self._embedder.embed(summaries)

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        for c in communities:
            ids.append(c.community_id)
            docs.append(c.summary or "")
            metas.append(
                {
                    "user_id": user_id,
                    "level": int(c.level),
                    "community_id": c.community_id,
                }
            )

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metas,
        )

        rows = []
        for c in communities:
            rows.append(
                (
                    c.community_id,
                    user_id,
                    int(c.level),
                    c.summary or "",
                    json.dumps(list(c.member_chunk_ids)),
                )
            )

        self._db.executemany(
            "INSERT OR REPLACE INTO kg_communities"
            "(community_id, user_id, level, summary, member_chunks) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._db.commit()

    def delete_user(self, user_id: str) -> None:
        """Wipe community summaries for *user_id* from both Chroma and SQLite."""
        try:
            self._collection.delete(where={"user_id": {"$eq": user_id}})
        except Exception as exc:  # noqa: BLE001 - some chroma stubs may not support delete
            logger.warning("CommunityStore: chroma delete failed for %s: %s", user_id, exc)

        self._db.execute(
            "DELETE FROM kg_communities WHERE user_id = ?",
            (user_id,),
        )
        self._db.commit()

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def query(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        levels: Optional[list[int]] = None,
    ) -> list[tuple[str, float]]:
        """Return (community_id, similarity_score) pairs, highest first."""
        where = _build_where(user_id=user_id, levels=levels)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["distances"],
        )

        ids_outer = results.get("ids") or [[]]
        dist_outer = results.get("distances") or [[]]
        ids = ids_outer[0] if ids_outer else []
        distances = dist_outer[0] if dist_outer else []

        pairs: list[tuple[str, float]] = []
        for cid, dist in zip(ids, distances):
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            pairs.append((cid, score))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def get_community(self, community_id: str) -> Optional[dict]:
        """Hydrate a community row from SQLite. Returns None if absent."""
        cur = self._db.execute(
            "SELECT community_id, level, summary, member_chunks "
            "FROM kg_communities WHERE community_id = ?",
            (community_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        try:
            members = json.loads(row["member_chunks"]) if row["member_chunks"] else []
        except (TypeError, ValueError):
            members = []
        return {
            "community_id": row["community_id"],
            "level": int(row["level"]),
            "summary": row["summary"] or "",
            "member_chunks": members,
        }


def _build_where(user_id: str, levels: Optional[list[int]]) -> dict:
    """Build a Chroma `where` filter scoped by user_id and (optionally) level."""
    conditions: list[dict] = [{"user_id": {"$eq": user_id}}]
    if levels:
        if len(levels) == 1:
            conditions.append({"level": {"$eq": int(levels[0])}})
        else:
            conditions.append({"level": {"$in": [int(x) for x in levels]}})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Convenience: end-to-end pipeline
# ---------------------------------------------------------------------------


def detect_and_summarize(
    kg_store: "KGStore",
    llm: "LLMProvider",
    db: "Database",
    embedder: "EmbeddingProvider",
    chroma_path: str | Path,
    user_id: str,
    cfg: "KGConfig",
) -> list[Community]:
    """Detect → summarize → upsert in one call. Returns the communities.

    Threshold note: ``min_chunk_members`` is not yet present on
    :class:`KGConfig`; if a future revision adds it, prefer that. For now we
    rely on :class:`CommunityDetector`'s default (currently
    :data:`_MIN_CHUNK_MEMBERS`) which drops 1-chunk singletons.
    """
    detector_kwargs: dict = {
        "kg_store": kg_store,
        "leiden_seed": cfg.leiden_seed,
        "levels": list(cfg.community_levels),
    }
    # Pull from KGConfig if a sibling Wave adds the field later; otherwise
    # the constructor default applies.
    cfg_min_chunk = getattr(cfg, "min_chunk_members", None)
    if cfg_min_chunk is not None:
        detector_kwargs["min_chunk_members"] = int(cfg_min_chunk)
    detector = CommunityDetector(**detector_kwargs)
    communities = detector.detect(user_id)

    if not communities:
        return communities

    summarizer = CommunitySummarizer(
        llm=llm,
        db=db,
        max_workers=cfg.parallel_workers,
    )
    summarizer.summarize_all(communities)

    store = CommunityStore(
        db=db,
        embedder=embedder,
        chroma_path=chroma_path,
    )
    store.upsert(user_id, communities)

    return communities
