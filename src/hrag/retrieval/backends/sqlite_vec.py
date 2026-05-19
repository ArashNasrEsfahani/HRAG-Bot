"""sqlite-vec backend — real implementation.

Drop-in alternative to :class:`ChromaBackend` backed by the
`sqlite-vec <https://github.com/asg017/sqlite-vec>`_ extension. Persistence
lives at ``persist_path/vec_store.db`` (a single SQLite file holding both the
``vec0`` virtual table and a typed metadata mirror).

Why a separate metadata mirror table?
- ``vec0`` virtual tables only store the embedding column. To support the
  Chroma-style ``where`` filter (flat key equality plus ``$and`` / ``$eq`` /
  ``$in`` operators), we mirror the filter-relevant metadata keys
  (``user_id``, ``source_type``, ``doc_id``, ``chunk_index``, ``excluded``)
  into typed columns on ``chunks_meta`` and JOIN at query time.
- The full metadata dict is also kept as JSON so future filters can be added
  without a schema migration.

Distance semantics
- The virtual table is created with ``distance_metric=cosine`` so the raw
  ``distance`` column already matches Chroma's convention (``1 -
  cosine_similarity``). :class:`hrag.retrieval.vector.VectorStore` then
  converts to similarity via ``score = 1 - distance``; everything downstream
  keeps working unchanged.

The embedding dimension is fixed on first upsert (it is part of the
``CREATE VIRTUAL TABLE`` DDL). Subsequent upserts must use the same
dimension; mismatches raise :class:`ValueError`.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any, Optional


_VEC_TABLE = "vec_chunks"
_META_TABLE = "chunks_meta"

# Metadata keys mirrored as typed columns so the where-filter JOIN is fast.
# Anything outside this list still lives in ``metadata_json`` and can be
# read back, just not filtered without a schema bump.
_INDEXED_COLS: tuple[str, ...] = (
    "user_id",
    "doc_id",
    "source_type",
    "chunk_index",
    "excluded",
)


class SqliteVecBackend:
    """Vector backend backed by sqlite-vec.

    See module docstring for design notes.
    """

    name = "sqlite_vec"

    def __init__(self, persist_path: str | Path) -> None:
        # Deferred import so module load is cheap even when sqlite-vec is not
        # installed (the dep is opt-in via the ``sqlite-vec`` extra).
        try:
            import sqlite_vec  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "sqlite-vec is required for SqliteVecBackend. "
                "Install with: pip install sqlite-vec"
            ) from e

        self._persist_path = Path(persist_path)
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self._persist_path / "vec_store.db"

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        # Re-disable extension loading after the sqlite-vec extension is in;
        # belt-and-braces hardening per sqlite-vec quickstart recommendation.
        self._conn.enable_load_extension(False)

        # Always (re)create the meta table on open — IF NOT EXISTS keeps it
        # idempotent. The vec0 table can only be created once we know the
        # embedding dim, so it's deferred to the first upsert. If a previous
        # run already created it, ``_dim`` is sniffed from the schema.
        self._init_meta_table()
        self._dim: Optional[int] = self._sniff_dim()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _init_meta_table(self) -> None:
        # rowid is the link to vec_chunks (vec0 rowid == meta rowid).
        # ``id`` is the public chunk id (UNIQUE so INSERT OR REPLACE works).
        # Typed mirror columns make filter pushdown straightforward.
        col_defs = ", ".join(f"{c} TEXT" if c not in ("chunk_index", "excluded") else f"{c} INTEGER"
                             for c in _INDEXED_COLS)
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {_META_TABLE} ("
            f"  rowid INTEGER PRIMARY KEY AUTOINCREMENT,"
            f"  id TEXT UNIQUE NOT NULL,"
            f"  document TEXT,"
            f"  metadata_json TEXT NOT NULL,"
            f"  {col_defs}"
            f")"
        )
        with self._conn:
            self._conn.execute(ddl)
            for col in _INDEXED_COLS:
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{_META_TABLE}_{col} "
                    f"ON {_META_TABLE}({col})"
                )

    def _sniff_dim(self) -> Optional[int]:
        """Return the dim of an existing vec_chunks table, or None if absent."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (_VEC_TABLE,),
        ).fetchone()
        if row is None or not row[0]:
            return None
        sql = row[0]
        # The DDL stored by SQLite is the exact CREATE statement we issued.
        # Parse the ``FLOAT[<dim>]`` token; fall back to None on any oddity.
        marker = "FLOAT["
        i = sql.find(marker)
        if i == -1:
            return None
        j = sql.find("]", i + len(marker))
        if j == -1:
            return None
        try:
            return int(sql[i + len(marker): j])
        except ValueError:
            return None

    def _ensure_vec_table(self, dim: int) -> None:
        if self._dim is not None:
            if dim != self._dim:
                raise ValueError(
                    f"Embedding dim mismatch: backend was initialised with "
                    f"dim={self._dim}, got dim={dim}."
                )
            return
        # First-use DDL — sets the dim forever.
        ddl = (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {_VEC_TABLE} "
            f"USING vec0(embedding FLOAT[{dim}] distance_metric=cosine)"
        )
        with self._conn:
            self._conn.execute(ddl)
        self._dim = dim

    # ------------------------------------------------------------------
    # VectorBackend protocol
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        n = len(ids)
        if not (len(embeddings) == len(documents) == len(metadatas) == n):
            raise ValueError(
                "ids, embeddings, documents, metadatas must all have equal length"
            )

        dim = len(embeddings[0])
        self._ensure_vec_table(dim)

        # Two-step upsert per row so we can keep the rowid in sync between the
        # vec0 table and the meta mirror:
        #   1) INSERT OR REPLACE into chunks_meta — captures the rowid.
        #   2) DELETE+INSERT into vec_chunks at that rowid (vec0 doesn't
        #      support UPSERT on rowid directly).
        # Wrapped in a single transaction for atomicity + speed.
        try:
            with self._conn:
                for cid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                    if len(emb) != dim:
                        raise ValueError(
                            f"All embeddings must have dim={dim}; got {len(emb)} for id={cid}"
                        )

                    # Step 1 — meta row. Look up an existing rowid first so we
                    # can keep it stable (REPLACE would change it and orphan
                    # the vec row).
                    existing = self._conn.execute(
                        f"SELECT rowid FROM {_META_TABLE} WHERE id = ?", (cid,)
                    ).fetchone()

                    col_values = [meta.get(c) for c in _INDEXED_COLS]
                    placeholders = ", ".join(["?"] * (3 + len(_INDEXED_COLS)))
                    cols = "id, document, metadata_json, " + ", ".join(_INDEXED_COLS)
                    if existing is None:
                        cur = self._conn.execute(
                            f"INSERT INTO {_META_TABLE} ({cols}) VALUES ({placeholders})",
                            (cid, doc, json.dumps(meta), *col_values),
                        )
                        rowid = cur.lastrowid
                    else:
                        rowid = existing[0]
                        set_clause = "document = ?, metadata_json = ?, " + ", ".join(
                            f"{c} = ?" for c in _INDEXED_COLS
                        )
                        self._conn.execute(
                            f"UPDATE {_META_TABLE} SET {set_clause} WHERE rowid = ?",
                            (doc, json.dumps(meta), *col_values, rowid),
                        )

                    # Step 2 — vec row at the same rowid.
                    packed = struct.pack(f"{dim}f", *emb)
                    # DELETE-then-INSERT is the safest cross-version pattern;
                    # newer sqlite-vec supports UPDATE but we don't rely on it.
                    self._conn.execute(
                        f"DELETE FROM {_VEC_TABLE} WHERE rowid = ?", (rowid,)
                    )
                    self._conn.execute(
                        f"INSERT INTO {_VEC_TABLE}(rowid, embedding) VALUES (?, ?)",
                        (rowid, packed),
                    )
        except sqlite3.Error as e:
            # Surface SQL errors with enough context to debug; don't bury them.
            raise RuntimeError(f"sqlite-vec upsert failed: {e}") from e

    def delete_where(self, where: dict) -> None:
        # Find matching rowids in the meta table, then delete from both.
        clause, params = _compile_where(where)
        if clause is None:
            # Empty where → no-op (mirrors Chroma's behaviour where an empty
            # filter would otherwise delete everything; we err on the side of
            # safety).
            return
        rows = self._conn.execute(
            f"SELECT rowid FROM {_META_TABLE} WHERE {clause}", params
        ).fetchall()
        if not rows:
            return
        rowids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(rowids))
        with self._conn:
            self._conn.execute(
                f"DELETE FROM {_VEC_TABLE} WHERE rowid IN ({placeholders})", rowids
            )
            self._conn.execute(
                f"DELETE FROM {_META_TABLE} WHERE rowid IN ({placeholders})", rowids
            )

    def query_one(
        self,
        embedding: list[float],
        top_k: int,
        where: Optional[dict],
    ) -> tuple[list[str], list[float]]:
        if self._dim is None:
            # No upserts yet — nothing to return. Mirrors Chroma on an empty
            # collection.
            return [], []
        if len(embedding) != self._dim:
            raise ValueError(
                f"Query embedding dim {len(embedding)} != backend dim {self._dim}"
            )

        packed = struct.pack(f"{self._dim}f", *embedding)

        # vec0 KNN requires its own ``k`` parameter; SQL LIMIT alone is not
        # enough (the virtual table does the actual nearest-neighbour scan).
        # When a where filter is present we over-fetch by 4x to leave headroom
        # for filter pruning (matches the heuristic used elsewhere in HRAG).
        if where:
            clause, params = _compile_where(where)
            if clause is None:
                # Filter compiled away to nothing (only operators we don't
                # support) — treat as unfiltered.
                clause = None
                params = []
        else:
            clause = None
            params = []

        fetch_k = top_k * 4 if clause else top_k

        sql_parts = [
            "SELECT cm.id, v.distance",
            f"FROM {_VEC_TABLE} v",
            f"JOIN {_META_TABLE} cm USING (rowid)",
            "WHERE v.embedding MATCH ? AND k = ?",
        ]
        sql_params: list[Any] = [packed, fetch_k]
        if clause:
            sql_parts.append(f"AND ({clause})")
            sql_params.extend(params)
        sql_parts.append("ORDER BY v.distance")
        sql_parts.append("LIMIT ?")
        sql_params.append(top_k)

        sql = " ".join(sql_parts)
        rows = self._conn.execute(sql, sql_params).fetchall()
        ids: list[str] = [r[0] for r in rows]
        distances: list[float] = [float(r[1]) for r in rows]
        return ids, distances

    def update_metadata(self, ids: list[str], metadatas: list[dict]) -> None:
        if not ids:
            return
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas must have equal length")

        with self._conn:
            for cid, meta in zip(ids, metadatas):
                row = self._conn.execute(
                    f"SELECT metadata_json FROM {_META_TABLE} WHERE id = ?",
                    (cid,),
                ).fetchone()
                if row is None:
                    # Mirror Chroma's best-effort semantic — silently skip
                    # rows we don't know about rather than erroring.
                    continue
                existing = json.loads(row[0]) if row[0] else {}
                existing.update(meta)
                col_values = [existing.get(c) for c in _INDEXED_COLS]
                set_clause = "metadata_json = ?, " + ", ".join(
                    f"{c} = ?" for c in _INDEXED_COLS
                )
                self._conn.execute(
                    f"UPDATE {_META_TABLE} SET {set_clause} WHERE id = ?",
                    (json.dumps(existing), *col_values, cid),
                )

    def count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {_META_TABLE}").fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __del__(self) -> None:  # pragma: no cover - GC timing dependent
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# where-filter compilation
# ---------------------------------------------------------------------------


def _compile_where(where: dict) -> tuple[Optional[str], list[Any]]:
    """Translate Chroma-style ``where`` dicts into a SQL WHERE clause.

    Supported shapes (a strict subset, matching what HRAG actually emits):
        {"key": "value"}                       — flat equality
        {"key": {"$eq": "value"}}              — explicit eq
        {"key": {"$in": ["a", "b"]}}           — IN list
        {"$and": [<sub-clause>, <sub-clause>]} — conjunction
        {"$or":  [<sub-clause>, <sub-clause>]} — disjunction (defensive)

    Anything else is ignored (returns no clause for that term) — callers
    should validate at a higher level if strict behaviour is needed.

    Returns ``(clause, params)`` or ``(None, [])`` if the filter would be a
    no-op. Only ``_INDEXED_COLS`` keys are filterable; unknown keys are
    treated as no-ops so they don't accidentally hide data.
    """
    if not where:
        return None, []

    clause, params = _compile_term(where)
    if not clause:
        return None, []
    return clause, params


def _compile_term(term: dict) -> tuple[str, list[Any]]:
    """Compile a single where-term (one dict). Returns ('', []) on no-op."""
    if "$and" in term:
        subs = term["$and"]
        compiled = [_compile_term(s) for s in subs if isinstance(s, dict)]
        compiled = [(c, p) for c, p in compiled if c]
        if not compiled:
            return "", []
        clause = " AND ".join(f"({c})" for c, _ in compiled)
        params: list[Any] = []
        for _, ps in compiled:
            params.extend(ps)
        return clause, params

    if "$or" in term:
        subs = term["$or"]
        compiled = [_compile_term(s) for s in subs if isinstance(s, dict)]
        compiled = [(c, p) for c, p in compiled if c]
        if not compiled:
            return "", []
        clause = " OR ".join(f"({c})" for c, _ in compiled)
        params = []
        for _, ps in compiled:
            params.extend(ps)
        return clause, params

    # Flat key terms — possibly several in the same dict.
    parts: list[str] = []
    params = []
    for key, val in term.items():
        if key in ("$and", "$or"):
            continue  # already handled above
        if key not in _INDEXED_COLS:
            # Unknown key — silently skip to avoid masking rows by mistake.
            continue
        if isinstance(val, dict):
            if "$eq" in val:
                parts.append(f"{key} = ?")
                params.append(val["$eq"])
            elif "$in" in val:
                items = list(val["$in"])
                if not items:
                    # Empty $in matches nothing; force a contradiction.
                    parts.append("1 = 0")
                else:
                    placeholders = ",".join("?" * len(items))
                    parts.append(f"{key} IN ({placeholders})")
                    params.extend(items)
            elif "$ne" in val:
                parts.append(f"{key} != ?")
                params.append(val["$ne"])
            else:
                # Unsupported operator — skip rather than error.
                continue
        else:
            # Bare equality shorthand.
            parts.append(f"{key} = ?")
            params.append(val)

    if not parts:
        return "", []
    return " AND ".join(parts), params
