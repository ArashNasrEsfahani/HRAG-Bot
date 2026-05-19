"""Clean placeholder-string + empty-string pollution from the live KG.

Removes phrase nodes whose label/node_id is empty/whitespace OR equals one of
the LLM placeholder values that ``TripleExtractor._drop_reason`` now filters
at extraction time. Mirrors the cleanup into both the SQLite tables and the
pickled NetworkX graph.

Usage:
    py tools/clean_kg_pollution.py             # apply
    py tools/clean_kg_pollution.py --dry-run   # report only

Idempotent.
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path

# Mirror the placeholder list from hrag.kg.builder so this script stays in
# sync without importing the module (avoids loading the orchestrator graph).
_PLACEHOLDER_VALUES = frozenset(
    {
        "<empty>", "<none>", "<null>", "<n/a>", "<unknown>", "<>",
        "none", "null", "n/a", "na", "unknown", "unspecified",
        "tbd", "todo", "placeholder", "...",
    }
)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "store.sqlite"
GRAPH_PATH = ROOT / "data" / "kg" / "graph.pkl"


def _is_pollution(value: str | None) -> bool:
    if value is None:
        return True
    s = value.strip().lower()
    return (not s) or (s in _PLACEHOLDER_VALUES)


def _print(label: str, value) -> None:
    print(f"  {label:<40} {value}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report only; no writes.")
    args = parser.parse_args()

    print(f"[clean_kg_pollution] db={DB_PATH}")
    print(f"[clean_kg_pollution] graph={GRAPH_PATH}")
    print(f"[clean_kg_pollution] mode={'DRY RUN' if args.dry_run else 'APPLY'}")
    print()

    if not DB_PATH.exists():
        print(f"ERROR: SQLite db not found at {DB_PATH}", flush=True)
        return 1

    print("[1/5] Counting BEFORE...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    placeholder_list = ",".join(["?"] * len(_PLACEHOLDER_VALUES))
    placeholder_args = list(_PLACEHOLDER_VALUES)

    nodes_before = c.execute(
        f"""
        SELECT COUNT(*) FROM kg_nodes
        WHERE node_type='phrase' AND (
            LENGTH(TRIM(COALESCE(label,'')))=0
            OR LENGTH(TRIM(node_id))=0
            OR LOWER(TRIM(label)) IN ({placeholder_list})
            OR LOWER(TRIM(node_id)) IN ({placeholder_list})
        )
        """,
        placeholder_args + placeholder_args,
    ).fetchone()[0]

    edges_before = c.execute(
        f"""
        SELECT COUNT(*) FROM kg_edges
        WHERE LENGTH(TRIM(COALESCE(src,'')))=0
           OR LENGTH(TRIM(COALESCE(dst,'')))=0
           OR LENGTH(TRIM(COALESCE(relation,'')))=0
           OR LOWER(TRIM(src)) IN ({placeholder_list})
           OR LOWER(TRIM(dst)) IN ({placeholder_list})
           OR LOWER(TRIM(relation)) IN ({placeholder_list})
        """,
        placeholder_args + placeholder_args + placeholder_args,
    ).fetchone()[0]

    _print("kg_nodes (phrase) to drop:", nodes_before)
    _print("kg_edges to drop:", edges_before)

    print()
    print("[2/5] Loading pickled graph...", flush=True)
    if not GRAPH_PATH.exists():
        print(f"  WARNING: no graph.pkl at {GRAPH_PATH}; SQLite-only cleanup.", flush=True)
        graph = None
    else:
        with GRAPH_PATH.open("rb") as f:
            graph = pickle.load(f)
        _print("nodes in graph:", graph.number_of_nodes())
        _print("edges in graph:", graph.number_of_edges())

    print()
    print("[3/5] Identifying nodes to remove from in-memory graph...", flush=True)
    to_remove: list[str] = []
    if graph is not None:
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "phrase":
                continue
            if _is_pollution(node_id) or _is_pollution(attrs.get("label", "")):
                to_remove.append(node_id)
    _print("phrase nodes to remove:", len(to_remove))
    for nid in to_remove[:10]:
        print(f"      - {nid!r}", flush=True)

    if args.dry_run:
        print()
        print("[dry-run] Skipping writes. Counts above show the impact.", flush=True)
        conn.close()
        return 0

    print()
    print("[4/5] Applying changes...", flush=True)
    if graph is not None:
        for nid in to_remove:
            graph.remove_node(nid)  # auto-removes incident edges
        with GRAPH_PATH.open("wb") as f:
            pickle.dump(graph, f)
        _print("graph nodes after:", graph.number_of_nodes())
        _print("graph edges after:", graph.number_of_edges())

    deleted_nodes = c.execute(
        f"""
        DELETE FROM kg_nodes
        WHERE node_type='phrase' AND (
            LENGTH(TRIM(COALESCE(label,'')))=0
            OR LENGTH(TRIM(node_id))=0
            OR LOWER(TRIM(label)) IN ({placeholder_list})
            OR LOWER(TRIM(node_id)) IN ({placeholder_list})
        )
        """,
        placeholder_args + placeholder_args,
    ).rowcount

    deleted_edges = c.execute(
        f"""
        DELETE FROM kg_edges
        WHERE LENGTH(TRIM(COALESCE(src,'')))=0
           OR LENGTH(TRIM(COALESCE(dst,'')))=0
           OR LENGTH(TRIM(COALESCE(relation,'')))=0
           OR LOWER(TRIM(src)) IN ({placeholder_list})
           OR LOWER(TRIM(dst)) IN ({placeholder_list})
           OR LOWER(TRIM(relation)) IN ({placeholder_list})
        """,
        placeholder_args + placeholder_args + placeholder_args,
    ).rowcount
    conn.commit()
    _print("kg_nodes deleted:", deleted_nodes)
    _print("kg_edges deleted:", deleted_edges)

    print()
    print("[5/5] AFTER snapshot — top-10 most-connected phrase nodes:", flush=True)
    for src, n in c.execute(
        "SELECT src, COUNT(*) FROM kg_edges GROUP BY src ORDER BY 2 DESC LIMIT 10"
    ):
        print(f"      {n:4d}  {src!r}", flush=True)

    conn.close()
    print()
    print("[clean_kg_pollution] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
