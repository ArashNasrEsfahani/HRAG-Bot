"""Backfill the ``metadata.has_math`` tag on existing chunks.

Re-runs :func:`hrag.ingest.math_detect.has_math` over every row in
``chunks``. When the detector disagrees with the stored value, the SQLite
row is UPDATEd and the corresponding ChromaDB entry receives an
``update_metadata`` call.

Idempotent: only writes rows whose computed value differs from what's
already stored. Safe to re-run.

Usage::

    python scripts/backfill_has_math.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python scripts/backfill_has_math.py` from the project root without
# installing the package, mirroring the other one-off scripts in this repo.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from hrag.config import load_config
from hrag.db.connection import Database
from hrag.ingest.math_detect import has_math
from hrag.retrieval.backends.chroma import ChromaBackend


_CHROMA_BATCH = 256


def main() -> int:
    cfg = load_config()
    db_path = cfg.resolve(cfg.storage.sqlite_path)
    chroma_path = cfg.resolve(cfg.storage.chroma_path)

    print(f"backfill_has_math: sqlite={db_path}  chroma={chroma_path}", flush=True)

    db = Database(db_path)
    try:
        backend = ChromaBackend(chroma_path)
        chroma_ok = True
    except Exception as exc:  # noqa: BLE001 — best-effort; SQLite stays authoritative
        print(f"  chroma backend unavailable: {exc!r}; SQLite-only backfill", flush=True)
        backend = None
        chroma_ok = False

    rows = db.execute("SELECT chunk_id, text, metadata FROM chunks").fetchall()
    total = len(rows)
    if total == 0:
        print("backfill_has_math: no chunks found, nothing to do.", flush=True)
        return 0

    pending_chroma: list[tuple[str, bool]] = []
    n_diff = 0
    n_true_now = 0
    n_true_was = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("scanning chunks", total=total)

        with db.conn:  # one transaction wrapping the SQLite writes
            for row in rows:
                chunk_id = row["chunk_id"]
                text = row["text"] or ""
                try:
                    meta = json.loads(row["metadata"]) if row["metadata"] else {}
                except (TypeError, ValueError):
                    meta = {}
                old = bool(meta.get("has_math", False))
                if old:
                    n_true_was += 1
                new = bool(has_math(text))
                if new:
                    n_true_now += 1
                if new != old or "has_math" not in meta:
                    meta["has_math"] = new
                    db.conn.execute(
                        "UPDATE chunks SET metadata = ? WHERE chunk_id = ?",
                        (json.dumps(meta), chunk_id),
                    )
                    n_diff += 1
                    if chroma_ok:
                        pending_chroma.append((chunk_id, new))
                progress.advance(task)

        # Batch the Chroma metadata updates. Each batch is a single .update().
        if chroma_ok and pending_chroma:
            chroma_task = progress.add_task(
                "updating chroma metadata", total=len(pending_chroma)
            )
            for i in range(0, len(pending_chroma), _CHROMA_BATCH):
                chunk_batch = pending_chroma[i : i + _CHROMA_BATCH]
                ids = [cid for cid, _ in chunk_batch]
                metas = [{"has_math": bool(v)} for _, v in chunk_batch]
                try:
                    backend.update_metadata(ids, metas)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  chroma update_metadata raised on batch {i}: {exc!r}",
                        flush=True,
                    )
                progress.advance(chroma_task, advance=len(chunk_batch))

    print(
        f"Tagged {n_true_now} of {total} chunks with has_math=True "
        f"(was {n_true_was}); {n_diff} rows updated.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
