"""Rebuild community summaries only (skip triple re-extraction).

Why this exists: `hrag rebuild-kg` re-extracts triples from every chunk
(~50+ min for 25 docs) before doing communities. When the KG is already
fresh and only the community layer is stale (e.g. SQLite mirror missing),
this is overkill. This script runs just the Leiden + LLM summarization
phase and writes both Chroma and SQLite, with per-community progress.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from hrag.cli import load_config
from hrag.kg.communities import (
    CommunityDetector,
    CommunityStore,
    CommunitySummarizer,
)
from hrag.orchestrator import Orchestrator


def main() -> int:
    console = Console()
    cfg = load_config()
    if not cfg.kg.enabled:
        console.print("[red]kg.enabled=false[/red] — nothing to do.")
        return 1
    if not cfg.kg.use_communities:
        console.print(
            "[yellow]kg.use_communities=false[/yellow] — set it to true in config.yaml "
            "before running this script."
        )
        return 1

    print("Initialising orchestrator (loads KG from SQLite)...", flush=True)
    orch = Orchestrator(cfg)
    if orch.kg_store is None or orch.community_store is None:
        console.print("[red]KG layer failed to init[/red]")
        return 1

    user_id = cfg.user.default_user_id
    chroma_path = cfg.resolve(cfg.storage.chroma_path)

    # --- 1. Wipe stale community summaries (both Chroma and SQLite) ---
    print("Wiping stale community summaries...", flush=True)
    orch.community_store.delete_user(user_id)

    # --- 2. Detect communities ---
    print("Detecting communities (Leiden at levels {})...".format(cfg.kg.community_levels), flush=True)
    t0 = time.time()
    detector = CommunityDetector(
        kg_store=orch.kg_store,
        leiden_seed=cfg.kg.leiden_seed,
        levels=list(cfg.kg.community_levels),
    )
    communities = detector.detect(user_id)
    dt = time.time() - t0
    print(f"  -> {len(communities)} communities to summarise (Leiden took {dt:.1f}s)", flush=True)
    if not communities:
        console.print("[yellow]No communities passed filters — nothing to summarise.[/yellow]")
        return 0

    # Per-level breakdown
    by_level: dict[int, int] = {}
    for c in communities:
        by_level[c.level] = by_level.get(c.level, 0) + 1
    for lvl in sorted(by_level):
        print(f"     level {lvl}: {by_level[lvl]} communities", flush=True)

    # --- 3. Summarise each community concurrently with a progress bar ---
    summarizer = CommunitySummarizer(
        llm=orch.llm,
        db=orch.db,
        max_workers=cfg.kg.parallel_workers,
    )
    by_id = {c.community_id: c for c in communities}

    progress = Progress(
        TextColumn("[bold]summarising[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("• ETA"),
        TimeRemainingColumn(),
        console=console,
    )
    failures = 0
    with progress:
        task = progress.add_task("summarise", total=len(communities))
        with ThreadPoolExecutor(max_workers=summarizer._max_workers) as pool:
            futures = {
                pool.submit(summarizer._summarize_one, c): c.community_id
                for c in communities
            }
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    summary = fut.result()
                except Exception as exc:  # noqa: BLE001
                    summary = "<summary unavailable>"
                    failures += 1
                    console.print(f"[red]fail[/red] {cid}: {exc}")
                by_id[cid].summary = summary
                progress.advance(task)

    if failures:
        console.print(f"[yellow]{failures} community summarisations failed[/yellow]")

    # --- 4. Upsert into Chroma + SQLite mirror ---
    print("Upserting {} community summaries to Chroma + SQLite...".format(len(communities)), flush=True)
    orch.community_store.upsert(user_id, communities)

    # --- 5. Verify ---
    n_sqlite = orch.db.execute(
        "SELECT COUNT(*) FROM kg_communities WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    print(f"SQLite kg_communities now has {n_sqlite} rows for user_id={user_id!r}", flush=True)

    orch.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
