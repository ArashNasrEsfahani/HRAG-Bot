"""CLI entry-point for HRAG-Bot.

Commands
--------
hrag init          – initialise the SQLite database
hrag ingest <path> – ingest a file or directory
hrag chat          – interactive REPL
hrag list-docs     – list ingested documents
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hrag.config import load_config
from hrag.db.connection import init_db
from hrag.orchestrator import ChatResult, Orchestrator
from hrag.providers.llm import LLMProviderError
from hrag.types import RetrievalResult

console = Console()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """HRAG-Bot: hierarchical RAG chatbot."""
    ctx.ensure_object(dict)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cli.command("init")
def cmd_init() -> None:
    """Initialise the SQLite database and ChromaDB directories."""
    try:
        cfg = load_config()
        db_path = cfg.resolve(cfg.storage.sqlite_path)
        init_db(db_path, cfg.user.default_user_id)
        console.print(f"Initialized at [bold]{db_path}[/bold]")
        console.print(
            "Tip: enable Nougat OCR for academic PDFs with "
            "[bold]ingest.use_nougat: true[/bold] "
            "(pip install nougat-ocr)."
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@cli.command("ingest")
@click.argument("path", type=click.Path(exists=False))
@click.option("--user", "-u", default=None, help="User ID (defaults to config value).")
@click.option("--recursive", "-r", is_flag=True, default=False, help="Recurse into subdirectories.")
def cmd_ingest(path: str, user: Optional[str], recursive: bool) -> None:
    """Ingest a file or directory into the vector store."""
    target = Path(path)
    if not target.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        sys.exit(1)

    try:
        cfg = load_config()
        orch = Orchestrator(cfg)
        user_id = user or cfg.user.default_user_id

        if target.is_dir():
            docs = orch.ingest.ingest_directory(target, user_id, recursive=recursive)
            total_chunks = sum(
                orch.db.execute(
                    "SELECT COUNT(*) as n FROM chunks WHERE doc_id = ?", (d.doc_id,)
                ).fetchone()["n"]
                for d in docs
            )
            console.print(
                f"Ingested [bold]{len(docs)}[/bold] documents "
                f"([bold]{total_chunks}[/bold] chunks total)."
            )
        else:
            doc = orch.ingest.ingest_path(target, user_id)
            chunk_count = orch.db.execute(
                "SELECT COUNT(*) as n FROM chunks WHERE doc_id = ?", (doc.doc_id,)
            ).fetchone()["n"]
            console.print(
                f"Ingested [bold]1[/bold] document "
                f"([bold]{chunk_count}[/bold] chunks total)."
            )

        # Phase 2: build community summaries after ingest, when both KG and
        # the use_communities flag are enabled. Default lightweight setup
        # leaves use_communities=false to skip the slow Leiden + LLM
        # summarization step; KG-PPR retrieval still works without it.
        if (
            cfg.kg.enabled
            and cfg.kg.use_communities
            and orch.kg_store is not None
            and orch.community_store is not None
        ):
            from hrag.kg.communities import detect_and_summarize  # noqa: PLC0415

            chroma_path = cfg.resolve(cfg.storage.chroma_path)
            orch.community_store.delete_user(user_id)  # rebuild from scratch
            console.print("Detecting communities...")
            communities = detect_and_summarize(
                orch.kg_store, orch.llm, orch.db, orch.embedder,
                chroma_path, user_id, cfg.kg,
            )
            console.print(f"  -> {len(communities)} communities summarized.")
        elif cfg.kg.enabled and not cfg.kg.use_communities:
            console.print(
                "[dim]Skipping community detection (kg.use_communities=false). "
                "KG-PPR retrieval still active.[/dim]"
            )

        orch.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# rebuild-kg
# ---------------------------------------------------------------------------


@cli.command("rebuild-kg")
@click.option("--user", "-u", default=None)
def cmd_rebuild_kg(user: Optional[str]) -> None:
    """Rebuild the knowledge graph and community summaries from existing chunks.

    Use this after editing chunk content directly, or after pulling a new
    version of the prompt template, without re-running expensive PDF parsing.
    """
    try:
        cfg = load_config()
        if not cfg.kg.enabled:
            console.print("[yellow]Warning:[/yellow] kg.enabled=false in config — nothing to rebuild.")
            return

        orch = Orchestrator(cfg)
        user_id = user or cfg.user.default_user_id

        if orch.kg_store is None or orch.community_store is None:
            console.print(
                "[red]Error:[/red] KG layer failed to initialize. "
                "Check that scipy/networkx/leidenalg/chromadb are installed."
            )
            sys.exit(1)

        # 1. Re-extract triples from all chunks for this user
        rows = orch.db.execute(
            "SELECT chunk_id, doc_id, user_id, text, title, section, subsection, "
            "chunk_index, token_count, source_type FROM chunks WHERE user_id=? AND excluded=0",
            (user_id,),
        ).fetchall()
        if not rows:
            console.print("No chunks to process. Run `hrag ingest` first.")
            return

        from hrag.types import Chunk  # noqa: PLC0415

        chunks_by_doc: dict[str, list[Chunk]] = {}
        for r in rows:
            chunks_by_doc.setdefault(r["doc_id"], []).append(
                Chunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    user_id=r["user_id"],
                    text=r["text"],
                    embedding_text=r["text"],
                    title=r["title"] or "",
                    section=r["section"] or "",
                    subsection=r["subsection"] or "",
                    chunk_index=r["chunk_index"],
                    token_count=r["token_count"],
                    source_type=r["source_type"],
                )
            )

        # 2. Re-extract triples per doc
        from hrag.kg.builder import TripleExtractor  # noqa: PLC0415

        console.print("Wiping existing KG and community summaries...")
        extractor = TripleExtractor(orch.llm, max_workers=cfg.kg.parallel_workers, db=orch.db)
        for doc_id, chunks in chunks_by_doc.items():
            console.print(f"  Extracting triples for {doc_id} ({len(chunks)} chunks)...")
            triples = extractor.extract_batch(chunks)
            cid_map = {c.chunk_id: c.doc_id for c in chunks}
            orch.kg_store.upsert_triples(user_id, doc_id, triples, cid_map)
            console.print(f"  → {len(triples)} triples")

        # 3. Re-detect + summarize communities (gated on use_communities flag)
        if cfg.kg.use_communities:
            console.print("Detecting communities...")
            from hrag.kg.communities import detect_and_summarize  # noqa: PLC0415

            chroma_path = cfg.resolve(cfg.storage.chroma_path)
            # Wipe stale summaries before regenerating
            orch.community_store.delete_user(user_id)
            communities = detect_and_summarize(
                orch.kg_store, orch.llm, orch.db, orch.embedder,
                chroma_path, user_id, cfg.kg,
            )
            console.print(f"[green]Done.[/green] {len(communities)} communities summarized.")
        else:
            console.print(
                "[green]Done.[/green] KG triples re-extracted; community step "
                "skipped (kg.use_communities=false)."
            )

    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------


@cli.group("taxonomy")
def cmd_taxonomy() -> None:
    """Manage the hierarchical document-taxonomy tree (Phase 2b)."""


@cmd_taxonomy.command("build")
@click.option("--user", "-u", default=None, help="User ID (defaults to config value).")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-summarize every doc even if cached summary exists.",
)
def cmd_taxonomy_build(user: Optional[str], force: bool) -> None:
    """Build (or rebuild) the taxonomy tree from the current corpus.

    Summarizes every document with the LLM, asks the LLM to propose a
    hierarchical category tree, then files each doc under a leaf.
    """
    try:
        cfg = load_config()
        if not cfg.taxonomy.enabled:
            console.print(
                "[yellow]Warning:[/yellow] taxonomy.enabled=false in config — "
                "nothing to build. Flip it in config.yaml and rerun."
            )
            return

        orch = Orchestrator(cfg)
        user_id = user or cfg.user.default_user_id

        if orch.taxonomy_store is None:
            console.print("[red]Error:[/red] taxonomy store failed to initialize.")
            sys.exit(1)

        from hrag.taxonomy.builder import TaxonomyBuilder  # noqa: PLC0415

        builder = TaxonomyBuilder(
            orch.db, orch.llm, orch.embedder, orch.taxonomy_store, cfg.taxonomy,
        )
        console.print(f"[cyan]Building taxonomy for user {user_id}...[/cyan]")
        summary = builder.build_for_user(user_id, force=force)
        console.print(
            f"[green]Done.[/green] {summary.get('docs_processed', 0)} docs · "
            f"{summary.get('nodes_created', 0)} nodes "
            f"({summary.get('leaves', 0)} leaves) in "
            f"{summary.get('duration_s', 0.0):.1f}s."
        )
        orch.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        sys.exit(1)


@cmd_taxonomy.command("show")
@click.option("--user", "-u", default=None, help="User ID (defaults to config value).")
def cmd_taxonomy_show(user: Optional[str]) -> None:
    """Print the user's taxonomy tree as a text outline."""
    try:
        cfg = load_config()
        orch = Orchestrator(cfg)
        user_id = user or cfg.user.default_user_id

        if orch.taxonomy_store is None:
            console.print("[red]Error:[/red] taxonomy disabled — set taxonomy.enabled=true.")
            sys.exit(1)

        nodes = orch.taxonomy_store.list_nodes(user_id)
        if not nodes:
            console.print("[dim]No taxonomy yet. Run `hrag taxonomy build` first.[/dim]")
            return

        children: dict[str, list[str]] = {}
        nodes_by_id = {n.node_id: n for n in nodes}
        for n in nodes:
            children.setdefault(n.parent_id or "", []).append(n.node_id)

        from rich.tree import Tree  # noqa: PLC0415

        roots = [n for n in nodes if n.parent_id is None]
        if not roots:
            console.print("[dim](no root node found)[/dim]")
            return

        def _build(node, parent_branch) -> None:
            label_text = (
                f"[bold]{node.label}[/bold] "
                f"[dim](docs={node.doc_count}, id={node.node_id[:10]})[/dim]"
            )
            if node.is_leaf and node.doc_count:
                docs = orch.taxonomy_store.get_docs_at(node.node_id)[:4]
                doc_titles = []
                if docs:
                    rows = orch.db.execute(
                        f"SELECT doc_id, title FROM documents WHERE doc_id IN "
                        f"({','.join('?' * len(docs))})",
                        docs,
                    ).fetchall()
                    doc_titles = [(r["title"] or r["doc_id"])[:60] for r in rows]
                if doc_titles:
                    label_text += " · " + "; ".join(doc_titles)
            branch = parent_branch.add(label_text)
            for cid in children.get(node.node_id, []):
                _build(nodes_by_id[cid], branch)

        rich_tree = Tree(f"[bold cyan]Taxonomy for {user_id}[/bold cyan]")
        for r in roots:
            _build(r, rich_tree)
        console.print(rich_tree)
        orch.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@cmd_taxonomy.command("clear")
@click.option("--user", "-u", default=None, help="User ID (defaults to config value).")
@click.confirmation_option(prompt="Wipe the user's entire taxonomy tree?")
def cmd_taxonomy_clear(user: Optional[str]) -> None:
    """Wipe the user's taxonomy tree (nodes + assignments + doc meta)."""
    try:
        cfg = load_config()
        orch = Orchestrator(cfg)
        user_id = user or cfg.user.default_user_id
        if orch.taxonomy_store is None:
            console.print("[red]Error:[/red] taxonomy disabled.")
            sys.exit(1)
        orch.taxonomy_store.clear(user_id)
        orch.db.commit()
        console.print(f"[green]Cleared taxonomy for {user_id}.[/green]")
        orch.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# chat (interactive REPL)
# ---------------------------------------------------------------------------


@cli.command("chat")
@click.option("--user", "-u", default=None, help="User ID (defaults to config value).")
@click.option(
    "--fast",
    is_flag=True,
    default=False,
    help="Fast preset: top_k=4, no rerank. Lowest latency, lowest heat.",
)
@click.option(
    "--no-rerank",
    is_flag=True,
    default=False,
    help="Skip the reranker entirely (fastest, slightly worse quality).",
)
@click.option(
    "--reranker",
    type=click.Choice(["cross_encoder", "llm", "batched_llm"], case_sensitive=False),
    default=None,
    help="Override which reranker to use (config default: cross_encoder).",
)
@click.option(
    "--retriever",
    type=click.Choice(
        ["vector", "bm25", "hybrid", "kg_ppr", "community", "router", "taxonomy"],
        case_sensitive=False,
    ),
    default=None,
    help="Override which retriever to use (config default: vector).",
)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Override top_k_vector (number of candidate chunks before rerank).",
)
@click.option(
    "--gate/--no-gate",
    default=None,
    help="Enable/disable the Phase 4 retrieval gate for this session.",
)
@click.option(
    "--clue/--no-clue",
    default=None,
    help="Enable/disable Phase 4 clue generation for this session.",
)
@click.option(
    "--mask-uncertain/--no-mask-uncertain",
    default=None,
    help="Render visible markers for [UNCERTAIN] tokens for this session.",
)
@click.option(
    "--think/--no-think",
    default=None,
    help="Toggle LLM reasoning tokens (Ollama-only). --no-think suppresses thinking "
    "for big latency wins on short prompts (gate, clue, classifier). Default: config.",
)
def cmd_chat(
    user: Optional[str],
    fast: bool,
    no_rerank: bool,
    reranker: Optional[str],
    retriever: Optional[str],
    top_k: Optional[int],
    gate: Optional[bool],
    clue: Optional[bool],
    mask_uncertain: Optional[bool],
    think: Optional[bool],
) -> None:
    """Open an interactive chat REPL."""
    try:
        cfg = load_config()
        if fast:
            cfg.retrieval.rerank_enabled = False
            cfg.retrieval.top_k_vector = 4
            cfg.retrieval.top_k_final = 4
        if no_rerank:
            cfg.retrieval.rerank_enabled = False
        if reranker is not None:
            cfg.retrieval.reranker = reranker
            # Adjust threshold scale automatically when switching reranker types.
            if reranker in ("llm", "batched_llm"):
                cfg.retrieval.rerank_threshold = 2.0
            else:
                cfg.retrieval.rerank_threshold = 0.0
        if retriever is not None:
            cfg.retrieval.retriever = retriever
        if top_k is not None:
            cfg.retrieval.top_k_vector = top_k
        # Phase 4 compaction overrides
        if gate is not None:
            cfg.compaction.gate_enabled = gate
        if clue is not None:
            cfg.compaction.clue_enabled = clue
        if mask_uncertain is not None:
            cfg.compaction.mask_uncertain = mask_uncertain
        if think is not None:
            cfg.llm.think = think
        orch = Orchestrator(cfg)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Failed to initialise orchestrator: {exc}")
        sys.exit(1)

    try:
        orch.llm.verify_ready()
    except LLMProviderError as exc:
        console.print(f"[red]LLM not ready:[/red] {exc}")
        console.print("[dim]Fix the issue above and run `hrag chat` again.[/dim]")
        orch.close()
        sys.exit(1)

    user_id = user or cfg.user.default_user_id
    session_id: Optional[str] = None
    last_result: Optional[ChatResult] = None

    console.print(
        Panel(
            "[bold cyan]HRAG-Bot[/bold cyan]  –  type [dim]/help[/dim] for commands, "
            "[dim]/exit[/dim] to quit.",
            expand=False,
        )
    )

    def _fire_auto_extract() -> None:
        """Hand the closed session to the background extractor, if enabled."""
        if orch.auto_extractor is not None and session_id is not None:
            try:
                orch.auto_extractor.on_session_close(user_id, session_id)
                console.print(
                    "[dim](auto-extract scheduled for session "
                    f"{session_id[:8]})[/dim]"
                )
            except Exception:  # noqa: BLE001 - never block exit
                pass

    while True:
        try:
            raw = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            _fire_auto_extract()
            break

        if not raw:
            continue

        # ---- Slash commands ----
        if raw.startswith("/"):
            # Preserve case of arguments — only the command name is normalised.
            head, _, args = raw.partition(" ")
            cmd = head.lower()
            args = args.strip()

            if cmd in ("/exit", "/quit"):
                console.print("[dim]Goodbye.[/dim]")
                _fire_auto_extract()
                break

            if cmd == "/help":
                console.print(
                    "[bold]Available commands:[/bold]\n"
                    "  /help              – show this message\n"
                    "  /status            – show session status (alias: /statusline)\n"
                    "  /sources           – show full text of last response's top-5 sources\n"
                    "  /stats             – Show corpus / KG statistics.\n"
                    "  /remember <text>   – save an episodic memory (competes with docs at retrieval)\n"
                    "  /recall <query>    – search episodic memories only (top 5)\n"
                    "  /forget <q|id>     – tombstone matching memory(ies) (asks for confirmation)\n"
                    "  /profile           – show structured user profile\n"
                    "  /profile add <polarity> <topic>=<value>  – manual upsert\n"
                    "  /profile forget <pref_id>                – delete one preference\n"
                    "  /exit, /quit       – exit the REPL"
                )
                continue

            if cmd in ("/status", "/statusline"):
                _print_status(orch, user_id, session_id, last_result)
                continue

            if cmd == "/sources":
                if last_result is None:
                    console.print("[dim]No previous response.[/dim]")
                else:
                    _print_sources_full(last_result.sources[:5])
                continue

            if cmd == "/stats":
                _print_stats(orch, console)
                continue

            # ---- Phase 3 memory commands ----
            if cmd == "/remember":
                if not args:
                    console.print("[dim]usage: /remember <text>[/dim]")
                    continue
                try:
                    memory_id = orch.memory_store.add(
                        user_id, args, session_id=session_id, source="user"
                    )
                    console.print(f"[green][memory] saved[/green] [dim]{memory_id}[/dim]")
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]Error:[/red] {exc}")
                continue

            if cmd == "/recall":
                if not args:
                    console.print("[dim]usage: /recall <query>[/dim]")
                    continue
                try:
                    hits = orch.retriever.retrieve(
                        args, user_id, top_k=5, source_types=["episodic"]
                    )
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]Error:[/red] {exc}")
                    continue
                if not hits:
                    console.print("[dim]No matching memories.[/dim]")
                    continue
                _print_sources_summary(hits)
                continue

            if cmd == "/forget":
                if not args:
                    console.print("[dim]usage: /forget <query-or-chunk-id>[/dim]")
                    continue
                _handle_forget(orch, user_id, args)
                continue

            if cmd == "/profile":
                _handle_profile(orch, user_id, args, session_id)
                continue

            console.print(f"[dim]Unknown command: {raw}[/dim]")
            continue

        # ---- Regular question ----
        try:
            with _PipelineProgress(console) as report:
                result = orch.chat(
                    raw,
                    user_id,
                    session_id=session_id,
                    progress=report,
                    stream=True,
                )
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
            continue
        except LLMProviderError as exc:
            console.print(f"[red]LLM error:[/red] {exc}")
            console.print("[dim]Type your question again to retry, or /exit to quit.[/dim]")
            continue
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            continue

        # Pin session for the rest of this REPL session.
        session_id = result.session_id
        last_result = result

        # Streaming already printed the answer live; just close the line.
        console.print()

        # Show top-3 sources summary
        if result.sources:
            _print_sources_summary(result.sources[:3])
        console.print()

    orch.close()


# ---------------------------------------------------------------------------
# embeddings-list / embeddings-current
# ---------------------------------------------------------------------------


@cli.command("embeddings-list")
def cmd_embeddings_list() -> None:
    """Show the curated embedding model presets.

    Switching ``embeddings.model`` requires ``hrag init && hrag ingest <corpus>``
    to re-embed the entire corpus under the new model. Existing embeddings are NOT
    compatible with a different model.
    """
    from hrag.config import EmbeddingsConfig  # noqa: PLC0415

    cfg = EmbeddingsConfig()
    table = Table(
        title="Curated embedding model presets",
        show_lines=True,
        show_header=True,
    )
    table.add_column("Label", style="bold", no_wrap=False)
    table.add_column("Model ID", style="cyan", no_wrap=False)
    table.add_column("Dim", justify="right", style="magenta")
    for entry in cfg.suggested_models:
        table.add_row(
            entry.get("label", ""),
            entry.get("model", ""),
            str(entry.get("dim", "")),
        )
    console.print(table)
    console.print(
        "[dim]Switching [bold]embeddings.model[/bold] requires "
        "[bold]hrag init && hrag ingest <corpus>[/bold] to re-embed.[/dim]"
    )


@cli.command("embeddings-current")
def cmd_embeddings_current() -> None:
    """Show the currently configured embedding model and its dimension.

    Also performs a best-effort check: reads one vector from ChromaDB and
    compares its length to ``embeddings.dim``. A mismatch suggests the index
    was built under a different model.
    """
    try:
        cfg = load_config()
        from hrag.providers.embeddings import dimension_for_model  # noqa: PLC0415

        known_dim = dimension_for_model(cfg.embeddings.model)
        console.print(f"Model  : [bold]{cfg.embeddings.model}[/bold]")
        console.print(f"Dim    : [bold]{cfg.embeddings.dim}[/bold] (config)")
        if known_dim is not None:
            match = "✓ matches curated list" if known_dim == cfg.embeddings.dim else f"[yellow]⚠ curated list says {known_dim}[/yellow]"
            console.print(f"Verify : {match}")
        else:
            console.print("[dim]Model not in curated list — dimension taken from config as-is.[/dim]")

        # Best-effort: peek at Chroma to verify live index dimension.
        try:
            import chromadb  # noqa: PLC0415

            chroma_path = str(cfg.resolve(cfg.storage.chroma_path))
            client = chromadb.PersistentClient(path=chroma_path)
            col = client.get_or_create_collection("hrag_chunks")
            result = col.get(limit=1, include=["embeddings"])
            embs = result.get("embeddings") or []
            if embs and embs[0] is not None:
                live_dim = len(embs[0])
                status = (
                    "[green]✓ matches config[/green]"
                    if live_dim == cfg.embeddings.dim
                    else f"[red]✗ mismatch — index has {live_dim}d, config says {cfg.embeddings.dim}d. Re-ingest required.[/red]"
                )
                console.print(f"Index  : {live_dim}d — {status}")
            else:
                console.print("[dim]Index  : (no vectors yet)[/dim]")
        except Exception as chroma_err:  # noqa: BLE001
            console.print(f"[dim]Index  : (could not read Chroma — {chroma_err})[/dim]")

    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# list-docs
# ---------------------------------------------------------------------------


@cli.command("list-docs")
@click.option("--user", "-u", default=None, help="User ID (defaults to config value).")
def cmd_list_docs(user: Optional[str]) -> None:
    """List ingested documents from the database."""
    try:
        cfg = load_config()
        db_path = cfg.resolve(cfg.storage.sqlite_path)
        db = init_db(db_path, cfg.user.default_user_id)
        user_id = user or cfg.user.default_user_id

        rows = db.execute(
            """
            SELECT d.doc_id, d.title, d.source_type, d.ingested_at,
                   COUNT(c.chunk_id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.doc_id AND c.excluded = 0
            WHERE d.user_id = ?
            GROUP BY d.doc_id
            ORDER BY d.ingested_at DESC
            """,
            (user_id,),
        ).fetchall()

        if not rows:
            console.print("[dim]No documents found.[/dim]")
            return

        table = Table(title=f"Documents for user [bold]{user_id}[/bold]", show_lines=False)
        table.add_column("Title", style="bold", no_wrap=False)
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Chunks", justify="right", style="magenta")
        table.add_column("Ingested At", style="dim", no_wrap=True)

        for r in rows:
            table.add_row(
                r["title"] or r["doc_id"],
                r["source_type"],
                str(r["chunk_count"]),
                (r["ingested_at"] or "")[:19],
            )

        console.print(table)
        db.close()

    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Source display helpers
# ---------------------------------------------------------------------------


def _print_sources_summary(sources: list[RetrievalResult]) -> None:
    """Print a compact dim panel showing top sources after an answer."""
    lines: list[str] = []
    for i, r in enumerate(sources, start=1):
        chunk = r.chunk
        title = chunk.title or "Untitled"
        section = chunk.section or "—"
        score = (
            f"rerank={r.rerank_score}"
            if r.rerank_score is not None
            else f"score={r.score:.3f}"
        )
        lines.append(f"[{i}] {title} — {section}  [dim]({score})[/dim]")
    console.print(Panel("\n".join(lines), title="[dim]Sources[/dim]", expand=False))


# ---------------------------------------------------------------------------
# /status command
# ---------------------------------------------------------------------------


def _print_status(
    orch: Orchestrator,
    user_id: str,
    session_id: Optional[str],
    last_result: Optional[ChatResult],
) -> None:
    """Show session info: turns, tokens used, budget, indexed docs, model."""
    cfg = orch.config

    # --- session-level stats ---
    if session_id is None:
        turns = 0
        history_tokens = 0
    else:
        rows = orch.db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY message_id",
            (session_id,),
        ).fetchall()
        turns = sum(1 for r in rows if r["role"] == "user")
        # Use the chunker's tokenizer for a consistent count.
        try:
            from hrag.ingest.chunker import count_tokens

            history_tokens = sum(count_tokens(r["content"]) for r in rows)
        except Exception:
            # Fall back to char/4 heuristic.
            history_tokens = sum(len(r["content"]) for r in rows) // 4

    budget = cfg.context.history_token_budget
    pct = (history_tokens / budget * 100.0) if budget > 0 else 0.0
    bar = _PipelineProgress._make_bar(min(history_tokens, budget), max(budget, 1), width=24)

    # --- corpus stats for this user ---
    docs_row = orch.db.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE user_id = ?", (user_id,)
    ).fetchone()
    chunks_row = orch.db.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE user_id = ? AND excluded = 0",
        (user_id,),
    ).fetchone()

    table = Table(title="[bold]Session status[/bold]", show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()

    table.add_row("user", user_id)
    table.add_row("session", (session_id or "(none — first turn will create one)")[:16])
    table.add_row("turns", str(turns))
    table.add_row(
        "history tokens",
        f"{history_tokens} / {budget}  {bar}  [bold]{pct:.0f}%[/bold]",
    )
    table.add_row("indexed docs", str(docs_row["n"]))
    table.add_row("indexed chunks", str(chunks_row["n"]))
    table.add_row("llm", f"{cfg.llm.provider} / {cfg.llm.model}")
    table.add_row("embedder", f"{cfg.embeddings.provider} / {cfg.embeddings.model}")
    table.add_row("retriever", cfg.retrieval.retriever)
    table.add_row(
        "reranker",
        cfg.retrieval.reranker if cfg.retrieval.rerank_enabled else "off",
    )
    table.add_row(
        "k",
        f"top_k_vector={cfg.retrieval.top_k_vector}, top_k_final={cfg.retrieval.top_k_final}, "
        f"threshold={cfg.retrieval.rerank_threshold}",
    )
    if last_result is not None:
        table.add_row("last sources", str(len(last_result.sources)))
    console.print(table)


# ---------------------------------------------------------------------------
# Live pipeline progress display
# ---------------------------------------------------------------------------


class _PipelineProgress:
    """Context manager that prints live progress for an Orchestrator.chat() call.

    Use as:
        with _PipelineProgress(console) as report:
            result = orch.chat(question, user_id, progress=report)

    Shows: retrieval timing, per-chunk rerank scores with progress bar,
    LLM generation timing, and a total at the end.
    """

    def __init__(self, console_obj: Console) -> None:
        self._console = console_obj
        self._status = None
        self._rerank_seen = 0
        self._rerank_total = 0

    def __enter__(self) -> "_PipelineProgress":
        from rich.status import Status

        self._status = Status("Starting...", console=self._console, spinner="dots")
        self._status.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._status is not None:
            self._status.__exit__(exc_type, exc, tb)
            self._status = None

    def __call__(self, event: str, payload: dict) -> None:
        """Receives orchestrator events. Signature matches ProgressCallback."""
        if event == "start":
            self._set("[cyan]embedding the question (~1 sentence) + vector search...[/cyan]")
        elif event == "query_rewrite":
            rewriter = payload.get("rewriter", "?")
            rewritten = payload.get("rewritten", "").replace("\n", " | ")
            self._console.print(
                f"  [dim]rewrite ({rewriter}):[/dim] {rewritten}"
            )
        elif event == "retrieve":
            self._console.print(
                f"  [dim]retrieve:[/dim] {payload['n_results']} candidates "
                f"in [bold]{payload['duration_s']:.2f}s[/bold]"
            )
            self._rerank_total = payload["n_results"]
            self._rerank_seen = 0
        elif event == "rerank_step":
            self._rerank_seen = payload["i"]
            n = payload["n"]
            score = payload["score"]
            bar = self._make_bar(self._rerank_seen, n)
            score_str = (
                f"{score:.2f}" if isinstance(score, float) else str(score)
            )
            self._set(
                f"[cyan]reranking[/cyan] {bar} "
                f"{self._rerank_seen}/{n}  "
                f"[dim](latest score: {score_str})[/dim]"
            )
        elif event == "rerank_done":
            fallback = payload.get("fallback_used")
            note = (
                f"  [yellow](rerank dropped all; fell back to top-{payload['kept']} vector)[/yellow]"
                if fallback
                else ""
            )
            self._console.print(
                f"  [dim]rerank:[/dim] kept [bold]{payload['kept']}[/bold] "
                f"of {self._rerank_total} in [bold]{payload['duration_s']:.2f}s[/bold]{note}"
            )
            self._set("[cyan]calling LLM (Gemma)...[/cyan]")
        elif event == "generate_start":
            # Stop the spinner so streamed tokens print cleanly below.
            if self._status is not None:
                self._status.__exit__(None, None, None)
                self._status = None
            self._console.print()
            self._console.print("[bold green]Assistant:[/bold green]")
        elif event == "generate_token":
            # Print token without newline, no markup so it renders verbatim.
            token = payload.get("token", "")
            if token:
                self._console.print(token, end="", soft_wrap=True, highlight=False, markup=False)
        elif event == "generate":
            self._console.print()  # newline after streamed answer
            self._console.print(
                f"  [dim]generate:[/dim] {payload['answer_chars']} chars "
                f"in [bold]{payload['duration_s']:.2f}s[/bold]"
            )
        elif event == "done":
            self._console.print(
                f"  [dim]TOTAL:[/dim] [bold green]{payload['total_s']:.2f}s[/bold green]"
            )

    def _set(self, msg: str) -> None:
        if self._status is not None:
            self._status.update(msg)

    @staticmethod
    def _make_bar(done: int, total: int, width: int = 20) -> str:
        if total <= 0:
            return ""
        filled = int(width * done / total)
        return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_stats(orch: Orchestrator, console: Console) -> None:
    """Print corpus-level stats from SQLite (no LLM, no retrieval)."""
    db = orch.db
    cfg = orch.config

    # Document counts by source_type
    doc_rows = db.execute(
        "SELECT source_type, COUNT(*) FROM documents GROUP BY source_type"
    ).fetchall()
    chunks_total = db.execute(
        "SELECT COUNT(*) FROM chunks WHERE excluded=0"
    ).fetchone()[0]
    docs_total = db.execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]

    # KG counts (zero if kg.enabled=False)
    phrase_n = db.execute(
        "SELECT COUNT(*) FROM kg_nodes WHERE node_type='phrase'"
    ).fetchone()[0]
    passage_n = db.execute(
        "SELECT COUNT(*) FROM kg_nodes WHERE node_type='passage'"
    ).fetchone()[0]
    edges_n = db.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
    communities_n = db.execute("SELECT COUNT(*) FROM kg_communities").fetchone()[0]

    # Per-doc chunk count (top 10)
    per_doc = db.execute(
        """
        SELECT d.title, COUNT(*) AS n
        FROM chunks ch JOIN documents d ON ch.doc_id = d.doc_id
        WHERE ch.excluded = 0
        GROUP BY d.title
        ORDER BY n DESC
        LIMIT 10
        """
    ).fetchall()

    t = Table(title="Corpus stats", show_lines=False)
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("Documents (total)", str(docs_total))
    for row in doc_rows:
        t.add_row(f"  by source_type={row[0]}", str(row[1]))
    t.add_row("Active chunks", str(chunks_total))
    t.add_row("KG enabled", str(cfg.kg.enabled))
    if cfg.kg.enabled:
        t.add_row("KG phrase nodes", str(phrase_n))
        t.add_row("KG passage nodes", str(passage_n))
        t.add_row("KG edges", str(edges_n))
        t.add_row("Communities", str(communities_n))
    console.print(t)

    if per_doc:
        t2 = Table(title="Top docs by chunk count", show_lines=False)
        t2.add_column("title")
        t2.add_column("chunks", justify="right")
        for title, n in per_doc:
            t2.add_row(title or "(untitled)", str(n))
        console.print(t2)


def _print_sources_full(sources: list[RetrievalResult]) -> None:
    """Print full source text for /sources command."""
    for i, r in enumerate(sources, start=1):
        chunk = r.chunk
        rr = (
            f", rerank={r.rerank_score:.2f}"
            if r.rerank_score is not None
            else ""
        )
        header = (
            f"[bold][Source {i}][/bold]  "
            f"{chunk.title or 'Untitled'} — {chunk.section or 'N/A'}  "
            f"[dim](vec={r.score:.3f}{rr}, {chunk.token_count}tok)[/dim]"
        )
        console.print(Panel(chunk.text, title=header, expand=False))


# ---------------------------------------------------------------------------
# Phase 3 — /forget and /profile slash-command helpers
# ---------------------------------------------------------------------------


_HEX_RE = __import__("re").compile(r"^[0-9a-f]{8,}$")


def _handle_forget(orch: Orchestrator, user_id: str, arg: str) -> None:
    """Resolve and tombstone a memory by chunk_id, memory_id, or semantic query.

    Resolution order:
      1. Exact chunk_id match → forget that one chunk.
      2. ``episodic:...`` memory_id prefix → forget the whole memory.
      3. Anything else → semantic /recall over episodic, ask for confirmation.
    """
    if arg.startswith("episodic:"):
        n = orch.memory_store.forget_memory(user_id, arg)
        if n == 0:
            console.print("[dim]No matching memory.[/dim]")
        else:
            console.print(f"[green]Forgot {n} chunk(s) under {arg}.[/green]")
        return

    # Plausible chunk_id (long hex). Resolve via SQL.
    if _HEX_RE.match(arg.lower()):
        ok = orch.memory_store.forget(user_id, arg)
        if ok:
            console.print(f"[green]Forgot chunk[/green] [dim]{arg}[/dim]")
        else:
            console.print("[dim]No chunk with that id.[/dim]")
        return

    # Fall back to semantic match.
    chunk_ids = orch.memory_store.forget_by_query(
        user_id, arg, top_k=3, retriever=orch.retriever
    )
    if not chunk_ids:
        console.print("[dim]No matching memories.[/dim]")
        return

    if orch.config.memory.forget_confirm:
        console.print(f"Forget {len(chunk_ids)} matching memory chunk(s)? [y/N]: ", end="")
        try:
            ans = console.input("").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return
        if ans not in ("y", "yes"):
            console.print("[dim]Cancelled.[/dim]")
            return

    for cid in chunk_ids:
        orch.memory_store.forget(user_id, cid)
    console.print(f"[green]Tombstoned {len(chunk_ids)} chunk(s).[/green]")


def _handle_profile(
    orch: Orchestrator,
    user_id: str,
    args: str,
    session_id: Optional[str],
) -> None:
    """Dispatch /profile, /profile add <polarity> <topic>=<value>, /profile forget <id>."""
    if not args:
        prefs = orch.profile_store.list_all(user_id)
        if not prefs:
            console.print("[dim](no profile yet)[/dim]")
            return
        table = Table(title=f"Profile for [bold]{user_id}[/bold]", show_lines=False)
        table.add_column("id", style="dim", justify="right")
        table.add_column("polarity", style="cyan")
        table.add_column("topic", style="bold")
        table.add_column("value")
        table.add_column("conf", justify="right")
        table.add_column("updated", style="dim")
        for p in prefs:
            table.add_row(
                str(p.pref_id),
                p.polarity,
                p.topic,
                p.value,
                f"{p.confidence:.2f}",
                (p.last_updated or "")[:19],
            )
        console.print(table)
        return

    sub, _, rest = args.partition(" ")
    sub = sub.lower()
    rest = rest.strip()

    if sub == "forget":
        try:
            pref_id = int(rest)
        except ValueError:
            console.print("[dim]usage: /profile forget <pref_id>[/dim]")
            return
        if orch.profile_store.delete(user_id, pref_id):
            console.print(f"[green]Deleted preference {pref_id}.[/green]")
        else:
            console.print("[dim]No preference with that id.[/dim]")
        return

    if sub == "add":
        # Expected: <polarity> <topic>=<value>
        if not rest:
            console.print('[dim]usage: /profile add <polarity> <topic>="<value>"[/dim]')
            return
        pol, _, kv = rest.partition(" ")
        pol = pol.strip().lower()
        topic_raw, eq, value = kv.partition("=")
        if not eq or not topic_raw.strip():
            console.print('[dim]usage: /profile add <polarity> <topic>="<value>"[/dim]')
            return
        topic = topic_raw.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")
        try:
            pid = orch.profile_store.upsert(
                user_id=user_id,
                polarity=pol,
                topic=topic,
                value=value,
                confidence=1.0,
                source_session_id=session_id,
            )
            console.print(f"[green]Upserted preference {pid}.[/green]")
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
        return

    console.print(f"[dim]Unknown /profile subcommand: {sub!r}[/dim]")


# ---------------------------------------------------------------------------
# Phase 3 — top-level commands: hrag remember, hrag memory list, hrag memory extract
# ---------------------------------------------------------------------------


_TEXT_EXT = {".md", ".markdown", ".txt"}


def _iter_memory_texts_from_path(
    path: Path,
    *,
    split_paragraphs: bool,
) -> list[dict]:
    """Walk *path* and return a list of memory items.

    Each item is ``{"text": str, "title": str | None, "source_path": str}``.
    For ``.md`` files when ``split_paragraphs`` is true, splits on lines that
    start with ``# `` or ``## ``. For ``.txt``, each non-empty line is one
    memory. Directories recurse over ``.md`` / ``.markdown`` / ``.txt``.
    """
    items: list[dict] = []

    def _emit_md(file: Path) -> None:
        raw = file.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            return
        if not split_paragraphs:
            items.append({"text": raw, "title": file.stem, "source_path": str(file)})
            return
        # Split on headings ``# ...`` / ``## ...`` etc.
        import re  # noqa: PLC0415

        chunks = re.split(r"(?m)^#{1,6}\s+", raw)
        # First chunk is anything before the first heading; the rest are
        # heading bodies with the heading text consumed by the regex.
        for piece in chunks:
            piece = piece.strip()
            if not piece:
                continue
            head_line = piece.split("\n", 1)[0].strip()
            items.append(
                {
                    "text": piece,
                    "title": head_line[:80] or file.stem,
                    "source_path": str(file),
                }
            )

    def _emit_txt(file: Path) -> None:
        for ln in file.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if ln:
                items.append({"text": ln, "title": None, "source_path": str(file)})

    def _emit_file(file: Path) -> None:
        ext = file.suffix.lower()
        if ext in (".md", ".markdown"):
            _emit_md(file)
        elif ext == ".txt":
            _emit_txt(file)

    if path.is_file():
        _emit_file(path)
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in _TEXT_EXT:
                _emit_file(p)
    return items


@cli.command("remember")
@click.argument("target", type=str)
@click.option("--user", "-u", default=None, help="User ID (defaults to config value).")
@click.option(
    "--title",
    default=None,
    help="Override the auto-derived title (only used when TARGET is inline text).",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Repeatable tag attached to each saved memory.",
)
@click.option(
    "--per-file",
    "per_file",
    is_flag=True,
    default=False,
    help="One memory per file (default: split on ## headings for .md, per-line for .txt).",
)
def cmd_remember(
    target: str,
    user: Optional[str],
    title: Optional[str],
    tags: tuple[str, ...],
    per_file: bool,
) -> None:
    """Save one or many episodic memories.

    TARGET can be:
      • inline text       e.g. `hrag remember "Postgres preferred over MySQL"`
      • a .md / .txt file
      • a directory walked recursively for .md / .txt
    """
    try:
        cfg = load_config()
        orch = Orchestrator(cfg)
        user_id = user or cfg.user.default_user_id

        path = Path(target)
        if path.exists():
            split_paragraphs = (not per_file) and cfg.memory.bulk_chunk_per_paragraph
            items = _iter_memory_texts_from_path(
                path, split_paragraphs=split_paragraphs
            )
            for item in items:
                if tags:
                    item["tags"] = list(tags)
            if not items:
                console.print("[yellow]No .md or .txt content found at that path.[/yellow]")
                orch.close()
                return

            from rich.progress import (  # noqa: PLC0415
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            progress = Progress(
                TextColumn("[bold]remember[/bold]"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("• ETA"),
                TimeRemainingColumn(),
                console=console,
            )
            saved = 0
            with progress:
                task = progress.add_task("save", total=len(items))
                for item in items:
                    try:
                        orch.memory_store.add(
                            user_id,
                            item["text"],
                            title=item.get("title"),
                            tags=item.get("tags"),
                            source="bulk",
                        )
                        saved += 1
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"[red]skip[/red] {item.get('source_path','?')}: {exc}")
                    progress.advance(task)
            console.print(
                f"[green]Saved {saved} memor{'y' if saved == 1 else 'ies'}[/green] "
                f"from [bold]{path}[/bold]."
            )
        else:
            # Treat the argument as inline text.
            memory_id = orch.memory_store.add(
                user_id,
                target,
                title=title,
                tags=list(tags) if tags else None,
                source="user",
            )
            console.print(f"[green][memory] saved[/green] [dim]{memory_id}[/dim]")

        orch.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@cli.group("memory")
def cmd_memory() -> None:
    """Inspect or extract per-user memory."""


@cmd_memory.command("list")
@click.option("--user", "-u", default=None, help="User ID (defaults to config value).")
@click.option("--limit", "-n", default=20, type=int, help="Max rows to show.")
def cmd_memory_list(user: Optional[str], limit: int) -> None:
    """List recently-added episodic memories for a user."""
    try:
        cfg = load_config()
        db_path = cfg.resolve(cfg.storage.sqlite_path)
        db = init_db(db_path, cfg.user.default_user_id)
        user_id = user or cfg.user.default_user_id

        rows = db.execute(
            """
            SELECT c.chunk_id, c.title, c.text, c.token_count, d.ingested_at
            FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.user_id = ? AND c.source_type = 'episodic' AND c.excluded = 0
            ORDER BY d.ingested_at DESC, c.chunk_index ASC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

        if not rows:
            console.print("[dim]No episodic memories yet.[/dim]")
            return

        table = Table(title=f"Memories for user [bold]{user_id}[/bold]", show_lines=False)
        table.add_column("chunk_id", style="dim")
        table.add_column("title", style="bold")
        table.add_column("text")
        table.add_column("tok", justify="right", style="magenta")
        table.add_column("ingested", style="dim")
        for r in rows:
            preview = (r["text"] or "")[:120].replace("\n", " ")
            table.add_row(
                (r["chunk_id"] or "")[:16],
                r["title"] or "—",
                preview,
                str(r["token_count"]),
                (r["ingested_at"] or "")[:19],
            )
        console.print(table)
        db.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@cmd_memory.command("extract")
@click.option("--session", "session_id", required=True, help="Session id to mine.")
@click.option("--user", "-u", default=None, help="User ID (defaults to config value).")
@click.option(
    "--auto-apply",
    is_flag=True,
    default=False,
    help="Upsert every candidate above min_confidence without asking.",
)
def cmd_memory_extract(
    session_id: str,
    user: Optional[str],
    auto_apply: bool,
) -> None:
    """Run PreferenceExtractor over a session's messages and upsert preferences."""
    try:
        cfg = load_config()
        orch = Orchestrator(cfg)
        user_id = user or cfg.user.default_user_id

        from hrag.memory.extractor import PreferenceExtractor  # noqa: PLC0415

        rows = orch.db.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? AND user_id = ? ORDER BY message_id",
            (session_id, user_id),
        ).fetchall()
        if not rows:
            console.print("[dim]No messages for that session.[/dim]")
            orch.close()
            return

        conversation = [(r["role"], r["content"]) for r in rows]
        extractor = PreferenceExtractor(orch.llm)
        console.print(
            f"[dim]Extracting from {len(conversation)} messages...[/dim]"
        )
        candidates = extractor.extract(conversation)
        min_conf = cfg.memory.auto_extract_min_confidence

        if not candidates:
            console.print("[dim]No preferences detected.[/dim]")
            orch.close()
            return

        applied = 0
        for c in candidates:
            ok_conf = c.confidence >= min_conf
            badge = "[green]ok[/green]" if ok_conf else "[yellow]low-conf[/yellow]"
            console.print(
                f"  {badge} [cyan]{c.polarity}[/cyan] {c.topic} = {c.value!r} "
                f"[dim](conf={c.confidence:.2f})[/dim]"
            )

            if not ok_conf:
                continue
            if not auto_apply:
                ans = console.input("    Apply? [Y/n]: ").strip().lower()
                if ans in ("n", "no"):
                    continue
            orch.profile_store.upsert(
                user_id=user_id,
                polarity=c.polarity,
                topic=c.topic,
                value=c.value,
                confidence=c.confidence,
                source_session_id=session_id,
            )
            applied += 1

        console.print(f"[green]Upserted {applied} preference(s).[/green]")
        orch.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Feedback analytics helpers  (Phase 6-B2)
# ---------------------------------------------------------------------------


def _feedback_summary(db) -> dict:
    """Return aggregate feedback statistics.

    Delegates to ``hrag.feedback_stats.feedback_summary`` — kept here as a
    thin alias so existing CLI code that calls ``_feedback_summary(db)``
    continues to work without change.
    """
    from hrag.feedback_stats import feedback_summary  # noqa: PLC0415
    return feedback_summary(db)


# ---------------------------------------------------------------------------
# feedback-stats  (Phase 6-B2)
# ---------------------------------------------------------------------------


@cli.command("feedback-stats")
@click.option("--user", "-u", default=None, help="Filter to this user_id (default: all users).")
def cmd_feedback_stats(user: Optional[str]) -> None:
    """Print aggregate thumbs-up / thumbs-down statistics.

    Shows total counts, ratio, and the top 5 most-recently negatively-rated
    questions (by joining feedback to the preceding user message in the same
    session).
    """
    try:
        cfg = load_config()
        db_path = cfg.resolve(cfg.storage.sqlite_path)
        console.print("[dim]reading schema + querying feedback table...[/dim]")
        db = init_db(db_path, cfg.user.default_user_id)

        # If user filter specified, restrict to that user.
        if user:
            # Temporarily narrow the view via an application-level filter.
            # We do this by overriding the queries with a user filter rather
            # than monkey-patching _feedback_summary — simpler and testable.
            up_row = db.execute(
                "SELECT COUNT(*) AS n FROM feedback WHERE rating = 1 AND user_id = ?",
                (user,),
            ).fetchone()
            down_row = db.execute(
                "SELECT COUNT(*) AS n FROM feedback WHERE rating = -1 AND user_id = ?",
                (user,),
            ).fetchone()
            sessions_row = db.execute(
                "SELECT COUNT(DISTINCT session_id) AS n FROM feedback "
                "WHERE rating != 0 AND user_id = ?",
                (user,),
            ).fetchone()
            neg_rows = db.execute(
                """
                SELECT f.message_id, f.session_id, f.created_at
                FROM feedback f
                WHERE f.rating = -1 AND f.user_id = ?
                ORDER BY f.created_at DESC
                LIMIT 5
                """,
                (user,),
            ).fetchall()
            thumbs_up = up_row["n"] if up_row else 0
            thumbs_down = down_row["n"] if down_row else 0
            total = thumbs_up + thumbs_down
            sessions = sessions_row["n"] if sessions_row else 0
            top_negative: list[dict] = []
            for row in neg_rows:
                user_msg_row = db.execute(
                    """
                    SELECT content FROM messages
                    WHERE session_id = ? AND role = 'user'
                      AND message_id < CAST(? AS INTEGER)
                    ORDER BY message_id DESC
                    LIMIT 1
                    """,
                    (row["session_id"], row["message_id"]),
                ).fetchone()
                question = (
                    user_msg_row["content"] if user_msg_row else "(question not found)"
                )
                top_negative.append(
                    {
                        "question": question,
                        "session_id": row["session_id"],
                        "created_at": row["created_at"] or "",
                    }
                )
            stats = {
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "total": total,
                "sessions": sessions,
                "top_negative": top_negative,
            }
        else:
            stats = _feedback_summary(db)

        up = stats["thumbs_up"]
        down = stats["thumbs_down"]
        total = stats["total"]
        ratio_up = (up / total * 100.0) if total > 0 else 0.0
        ratio_down = (down / total * 100.0) if total > 0 else 0.0

        summary_lines = [
            f"[green]+1 thumbs up:[/green]   {up:>6}  ({ratio_up:.1f}%)",
            f"[red]-1 thumbs down:[/red]  {down:>6}  ({ratio_down:.1f}%)",
            f"[bold]Total:[/bold]           {total:>6}   across {stats['sessions']} session(s)",
        ]
        console.print(
            Panel(
                "\n".join(summary_lines),
                title="[bold]Feedback summary[/bold]",
                expand=False,
            )
        )

        if stats["top_negative"]:
            console.print("\n[bold]Top 5 negatively-rated questions:[/bold]")
            for item in stats["top_negative"]:
                q = (item["question"] or "").replace("\n", " ")[:120]
                sid = (item["session_id"] or "")[:12]
                ts = (item["created_at"] or "")[:10]
                console.print(f'  [dim]•[/dim] "{q}" [dim](session {sid}, {ts})[/dim]')
        else:
            console.print("[dim]No negative feedback yet.[/dim]")

        db.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# feedback-export  (Phase 6-B2)
# ---------------------------------------------------------------------------


@cli.command("feedback-export")
@click.option(
    "--rating",
    "rating_filter",
    type=click.Choice(["up", "down"], case_sensitive=False),
    required=True,
    help="Export only thumbs-up ('up', rating=+1) or thumbs-down ('down', rating=-1) rows.",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(writable=True, dir_okay=False),
    help="Output JSONL file path.",
)
@click.option("--user", "-u", default=None, help="Filter to this user_id (default: all users).")
@click.option(
    "--limit",
    "limit",
    type=int,
    default=None,
    help="Maximum number of rows to export (default: unlimited).",
)
def cmd_feedback_export(
    rating_filter: str,
    out: str,
    user: Optional[str],
    limit: Optional[int],
) -> None:
    """Export feedback-filtered assistant messages as JSONL.

    Output format matches export-training-pairs: one JSON object per line with
    user_message, assistant_message, rating, note, session_id, created_at, and
    retrieved_chunks. Only rows with the specified rating (+1 for 'up', -1 for
    'down') are written.
    """
    import json as _json  # noqa: PLC0415

    try:
        cfg = load_config()
        db_path = cfg.resolve(cfg.storage.sqlite_path)
        console.print(f"[dim]querying feedback (rating={rating_filter!r})...[/dim]")
        db = init_db(db_path, cfg.user.default_user_id)

        rating_int = 1 if rating_filter.lower() == "up" else -1

        # Build query — optionally filter by user and limit.
        limit_clause = f"LIMIT {limit}" if limit is not None else ""
        if user:
            rows = db.execute(
                f"""
                SELECT f.feedback_id, f.message_id, f.session_id, f.user_id,
                       f.rating, f.note, f.created_at,
                       m.content AS assistant_message
                FROM feedback f
                JOIN messages m ON m.message_id = CAST(f.message_id AS INTEGER)
                WHERE f.rating = ? AND f.user_id = ?
                ORDER BY f.created_at ASC
                {limit_clause}
                """,
                (rating_int, user),
            ).fetchall()
        else:
            rows = db.execute(
                f"""
                SELECT f.feedback_id, f.message_id, f.session_id, f.user_id,
                       f.rating, f.note, f.created_at,
                       m.content AS assistant_message
                FROM feedback f
                JOIN messages m ON m.message_id = CAST(f.message_id AS INTEGER)
                WHERE f.rating = ?
                ORDER BY f.created_at ASC
                {limit_clause}
                """,
                (rating_int,),
            ).fetchall()

        if not rows:
            console.print("[dim]No feedback rows matched — JSONL file will be empty.[/dim]")

        out_path = Path(out)
        written = 0

        from rich.progress import (  # noqa: PLC0415
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            TextColumn("[bold]feedback-export[/bold]"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            disable=len(rows) < 2,
        )

        with out_path.open("w", encoding="utf-8") as fh, progress:
            task = progress.add_task("rows", total=len(rows))
            for r in rows:
                user_msg_row = db.execute(
                    """
                    SELECT content FROM messages
                    WHERE session_id = ? AND role = 'user'
                      AND message_id < CAST(? AS INTEGER)
                    ORDER BY message_id DESC
                    LIMIT 1
                    """,
                    (r["session_id"], r["message_id"]),
                ).fetchone()
                user_message = user_msg_row["content"] if user_msg_row else ""
                record = {
                    "message_id": r["message_id"],
                    "user_message": user_message,
                    "assistant_message": r["assistant_message"],
                    "rating": r["rating"],
                    "note": r["note"],
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "retrieved_chunks": [],
                }
                fh.write(_json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                progress.advance(task)

        console.print(
            f"[green]Exported {written} row(s)[/green] to [bold]{out_path}[/bold]."
        )
        db.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# export-training-pairs  (Track D1)
# ---------------------------------------------------------------------------


@cli.command("export-training-pairs")
@click.option(
    "--out",
    required=True,
    type=click.Path(writable=True, dir_okay=False),
    help="Output JSONL file path (one JSON object per line).",
)
@click.option("--user", "-u", default=None, help="Filter to this user_id (default: all users).")
@click.option(
    "--min-rating",
    type=int,
    default=-1,
    show_default=True,
    help="Minimum rating to include (-1 = all rated, 0 = only neutral+good, 1 = only thumbs-up).",
)
def cmd_export_training_pairs(out: str, user: Optional[str], min_rating: int) -> None:
    """Export rated assistant messages as training pairs to a JSONL file.

    Each line is a JSON object with user_message, assistant_message, rating,
    note, session_id, created_at, and an empty retrieved_chunks list (reserved
    for future per-message source persistence).
    """
    import json as _json  # noqa: PLC0415

    try:
        cfg = load_config()
        db_path = cfg.resolve(cfg.storage.sqlite_path)
        db = init_db(db_path, cfg.user.default_user_id)

        # Build query — optionally filter by user.
        if user:
            rows = db.execute(
                """
                SELECT f.feedback_id, f.message_id, f.session_id, f.user_id,
                       f.rating, f.note, f.created_at,
                       m.content AS assistant_message
                FROM feedback f
                JOIN messages m ON m.message_id = CAST(f.message_id AS INTEGER)
                WHERE f.rating >= ? AND f.user_id = ?
                ORDER BY f.created_at ASC
                """,
                (min_rating, user),
            ).fetchall()
        else:
            rows = db.execute(
                """
                SELECT f.feedback_id, f.message_id, f.session_id, f.user_id,
                       f.rating, f.note, f.created_at,
                       m.content AS assistant_message
                FROM feedback f
                JOIN messages m ON m.message_id = CAST(f.message_id AS INTEGER)
                WHERE f.rating >= ?
                ORDER BY f.created_at ASC
                """,
                (min_rating,),
            ).fetchall()

        if not rows:
            console.print("[dim]No feedback rows matched — JSONL file will be empty.[/dim]")

        out_path = Path(out)
        written = 0

        from rich.progress import (  # noqa: PLC0415
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            TextColumn("[bold]export[/bold]"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            disable=len(rows) < 2,
        )

        with out_path.open("w", encoding="utf-8") as fh, progress:
            task = progress.add_task("rows", total=len(rows))
            for r in rows:
                # Look up the preceding user message in the same session.
                user_msg_row = db.execute(
                    """
                    SELECT content FROM messages
                    WHERE session_id = ? AND role = 'user'
                      AND message_id < CAST(? AS INTEGER)
                    ORDER BY message_id DESC
                    LIMIT 1
                    """,
                    (r["session_id"], r["message_id"]),
                ).fetchone()
                user_message = user_msg_row["content"] if user_msg_row else ""
                record = {
                    "message_id": r["message_id"],
                    "user_message": user_message,
                    "assistant_message": r["assistant_message"],
                    "rating": r["rating"],
                    "note": r["note"],
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "retrieved_chunks": [],  # reserved — not yet persisted per message
                }
                fh.write(_json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                progress.advance(task)

        console.print(
            f"[green]Exported {written} training pair(s)[/green] to [bold]{out_path}[/bold]."
        )
        db.close()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# gui
# ---------------------------------------------------------------------------


@cli.command("gui")
@click.option("--host", default="localhost", help="Bind address.")
@click.option("--port", default=8501, type=int, help="Port to serve on.")
@click.option(
    "--browser/--no-browser",
    default=True,
    help="Auto-open the browser when the server is ready.",
)
def cmd_gui(host: str, port: int, browser: bool) -> None:
    """Launch the Streamlit dashboard.

    Requires the ``gui`` optional dependencies: ``pip install -e .[gui]``.
    Shells out to ``streamlit run`` because Streamlit must own the process.
    """
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    try:
        import streamlit  # noqa: F401, PLC0415
    except ImportError:
        console.print(
            "[red]Streamlit is not installed.[/red]\n"
            "Install the GUI extras: [bold]pip install -e .[gui][/bold]"
        )
        sys.exit(1)

    streamlit_bin = shutil.which("streamlit")
    if streamlit_bin is None:
        console.print(
            "[red]Could not locate the 'streamlit' executable on PATH.[/red]\n"
            "Try running [bold]python -m streamlit run <app>[/bold] manually."
        )
        sys.exit(1)

    app_path = Path(__file__).parent / "gui" / "app.py"
    if not app_path.exists():
        console.print(f"[red]GUI app missing at {app_path}[/red]")
        sys.exit(1)

    cmd = [
        streamlit_bin,
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "false" if browser else "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    console.print(f"[dim]launching:[/dim] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        console.print("[dim]GUI stopped.[/dim]")


# ---------------------------------------------------------------------------
# web — Claude/ChatGPT-style FastAPI SPA (replaces Streamlit)
# ---------------------------------------------------------------------------


@cli.command("web")
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", default=8000, type=int, help="Port to serve on.")
@click.option("--reload/--no-reload", default=False, help="Auto-reload on source change (dev only).")
@click.option("--browser/--no-browser", default=True, help="Auto-open the browser.")
def cmd_web(host: str, port: int, reload: bool, browser: bool) -> None:
    """Launch the FastAPI-backed web chat UI.

    Replaces the legacy Streamlit GUI with a hand-crafted SPA modelled on
    claude.ai / ChatGPT. Streams tokens over SSE; supports markdown, code,
    sources, sessions, and live settings.
    """
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        console.print(
            "[red]uvicorn is not installed.[/red]\n"
            "Install with: [bold]pip install fastapi uvicorn[standard][/bold]"
        )
        sys.exit(1)

    import webbrowser  # noqa: PLC0415
    import threading  # noqa: PLC0415

    url = f"http://{host}:{port}"
    console.print(f"[bold]HRAG web[/bold] — serving at [cyan]{url}[/cyan]")
    if browser:
        # Open in the default browser after a short delay so the server is up.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(
            "hrag.web.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=False,
        )
    except KeyboardInterrupt:
        console.print("[dim]web stopped.[/dim]")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
