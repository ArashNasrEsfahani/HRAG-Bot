"""Episodic memories — list, semantic search, /remember, /forget."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from hrag.gui.state import (
    apply_chrome,
    chip,
    current_user_id,
    empty_state,
    get_orchestrator,
    kpi_card,
    page_header,
)


def _render() -> None:
    apply_chrome(page_icon="📚", page_title="Memories · HRAG-Bot")
    page_header(
        "📚 Memories",
        icon="📚",
        subtitle="Episodic notes saved via /remember. They compete with documents in retrieval.",
        tips=[
            "Bulk-import a folder by dragging <code>.md</code> / <code>.txt</code> files into <b>➕ Add</b>.",
            "Markdown files split on <code>## </code> headings; toggle <b>One memory per file</b> to disable.",
            "<b>🔍 Search</b> uses the same retriever the chat path uses — what you see here is what the bot retrieves.",
            "Tombstoned memories (<code>excluded=1</code>) are preserved in SQLite but hidden from retrieval.",
        ],
    )

    orch = get_orchestrator()
    user_id = current_user_id()

    # ----- Add tab / Browse tab / Search tab -----
    tab_browse, tab_add, tab_search = st.tabs(["Browse", "➕ Add", "🔍 Search"])

    # ---- Browse -------------------------------------------------------------
    with tab_browse:
        # KPI strip
        try:
            stats = orch.db.execute(
                "SELECT "
                "SUM(CASE WHEN c.excluded=0 THEN 1 ELSE 0 END) AS active, "
                "SUM(CASE WHEN c.excluded=1 THEN 1 ELSE 0 END) AS tomb, "
                "MAX(d.ingested_at) AS last_at "
                "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id "
                "WHERE c.user_id = ? AND c.source_type = 'episodic'",
                (user_id,),
            ).fetchone()
            active_count = int(stats["active"] or 0)
            tomb_count = int(stats["tomb"] or 0)
            last_at_raw = stats["last_at"]
            last_at = last_at_raw[:10] if last_at_raw else "—"
        except Exception:  # noqa: BLE001
            active_count, tomb_count, last_at = 0, 0, "—"

        kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
        kpi_card(kpi_c1, label="Active memories", value=active_count, icon="📚")
        kpi_card(kpi_c2, label="Tombstoned", value=tomb_count, icon="🗑️", tone="warn")
        kpi_card(kpi_c3, label="Last added", value=last_at, icon="🕒", tone="info")

        col_l, col_r = st.columns([3, 1])
        limit = col_r.number_input("Show", min_value=10, max_value=2000, value=100, step=10)
        include_excluded = col_r.checkbox("Show tombstoned (excluded)", value=False)

        where_excluded = "" if include_excluded else "AND c.excluded = 0"
        rows = orch.db.execute(
            f"""
            SELECT c.chunk_id, c.doc_id, c.title, c.text, c.token_count,
                   c.excluded, d.ingested_at
            FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.user_id = ? AND c.source_type = 'episodic' {where_excluded}
            ORDER BY d.ingested_at DESC, c.chunk_index ASC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

        if not rows:
            with col_l:
                empty_state(
                    icon="📚",
                    title="No memories yet",
                    message="Save your first one in the ➕ Add tab — or bulk-import a folder of notes.",
                )
        else:
            df = pd.DataFrame(
                {
                    "chunk_id": [r["chunk_id"][:24] for r in rows],
                    "title": [r["title"] or "—" for r in rows],
                    "text": [(r["text"] or "").replace("\n", " ")[:140] for r in rows],
                    "tok": [r["token_count"] for r in rows],
                    "ingested": [(r["ingested_at"] or "")[:19] for r in rows],
                    "excluded": [bool(r["excluded"]) for r in rows],
                }
            )
            col_l.dataframe(df, width="stretch", hide_index=True)

            with col_l.expander("Forget a specific memory by chunk_id"):
                cid_input = st.text_input("chunk_id prefix")
                if st.button("Forget", type="primary", disabled=not cid_input.strip()):
                    match = next(
                        (r["chunk_id"] for r in rows if r["chunk_id"].startswith(cid_input.strip())),
                        None,
                    )
                    if match is None:
                        st.error("No memory with that chunk_id prefix in the current view.")
                    else:
                        ok = orch.memory_store.forget(user_id, match)
                        if ok:
                            st.success(f"Tombstoned {match}.")
                            st.rerun()
                        else:
                            st.error("Could not forget that chunk.")

    # ---- Add ----------------------------------------------------------------
    with tab_add:
        st.markdown("#### Save a new memory")
        text = st.text_area(
            "Memory text",
            placeholder="e.g. Postgres is preferred over MySQL for new projects.",
            height=120,
        )
        title = st.text_input("Title (optional)", placeholder="defaults to first 60 chars")
        tags_raw = st.text_input("Tags (optional, comma-separated)")
        if st.button("💾 Save", type="primary", disabled=not text.strip()):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()] or None
            try:
                mid = orch.memory_store.add(
                    user_id, text, title=title or None, tags=tags, source="gui"
                )
                st.success(f"Saved {mid}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed: {exc}")

        st.divider()
        st.markdown("#### Bulk import")
        st.caption(
            "Drop one or more `.md` / `.txt` files. Markdown files are split on `## ` "
            "headings by default; text files become one memory per non-empty line."
        )
        uploaded = st.file_uploader(
            "Drop files here",
            type=["md", "markdown", "txt"],
            accept_multiple_files=True,
        )
        per_file = st.checkbox(
            "One memory per file (don't split on headings)",
            value=not orch.config.memory.bulk_chunk_per_paragraph,
        )

        if uploaded:
            exts = {f".{f.name.rsplit('.', 1)[-1].lower()}" for f in uploaded if "." in f.name}
            _tone_map = {".md": "violet", ".markdown": "violet", ".txt": "info"}
            chips_html = " ".join(
                chip(ext, _tone_map.get(ext, "default")) for ext in sorted(exts)
            )
            st.markdown(chips_html, unsafe_allow_html=True)

        if uploaded and st.button("📥 Import"):
            from hrag.cli import _iter_memory_texts_from_path  # noqa: PLC0415
            import tempfile  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415

            total_saved = 0
            progress = st.progress(0.0, text="Importing…")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                for f in uploaded:
                    dest = tmp_path / f.name
                    dest.write_bytes(f.getbuffer())
                items = _iter_memory_texts_from_path(
                    tmp_path,
                    split_paragraphs=not per_file,
                )

                for i, item in enumerate(items, 1):
                    try:
                        orch.memory_store.add(
                            user_id,
                            item["text"],
                            title=item.get("title"),
                            source="bulk",
                        )
                        total_saved += 1
                    except Exception:  # noqa: BLE001
                        pass
                    progress.progress(i / max(len(items), 1), text=f"{i}/{len(items)}")
            progress.empty()
            st.success(f"Imported {total_saved} memories from {len(uploaded)} file(s).")

    # ---- Search -------------------------------------------------------------
    with tab_search:
        st.markdown("#### Semantic search over memories")
        q = st.text_input("Query", placeholder="What did I save about databases?")
        topk = st.slider("Top-K", 1, 20, 5)
        col_a, col_b = st.columns([1, 1])
        do_search = col_a.button("🔍 Recall", disabled=not q.strip())
        do_forget = col_b.button(
            "🗑️ Forget all hits", disabled=not q.strip(),
        )

        if do_search or do_forget:
            hits = orch.retriever.retrieve(
                q, user_id, top_k=topk, source_types=["episodic"]
            )
            if not hits:
                empty_state(
                    icon="🔍",
                    title="No matching memories",
                    message=f'Nothing in this user\'s memory matched: "{q}"',
                )
                return

            for i, h in enumerate(hits, 1):
                with st.container(border=True):
                    rr = (
                        f", rerank={h.rerank_score:.2f}"
                        if h.rerank_score is not None
                        else ""
                    )
                    st.markdown(
                        f"**[{i}] {h.chunk.title or '—'}** "
                        f"`{h.chunk.chunk_id[:24]}` · score={h.score:.3f}{rr}"
                    )
                    st.write(h.chunk.text)

            if do_forget:
                if not orch.config.memory.forget_confirm or st.checkbox(
                    f"Confirm: forget {len(hits)} chunk(s)?", value=False, key="confirm_forget"
                ):
                    for h in hits:
                        orch.memory_store.forget(user_id, h.chunk.chunk_id)
                    st.success(f"Tombstoned {len(hits)} chunk(s).")


_render()
