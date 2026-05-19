from __future__ import annotations

from pathlib import Path

import streamlit as st

from hrag.gui.state import (
    apply_chrome,
    chip,
    current_user_id,
    doc_type_icon,
    empty_state,
    get_orchestrator,
    kpi_card,
    page_header,
)


def _render() -> None:
    apply_chrome(page_icon="📄", page_title="Documents · HRAG-Bot")
    page_header(
        "📄 Documents",
        icon="📄",
        subtitle="Source-of-truth corpus (PDF · MD · DOCX · TXT). Episodic notes are on the Memories page.",
        tips=[
            "Drop multiple files at once — they ingest in batch with a progress bar.",
            "<b>Ingest path on disk</b> walks folders recursively if you turn the toggle on.",
            "Deleting a document tombstones its chunks (<code>excluded=1</code>) AND removes the Chroma vectors.",
            "Big PDFs may take a while — academic papers chunk into ~50–200 pieces each.",
        ],
    )

    orch = get_orchestrator()
    user_id = current_user_id()

    tab_browse, tab_ingest = st.tabs(["Browse", "📥 Ingest"])

    # ---- Browse -------------------------------------------------------------
    with tab_browse:
        rows = orch.db.execute(
            """
            SELECT d.doc_id, d.title, d.source_type, d.source_path, d.ingested_at,
                   COUNT(c.chunk_id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.doc_id AND c.excluded = 0
            WHERE d.user_id = ? AND d.source_type = 'document'
            GROUP BY d.doc_id
            ORDER BY d.ingested_at DESC
            """,
            (user_id,),
        ).fetchall()

        k1, k2, k3 = st.columns(3)
        kpi_card(k1, label="Documents", value=len(rows), icon="📄", tone="accent")
        kpi_card(k2, label="Total chunks", value=sum(r["chunk_count"] for r in rows), icon="🧩", tone="info")
        kpi_card(k3, label="Last ingested", value=(rows[0]["ingested_at"][:10] if rows else "—"), icon="🕒", tone="good")

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

        if not rows:
            empty_state(
                icon="📄",
                title="No documents ingested yet",
                message="Drag-and-drop PDFs / Markdown / DOCX / TXT into the ➕ Ingest tab — or point HRAG at a folder on disk.",
            )
        else:
            for r in rows:
                icon = doc_type_icon(r["source_path"])
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([0.1, 0.55, 0.2, 0.15])
                    with c1:
                        st.markdown(
                            f"<div style='font-size:1.8rem;text-align:center;'>{icon}</div>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f"**{r['title'] or '—'}**  \n<span style='color:#9ca3af;font-size:0.82rem;'>"
                            f"<code>{r['doc_id'][:16]}</code> · {(r['source_path'] or '')[:60]}</span>",
                            unsafe_allow_html=True,
                        )
                    with c3:
                        chunks_chip = chip(f"{r['chunk_count']} chunks", "violet")
                        date_chip = chip((r["ingested_at"] or "")[:10] or "—", "muted")
                        st.markdown(
                            f"{chunks_chip} {date_chip}",
                            unsafe_allow_html=True,
                        )
                    with c4:
                        if st.button("🗑️", key=f"del_{r['doc_id']}", help="Delete this document"):
                            n = orch.db.execute(
                                "UPDATE chunks SET excluded = 1 WHERE doc_id = ? AND user_id = ?",
                                (r["doc_id"], user_id),
                            ).rowcount
                            orch.db.execute(
                                "DELETE FROM documents WHERE doc_id = ? AND user_id = ?",
                                (r["doc_id"], user_id),
                            )
                            orch.db.commit()
                            try:
                                orch.vector_store.delete_doc(user_id, r["doc_id"])
                            except Exception:  # noqa: BLE001
                                pass
                            st.toast(f"Removed {r['title'] or r['doc_id'][:8]} ({n} chunks)", icon="🗑️")
                            st.rerun()

    # ---- Ingest -------------------------------------------------------------
    with tab_ingest:
        st.markdown("#### Drag-and-drop ingest")
        st.markdown(
            f"Supported: {chip('PDF 📕', 'bad')} {chip('DOCX 📘', 'info')} "
            f"{chip('Markdown 📝', 'violet')} {chip('Text 📄', 'muted')}",
            unsafe_allow_html=True,
        )
        uploads = st.file_uploader(
            "Drop PDFs / MD / DOCX / TXT here",
            type=["pdf", "md", "markdown", "docx", "txt"],
            accept_multiple_files=True,
        )
        if uploads and st.button("📥 Ingest uploads", type="primary"):
            import tempfile  # noqa: PLC0415

            progress = st.progress(0.0, text="Ingesting…")
            ingested = 0
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                for i, f in enumerate(uploads, 1):
                    dest = tmp_path / f.name
                    dest.write_bytes(f.getbuffer())
                    try:
                        orch.ingest.ingest_path(dest, user_id)
                        ingested += 1
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"{f.name}: {exc}")
                    progress.progress(
                        i / max(len(uploads), 1), text=f"{i}/{len(uploads)}: {f.name}"
                    )
            progress.empty()
            st.success(f"Ingested {ingested}/{len(uploads)} file(s).")

        st.divider()
        st.markdown("#### Or ingest a path on disk")
        path_str = st.text_input(
            "Path",
            placeholder=r"D:\Selected Dynamic Papers",
        )
        recursive = st.checkbox("Recurse into subdirectories", value=True)
        if st.button("📥 Ingest path", disabled=not path_str.strip()):
            p = Path(path_str)
            if not p.exists():
                st.error(f"Path does not exist: {p}")
                return
            try:
                if p.is_dir():
                    with st.spinner(f"Walking {p}…"):
                        docs = orch.ingest.ingest_directory(
                            p, user_id, recursive=recursive
                        )
                    st.success(f"Ingested {len(docs)} document(s).")
                else:
                    with st.spinner(f"Ingesting {p.name}…"):
                        orch.ingest.ingest_path(p, user_id)
                    st.success(f"Ingested {p.name}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed: {exc}")


_render()
