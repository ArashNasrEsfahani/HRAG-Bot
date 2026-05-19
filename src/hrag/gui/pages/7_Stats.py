"""Corpus + KG stats with simple charts."""

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
    section_title,
)


def _render() -> None:
    apply_chrome(page_icon="📊", page_title="Stats · HRAG-Bot")
    page_header(
        "📊 Stats",
        icon="📊",
        subtitle="Corpus + KG + per-doc breakdowns for the active user.",
        tips=[
            "<b>Tombstoned</b> counts deleted-but-preserved chunks (<code>excluded=1</code>).",
            "The KG block only renders when <code>kg.enabled=true</code> in <code>config.yaml</code>.",
            "<b>Top documents by chunk count</b> is a quick way to spot bloated PDFs that may pollute retrieval.",
        ],
    )

    orch = get_orchestrator()
    user_id = current_user_id()
    cfg = orch.config

    # ---- KPIs ---------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    n_docs = orch.db.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE user_id = ? AND source_type = 'document'",
        (user_id,),
    ).fetchone()["n"]
    n_chunks = orch.db.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE user_id = ? AND source_type = 'document' AND excluded = 0",
        (user_id,),
    ).fetchone()["n"]
    n_mem = orch.db.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE user_id = ? AND source_type = 'episodic' AND excluded = 0",
        (user_id,),
    ).fetchone()["n"]
    n_excluded = orch.db.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE user_id = ? AND excluded = 1",
        (user_id,),
    ).fetchone()["n"]
    kpi_card(c1, "Documents", n_docs, icon="📄", tone="accent")
    kpi_card(c2, "Doc chunks", n_chunks, icon="🧩", tone="info")
    kpi_card(c3, "Active memories", n_mem, icon="📚", tone="good")
    kpi_card(c4, "Tombstoned", n_excluded, icon="🗑️", tone="warn")

    st.divider()

    # ---- KG counts ----------------------------------------------------------
    if cfg.kg.enabled:
        section_title("Knowledge graph", icon="🕸️", caption="Phrase / passage nodes plus the edges and communities derived from them.")
        kg_phrase = orch.db.execute(
            "SELECT COUNT(*) AS n FROM kg_nodes WHERE user_id = ? AND node_type='phrase'",
            (user_id,),
        ).fetchone()["n"]
        kg_passage = orch.db.execute(
            "SELECT COUNT(*) AS n FROM kg_nodes WHERE user_id = ? AND node_type='passage'",
            (user_id,),
        ).fetchone()["n"]
        kg_edges = orch.db.execute(
            "SELECT COUNT(*) AS n FROM kg_edges WHERE user_id = ?",
            (user_id,),
        ).fetchone()["n"]
        kg_comm = orch.db.execute(
            "SELECT COUNT(*) AS n FROM kg_communities WHERE user_id = ?",
            (user_id,),
        ).fetchone()["n"]
        col_a, col_b, col_c, col_d = st.columns(4)
        kpi_card(col_a, "Phrase nodes", kg_phrase, icon="🔤", tone="accent")
        kpi_card(col_b, "Passage nodes", kg_passage, icon="📑", tone="info")
        kpi_card(col_c, "Edges", kg_edges, icon="🔗", tone="info")
        kpi_card(col_d, "Communities", kg_comm, icon="🌐", tone="good")
        st.divider()
    else:
        st.markdown(
            "<div style='padding:14px 16px;border-radius:12px;background:rgba(255,255,255,0.03);"
            "border:1px dashed rgba(255,255,255,0.16);color:#9ca3af;font-size:0.9rem;'>"
            "🕸️ Knowledge graph is disabled. Set <code>kg.enabled: true</code> in <code>config.yaml</code> "
            "and run <code>hrag rebuild-kg</code> to populate it.</div>",
            unsafe_allow_html=True,
        )
        st.divider()

    # ---- Top documents by chunk count ---------------------------------------
    section_title("Top documents by chunk count", icon="📊", caption="A quick way to spot bloated PDFs that may pollute retrieval.")
    rows = orch.db.execute(
        """
        SELECT d.title, COUNT(*) AS n
        FROM chunks ch JOIN documents d ON ch.doc_id = d.doc_id
        WHERE ch.excluded = 0 AND ch.user_id = ? AND ch.source_type = 'document'
        GROUP BY d.title
        ORDER BY n DESC
        LIMIT 20
        """,
        (user_id,),
    ).fetchall()
    if rows:
        df = pd.DataFrame(
            {"title": [r["title"] or "(untitled)" for r in rows], "chunks": [r["n"] for r in rows]}
        )
        st.bar_chart(df.set_index("title"), width="stretch")
    else:
        empty_state(
            icon="📊",
            title="No document chunks yet",
            message="Ingest a document from the Documents page to see this chart.",
            cta_label="Ingest documents",
            cta_page="pages/3_Documents.py",
        )

    # ---- Source-type split --------------------------------------------------
    section_title("Active chunks by source type", icon="🧬")
    split = orch.db.execute(
        "SELECT source_type, COUNT(*) AS n FROM chunks "
        "WHERE excluded = 0 AND user_id = ? GROUP BY source_type",
        (user_id,),
    ).fetchall()
    if split:
        parts = " ".join(
            chip(f"{r['source_type']} · {r['n']}", "violet" if r["source_type"] == "episodic" else "accent")
            for r in split
        )
        st.markdown(parts, unsafe_allow_html=True)
    else:
        empty_state(icon="🧬", title="No chunks yet", message="Ingest something to see how chunks split across source types.")


_render()
