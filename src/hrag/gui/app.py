"""HRAG-Bot dashboard — Streamlit entry point (home page).

Launch with: ``hrag gui``
Or directly:  ``streamlit run src/hrag/gui/app.py``
"""

from __future__ import annotations

import streamlit as st

from hrag.gui.state import (
    apply_chrome,
    chip,
    current_user_id,
    get_orchestrator,
    kpi_card,
    nav_card,
    page_header,
    section_title,
)


def main() -> None:
    apply_chrome(page_icon="🧠", page_title="HRAG-Bot")
    page_header(
        "🧠 HRAG-Bot dashboard",
        icon="🧠",
        subtitle="Hierarchical RAG with per-user memory. Pick a page from the sidebar.",
        tips=[
            "First time? Run <code>hrag init</code> then ingest some files from the <b>📄 Documents</b> page.",
            "Try <b>📚 Memories → ➕ Add → drop files</b> to bulk-import a folder of notes.",
            "Profile preferences are rendered into <i>every</i> answer prompt — set them on the <b>👤 Profile</b> page.",
            "Streaming chat lives on the <b>💬 Chat</b> page — you'll see the LLM <b>think</b> → <b>write</b> → <b>done</b> phases live.",
        ],
    )

    orch = get_orchestrator()
    user_id = current_user_id()
    cfg = orch.config

    # ---- KPI strip ----------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    docs_row = orch.db.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE user_id = ? AND source_type = 'document'",
        (user_id,),
    ).fetchone()
    memories_row = orch.db.execute(
        "SELECT COUNT(*) AS n FROM chunks "
        "WHERE user_id = ? AND source_type = 'episodic' AND excluded = 0",
        (user_id,),
    ).fetchone()
    sessions_row = orch.db.execute(
        "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    prefs_row = orch.db.execute(
        "SELECT COUNT(*) AS n FROM preferences WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    kpi_card(col1, "Documents", docs_row["n"], icon="📄", tone="accent")
    kpi_card(col2, "Memories", memories_row["n"], icon="📚", tone="accent")
    kpi_card(col3, "Sessions", sessions_row["n"], icon="💬", tone="accent")
    kpi_card(col4, "Profile entries", prefs_row["n"], icon="👤", tone="accent")

    st.divider()

    # ---- Config snapshot ----------------------------------------------------
    st.subheader("Configuration")
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        section_title("LLM", icon="🤖")
        st.code(
            f"provider : {cfg.llm.provider}\n"
            f"model    : {cfg.llm.model}\n"
            f"temp     : {cfg.llm.temperature}",
            language="yaml",
        )
        section_title("Retrieval", icon="🔍")
        st.code(
            f"retriever     : {cfg.retrieval.retriever}\n"
            f"reranker      : {cfg.retrieval.reranker if cfg.retrieval.rerank_enabled else 'off'}\n"
            f"top_k_vector  : {cfg.retrieval.top_k_vector}\n"
            f"top_k_final   : {cfg.retrieval.top_k_final}",
            language="yaml",
        )
    with cfg_col2:
        section_title("Embeddings", icon="🧬")
        st.code(
            f"provider : {cfg.embeddings.provider}\n"
            f"model    : {cfg.embeddings.model}\n"
            f"dim      : {cfg.embeddings.dim}",
            language="yaml",
        )
        with st.expander("Phase 2 / 3 toggles", expanded=False):
            kg_enabled_chip = chip("on", "good") if cfg.kg.enabled else chip("off", "muted")
            kg_communities_chip = chip("on", "good") if cfg.kg.use_communities else chip("off", "muted")
            memory_extract_chip = chip("on", "good") if cfg.memory.auto_extract else chip("off", "muted")
            st.markdown(
                f"kg.enabled {kg_enabled_chip} &nbsp; "
                f"kg.use_communities {kg_communities_chip} &nbsp; "
                f"memory.auto_extract {memory_extract_chip}",
                unsafe_allow_html=True,
            )
            st.code(
                f"kg.enabled              : {cfg.kg.enabled}\n"
                f"kg.use_communities      : {cfg.kg.use_communities}\n"
                f"memory.auto_extract     : {cfg.memory.auto_extract}\n"
                f"memory.profile_max_items: {cfg.memory.profile_max_items}",
                language="yaml",
            )

    st.divider()

    # ---- Navigation cards ---------------------------------------------------
    st.subheader("Where to next?")
    nav_cols = st.columns(3)
    cards = [
        ("💬", "Chat", "Streaming chat over your corpus + memories.", "pages/1_Chat.py"),
        ("📚", "Memories", "Browse · add · semantic-recall · forget.", "pages/2_Memories.py"),
        ("📄", "Documents", "Browse · drag-and-drop ingest · delete.", "pages/3_Documents.py"),
        ("👤", "Profile", "Structured prefs rendered into every prompt.", "pages/4_Profile.py"),
        ("👥", "Users", "Switch user · create user · per-user activity.", "pages/5_Users.py"),
        ("📊", "Stats", "Corpus + KG counts + per-doc chunk chart.", "pages/7_Stats.py"),
    ]
    for i, (icon, label, desc, page) in enumerate(cards):
        nav_card(nav_cols[i % 3], icon, label, desc, page)


if __name__ == "__main__":
    main()
