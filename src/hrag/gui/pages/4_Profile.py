"""Structured user profile — view, add, delete, run extraction over a session."""

from __future__ import annotations

from collections import Counter

import streamlit as st

from hrag.gui.state import (
    apply_chrome,
    chip,
    current_user_id,
    empty_state,
    get_orchestrator,
    kpi_card,
    page_header,
    polarity_chip,
)


_POLARITIES = ("fact", "style", "like", "dislike")

_POLARITY_META = {
    "fact":    ("📋", "info"),
    "style":   ("🎨", "violet"),
    "like":    ("👍", "good"),
    "dislike": ("👎", "bad"),
}


def _render() -> None:
    apply_chrome(page_icon="👤", page_title="Profile · HRAG-Bot")
    page_header(
        "👤 User profile",
        icon="👤",
        subtitle="Structured preferences rendered verbatim into every answer prompt.",
        tips=[
            "Polarities: <code>fact</code> · <code>style</code> · <code>like</code> · <code>dislike</code>.",
            "Confidence below the threshold (default 0.5) is hidden from the prompt — useful for soft hints.",
            "<b>🪄 Extract from session</b> mines an existing chat session via the LLM and asks before applying.",
            "The <i>Rendered preview</i> block in <b>View</b> shows exactly what the LLM sees each turn.",
        ],
    )

    orch = get_orchestrator()
    user_id = current_user_id()

    tab_view, tab_add, tab_extract = st.tabs(["View", "➕ Add", "🪄 Extract from session"])

    # ---- View ---------------------------------------------------------------
    with tab_view:
        prefs = orch.profile_store.list_all(user_id)
        if not prefs:
            empty_state(
                icon="👤",
                title="No profile entries yet",
                message="Add a structured preference in ➕ Add — or run extraction over an existing chat session in 🪄 Extract from session.",
            )
        else:
            ctr = Counter(p.polarity for p in prefs)
            kpi_cols = st.columns(4)
            for col, pol in zip(kpi_cols, _POLARITIES):
                icon, tone = _POLARITY_META[pol]
                kpi_card(col, pol.capitalize(), ctr.get(pol, 0), icon=icon, tone=tone)

            st.write("")

            for p in prefs:
                with st.container(border=True):
                    cols = st.columns([0.18, 0.32, 0.32, 0.10, 0.08])
                    cols[0].markdown(polarity_chip(p.polarity), unsafe_allow_html=True)
                    cols[1].markdown(f"**{p.topic}**")
                    if p.value:
                        cols[2].write(p.value)
                    else:
                        cols[2].markdown(
                            "<span style='color:#9ca3af'>(empty)</span>",
                            unsafe_allow_html=True,
                        )
                    conf_tone = "good" if p.confidence >= 0.7 else "warn"
                    cols[3].markdown(
                        chip(f"{p.confidence:.2f}", conf_tone),
                        unsafe_allow_html=True,
                    )
                    if cols[4].button("🗑️", key=f"delp_{p.pref_id}"):
                        if orch.profile_store.delete(user_id, p.pref_id):
                            st.toast(f"Deleted pref {p.pref_id}.")
                            st.rerun()

        st.divider()
        st.markdown("**Rendered preview** (this is exactly what the LLM sees):")
        st.code(
            orch.context_builder.build(user_id)["user_profile"], language=None
        )

    # ---- Add ----------------------------------------------------------------
    with tab_add:
        polarity = st.selectbox("Polarity", _POLARITIES, index=0)
        topic = st.text_input("Topic", placeholder="e.g. occupation, code language")
        value = st.text_input("Value", placeholder="e.g. data engineer, Python over R")
        confidence = st.slider("Confidence", 0.0, 1.0, 1.0, 0.05)
        st.markdown(
            f"Preview: {polarity_chip(polarity)} **{topic or '(topic)'}** — {value or '(value)'}",
            unsafe_allow_html=True,
        )
        if st.button(
            "💾 Upsert", type="primary", disabled=not topic.strip()
        ):
            try:
                pid = orch.profile_store.upsert(
                    user_id=user_id,
                    polarity=polarity,
                    topic=topic.strip(),
                    value=value.strip(),
                    confidence=confidence,
                )
                st.success(f"Upserted pref {pid}.")
            except ValueError as exc:
                st.error(str(exc))

    # ---- Extract ------------------------------------------------------------
    with tab_extract:
        st.caption(
            "Run `PreferenceExtractor` over a session's messages. "
            "Candidates above `auto_extract_min_confidence` "
            f"(currently {orch.config.memory.auto_extract_min_confidence}) "
            "are eligible to upsert."
        )

        sessions = orch.db.execute(
            """
            SELECT s.session_id, s.started_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS n_msgs
            FROM sessions s
            WHERE s.user_id = ?
            ORDER BY s.started_at DESC
            LIMIT 50
            """,
            (user_id,),
        ).fetchall()
        if not sessions:
            st.info("No sessions yet for this user.")
            return

        labels = [
            f"{r['session_id'][:12]} · {(r['started_at'] or '')[:19]} · {r['n_msgs']} msgs"
            for r in sessions
        ]
        idx = st.selectbox("Pick a session", range(len(labels)), format_func=lambda i: labels[i])
        target_sid = sessions[idx]["session_id"]

        auto_apply = st.checkbox("Auto-apply all candidates above threshold", value=False)
        if st.button("🪄 Run extraction", type="primary"):
            from hrag.memory.extractor import PreferenceExtractor  # noqa: PLC0415

            rows = orch.db.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? AND user_id = ? ORDER BY message_id",
                (target_sid, user_id),
            ).fetchall()
            conversation = [(r["role"], r["content"]) for r in rows]

            with st.spinner("LLM extracting preferences…"):
                cands = PreferenceExtractor(orch.llm).extract(conversation)

            if not cands:
                st.info("No preferences detected.")
                return

            min_conf = orch.config.memory.auto_extract_min_confidence
            applied = 0
            for c in cands:
                with st.container(border=True):
                    cols = st.columns([1, 2, 2, 1, 1])
                    cols[0].markdown(polarity_chip(c.polarity), unsafe_allow_html=True)
                    cols[1].write(c.topic)
                    cols[2].write(c.value)
                    cols[3].write(f"{c.confidence:.2f}")
                    if c.confidence >= min_conf:
                        if auto_apply or cols[4].button("Apply", key=f"apply_{c.topic}_{c.polarity}"):
                            orch.profile_store.upsert(
                                user_id=user_id,
                                polarity=c.polarity,
                                topic=c.topic,
                                value=c.value,
                                confidence=c.confidence,
                                source_session_id=target_sid,
                            )
                            applied += 1
                    else:
                        cols[4].caption("low-conf")

            if applied:
                st.success(f"Applied {applied} preference(s).")


_render()
