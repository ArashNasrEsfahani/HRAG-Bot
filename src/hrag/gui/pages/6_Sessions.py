from __future__ import annotations

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
    apply_chrome(page_icon="💬", page_title="Sessions · HRAG-Bot")
    page_header(
        "💬 Sessions",
        icon="💬",
        subtitle="Every chat session ever had with this user. Replay any conversation in one click.",
        tips=[
            "Click <b>👁️ View</b> on any session card to inspect its full message log.",
            "<b>🪄 Extract preferences</b> jumps to the Profile page wired to this session.",
            "Deleting a session is permanent — both messages and the session row go.",
            "Sessions live in SQLite, so they survive process restarts and config changes.",
        ],
    )

    orch = get_orchestrator()
    user_id = current_user_id()

    sessions = orch.db.execute(
        """
        SELECT s.session_id, s.started_at, s.ended_at, s.summary,
               (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS n_msgs
        FROM sessions s
        WHERE s.user_id = ?
        ORDER BY s.started_at DESC
        """,
        (user_id,),
    ).fetchall()

    if not sessions:
        empty_state(
            icon="💬",
            title="No sessions yet",
            message="Start a conversation in 💬 Chat — every turn is saved here automatically.",
            cta_label="Go to Chat",
            cta_page="pages/1_Chat.py",
        )
        return

    k1, k2, k3 = st.columns(3)
    kpi_card(k1, label="Total sessions", value=len(sessions), icon="💬", tone="accent")
    kpi_card(
        k2,
        label="Total messages",
        value=sum(r["n_msgs"] for r in sessions),
        icon="✉️",
        tone="info",
    )
    kpi_card(
        k3,
        label="Last activity",
        value=sessions[0]["started_at"][:10] if sessions else "—",
        icon="🕒",
        tone="good",
    )

    st.divider()

    for s in sessions:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([0.10, 0.55, 0.20, 0.15])
            with c1:
                st.markdown(
                    "<div style='font-size:1.6rem;text-align:center;'>💬</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                summary = (s["summary"] or "(no summary)")[:90]
                st.markdown(
                    f"**`{s['session_id'][:12]}`**  \n"
                    f"<span style='color:#9ca3af;font-size:0.82rem;'>"
                    f"started {(s['started_at'] or '')[:19]} · {summary}</span>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(chip(f"{s['n_msgs']} msgs", "violet"), unsafe_allow_html=True)
            with c4:
                if st.button("👁️ View", key=f"viewsess_{s['session_id']}"):
                    st.session_state["selected_sid"] = s["session_id"]
                    st.rerun()

    sid = st.session_state.get("selected_sid")
    if sid is None:
        sid = sessions[0]["session_id"]

    st.divider()
    section_title("Conversation", icon="💬", caption=f"session {sid[:12]}")

    rows = orch.db.execute(
        "SELECT role, content, created_at FROM messages "
        "WHERE session_id = ? ORDER BY message_id",
        (sid,),
    ).fetchall()

    for r in rows:
        with st.chat_message(r["role"]):
            st.markdown(r["content"])
            st.caption((r["created_at"] or "")[:19])

    st.divider()
    col_a, col_b = st.columns(2)
    if col_a.button("🪄 Extract preferences from this session"):
        st.session_state["_extract_session_id"] = sid
        st.switch_page("pages/4_Profile.py")
    if col_b.button("🗑️ Delete session and its messages", type="secondary"):
        orch.db.execute(
            "DELETE FROM messages WHERE session_id = ? AND user_id = ?",
            (sid, user_id),
        )
        orch.db.execute(
            "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
            (sid, user_id),
        )
        orch.db.commit()
        st.session_state.pop("selected_sid", None)
        st.success("Session deleted.")
        st.rerun()


_render()
