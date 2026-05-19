from __future__ import annotations

import streamlit as st

from hrag.gui.state import (
    apply_chrome,
    avatar_html,
    chip,
    current_user_id,
    empty_state,
    get_orchestrator,
    page_header,
    section_title,
    set_user_id,
)


def _render() -> None:
    apply_chrome(page_icon="👥", page_title="Users · HRAG-Bot")
    page_header(
        "👥 Users",
        icon="👥",
        subtitle="Every chunk, memory, preference, and session is scoped by user_id.",
        tips=[
            "Switching the active user updates the sidebar pill on every page.",
            "Per-user knowledge graphs are fully isolated — no cross-talk between users.",
            "Create a user just by typing an id and clicking <b>➕ Create</b>; the row is appended to the <code>users</code> table.",
            "The default user comes from <code>config.yaml</code> → <code>user.default_user_id</code>.",
        ],
    )

    orch = get_orchestrator()
    active = current_user_id()

    rows = orch.db.execute(
        """
        SELECT u.user_id, u.display_name, u.created_at,
               (SELECT COUNT(*) FROM documents WHERE user_id = u.user_id AND source_type='document') AS docs,
               (SELECT COUNT(*) FROM chunks WHERE user_id = u.user_id AND source_type='episodic' AND excluded=0) AS memories,
               (SELECT COUNT(*) FROM preferences WHERE user_id = u.user_id) AS prefs,
               (SELECT COUNT(*) FROM sessions WHERE user_id = u.user_id) AS sessions
        FROM users u
        ORDER BY u.created_at ASC
        """,
    ).fetchall()

    if not rows:
        empty_state(
            icon="👥",
            title="No users yet",
            message="Run `hrag init` from the command line to bootstrap the default user.",
        )
    else:
        section_title("Users", icon="👥")
        for r in rows:
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([0.10, 0.40, 0.35, 0.15])
                with col1:
                    st.markdown(avatar_html(r["user_id"], size="lg"), unsafe_allow_html=True)
                with col2:
                    active_mark = f" {chip('● active', 'good')}" if r["user_id"] == active else ""
                    st.markdown(
                        f"**{r['user_id']}**{active_mark}  \n"
                        f"<span style='color:#9ca3af;font-size:0.85rem;'>{r['display_name'] or '—'} · "
                        f"created {(r['created_at'] or '')[:10]}</span>",
                        unsafe_allow_html=True,
                    )
                with col3:
                    c_docs = chip(f"📄 {r['docs']}", "accent")
                    c_mem = chip(f"📚 {r['memories']}", "violet")
                    c_sess = chip(f"💬 {r['sessions']}", "info")
                    c_prefs = chip(f"👤 {r['prefs']}", "good")
                    st.markdown(
                        f"{c_docs} {c_mem} {c_sess} {c_prefs}",
                        unsafe_allow_html=True,
                    )
                with col4:
                    if r["user_id"] == active:
                        st.markdown(chip("active", "good"), unsafe_allow_html=True)
                    else:
                        if st.button("✅ Switch", key=f"sw_{r['user_id']}", type="primary"):
                            set_user_id(r["user_id"])
                            st.rerun()

    st.divider()

    with st.expander("➕ Create a new user", expanded=False):
        new_id = st.text_input("user_id (lowercase, no spaces)")
        new_name = st.text_input("display name (optional)")
        if st.button("➕ Create", type="primary", disabled=not new_id.strip()):
            uid = new_id.strip().lower().replace(" ", "_")
            try:
                orch.db.ensure_user(uid, new_name or None)
                orch.db.commit()
                set_user_id(uid)
                st.success(f"Created and switched to '{uid}'.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed: {exc}")


_render()
