"""Taxonomy — hierarchical category tree over your documents.

Edit labels, move nodes, file docs into the tree.
"""

from __future__ import annotations

import streamlit as st

from hrag.gui.state import (
    apply_chrome,
    chip,
    current_user_id,
    empty_state,
    get_orchestrator,
    page_header,
    section_title,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _indented_label(node, nodes_by_id: dict) -> str:
    """Return a depth-indented display string for use in selectboxes."""
    indent = "    " * node.depth  # non-breaking spaces
    return f"{indent}{node.label} (docs={node.doc_count})"


def _path_from_root(node, nodes_by_id: dict) -> str:
    """Walk parent_ids upward and return a slash-delimited path string."""
    parts: list[str] = []
    cursor = node
    while cursor is not None:
        parts.append(cursor.label)
        parent_id = cursor.parent_id
        cursor = nodes_by_id.get(parent_id) if parent_id else None
    parts.reverse()
    return " / ".join(parts)


def _graphviz_tree(nodes, tree: dict[str, list[str]]) -> str:
    """Build a Graphviz DOT source string for the full taxonomy tree."""
    lines: list[str] = [
        "digraph taxonomy {",
        "  rankdir=TB;",
        "  node [fontname=Helvetica fontsize=11];",
        "  edge [color=\"#6b7280\" arrowsize=0.6];",
    ]
    for node in nodes:
        nid = node.node_id.replace('"', "_").replace(":", "_").replace(" ", "_")
        label_safe = node.label.replace('"', "'")
        display = f"{label_safe}\\n({node.doc_count} docs)"
        if node.is_leaf:
            shape = "box"
            fill = "#16a34a"
            font_color = "#ffffff"
        else:
            shape = "ellipse"
            fill = "#1e3a5f"
            font_color = "#e5e7eb"
        lines.append(
            f'  "{nid}" [label="{display}" shape={shape} style="filled" '
            f'fillcolor="{fill}" fontcolor="{font_color}"];'
        )
    for parent_id, children in tree.items():
        pid = parent_id.replace('"', "_").replace(":", "_").replace(" ", "_")
        for child_id in children:
            cid = child_id.replace('"', "_").replace(":", "_").replace(" ", "_")
            lines.append(f'  "{pid}" -> "{cid}";')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _render() -> None:
    apply_chrome(page_icon="🌳", page_title="Taxonomy · HRAG-Bot")
    page_header(
        "🌳 Taxonomy",
        icon="🌳",
        subtitle="Hierarchical category tree over your documents. Edit labels, move nodes, file docs.",
        tips=[
            "LLM proposes; you refine.",
            "Click a node to inspect/edit.",
            "Drag-and-drop isn't here yet — use the <b>Move to</b> dropdown.",
            "Run <b>Rebuild</b> to re-propose from the current corpus.",
        ],
    )

    orch = get_orchestrator()
    user_id = current_user_id()

    # Guard: taxonomy feature requires taxonomy_store to be wired up.
    if orch.taxonomy_store is None:
        st.warning(
            "Taxonomy store is not initialised. "
            "Enable `taxonomy.enabled: true` in `config.yaml` and restart."
        )
        return

    ts = orch.taxonomy_store

    # ---- Action bar ---------------------------------------------------------
    section_title("Actions", icon="⚙️")
    act1, act2, act3, act4 = st.columns(4)

    with act1:
        if st.button("🔨 Propose / Rebuild", type="primary", use_container_width=True):
            try:
                from hrag.taxonomy.builder import TaxonomyBuilder  # noqa: PLC0415

                builder = TaxonomyBuilder(
                    orch.db,
                    orch.llm,
                    orch.embedder,
                    ts,
                    orch.config.taxonomy,
                )
                with st.spinner("Building taxonomy tree… (LLM-heavy, may take a minute)"):
                    builder.build_for_user(user_id)
                st.toast("Tree rebuilt", icon="✅")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with act2:
        if st.button("🔁 Recompute centroids", use_container_width=True):
            try:
                with st.spinner("Recomputing centroids…"):
                    ts.recompute_all_centroids(user_id)
                st.toast("Centroids refreshed", icon="✅")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with act3:
        if st.button("🗑️ Clear tree", use_container_width=True):
            st.session_state["confirm_clear_tree"] = True
            st.rerun()
        if st.session_state.get("confirm_clear_tree"):
            st.warning("This deletes ALL taxonomy nodes and assignments for this user.")
            if st.button("Confirm clear", key="confirm_clear_tree_yes", type="primary"):
                try:
                    ts.clear(user_id)
                    st.session_state.pop("confirm_clear_tree", None)
                    st.toast("Tree cleared", icon="🗑️")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

    with act4:
        if st.button("📥 Assign all unfiled", use_container_width=True):
            try:
                from hrag.taxonomy.assigner import DocAssigner  # noqa: PLC0415

                assigner = DocAssigner(
                    orch.db,
                    orch.llm,
                    orch.embedder,
                    ts,
                    orch.config.taxonomy,
                )
                with st.spinner("Assigning unfiled docs…"):
                    assigner.assign_all(user_id)
                st.toast("Unfiled docs assigned", icon="✅")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    st.divider()

    # ---- Load tree data -----------------------------------------------------
    nodes = ts.list_nodes(user_id)
    tree = ts.get_tree(user_id)
    nodes_by_id: dict[str, object] = {n.node_id: n for n in nodes}

    if not nodes:
        empty_state(
            icon="🌳",
            title="No taxonomy tree yet",
            message=(
                "Click 🔨 Propose / Rebuild to let the LLM generate a tree from your corpus, "
                "or add nodes manually using the edit panel."
            ),
        )
        return

    # ---- Main layout: tree (60%) | edit panel (40%) ------------------------
    col_tree, col_edit = st.columns([0.60, 0.40])

    # ---- Tree visualisation -------------------------------------------------
    with col_tree:
        section_title("Tree", icon="🌳")

        dot_source = _graphviz_tree(nodes, tree)

        # Try streamlit-agraph first; fall back to graphviz_chart.
        try:
            from streamlit_agraph import Config as AConfig  # noqa: PLC0415
            from streamlit_agraph import Edge, Node, agraph

            _nodes = []
            _edges = []
            for n in nodes:
                color = "#16a34a" if n.is_leaf else "#1e3a5f"
                _nodes.append(
                    Node(
                        id=n.node_id,
                        label=f"{n.label}\n({n.doc_count})",
                        color=color,
                        size=20,
                    )
                )
            for parent_id, children in tree.items():
                for child_id in children:
                    _edges.append(Edge(source=parent_id, target=child_id))
            agraph(
                nodes=_nodes,
                edges=_edges,
                config=AConfig(width=640, height=480, directed=True, physics=True),
            )
        except ImportError:
            # graphviz_chart is always available in Streamlit >= 1.18.
            st.graphviz_chart(dot_source, use_container_width=True)

        st.markdown(
            "<div style='margin-top:6px;margin-bottom:4px;color:#9ca3af;font-size:0.8rem;'>"
            "Internal nodes = ellipse · Leaves = filled box (green)</div>",
            unsafe_allow_html=True,
        )

        # Flat node picker (hierarchical-looking via indentation).
        section_title("Select node to edit", icon="✏️")

        if not nodes:
            st.caption("No nodes yet.")
        else:
            node_ids = [n.node_id for n in nodes]

            # Persist selection across reruns.
            default_index = 0
            if "selected_node_id" in st.session_state:
                sel = st.session_state["selected_node_id"]
                if sel in node_ids:
                    default_index = node_ids.index(sel)

            selected_id = st.selectbox(
                "Node",
                options=node_ids,
                index=default_index,
                format_func=lambda nid: _indented_label(nodes_by_id[nid], nodes_by_id),
                key="taxonomy_node_select",
                label_visibility="collapsed",
            )
            st.session_state["selected_node_id"] = selected_id

    # ---- Edit panel ---------------------------------------------------------
    with col_edit:
        selected_id = st.session_state.get("selected_node_id")
        if not selected_id or selected_id not in nodes_by_id:
            st.info("Select a node on the left to edit it.")
        else:
            node = nodes_by_id[selected_id]

            # Path from root
            path_str = _path_from_root(node, nodes_by_id)
            st.markdown(
                f"<div style='font-size:0.88rem;color:#9ca3af;margin-bottom:6px;'>"
                f"<code>{path_str}</code></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**{node.label}** "
                + chip("leaf" if node.is_leaf else "internal", "good" if node.is_leaf else "info")
                + " "
                + chip(f"depth {node.depth}", "muted"),
                unsafe_allow_html=True,
            )

            is_root = node.parent_id is None

            # -- Label edit ---------------------------------------------------
            new_label = st.text_input(
                "Label",
                value=node.label,
                key=f"label_{node.node_id}",
                disabled=is_root,
            )
            if not is_root and st.button("💾 Save label", key=f"save_label_{node.node_id}"):
                try:
                    ts.update_node(node.node_id, label=new_label.strip())
                    orch.db.commit()
                    st.toast("Label saved", icon="✅")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

            # -- Description edit ---------------------------------------------
            new_desc = st.text_area(
                "Description (used for LLM matching)",
                value=node.description,
                key=f"desc_{node.node_id}",
                height=90,
                disabled=is_root,
            )
            if not is_root and st.button(
                "💾 Save description", key=f"save_desc_{node.node_id}"
            ):
                try:
                    ts.update_node(node.node_id, description=new_desc.strip())
                    orch.db.commit()
                    st.toast("Description saved", icon="✅")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

            st.divider()

            # -- Move node ----------------------------------------------------
            if not is_root:
                other_node_ids = [
                    n.node_id
                    for n in nodes
                    if n.node_id != node.node_id
                ]
                if other_node_ids:
                    move_target = st.selectbox(
                        "Move under",
                        options=other_node_ids,
                        format_func=lambda nid: _indented_label(
                            nodes_by_id[nid], nodes_by_id
                        ),
                        key=f"move_target_{node.node_id}",
                    )
                    if st.button("📦 Move", key=f"move_btn_{node.node_id}"):
                        try:
                            ts.move_node(node.node_id, move_target)
                            orch.db.commit()
                            st.toast("Node moved", icon="✅")
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(str(exc))

            st.divider()

            # -- Add child ----------------------------------------------------
            with st.expander("➕ Add child node", expanded=False):
                with st.form(key=f"add_child_{node.node_id}"):
                    child_label = st.text_input("Label", key=f"cl_{node.node_id}")
                    child_desc = st.text_area(
                        "Description", key=f"cd_{node.node_id}", height=70
                    )
                    child_is_leaf = st.checkbox(
                        "Leaf node (docs can be assigned here)",
                        value=True,
                        key=f"cil_{node.node_id}",
                    )
                    if st.form_submit_button("Create"):
                        if not child_label.strip():
                            st.warning("Label is required.")
                        else:
                            try:
                                ts.add_node(
                                    user_id,
                                    node.node_id,
                                    child_label.strip(),
                                    child_desc.strip(),
                                    is_leaf=child_is_leaf,
                                )
                                orch.db.commit()
                                st.toast("Child node created", icon="✅")
                                st.rerun()
                            except Exception as exc:  # noqa: BLE001
                                st.error(str(exc))

            st.divider()

            # -- Delete node --------------------------------------------------
            if not is_root:
                with st.expander("🗑️ Delete this node", expanded=False):
                    st.warning(
                        "Deleting removes this node and its whole subtree. "
                        "Optionally reassign filed docs to another leaf first."
                    )
                    leaf_ids = [
                        n.node_id for n in nodes if n.is_leaf and n.node_id != node.node_id
                    ]
                    reassign_target = None
                    if leaf_ids:
                        use_reassign = st.checkbox(
                            "Reassign docs to another leaf",
                            key=f"use_reassign_{node.node_id}",
                        )
                        if use_reassign:
                            reassign_target = st.selectbox(
                                "Reassign to",
                                options=leaf_ids,
                                format_func=lambda nid: _indented_label(
                                    nodes_by_id[nid], nodes_by_id
                                ),
                                key=f"reassign_target_{node.node_id}",
                            )

                    confirm_key = f"confirm_delete_{node.node_id}"
                    if not st.session_state.get(confirm_key):
                        if st.button(
                            "Delete node",
                            key=f"del_init_{node.node_id}",
                            type="primary",
                        ):
                            st.session_state[confirm_key] = True
                            st.rerun()
                    else:
                        st.error("Are you sure? This cannot be undone.")
                        if st.button(
                            "Yes, delete", key=f"del_confirm_{node.node_id}", type="primary"
                        ):
                            try:
                                ts.delete_node(
                                    node.node_id,
                                    reassign_docs_to=reassign_target,
                                )
                                orch.db.commit()
                                st.session_state.pop(confirm_key, None)
                                st.session_state.pop("selected_node_id", None)
                                st.toast("Node deleted", icon="🗑️")
                                st.rerun()
                            except Exception as exc:  # noqa: BLE001
                                st.error(str(exc))
                        if st.button("Cancel", key=f"del_cancel_{node.node_id}"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()

            st.divider()

            # -- Docs filed under this node -----------------------------------
            section_title("Filed documents", icon="📄")
            include_desc = not node.is_leaf
            doc_ids = ts.get_docs_at(node.node_id, include_descendants=include_desc)

            if not doc_ids:
                st.caption("No documents filed here.")
            else:
                for doc_id in doc_ids:
                    doc_row = orch.db.execute(
                        "SELECT title, source_path FROM documents WHERE doc_id = ? LIMIT 1",
                        (doc_id,),
                    ).fetchone()
                    title = (doc_row["title"] if doc_row else None) or doc_id[:16]
                    path_hint = (doc_row["source_path"] if doc_row else "") or ""

                    col_dtitle, col_unassign = st.columns([0.82, 0.18])
                    with col_dtitle:
                        st.markdown(
                            f"<div style='font-size:0.88rem;'>"
                            f"<b>{title}</b><br>"
                            f"<span style='color:#9ca3af;font-size:0.78rem;'>"
                            f"{path_hint[:60]}</span></div>",
                            unsafe_allow_html=True,
                        )
                    with col_unassign:
                        if st.button(
                            "✖",
                            key=f"unassign_{node.node_id}_{doc_id}",
                            help="Unassign this doc from node",
                        ):
                            try:
                                ts.unassign_doc(user_id, doc_id, node_id=node.node_id)
                                orch.db.commit()
                                st.toast("Doc unassigned", icon="✅")
                                st.rerun()
                            except Exception as exc:  # noqa: BLE001
                                st.error(str(exc))


_render()
