"""Chat — streaming tokens + live phase badge (retrieve → rerank → write → done)."""

from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

from hrag.gui.components.taxonomy_tree import render_tree as _render_tree_component  # noqa: F401  (kept for legacy callers)
from hrag.gui.components.reasoning_trace import render as _render_reasoning_trace
from hrag.gui.state import (
    apply_chrome,
    chip,
    current_user_id,
    empty_state,
    get_orchestrator,
    page_header,
    render_source,
    stream_chat_events,
)


# ---------------------------------------------------------------------------
# Phase-badge HTML
# ---------------------------------------------------------------------------

_PHASE_LABELS = {
    "boot":       ("🧠", "Booting…",                "boot"),
    "rewrite":    ("📝", "Rewriting query",          "retrieve"),
    "route":      ("🧭", "Routing query",            "retrieve"),
    "taxonomy":   ("🌳", "Navigating taxonomy",      "retrieve"),
    "intent_greeting": ("👋", "Greeting — skipping retrieval",                "retrieve"),
    "intent_personal": ("👤", "Personal — memory only",                       "retrieve"),
    "intent_factual":  ("🔍", "Substantive question",                         "retrieve"),
    "intent_general":  ("🌍", "Off-corpus — answering from general knowledge", "retrieve"),
    "intent_unclear":  ("❓", "Unclear — asking for clarification",            "retrieve"),
    "retrieve":   ("🔍", "Retrieving",               "retrieve"),
    "rerank":     ("🎯", "Reranking",                "rerank"),
    "organize":   ("🧩", "Organizing (KG-MST)",      "rerank"),
    "wait_llm":   ("🤔", "LLM is thinking",          "rerank"),
    "write":      ("✍️", "LLM is writing",           "write"),
    "done":       ("✅", "Done",                     "done"),
}


def _phase_badge(phase: str, detail: str = "") -> str:
    icon, label, cls = _PHASE_LABELS.get(phase, ("🧠", phase, "boot"))
    pulsing = "" if phase == "done" else "<span class='hrag-thinking-dots'></span>"
    extra = f" <span style='color:#9ca3af;font-size:0.85rem'>· {detail}</span>" if detail else ""
    return (
        f"<div class='hrag-status-bar {cls}'>"
        f"<span class='dot'></span>"
        f"<span>{icon} <b>{label}</b>{pulsing}{extra}</span>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Taxonomy navigation visual — a tree that GROWS:
#
#   1. The query bubble appears at the top.
#   2. A trunk segment grows downward.
#   3. The level's nodes pop in (scale 0 → 1.1 → 1.0), one after another.
#   4. After a beat, the verdict animates: pruned nodes shrink + drain to grey,
#      kept nodes flash a green glow and ease back to rest.
#   5. From each kept node, a small branch grows downward toward the next row.
#   6. Repeat 2–5 for each level.
#   7. Finally, leaves "open" — cards rotate-in and the doc lists slide up.
#
# All timing is CSS animation-delay only; no JavaScript. Each row gets a
# CSS variable --t0 (the level's start time in seconds) and each node gets
# --i (its index within the row) so the staggered timing is data-driven.
# ---------------------------------------------------------------------------

# Per-level cycle length, in seconds. Five sub-phases:
#   stem grow (0.0–0.35) · node pop (0.35–0.95) · verdict (0.95–1.35)
#   · branch grow (1.35–1.65) · settle (1.65–1.80)
_TX_LEVEL_DUR = 1.8

_TAXONOMY_CSS = """
<style>
.tx-shell {
    background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 18px 18px 22px;
    margin: 8px 0 14px;
    color: #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
}
.tx-banner {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; flex-wrap: wrap;
    padding: 10px 14px;
    background: linear-gradient(135deg, rgba(16,185,129,0.18) 0%, rgba(14,165,233,0.12) 100%);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 10px;
    margin-bottom: 14px;
    /* fill-mode: both → 0% keyframe applies BEFORE the animation starts
       (so the element is hidden during the delay), and 100% keyframe holds
       AFTER (so it stays visible). If animations are disabled the base
       style (visible) wins — best of both worlds. */
    animation: tx-banner-in 0.5s 0.05s ease-out both;
}
@keyframes tx-banner-in {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.tx-banner .tx-banner-title {
    font-size: 0.95rem; font-weight: 600; color: #f3f4f6;
}
.tx-banner .tx-banner-stats {
    display: flex; gap: 10px; flex-wrap: wrap;
}
.tx-stat {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.78rem; color: #cbd5e1;
}
.tx-stat b { color: #f9fafb; }

.tx-stage {
    display: flex; flex-direction: column; align-items: center;
    position: relative;
}
.tx-query {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
    color: #0b1220; font-weight: 600;
    padding: 7px 16px; border-radius: 999px;
    box-shadow: 0 0 22px rgba(14,165,233,0.45);
    font-size: 0.85rem;
    /* Progressive reveal via fill-mode: both — hidden during delay, holds
       at final state after, falls back to (visible) base if animations are
       disabled entirely. */
    animation: tx-query-pop 0.55s 0.20s cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes tx-query-pop {
    0%   { opacity: 0; transform: scale(0.5) translateY(-6px); }
    65%  { opacity: 1; transform: scale(1.08); }
    100% { opacity: 1; transform: scale(1.0); }
}

/* Trunk segment that grows downward between levels (or query → level 0). */
.tx-stem {
    width: 2px;
    height: 26px;
    background: linear-gradient(to bottom,
        rgba(56,189,248,0.85) 0%,
        rgba(16,185,129,0.85) 100%);
    border-radius: 2px;
    transform-origin: top center;
    /* Progressive reveal — both = hidden during delay, holds open after. */
    animation: tx-stem-grow 0.35s var(--t0, 0s) ease-out both;
}
@keyframes tx-stem-grow {
    from { transform: scaleY(0); }
    to   { transform: scaleY(1); }
}

.tx-level-wrap {
    width: 100%;
    margin-top: 2px;
    /* Progressive: hidden during --t0 delay, visible during/after. */
    animation: tx-level-show 0.30s var(--t0, 0s) ease-out both;
}
@keyframes tx-level-show {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.tx-level-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #6b7280; margin: 6px 0 4px; text-align: center;
}
.tx-level {
    display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;
    align-items: stretch;
    position: relative;
}

/* A node has three sequential animations, all scheduled relative to --t0:
     1. pop:     0.35s + i*0.08s   — scale 0 → 1.1 → 1.0 entrance
     2. verdict: 0.95s              — kept glows, pruned shrinks
     3. (kept only) branch ::after grows downward starting at 1.35s          */
.tx-node {
    position: relative;
    flex: 0 1 auto; min-width: 150px; max-width: 240px;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 9px 12px;
    transform-origin: center top;
    will-change: transform, opacity, filter;
    /* both = hidden via 0%-keyframe during the per-node staggered delay,
       holds final state after. Base style (visible) is the fallback if no
       animation runs. */
    animation:
        tx-node-pop 0.45s calc(var(--t0, 0s) + 0.35s + var(--i, 0) * 0.08s)
            cubic-bezier(.34,1.56,.64,1) both;
}
.tx-node-label {
    font-size: 0.82rem; font-weight: 500; color: #cbd5e1;
    overflow: hidden; text-overflow: ellipsis;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    line-height: 1.25;
}
.tx-node-meta {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 5px;
    font-size: 0.7rem; color: #94a3b8;
}
.tx-score {
    background: rgba(255,255,255,0.06);
    padding: 1px 7px; border-radius: 999px;
    font-family: ui-monospace, monospace; font-weight: 600;
}
@keyframes tx-node-pop {
    0%   { opacity: 0; transform: scale(0.3) translateY(-8px); }
    60%  { opacity: 1; transform: scale(1.12) translateY(0); }
    100% { opacity: 1; transform: scale(1.0)  translateY(0); }
}

/* KEPT node: green glow flash, then settle.
   Base = the settled-kept appearance (so it's correct without animation). */
.tx-node.kept {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 60%, #047857 100%);
    border-color: #10b981;
    box-shadow: 0 0 14px rgba(16,185,129, 0.45),
                inset 0 0 0 1px rgba(255,255,255,0.04);
    animation:
        tx-node-pop 0.45s calc(var(--t0, 0s) + 0.35s + var(--i, 0) * 0.08s)
            cubic-bezier(.34,1.56,.64,1) both,
        tx-node-keep 0.6s calc(var(--t0, 0s) + 0.95s) ease-out both;
}
@keyframes tx-node-keep {
    0% {
        background: #1f2937;
        border-color: #374151;
        box-shadow: 0 0 0 rgba(16,185,129, 0.0);
        transform: scale(1.0);
    }
    45% {
        background: linear-gradient(135deg, #047857 0%, #10b981 60%, #34d399 100%);
        border-color: #6ee7b7;
        box-shadow: 0 0 28px rgba(16,185,129, 0.85),
                    inset 0 0 0 1px rgba(255,255,255,0.10);
        transform: scale(1.08);
    }
    100% {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 60%, #047857 100%);
        border-color: #10b981;
        box-shadow: 0 0 14px rgba(16,185,129, 0.45),
                    inset 0 0 0 1px rgba(255,255,255,0.04);
        transform: scale(1.0);
    }
}
.tx-node.kept .tx-node-label { color: #ecfdf5; font-weight: 600; }
.tx-node.kept .tx-node-meta  { color: #a7f3d0; }
.tx-node.kept .tx-score      { background: rgba(255,255,255,0.18); color: #ffffff; }
.tx-node.kept .tx-star       { color: #fde68a; margin-right: 4px; }

/* PRUNED node: at the verdict beat, shrink + drain to grey.
   Base = settled-pruned appearance, so even without animation the user
   sees grey + faded rather than identical-to-kept. */
.tx-node.pruned {
    opacity: 0.55;
    filter: grayscale(0.7);
    transform: scale(0.92);
    animation:
        tx-node-pop 0.45s calc(var(--t0, 0s) + 0.35s + var(--i, 0) * 0.08s)
            cubic-bezier(.34,1.56,.64,1) both,
        tx-node-prune 0.55s calc(var(--t0, 0s) + 0.95s) ease-in both;
}
@keyframes tx-node-prune {
    0%   { opacity: 1; filter: grayscale(0)   blur(0); transform: scale(1.0); }
    60%  { opacity: 0.55; filter: grayscale(0.6) blur(0.2px); transform: scale(0.88); }
    100% { opacity: 0.40; filter: grayscale(0.85) blur(0.4px); transform: scale(0.82); }
}

/* Branch line drawn DOWNWARD from each kept node toward the next row.
   Base height = 26px (final), animation overrides 0→26 if running. */
.tx-node.kept::after {
    content: "";
    position: absolute;
    left: 50%; bottom: -28px;
    width: 2px; height: 26px;
    background: linear-gradient(to bottom,
        rgba(16,185,129,0.95) 0%,
        rgba(16,185,129,0.25) 100%);
    transform: translateX(-50%);
    border-radius: 2px;
    animation: tx-branch-grow 0.40s calc(var(--t0, 0s) + 1.35s) ease-out both;
}
.tx-node.kept.tx-leaf-stop::after { display: none; }
@keyframes tx-branch-grow {
    from { height: 0; }
    to   { height: 26px; }
}

/* Leaves opening — final flourish. */
.tx-leaves-wrap {
    margin-top: 22px;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    width: 100%;
}
.tx-leaf-card {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    border: 1px solid rgba(16,185,129,0.45);
    border-radius: 10px;
    padding: 11px 13px;
    box-shadow: 0 0 12px rgba(16,185,129,0.18);
    transform-origin: top center;
    /* both = hidden during the post-descent delay, holds open after. */
    animation: tx-leaf-open 0.65s
        calc(var(--t-final, 0s) + var(--i, 0) * 0.12s)
        cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes tx-leaf-open {
    0%   { opacity: 0; transform: scale(0.45) rotate(-4deg); }
    55%  { opacity: 1; transform: scale(1.08) rotate(1deg); }
    100% { opacity: 1; transform: scale(1.0)  rotate(0deg); }
}
.tx-leaf-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
}
.tx-leaf-title { font-size: 0.85rem; font-weight: 600; color: #ecfdf5; }
.tx-leaf-badge {
    font-size: 0.7rem; padding: 1px 7px;
    background: rgba(16,185,129,0.22); color: #a7f3d0;
    border-radius: 999px;
}
.tx-leaf-docs {
    list-style: none; padding: 0; margin: 0;
    font-size: 0.76rem; color: #cbd5e1;
    overflow: hidden;
}
.tx-leaf-docs li {
    padding: 2px 0 2px 14px; position: relative;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    /* Per-doc slide-in. both = hidden during pre-roll then settled. */
    animation: tx-doc-slide 0.35s
        calc(var(--t-final, 0s) + var(--i-card, 0) * 0.12s + 0.25s
             + var(--i-doc, 0) * 0.05s) ease-out both;
}
.tx-leaf-docs li::before {
    content: "•"; color: #10b981;
    position: absolute; left: 4px; top: 2px;
}
@keyframes tx-doc-slide {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.tx-note {
    margin-top: 10px; padding: 8px 12px;
    background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.30);
    border-radius: 8px; color: #fde68a; font-size: 0.8rem;
}
</style>
"""


def _render_taxonomy_trace(payload: dict) -> str:
    """Build the HTML for the animated tree-navigation visual.

    Layout (top → bottom):
      banner → query bubble → [stem → level row → branch from kept] × N
      → leaves-opening row.

    Each row is offset in time by ``--t0`` (level index × _TX_LEVEL_DUR).
    Within a row, nodes pop in via ``--i`` stagger, then the kept/pruned
    verdict animates from the same row start. The deepest kept set has
    ``tx-leaf-stop`` set so its branch ::after is suppressed and the leaf
    cards take over the visual.
    """
    if not payload:
        return ""

    stats = payload.get("stats") or {}
    trace = payload.get("trace") or []
    leaves = payload.get("leaves") or []
    note = payload.get("note")

    total_docs = int(stats.get("total_docs") or 0)
    docs_opened = int(stats.get("docs_opened") or 0)
    leaves_picked = int(stats.get("leaves_picked") or len(leaves))
    nodes_considered = int(stats.get("nodes_considered") or 0)

    if total_docs > 0:
        pct = max(1, int(round(docs_opened * 100 / total_docs)))
        banner_title = (
            f"🎯 Opened <b>{docs_opened}</b> of <b>{total_docs}</b> documents "
            f"<span style='color:#a7f3d0'>({pct}%)</span> — "
            f"skipped {total_docs - docs_opened}"
        )
    else:
        banner_title = "🎯 Hierarchical descent"

    parts: list[str] = [_TAXONOMY_CSS, "<div class='tx-shell'>"]

    parts.append(
        f"<div class='tx-banner'>"
        f"<div class='tx-banner-title'>{banner_title}</div>"
        f"<div class='tx-banner-stats'>"
        f"<span class='tx-stat'><b>{leaves_picked}</b> leaves</span>"
        f"<span class='tx-stat'><b>{nodes_considered}</b> nodes considered</span>"
        f"</div></div>"
    )

    if not trace and not leaves:
        if note:
            parts.append(f"<div class='tx-note'>⚠ {_esc(note)}</div>")
        parts.append("</div>")
        return "".join(parts)

    parts.append("<div class='tx-stage'>")
    parts.append(
        "<div class='tx-query' aria-label='query'>🔎 query embedding</div>"
    )

    # Drop empty-considered levels so timing stays tight.
    levels = [
        (int(level.get("depth", i)), level.get("considered") or [])
        for i, level in enumerate(trace)
        if (level.get("considered") or [])
    ]
    last_idx = len(levels) - 1

    for i, (depth, considered) in enumerate(levels):
        t0 = 0.60 + i * _TX_LEVEL_DUR  # query bubble settles around 0.55s
        # Stem segment from previous level (or query bubble) down to this row.
        parts.append(
            f"<div class='tx-stem' style='--t0:{t0:.2f}s'></div>"
        )
        kept_count = sum(1 for ns in considered if ns.get("kept"))
        parts.append(
            f"<div class='tx-level-wrap' style='--t0:{t0:.2f}s'>"
            f"<div class='tx-level-label'>"
            f"level {depth} · {len(considered)} considered · "
            f"<span style='color:#10b981'>{kept_count} kept</span>"
            f"</div><div class='tx-level'>"
        )
        # Show every considered node, but cap at 12 to keep the row legible.
        shown = considered[:12]
        for ix, ns in enumerate(shown):
            kept = bool(ns.get("kept"))
            label = _esc(str(ns.get("label", "?"))[:80])
            score = float(ns.get("score") or 0.0)
            star = "<span class='tx-star'>★</span>" if kept else ""
            classes = ["tx-node", "kept" if kept else "pruned"]
            if kept and i == last_idx:
                # Last descent row: leaves take over visually; suppress branch.
                classes.append("tx-leaf-stop")
            cls = " ".join(classes)
            parts.append(
                f"<div class='{cls}' style='--t0:{t0:.2f}s; --i:{ix}'>"
                f"<div class='tx-node-label'>{star}{label}</div>"
                f"<div class='tx-node-meta'>"
                f"<span>{'kept' if kept else 'pruned'}</span>"
                f"<span class='tx-score'>{score:+.2f}</span>"
                f"</div></div>"
            )
        if len(considered) > len(shown):
            parts.append(
                f"<div class='tx-node pruned' "
                f"style='--t0:{t0:.2f}s; --i:{len(shown)}'>"
                f"<div class='tx-node-label'>+ {len(considered) - len(shown)} more</div>"
                f"<div class='tx-node-meta'><span>not shown</span></div>"
                f"</div>"
            )
        parts.append("</div></div>")

    # Picked leaves with their docs — final animation pass.
    if leaves:
        t_final = 0.60 + len(levels) * _TX_LEVEL_DUR + 0.10
        parts.append(
            f"<div class='tx-leaves-wrap' style='--t-final:{t_final:.2f}s'>"
        )
        for i, leaf in enumerate(leaves):
            label = _esc(str(leaf.get("label", "?")))
            doc_count = int(leaf.get("doc_count") or 0)
            doc_titles = leaf.get("doc_titles") or []
            docs_html = ""
            if doc_titles:
                items = "".join(
                    f"<li title='{_esc(t)}' style='--i-card:{i}; --i-doc:{j}'>"
                    f"{_esc(str(t)[:64])}</li>"
                    for j, t in enumerate(doc_titles)
                )
                docs_html = f"<ul class='tx-leaf-docs'>{items}</ul>"
                if doc_count > len(doc_titles):
                    docs_html += (
                        f"<div style='font-size:0.7rem;color:#64748b;margin-top:2px'>"
                        f"+ {doc_count - len(doc_titles)} more</div>"
                    )
            parts.append(
                f"<div class='tx-leaf-card' style='--i:{i}'>"
                f"<div class='tx-leaf-head'>"
                f"<span class='tx-leaf-title'>📂 {label}</span>"
                f"<span class='tx-leaf-badge'>"
                f"{doc_count} doc{'s' if doc_count != 1 else ''}</span>"
                f"</div>{docs_html}</div>"
            )
        parts.append("</div>")

    parts.append("</div>")  # /tx-stage

    if note:
        parts.append(f"<div class='tx-note'>⚠ {_esc(note)}</div>")

    parts.append("</div>")  # /tx-shell
    return "".join(parts)


def _esc(s: object) -> str:
    """Minimal HTML escape for label / title strings."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Streaming text card — visible blinking caret + "writing" header so the user
# can tell the LLM is actively producing tokens
# ---------------------------------------------------------------------------

_STREAMING_CSS = """
<style>
.hrag-streaming {
    position: relative;
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    border: 1px solid rgba(56,189,248,0.35);
    border-radius: 12px;
    padding: 12px 16px 14px;
    margin: 4px 0 8px;
    color: #f1f5f9;
    box-shadow: 0 0 18px rgba(56,189,248,0.18);
    overflow: hidden;
}
.hrag-streaming::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, #38bdf8 50%, transparent 100%);
    background-size: 200% 100%;
    animation: hrag-stream-sweep 1.6s linear infinite;
}
@keyframes hrag-stream-sweep {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.hrag-streaming-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 6px;
    font-size: 0.74rem; color: #7dd3fc; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.hrag-streaming-head .hrag-streaming-tokens {
    background: rgba(56,189,248,0.15);
    padding: 2px 8px; border-radius: 999px;
    font-family: ui-monospace, monospace; font-weight: 600;
    color: #bae6fd; letter-spacing: 0;
}
.hrag-streaming-body {
    font-size: 0.95rem; line-height: 1.55;
    white-space: pre-wrap; word-wrap: break-word;
}
.hrag-streaming-caret {
    display: inline-block; width: 8px; height: 1.05em;
    margin-left: 2px; vertical-align: text-bottom;
    background: #7dd3fc;
    animation: hrag-caret-blink 1s steps(2, end) infinite;
    border-radius: 1px;
}
@keyframes hrag-caret-blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}
</style>
"""


def _streaming_card(text: str, token_count: int) -> str:
    """Render the in-flight answer inside a styled card with a blinking caret."""
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        _STREAMING_CSS
        + "<div class='hrag-streaming'>"
        + "<div class='hrag-streaming-head'>"
        + "<span>✍️ writing answer</span>"
        + f"<span class='hrag-streaming-tokens'>{token_count} tok</span>"
        + "</div>"
        + "<div class='hrag-streaming-body'>"
        + safe
        + "<span class='hrag-streaming-caret'></span>"
        + "</div></div>"
    )


# ---------------------------------------------------------------------------
# Finalized answer card — visual successor to the streaming card. Same shell
# (gradient bg, padding, border-radius, box-shadow) so the transition out of
# streaming is a settle, not a swap. Caret + sweep animation are gone; accent
# colour shifts from cyan to emerald; the body is rendered as proper markdown
# (not raw escaped text); RAFT "Reasoning: … Answer: …" output is parsed so
# the user sees the Answer prominently and Reasoning collapsed.
# ---------------------------------------------------------------------------

_FINALIZED_CSS = """
<style>
.hrag-final {
    position: relative;
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    border: 1px solid rgba(16,185,129,0.45);
    border-radius: 12px;
    padding: 12px 16px 14px;
    margin: 4px 0 8px;
    color: #f1f5f9;
    box-shadow: 0 0 18px rgba(16,185,129,0.18);
    overflow: hidden;
    /* "Settle" animation eases the colour shift from cyan→emerald so the
       handoff from streaming-card is visually continuous. */
    animation: hrag-final-settle 0.55s ease both;
}
@keyframes hrag-final-settle {
    from {
        border-color: rgba(56,189,248,0.45);
        box-shadow: 0 0 18px rgba(56,189,248,0.30);
    }
    to {
        border-color: rgba(16,185,129,0.45);
        box-shadow: 0 0 18px rgba(16,185,129,0.18);
    }
}
.hrag-final::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,
        rgba(52,211,153,0.0) 0%,
        rgba(52,211,153,0.85) 50%,
        rgba(16,185,129,0.0) 100%);
}
.hrag-final-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px;
    font-size: 0.74rem; color: #6ee7b7; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.hrag-final-head .hrag-final-meta {
    background: rgba(16,185,129,0.15);
    padding: 2px 8px; border-radius: 999px;
    font-family: ui-monospace, monospace; font-weight: 600;
    color: #a7f3d0; letter-spacing: 0;
    text-transform: none;
}
.hrag-final-body {
    font-size: 0.95rem; line-height: 1.6;
    color: #f1f5f9;
    word-wrap: break-word;
}
.hrag-final-body p { margin: 0.5em 0; }
.hrag-final-body p:first-child { margin-top: 0; }
.hrag-final-body p:last-child  { margin-bottom: 0; }
.hrag-final-body h1, .hrag-final-body h2, .hrag-final-body h3 {
    color: #ecfdf5;
    margin: 0.8em 0 0.4em;
    line-height: 1.3;
}
.hrag-final-body h1 { font-size: 1.25rem; }
.hrag-final-body h2 { font-size: 1.10rem; }
.hrag-final-body h3 { font-size: 1.00rem; font-weight: 600; }
.hrag-final-body ul, .hrag-final-body ol {
    margin: 0.4em 0; padding-left: 1.6em;
}
.hrag-final-body li { margin: 0.2em 0; }
.hrag-final-body code {
    background: rgba(255,255,255,0.08);
    padding: 1px 6px; border-radius: 4px;
    font-size: 0.88em;
    font-family: ui-monospace, SF Mono, Menlo, monospace;
    color: #fde68a;
}
.hrag-final-body pre {
    background: rgba(0,0,0,0.35);
    padding: 10px 12px; border-radius: 8px;
    overflow-x: auto;
    font-size: 0.85em;
}
.hrag-final-body pre code {
    background: transparent; padding: 0; color: #f1f5f9;
}
.hrag-final-body blockquote {
    border-left: 3px solid rgba(16,185,129,0.5);
    padding-left: 12px;
    margin: 0.5em 0;
    color: #cbd5e1;
    font-style: italic;
}
.hrag-final-body strong { color: #ffffff; font-weight: 700; }
.hrag-final-body em { color: #fde68a; }
.hrag-final-body a { color: #7dd3fc; text-decoration: underline; }

/* Reasoning details — collapsed by default, soft divider above */
.hrag-final-reasoning {
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px dashed rgba(255,255,255,0.12);
}
.hrag-final-reasoning summary {
    cursor: pointer;
    list-style: none;
    font-size: 0.74rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    user-select: none;
    display: inline-flex; align-items: center; gap: 6px;
    padding: 2px 4px;
    border-radius: 4px;
    transition: color 0.15s;
}
.hrag-final-reasoning summary::-webkit-details-marker { display: none; }
.hrag-final-reasoning summary::before {
    content: "▶";
    display: inline-block;
    font-size: 0.65em;
    color: #6ee7b7;
    transition: transform 0.18s;
}
.hrag-final-reasoning[open] summary::before {
    transform: rotate(90deg);
}
.hrag-final-reasoning summary:hover { color: #cbd5e1; }
.hrag-final-reasoning-body {
    margin-top: 8px;
    padding: 10px 12px;
    background: rgba(0,0,0,0.25);
    border-radius: 8px;
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.hrag-final-reasoning-body code {
    background: rgba(255,255,255,0.07); padding: 0 4px; border-radius: 3px;
}
</style>
"""


# RAFT format: "Reasoning: …\n\nAnswer: …" — produced by prompts/answer.md.
# We split into (answer, reasoning) so the UI can lead with the Answer.
import re as _re  # noqa: E402  — local alias keeps the module-import block tidy
_RAFT_RE = _re.compile(
    r"^\s*Reasoning\s*:\s*(.*?)\s*Answer\s*:\s*(.*)\Z",
    _re.IGNORECASE | _re.DOTALL,
)


def _split_raft(text: str) -> tuple[str, str]:
    """Return ``(answer, reasoning)`` from RAFT-formatted output.

    Falls back to ``(text, "")`` when no Reasoning/Answer markers are found —
    so non-RAFT prompts (chitchat, greeting, personal, general, unclear) work
    unchanged.
    """
    if not text:
        return "", ""
    m = _RAFT_RE.match(text.strip())
    if m:
        reasoning = m.group(1).strip()
        answer = m.group(2).strip()
        return answer, reasoning
    return text.strip(), ""


def _md_to_html(text: str) -> str:
    """Render markdown → HTML with ``markdown-it-py`` (already a project dep).

    On any failure (very rare), fall back to an escaped <pre>-wrap so the
    user never sees an exception in the chat area.
    """
    if not text:
        return ""
    try:
        from markdown_it import MarkdownIt  # noqa: PLC0415
        # Start with commonmark, then opt into GFM-style features (tables,
        # strikethrough). `breaks` turns single newlines into <br>
        # (chat-style behaviour). `linkify` auto-detects bare URLs but
        # depends on optional linkify-it-py — enable only when available.
        opts: dict = {"breaks": True}
        try:
            import linkify_it  # noqa: F401, PLC0415
            opts["linkify"] = True
        except ImportError:
            pass
        md = MarkdownIt("commonmark", opts)
        md.enable(["table", "strikethrough"])
        return md.render(text)
    except Exception:  # noqa: BLE001
        safe = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return f"<div style='white-space:pre-wrap'>{safe}</div>"


def _finalized_answer_card(
    text: str,
    *,
    duration_s: float | None = None,
    token_count: int = 0,
) -> str:
    """Render the completed answer in the same shell as the streaming card.

    Visually continuous with ``_streaming_card``: same gradient background,
    padding, border-radius, box-shadow. Differences: emerald accent (vs cyan),
    no sweeping border / no caret, body is parsed markdown (not raw escaped
    text), RAFT format is split so Reasoning collapses under the Answer.
    """
    answer, reasoning = _split_raft(text)

    meta_bits: list[str] = []
    if duration_s is not None and duration_s > 0:
        meta_bits.append(f"{duration_s:.1f}s")
    if token_count > 0:
        meta_bits.append(f"{token_count} tok")
    meta_label = " · ".join(meta_bits) if meta_bits else "answered"

    body_html = _md_to_html(answer)

    reasoning_block = ""
    if reasoning:
        safe_reasoning = (
            reasoning.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        reasoning_block = (
            "<details class='hrag-final-reasoning'>"
            "<summary>reasoning</summary>"
            f"<div class='hrag-final-reasoning-body'>{safe_reasoning}</div>"
            "</details>"
        )

    return (
        _FINALIZED_CSS
        + "<div class='hrag-final'>"
        + "<div class='hrag-final-head'>"
        + "<span>✅ answer</span>"
        + f"<span class='hrag-final-meta'>{meta_label}</span>"
        + "</div>"
        + f"<div class='hrag-final-body'>{body_html}</div>"
        + reasoning_block
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _render() -> None:
    apply_chrome(page_icon="💬", page_title="Chat · HRAG-Bot")
    page_header(
        "💬 Chat",
        icon="💬",
        subtitle="Streaming chat over your documents + memories. Watch the LLM think and write in real time.",
        tips=[
            "Type <code>/remember &lt;text&gt;</code> to save a memory inline.",
            "Type <code>/recall &lt;query&gt;</code> to semantic-search memories only.",
            "Click 🆕 New session in the sidebar to start fresh — older sessions live in <b>💬 Sessions</b>.",
            "Expand the <b>Sources</b> block after each reply to audit what evidence the LLM saw.",
        ],
    )

    orch = get_orchestrator()
    user_id = current_user_id()

    # ---- Session-state init -------------------------------------------------
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[{role, content, sources}]

    # ---- Sidebar: chat controls --------------------------------------------
    with st.sidebar:
        st.markdown("### 💬 Chat controls")
        if st.button("🆕 New session", width="stretch"):
            st.session_state.chat_session_id = None
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.markdown("### 🔧 Diagnostics")
        if st.button(
            "♻️ Reload classifier / orchestrator",
            width="stretch",
            help=(
                "Rebuild the cached Orchestrator. Use after editing intent.py "
                "or any prompt file mid-session — clears @st.cache_resource so "
                "the next message uses fresh source. Costs ~10 s of cold-load."
            ),
        ):
            st.cache_resource.clear()
            st.session_state.pop("chat_history", None)
            st.session_state.pop("chat_session_id", None)
            st.toast(
                "Orchestrator rebuilt — next message uses fresh code.",
                icon="♻️",
            )
            st.rerun()

        active = st.session_state.chat_session_id or "new"
        st.markdown(
            f"<div style='font-size:0.8rem;color:#9ca3af;margin-top:6px'>"
            f"active session: <span style='background:#1f2937;color:#e5e7eb;"
            f"padding:1px 6px;border-radius:6px;font-family:ui-monospace,monospace;"
            f"font-size:0.75rem'>{active[:16]}</span></div>",
            unsafe_allow_html=True,
        )

        cfg = orch.config

        # Live toggle for Ollama think mode — mutates the cached orchestrator's
        # LLMConfig in place. `OllamaProvider._build_chat_kwargs` reads it on
        # every call, so the next message picks up the new value.
        if cfg.llm.provider == "ollama":
            current_think = bool(cfg.llm.think) if cfg.llm.think is not None else False
            new_think = st.toggle(
                "🧠 Think mode",
                value=current_think,
                help=(
                    "When ON, the LLM emits internal reasoning tokens before "
                    "the answer (slower, better on hard questions). When OFF, "
                    "it answers directly (much faster on short prompts like "
                    "gate / clue / classifier). Only models with the `thinking` "
                    "capability are affected."
                ),
                key="chat_think_mode",
            )
            if new_think != current_think:
                cfg.llm.think = new_think
                st.toast(
                    f"Think mode {'on' if new_think else 'off'}",
                    icon="🧠",
                )

        st.markdown("### 🔧 Pipeline")
        rerank_label = cfg.retrieval.reranker if cfg.retrieval.rerank_enabled else "off"
        _pill = (
            "background:#1f2937;color:#e5e7eb;padding:1px 6px;"
            "border-radius:6px;font-family:ui-monospace,monospace;font-size:0.8rem"
        )
        think_label = (
            "on" if cfg.llm.think
            else "off" if cfg.llm.think is False
            else "model default"
        )
        st.markdown(
            f"<div style='font-size:0.85rem;line-height:1.8;color:#cbd5e1'>"
            f"<b>LLM</b> · <span style='{_pill}'>{cfg.llm.provider}</span> "
            f"/ <span style='{_pill}'>{cfg.llm.model}</span><br>"
            f"<b>Think</b> · <span style='{_pill}'>{think_label}</span><br>"
            f"<b>Retriever</b> · <span style='{_pill}'>{cfg.retrieval.retriever}</span><br>"
            f"<b>Reranker</b> · <span style='{_pill}'>{rerank_label}</span><br>"
            f"<b>top_k</b> · <span style='{_pill}'>"
            f"{cfg.retrieval.top_k_vector} → {cfg.retrieval.top_k_final}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### 📚 Recent sessions")
        recents = orch.db.execute(
            "SELECT session_id, started_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS n "
            "FROM sessions s WHERE user_id = ? "
            "ORDER BY started_at DESC LIMIT 3",
            (user_id,),
        ).fetchall()
        if recents:
            for r in recents:
                sid = r["session_id"]
                started = r["started_at"] or ""
                n = r["n"]
                if st.button(
                    f"💬 {sid[:10]} · {n} msgs · {started[:10]}",
                    key=f"sess_{sid}",
                ):
                    st.session_state.chat_session_id = sid
                    st.session_state.chat_history = []
                    st.rerun()
        else:
            st.caption("No prior sessions yet.")

    # ---- Empty state (no messages yet) --------------------------------------
    if not st.session_state.chat_history:
        empty_state(
            icon="💬",
            title="Ready when you are",
            message="Type a question below. Use /remember to save a note · /recall to search memories.",
        )

    # ---- Replay history ------------------------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            # Prefer the unified reasoning-trace snapshot when available;
            # fall back to the legacy descend-only payload for old turns.
            if msg.get("trace"):
                _replay_html, _replay_h = _render_reasoning_trace(msg["trace"])
                components.html(_replay_html, height=_replay_h, scrolling=False)
            elif msg.get("descend"):
                _replay_html, _replay_h = _render_tree_component(msg["descend"])
                components.html(_replay_html, height=_replay_h, scrolling=False)
            # Assistant turns get the finalized-answer card (RAFT parsed,
            # markdown rendered, reasoning collapsed). User turns are plain.
            if msg["role"] == "assistant":
                st.markdown(
                    _finalized_answer_card(msg["content"]),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        render_source(i, src)

    # ---- Slash-command hint chips -------------------------------------------
    st.markdown(
        f"<div style='margin-bottom:6px;'>"
        f"{chip('/remember &lt;text&gt;', 'violet')} {chip('/recall &lt;query&gt;', 'info')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---- Input ---------------------------------------------------------------
    prompt = st.chat_input(
        "Ask anything · /remember <text> · /recall <query>"
    )
    if not prompt:
        return

    # ---- Slash shortcuts (inline-handled, no LLM) ----------------------------
    if prompt.startswith("/remember "):
        text = prompt[len("/remember "):].strip()
        if text:
            mid = orch.memory_store.add(
                user_id,
                text,
                session_id=st.session_state.chat_session_id,
                source="gui",
            )
            st.toast(f"💾 Memory saved · {mid[:24]}…", icon="✅")
        return

    if prompt.startswith("/recall "):
        query = prompt[len("/recall "):].strip()
        hits = orch.retriever.retrieve(
            query, user_id, top_k=5, source_types=["episodic"]
        )
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        body = (
            "\n".join(
                f"- **{h.chunk.title or 'Untitled'}** — {h.chunk.text[:160]}…"
                for h in hits
            )
            if hits
            else "_no matching memories_"
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"### Recall hits\n{body}",
                "sources": hits,
            }
        )
        st.rerun()
        return

    # ---- Normal streaming chat turn -----------------------------------------
    st.session_state.chat_history.append(
        {"role": "user", "content": prompt, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_slot = st.empty()
        trace_slot = st.empty()      # the live reasoning-trace iframe lives here
        text_slot  = st.empty()

        status_slot.markdown(_phase_badge("boot"), unsafe_allow_html=True)

        # Build up the reasoning-trace payload as events arrive. The component
        # is re-rendered every time a meaningful field is added (intent verdict,
        # taxonomy descend, memory snippets, swap). Each render replaces the
        # iframe at `trace_slot` — Streamlit handles the diff for us.
        trace_state: dict[str, object] = {"query": prompt}

        # Phase 4 compaction events accumulated during this turn.
        # Keys match the four progress event names; values are their payloads.
        compaction_events: dict[str, dict] = {}

        def _draw_trace() -> None:
            """Re-render the reasoning-trace iframe with whatever state we have."""
            try:
                html, height = _render_reasoning_trace(trace_state)
                with trace_slot:
                    components.html(html, height=height, scrolling=False)
            except Exception:  # noqa: BLE001  — UI rendering must never block the LLM
                pass

        stream = stream_chat_events(
            orch, prompt, user_id=user_id, session_id=st.session_state.chat_session_id
        )

        partial_text = ""
        token_count = 0
        t_first_token: float | None = None
        phase_seen: dict[str, float] = {}
        t_start = time.perf_counter()

        for ev in stream.events():
            kind = ev.event
            payload = ev.payload

            if kind == "query_rewrite":
                status_slot.markdown(_phase_badge("rewrite"), unsafe_allow_html=True)
                phase_seen["rewrite"] = time.perf_counter()

            elif kind == "router_classify":
                label = payload.get("label", "?")
                status_slot.markdown(
                    _phase_badge("route", detail=f"label = {label}"),
                    unsafe_allow_html=True,
                )

            elif kind == "intent_check":
                # Verdict in — populate the trace and render its first frame.
                if not payload.get("enabled"):
                    st.caption("ℹ️ intent gate disabled in config (everything will run as FACTUAL)")
                intent_label = payload.get("intent", "factual")
                trace_state["intent"]     = intent_label
                trace_state["confidence"] = float(payload.get("confidence") or 0.0)
                trace_state["source"]     = payload.get("source", "?")
                # raw_label carries the matched topic terms when source is
                # "named_topic" (e.g. "hipporag"). The reasoning trace shows
                # this in the thinking narration.
                trace_state["raw_label"]  = payload.get("raw_label")
                phase_key = f"intent_{intent_label}"
                if phase_key not in _PHASE_LABELS:
                    phase_key = "intent_factual"
                status_slot.markdown(
                    _phase_badge(
                        phase_key,
                        detail=f"{trace_state['confidence']:.2f} · {trace_state['source']}",
                    ),
                    unsafe_allow_html=True,
                )
                _draw_trace()

            elif kind == "intent_route":
                scope = payload.get("scope", "full")
                # If the orchestrator's post-retrieval check swapped FACTUAL
                # to GENERAL, mark it on the trace state so the narration line
                # reads "I tried, but found nothing useful — using general
                # knowledge" and the tree (if any) stays visible alongside.
                if payload.get("swapped_from") == "factual":
                    trace_state["intent"] = "general"
                    trace_state["swapped_from_factual"] = True
                    if payload.get("top_score") is not None:
                        trace_state["swap_top_score"] = float(payload.get("top_score") or 0.0)
                    status_slot.markdown(
                        _phase_badge(
                            "intent_general",
                            detail=(
                                f"no corpus match · top score "
                                f"{float(payload.get('top_score') or 0.0):.2f}"
                            ),
                        ),
                        unsafe_allow_html=True,
                    )
                    _draw_trace()
                # Greeting / unclear paths: nothing more to plumb here — the
                # trace's `renderSkip` branch handles them once intent_check
                # has filled in the verdict.
                _ = scope  # kept for future per-scope UI hooks

            elif kind == "taxonomy_descend":
                # The retriever delivered its descend trace — splice it into
                # the reasoning trace so the embedded tree visualization fills
                # in alongside the thinking narration.
                trace_state["taxonomy"] = payload
                st.session_state["last_descend"] = payload   # legacy compat
                stats = payload.get("stats") or {}
                opened = stats.get("docs_opened", 0)
                total = stats.get("total_docs", 0)
                detail = (
                    f"opened {opened} of {total} docs"
                    if total
                    else f"{len(payload.get('leaves') or [])} leaves"
                )
                status_slot.markdown(
                    _phase_badge("taxonomy", detail=detail),
                    unsafe_allow_html=True,
                )
                _draw_trace()

            elif kind == "personal_memories":
                # Memory previews arrived for a PERSONAL intent — render them
                # as cards inside the reasoning trace.
                trace_state["memories"] = payload.get("memories") or []
                _draw_trace()

            elif kind == "retrieve":
                status_slot.markdown(
                    _phase_badge(
                        "retrieve",
                        detail=(
                            f"{payload['n_results']} chunks · "
                            f"{payload['duration_s']:.2f}s"
                        ),
                    ),
                    unsafe_allow_html=True,
                )

            elif kind == "rerank_step":
                # Don't redraw per chunk (too chatty); just update the dot pulse.
                pass

            elif kind == "rerank_done":
                status_slot.markdown(
                    _phase_badge(
                        "rerank",
                        detail=(
                            f"kept {payload['kept']} in "
                            f"{payload['duration_s']:.2f}s"
                            + (" · fallback" if payload.get("fallback_used") else "")
                        ),
                    ),
                    unsafe_allow_html=True,
                )

            elif kind == "organize_done":
                status_slot.markdown(
                    _phase_badge(
                        "organize",
                        detail=f"{payload['input']} → {payload['output']}",
                    ),
                    unsafe_allow_html=True,
                )

            elif kind == "gate_check":
                compaction_events["gate_check"] = payload

            elif kind == "clue_generate":
                compaction_events["clue_generate"] = payload

            elif kind == "dialog_compact":
                compaction_events["dialog_compact"] = payload

            elif kind == "uncertain_render":
                compaction_events["uncertain_render"] = payload

            elif kind == "generate_start":
                status_slot.markdown(
                    _phase_badge("wait_llm"), unsafe_allow_html=True
                )

            elif kind == "generate_token":
                token = payload.get("token", "")
                if not token:
                    continue
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                    status_slot.markdown(_phase_badge("write"), unsafe_allow_html=True)
                token_count += 1
                partial_text += token
                # Throttle screen updates to ~30 fps to avoid websocket flood
                # while still feeling live. The blinking-caret CSS gives the
                # impression of typing even when redraws are throttled.
                now = time.perf_counter()
                if now - phase_seen.get("last_draw", 0.0) >= 0.033 or token in ("\n", "."):
                    text_slot.markdown(
                        _streaming_card(partial_text, token_count),
                        unsafe_allow_html=True,
                    )
                    phase_seen["last_draw"] = now

            elif kind == "generate":
                # Final flush — settle into the finalized-answer card so the
                # transition out of the streaming-card is visually continuous.
                # Same shell, accent shifts cyan→emerald, RAFT format gets
                # parsed (Reasoning collapsed, Answer prominent), markdown is
                # rendered instead of shown as raw text.
                if partial_text:
                    text_slot.markdown(
                        _finalized_answer_card(
                            partial_text,
                            duration_s=payload.get("duration_s"),
                            token_count=token_count,
                        ),
                        unsafe_allow_html=True,
                    )
                status_slot.markdown(
                    _phase_badge(
                        "done",
                        detail=f"generated in {payload['duration_s']:.2f}s · "
                        f"{payload['answer_chars']} chars",
                    ),
                    unsafe_allow_html=True,
                )

            elif kind == "done":
                total = payload.get("total_s", time.perf_counter() - t_start)
                ttft = (
                    f"TTFT {t_first_token - t_start:.2f}s · "
                    if t_first_token else ""
                )
                status_slot.markdown(
                    _phase_badge(
                        "done", detail=f"{ttft}total {total:.2f}s",
                    ),
                    unsafe_allow_html=True,
                )

        # Error path
        if stream.error is not None:
            text_slot.error(f"LLM error: {stream.error}")
            return

        result = stream.result
        if result is None:
            text_slot.warning("No result was produced.")
            return

        # Pin session, finalize text, render sources block. If the streaming
        # buffer and the canonical answer disagree (cleanup or post-processing
        # happened), re-render the finalized card with the canonical text.
        st.session_state.chat_session_id = result.session_id
        if partial_text != result.answer:
            text_slot.markdown(
                _finalized_answer_card(result.answer, token_count=token_count),
                unsafe_allow_html=True,
            )

        if result.sources:
            with st.expander(f"📚 Sources ({len(result.sources)})", expanded=False):
                for i, src in enumerate(result.sources, 1):
                    render_source(i, src)

        # Phase 4 compaction expander — only shown when at least one event fired.
        if compaction_events:
            with st.expander("🚦 Compaction", expanded=False):
                _pill = (
                    "background:#1f2937;color:#e5e7eb;padding:1px 6px;"
                    "border-radius:6px;font-family:ui-monospace,monospace;font-size:0.8rem"
                )
                rows: list[str] = []
                if "gate_check" in compaction_events:
                    g = compaction_events["gate_check"]
                    decision = g.get("decision", "?")
                    dur = g.get("duration_s", 0.0)
                    color = "#10b981" if decision == "RETRIEVE" else "#f59e0b"
                    rows.append(
                        f"<b>Gate</b> · "
                        f"<span style='color:{color};font-weight:700'>{decision}</span> "
                        f"<span style='{_pill}'>{dur:.2f}s</span>"
                    )
                if "clue_generate" in compaction_events:
                    c = compaction_events["clue_generate"]
                    clue_text = str(c.get("clue", ""))[:200]
                    dur = c.get("duration_s", 0.0)
                    safe_clue = (
                        clue_text
                        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    )
                    rows.append(
                        f"<b>Clue</b> · "
                        f"<span style='color:#cbd5e1;font-style:italic'>\"{safe_clue}\"</span> "
                        f"<span style='{_pill}'>{dur:.2f}s</span>"
                    )
                if "dialog_compact" in compaction_events:
                    d = compaction_events["dialog_compact"]
                    inp = d.get("input_turns", "?")
                    out = d.get("output_turns", "?")
                    dur = d.get("duration_s", 0.0)
                    rows.append(
                        f"<b>Dialog compaction</b> · "
                        f"<span style='{_pill}'>{inp} → {out} turns</span> "
                        f"<span style='{_pill}'>{dur:.2f}s</span>"
                    )
                if "uncertain_render" in compaction_events:
                    u = compaction_events["uncertain_render"]
                    count = u.get("count", 0)
                    rows.append(
                        f"<b>Uncertain markers</b> · "
                        f"<span style='{_pill}'>{count} [UNCERTAIN]</span>"
                    )
                st.markdown(
                    "<div style='font-size:0.87rem;line-height:2.0;color:#cbd5e1'>"
                    + "<br>".join(rows)
                    + "</div>",
                    unsafe_allow_html=True,
                )

        # The reasoning trace is already rendered above the answer (filled
        # progressively as events arrived). Persist the trace state in chat
        # history so replays show the same thing. Keep `descend` too as a
        # legacy compatibility key for prior session-state shape.
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
                "trace": dict(trace_state),
                "descend": st.session_state.get("last_descend"),
            }
        )


_render()
