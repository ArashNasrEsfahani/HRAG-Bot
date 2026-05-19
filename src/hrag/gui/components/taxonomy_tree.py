"""Taxonomy-tree visualization component (SVG + JS, mint/emerald theme).

Rendered via ``streamlit.components.v1.html`` into an iframe — gives us full
browser capabilities (SVG paths with stroke-dashoffset animation, CSS
keyframes, JavaScript-driven sequenced reveal) that ``st.markdown`` can't
match. The component is fully self-contained: a single HTML string with
embedded ``<style>`` and ``<script>``.

Usage from the chat page::

    from hrag.gui.components.taxonomy_tree import render_tree
    import streamlit.components.v1 as components

    html, height = render_tree(descend_payload)
    components.html(html, height=height, scrolling=False)

The animation sequence:
  1. Header banner fades in with the "Opened X of Y docs" stats.
  2. Query bubble pops in with a soft emerald glow + pulse.
  3. Stem line grows downward to the first level.
  4. Level-0 nodes pop in one by one (elastic ease).
  5. After a beat, pruned nodes drift down + fade; kept nodes get an aura.
  6. Edges from each kept node grow downward to the next row.
  7. Repeat for each level.
  8. Leaf "doc cards" expand at the bottom, each doc line slides up.
"""

from __future__ import annotations

import json
import secrets
from typing import Any


# Tunables for layout / pacing
_ROW_HEIGHT_PX = 120                      # vertical distance between levels
_NODE_W = 180                             # node card width
_NODE_H = 64                              # node card height
_LEAF_CARD_MIN_W = 240                    # minimum doc-card width
_LEAF_CARD_HEIGHT = 220                   # doc-card height
_HEADER_HEIGHT_PX = 96                    # banner height
_QUERY_HEIGHT_PX = 80                     # query-bubble vertical space
_BOTTOM_PAD_PX = 32                       # padding under everything

_DEFAULT_HEIGHT_PX = 480


def render_tree(payload: dict[str, Any]) -> tuple[str, int]:
    """Build the self-contained HTML and recommended iframe height.

    Returns:
        (html, height_px) — pass both to ``components.v1.html(html, height=h)``.
    """
    if not payload:
        return _empty_html(), 140

    trace = payload.get("trace") or []
    leaves = payload.get("leaves") or []
    note = payload.get("note")

    # Only count levels that actually had candidates considered.
    real_levels = [lv for lv in trace if (lv.get("considered") or [])]
    n_levels = max(1, len(real_levels))

    # Estimate iframe height: header + query + level rows + leaf cards + pad.
    height = (
        _HEADER_HEIGHT_PX
        + _QUERY_HEIGHT_PX
        + n_levels * _ROW_HEIGHT_PX
        + (_LEAF_CARD_HEIGHT if leaves else 0)
        + _BOTTOM_PAD_PX
        + 40  # safety margin so the bottom doc-row isn't clipped
    )
    # Note errors push the banner taller.
    if note:
        height += 36

    # Unique DOM id per render so multiple instances (history replay) don't
    # clash on the JS handler hookup.
    dom_id = f"hrag-tree-{secrets.token_hex(4)}"

    data_json = json.dumps(payload, ensure_ascii=False, default=str)
    return _build_html(dom_id, data_json, height), height


# ---------------------------------------------------------------------------
# HTML construction
# ---------------------------------------------------------------------------


def _empty_html() -> str:
    return (
        "<div style='padding:16px;font-family:system-ui;color:#065f46;"
        "background:linear-gradient(180deg,#f0fdf4,#ecfdf5);border:1px solid "
        "#a7f3d0;border-radius:12px;font-size:0.85rem'>"
        "🌱 No descent trace available yet."
        "</div>"
    )


def _build_html(dom_id: str, data_json: str, height: int) -> str:
    """Assemble the full HTML document. The data JSON is embedded inline so the
    iframe has no network dependencies."""
    # Triple-curly trick: in f-strings, '{{' renders as a literal '{'. CSS and
    # JS are NOT f-strings — they're regular strings concatenated at the end,
    # so we don't need to double-brace inside them. Only the small Python
    # `.format()` substitutions at the bottom use {{placeholder}} syntax.

    css = _CSS_BLOCK
    js = _JS_TEMPLATE.replace("__DOM_ID__", dom_id).replace(
        "__DATA_JSON__", data_json
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{css}</style>
</head>
<body>
<div id="{dom_id}" class="hrag-tree-shell">
  <div class="hrag-tree-banner">
    <div class="hrag-tree-banner-text">
      <span class="hrag-tree-emoji">🌿</span>
      <span class="hrag-tree-title">Hierarchical retrieval</span>
      <span class="hrag-tree-subtitle" data-role="subtitle">Loading…</span>
    </div>
    <div class="hrag-tree-stats" data-role="stats"></div>
  </div>
  <div class="hrag-tree-stage" data-role="stage">
    <svg class="hrag-tree-svg" data-role="svg" xmlns="http://www.w3.org/2000/svg"
         preserveAspectRatio="xMidYMin meet"></svg>
    <div class="hrag-tree-nodes" data-role="nodes"></div>
  </div>
  <div class="hrag-tree-leaves" data-role="leaves"></div>
  <div class="hrag-tree-note" data-role="note"></div>
  <!-- Sparkle layer for decorative particles -->
  <div class="hrag-tree-sparkles" data-role="sparkles"></div>
</div>
<script>{js}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Stylesheet — light emerald / mint theme with high-tech glow
# ---------------------------------------------------------------------------

_CSS_BLOCK = r"""
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: transparent;
  color: #064e3b;
}
.hrag-tree-shell {
  position: relative;
  background:
    radial-gradient(ellipse at top, rgba(16,185,129,0.08) 0%, transparent 60%),
    linear-gradient(180deg, #f0fdf4 0%, #ecfdf5 60%, #d1fae5 100%);
  border: 1px solid #a7f3d0;
  border-radius: 16px;
  padding: 14px 18px 20px;
  margin: 4px 0 12px;
  overflow: hidden;
  box-shadow:
    0 10px 30px rgba(16,185,129,0.10),
    inset 0 0 0 1px rgba(255,255,255,0.6);
}
/* Subtle animated grid backdrop — high-tech feel */
.hrag-tree-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(90deg, rgba(16,185,129,0.06) 1px, transparent 1px) 0 0 / 32px 32px,
    linear-gradient(0deg,  rgba(16,185,129,0.06) 1px, transparent 1px) 0 0 / 32px 32px;
  mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
  -webkit-mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
  pointer-events: none;
  animation: hrag-grid-shift 18s linear infinite;
}
@keyframes hrag-grid-shift {
  from { background-position: 0 0, 0 0; }
  to   { background-position: 64px 64px, 64px 64px; }
}

/* -- Banner --------------------------------------------------------------- */
.hrag-tree-banner {
  position: relative;
  display: flex; align-items: center; justify-content: space-between;
  gap: 14px; flex-wrap: wrap;
  padding: 12px 16px;
  border-radius: 12px;
  background: linear-gradient(135deg,
    rgba(167,243,208,0.55) 0%,
    rgba(110,231,183,0.35) 60%,
    rgba(52,211,153,0.25) 100%);
  border: 1px solid rgba(16,185,129,0.45);
  margin-bottom: 14px;
  box-shadow: 0 4px 18px rgba(16,185,129,0.18);
  animation: hrag-banner-in 0.6s 0.05s cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes hrag-banner-in {
  from { opacity: 0; transform: translateY(-12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.hrag-tree-banner-text {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  color: #065f46; font-weight: 600;
}
.hrag-tree-emoji {
  font-size: 1.2rem;
  filter: drop-shadow(0 0 6px rgba(16,185,129,0.6));
}
.hrag-tree-title {
  font-size: 0.98rem; letter-spacing: 0.01em;
}
.hrag-tree-subtitle {
  font-size: 0.85rem; color: #047857; font-weight: 500;
}
.hrag-tree-subtitle b { color: #064e3b; font-weight: 700; }
.hrag-tree-stats { display: flex; gap: 8px; flex-wrap: wrap; }
.hrag-tree-stat {
  background: rgba(255,255,255,0.7);
  border: 1px solid rgba(16,185,129,0.45);
  color: #065f46;
  font-size: 0.74rem; padding: 3px 10px; border-radius: 999px;
  font-weight: 600;
  box-shadow: 0 2px 6px rgba(16,185,129,0.10);
}
.hrag-tree-stat b { color: #064e3b; }

/* -- Stage (SVG + nodes layered on top) ----------------------------------- */
.hrag-tree-stage {
  position: relative;
  width: 100%;
}
.hrag-tree-svg {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  overflow: visible;
}
.hrag-tree-nodes {
  position: relative;
  width: 100%;
}

/* -- Query bubble --------------------------------------------------------- */
.hrag-query-bubble {
  position: absolute;
  display: inline-flex; align-items: center; gap: 7px;
  padding: 9px 18px;
  border-radius: 999px;
  background: linear-gradient(135deg, #34d399 0%, #10b981 60%, #059669 100%);
  color: #ecfdf5; font-weight: 700; font-size: 0.86rem;
  box-shadow:
    0 0 0 4px rgba(16,185,129,0.18),
    0 8px 28px rgba(16,185,129,0.35),
    inset 0 1px 0 rgba(255,255,255,0.35);
  transform-origin: center bottom;
  opacity: 0; transform: translateX(-50%) scale(0.4);
  animation: hrag-query-pop 0.6s 0.30s cubic-bezier(.34,1.56,.64,1) forwards,
             hrag-query-pulse 2.2s 1.2s ease-in-out infinite;
  white-space: nowrap;
}
@keyframes hrag-query-pop {
  0%   { opacity: 0; transform: translateX(-50%) scale(0.4); }
  60%  { opacity: 1; transform: translateX(-50%) scale(1.12); }
  100% { opacity: 1; transform: translateX(-50%) scale(1.0); }
}
@keyframes hrag-query-pulse {
  0%, 100% { box-shadow: 0 0 0 4px rgba(16,185,129,0.18), 0 8px 28px rgba(16,185,129,0.35); }
  50%      { box-shadow: 0 0 0 10px rgba(16,185,129,0.10), 0 8px 38px rgba(16,185,129,0.5); }
}

/* -- Tree node card ------------------------------------------------------- */
.hrag-node {
  position: absolute;
  width: 180px; min-height: 64px;
  padding: 8px 12px;
  border-radius: 12px;
  background: rgba(255,255,255,0.92);
  border: 1.5px solid #a7f3d0;
  color: #064e3b;
  font-size: 0.82rem;
  font-weight: 500;
  display: flex; flex-direction: column; justify-content: center; gap: 4px;
  opacity: 0; transform: translateY(-10px) scale(0.4);
  transform-origin: center top;
  box-shadow: 0 4px 14px rgba(16,185,129,0.10);
  animation: hrag-node-pop 0.55s cubic-bezier(.34,1.56,.64,1) forwards;
}
@keyframes hrag-node-pop {
  0%   { opacity: 0; transform: translateY(-12px) scale(0.4); }
  60%  { opacity: 1; transform: translateY(0) scale(1.08); }
  100% { opacity: 1; transform: translateY(0) scale(1.0); }
}
.hrag-node-label {
  font-weight: 700;
  overflow: hidden; text-overflow: ellipsis;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  line-height: 1.25;
  color: #064e3b;
}
.hrag-node-meta {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.7rem; color: #047857;
}
.hrag-node-score {
  background: rgba(16,185,129,0.15);
  color: #065f46;
  font-family: ui-monospace, SF Mono, Menlo, monospace;
  font-weight: 700;
  padding: 1px 7px; border-radius: 999px;
}

/* Kept node — bright glow, gradient, star. */
.hrag-node.kept {
  background: linear-gradient(135deg,
    #d1fae5 0%, #a7f3d0 55%, #6ee7b7 100%);
  border: 1.5px solid #10b981;
  color: #064e3b;
  box-shadow:
    0 0 22px rgba(16,185,129,0.55),
    inset 0 0 0 1px rgba(255,255,255,0.6);
  animation:
    hrag-node-pop 0.55s cubic-bezier(.34,1.56,.64,1) forwards,
    hrag-kept-glow 2.4s 0.6s ease-in-out infinite;
}
@keyframes hrag-kept-glow {
  0%, 100% { box-shadow: 0 0 22px rgba(16,185,129,0.55), inset 0 0 0 1px rgba(255,255,255,0.6); }
  50%      { box-shadow: 0 0 36px rgba(16,185,129,0.85), inset 0 0 0 1px rgba(255,255,255,0.8); }
}
.hrag-node.kept .hrag-node-score {
  background: rgba(255,255,255,0.75); color: #047857;
}
.hrag-node-star {
  position: absolute;
  top: -10px; right: -8px;
  width: 22px; height: 22px;
  border-radius: 50%;
  background: linear-gradient(135deg, #fde68a, #f59e0b);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem;
  box-shadow: 0 0 10px rgba(245,158,11,0.65);
  color: #78350f;
  animation: hrag-star-spin 4s linear infinite;
}
@keyframes hrag-star-spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}

/* Pruned node — drift down + fade. */
.hrag-node.pruned {
  background: rgba(241,245,249,0.85);
  border: 1.5px dashed #cbd5e1;
  color: #64748b;
  animation:
    hrag-node-pop 0.55s cubic-bezier(.34,1.56,.64,1) forwards,
    hrag-prune 0.7s 0.85s cubic-bezier(.55,.06,.68,.19) forwards;
}
@keyframes hrag-prune {
  0%   { opacity: 1; transform: translateY(0) scale(1.0) rotate(0deg); filter: grayscale(0); }
  100% { opacity: 0.35; transform: translateY(10px) scale(0.88) rotate(-3deg); filter: grayscale(0.6); }
}
.hrag-node.pruned .hrag-node-score { background: rgba(0,0,0,0.05); color: #64748b; }

/* -- SVG edges ------------------------------------------------------------ */
.hrag-edge {
  fill: none;
  stroke: url(#hragEdgeGradient);
  stroke-width: 2.2;
  stroke-linecap: round;
  filter: drop-shadow(0 0 4px rgba(16,185,129,0.45));
  stroke-dasharray: var(--edge-len, 200);
  stroke-dashoffset: var(--edge-len, 200);
  animation: hrag-edge-draw 0.55s var(--edge-delay, 0s) ease-out forwards;
}
@keyframes hrag-edge-draw {
  to { stroke-dashoffset: 0; }
}
.hrag-edge.pruned-edge {
  stroke: #cbd5e1;
  filter: none;
  animation:
    hrag-edge-draw 0.55s var(--edge-delay, 0s) ease-out forwards,
    hrag-edge-fade 0.6s calc(var(--edge-delay, 0s) + 0.7s) ease-out forwards;
  opacity: 1;
}
@keyframes hrag-edge-fade {
  to { opacity: 0.25; }
}

/* Animated traveling particle along edges (kept ones) */
.hrag-edge-particle {
  fill: #34d399;
  filter: drop-shadow(0 0 6px #34d399);
  offset-rotate: 0deg;
  animation: hrag-particle-travel var(--edge-particle-dur, 1.8s) var(--edge-particle-delay, 0s)
    linear infinite;
}
@keyframes hrag-particle-travel {
  from { offset-distance: 0%;  opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  to   { offset-distance: 100%; opacity: 0; }
}

/* -- Level label ---------------------------------------------------------- */
.hrag-level-label {
  position: absolute;
  left: 0;
  font-size: 0.66rem;
  color: #059669;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 8px;
  background: rgba(255,255,255,0.7);
  border: 1px solid rgba(16,185,129,0.3);
  border-radius: 6px;
  animation: hrag-label-in 0.4s var(--label-delay, 0s) ease-out both;
}
@keyframes hrag-label-in {
  from { opacity: 0; transform: translateX(-6px); }
  to   { opacity: 1; transform: translateX(0); }
}

/* -- Leaf doc cards (the payoff) ----------------------------------------- */
.hrag-tree-leaves {
  margin-top: 18px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}
.hrag-leaf-card {
  background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);
  border: 1.5px solid #6ee7b7;
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 6px 22px rgba(16,185,129,0.15);
  position: relative;
  overflow: hidden;
  transform-origin: top center;
  animation: hrag-leaf-open 0.7s var(--leaf-delay, 0s) cubic-bezier(.34,1.56,.64,1) both;
}
.hrag-leaf-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #34d399, #10b981, #059669);
  background-size: 200% 100%;
  animation: hrag-leaf-shimmer 3.5s linear infinite;
}
@keyframes hrag-leaf-shimmer {
  from { background-position: 200% 0; }
  to   { background-position: -200% 0; }
}
@keyframes hrag-leaf-open {
  0%   { opacity: 0; transform: scale(0.5) rotate(-3deg); }
  60%  { opacity: 1; transform: scale(1.06) rotate(1deg); }
  100% { opacity: 1; transform: scale(1.0)  rotate(0deg); }
}
.hrag-leaf-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 8px; gap: 8px;
}
.hrag-leaf-title {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.92rem; font-weight: 800; color: #064e3b;
}
.hrag-leaf-badge {
  background: rgba(16,185,129,0.2);
  color: #065f46;
  font-size: 0.7rem; font-weight: 700;
  padding: 2px 9px; border-radius: 999px;
}
.hrag-leaf-docs { list-style: none; margin: 0; padding: 0; font-size: 0.78rem; }
.hrag-leaf-docs li {
  padding: 4px 0 4px 18px;
  position: relative;
  color: #064e3b;
  font-family: ui-monospace, SF Mono, Menlo, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  animation: hrag-doc-slide 0.45s var(--doc-delay, 0s) ease-out both;
}
.hrag-leaf-docs li::before {
  content: "📄";
  position: absolute; left: 0; top: 3px;
  font-size: 0.8rem;
}
@keyframes hrag-doc-slide {
  from { opacity: 0; transform: translateX(-8px); }
  to   { opacity: 1; transform: translateX(0); }
}
.hrag-leaf-more {
  margin-top: 6px;
  font-size: 0.72rem;
  color: #047857;
  font-style: italic;
}

/* -- Note (e.g. tree empty fallback) ------------------------------------- */
.hrag-tree-note {
  margin-top: 12px;
  padding: 9px 14px;
  background: rgba(254,243,199,0.85);
  border: 1px solid #fcd34d;
  border-radius: 10px;
  color: #78350f;
  font-size: 0.82rem;
  display: none;
}

/* -- Sparkles (decorative floating particles) ---------------------------- */
.hrag-tree-sparkles {
  position: absolute; inset: 0; pointer-events: none;
  overflow: hidden;
}
.hrag-sparkle {
  position: absolute;
  width: 4px; height: 4px;
  border-radius: 50%;
  background: radial-gradient(circle, #34d399 0%, transparent 70%);
  filter: blur(0.4px);
  opacity: 0;
  animation: hrag-sparkle-float var(--sp-dur, 6s) var(--sp-delay, 0s) linear infinite;
}
@keyframes hrag-sparkle-float {
  0%   { opacity: 0; transform: translateY(0) scale(0.6); }
  10%  { opacity: 0.8; }
  90%  { opacity: 0.8; }
  100% { opacity: 0; transform: translateY(-180px) scale(1.4); }
}
"""


# ---------------------------------------------------------------------------
# JS — layout, sequencing, SVG path generation
# ---------------------------------------------------------------------------

_JS_TEMPLATE = r"""
(function() {
  const DATA = __DATA_JSON__;
  const root = document.getElementById("__DOM_ID__");
  if (!root || !DATA) return;

  // ---- layout constants (must match Python tunables) -------------------
  const ROW_H = 120;
  const NODE_W = 180;
  const NODE_H = 64;
  const QUERY_OFFSET_Y = 36;
  const STAGE_TOP_PAD = 18;

  // ---- DOM handles -----------------------------------------------------
  const subtitleEl = root.querySelector('[data-role="subtitle"]');
  const statsEl    = root.querySelector('[data-role="stats"]');
  const stageEl    = root.querySelector('[data-role="stage"]');
  const svgEl      = root.querySelector('[data-role="svg"]');
  const nodesEl    = root.querySelector('[data-role="nodes"]');
  const leavesEl   = root.querySelector('[data-role="leaves"]');
  const noteEl     = root.querySelector('[data-role="note"]');
  const sparklesEl = root.querySelector('[data-role="sparkles"]');

  // ---- banner ----------------------------------------------------------
  const stats = DATA.stats || {};
  const totalDocs   = stats.total_docs || 0;
  const docsOpened  = stats.docs_opened || 0;
  const leavesPicked= stats.leaves_picked || (DATA.leaves || []).length;
  const nodesCons   = stats.nodes_considered || 0;
  const pct = totalDocs > 0 ? Math.max(1, Math.round(docsOpened * 100 / totalDocs)) : 0;

  if (totalDocs > 0) {
    subtitleEl.innerHTML = `Opened <b>${docsOpened}</b> of <b>${totalDocs}</b> documents · `
      + `<b>${pct}%</b> of corpus · skipped ${totalDocs - docsOpened}`;
  } else {
    subtitleEl.innerHTML = `Beam descent`;
  }
  const statHtml = [
    `<span class="hrag-tree-stat"><b>${leavesPicked}</b> leaves</span>`,
    `<span class="hrag-tree-stat"><b>${nodesCons}</b> nodes considered</span>`,
  ].join("");
  statsEl.innerHTML = statHtml;

  // ---- gather levels ---------------------------------------------------
  const trace = (DATA.trace || []).filter(l => (l.considered || []).length > 0);
  const leaves = DATA.leaves || [];

  // Stage width is full container width — measured at runtime.
  const ro = new ResizeObserver(() => layout());
  ro.observe(root);

  // Layout state: filled in by layout().
  let stageW = 0;
  let stageH = 0;
  let queryX = 0, queryY = 0;
  const nodePositions = []; // [{lvl, idx, x, y, w, h, kept, label, score, id}]
  const edgeSpecs = [];     // [{from:{x,y}, to:{x,y}, kept, levelIdx, nodeIdx, len}]

  function layout() {
    const rect = stageEl.getBoundingClientRect();
    stageW = rect.width || 720;

    // Required stage height: query + per-level rows + small bottom gap.
    stageH = QUERY_OFFSET_Y + 24 + (trace.length || 1) * ROW_H + 12;
    stageEl.style.height = stageH + "px";
    svgEl.setAttribute("viewBox", `0 0 ${stageW} ${stageH}`);
    svgEl.style.width = stageW + "px";
    svgEl.style.height = stageH + "px";

    queryX = stageW / 2;
    queryY = QUERY_OFFSET_Y;

    nodePositions.length = 0;
    trace.forEach((level, lvlIdx) => {
      const considered = level.considered || [];
      const n = considered.length;
      const rowY = QUERY_OFFSET_Y + 28 + (lvlIdx + 1) * ROW_H;
      // Even spacing: account for node width so cards stay inside the stage.
      const spacing = stageW / (n + 1);
      considered.forEach((ns, idx) => {
        const cx = spacing * (idx + 1);
        const x = Math.max(NODE_W/2 + 6, Math.min(stageW - NODE_W/2 - 6, cx));
        nodePositions.push({
          lvl: lvlIdx, idx,
          x: x, y: rowY,
          w: NODE_W, h: NODE_H,
          kept: !!ns.kept,
          label: ns.label || "?",
          score: typeof ns.score === "number" ? ns.score : 0,
          id: ns.node_id || `n${lvlIdx}-${idx}`,
        });
      });
    });

    // Edges: from query → every level-0 considered, then from each kept
    // node at level L → every considered node at level L+1.
    edgeSpecs.length = 0;
    const byLevel = {};
    nodePositions.forEach(np => {
      (byLevel[np.lvl] = byLevel[np.lvl] || []).push(np);
    });

    // Query → level 0
    (byLevel[0] || []).forEach((np, idx) => {
      edgeSpecs.push({
        fromX: queryX, fromY: queryY + 14,
        toX: np.x, toY: np.y - np.h/2,
        kept: np.kept,
        levelIdx: 0, nodeIdx: idx,
      });
    });
    // Level L kept → level L+1 nodes
    for (let L = 0; L < trace.length - 1; L++) {
      const keptAtL = (byLevel[L] || []).filter(n => n.kept);
      const nextLevel = byLevel[L+1] || [];
      keptAtL.forEach((par) => {
        nextLevel.forEach((np, idx) => {
          edgeSpecs.push({
            fromX: par.x, fromY: par.y + par.h/2,
            toX: np.x, toY: np.y - np.h/2,
            kept: np.kept,
            levelIdx: L+1, nodeIdx: idx,
          });
        });
      });
    }
    render();
  }

  function curvedPath(x1, y1, x2, y2) {
    // Vertical cubic Bezier; gives the tree a softer, organic feel.
    const dy = (y2 - y1);
    const c1x = x1, c1y = y1 + dy * 0.55;
    const c2x = x2, c2y = y2 - dy * 0.55;
    return `M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}`;
  }

  function approxPathLen(x1, y1, x2, y2) {
    // Quick approximation — Bezier arc length ≈ chord length × 1.18.
    const dx = x2 - x1, dy = y2 - y1;
    return Math.sqrt(dx*dx + dy*dy) * 1.18;
  }

  function render() {
    // Clear prior nodes & edges (in case of resize).
    nodesEl.innerHTML = "";
    svgEl.innerHTML = `
      <defs>
        <linearGradient id="hragEdgeGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"  stop-color="#34d399"/>
          <stop offset="60%" stop-color="#10b981"/>
          <stop offset="100%" stop-color="#059669"/>
        </linearGradient>
        <filter id="hragGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="b"/>
          <feMerge>
            <feMergeNode in="b"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>`;

    // ---- query bubble ------------------------------------------------
    const qDiv = document.createElement("div");
    qDiv.className = "hrag-query-bubble";
    qDiv.style.left = queryX + "px";
    qDiv.style.top  = (queryY - 16) + "px";
    qDiv.innerHTML = `🔎 <span>query embedding</span>`;
    nodesEl.appendChild(qDiv);

    // ---- edges (drawn first so they sit BEHIND nodes) ----------------
    edgeSpecs.forEach((e, eIdx) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const d = curvedPath(e.fromX, e.fromY, e.toX, e.toY);
      const len = approxPathLen(e.fromX, e.fromY, e.toX, e.toY);
      path.setAttribute("d", d);
      path.setAttribute("class", e.kept ? "hrag-edge" : "hrag-edge pruned-edge");
      path.style.setProperty("--edge-len", len);
      // Edge delays: query→L0 at 0.5s, level-L→level-(L+1) at 0.5s+L*0.9s+offset
      const baseDelay = 0.55 + e.levelIdx * 0.9 + (e.nodeIdx * 0.05);
      path.style.setProperty("--edge-delay", baseDelay + "s");
      svgEl.appendChild(path);

      // Animated traveling particle along KEPT edges only.
      if (e.kept) {
        const particle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        particle.setAttribute("r", "3");
        particle.setAttribute("class", "hrag-edge-particle");
        particle.style.offsetPath = `path("${d}")`;
        particle.style.webkitOffsetPath = `path("${d}")`;
        particle.style.setProperty("--edge-particle-delay", (baseDelay + 0.5) + "s");
        particle.style.setProperty("--edge-particle-dur", "2.4s");
        svgEl.appendChild(particle);
      }
    });

    // ---- nodes (HTML-on-top-of-SVG so we get text wrap) --------------
    nodePositions.forEach(np => {
      const div = document.createElement("div");
      div.className = "hrag-node " + (np.kept ? "kept" : "pruned");
      div.style.left = (np.x - np.w/2) + "px";
      div.style.top  = (np.y - np.h/2) + "px";
      div.style.width = np.w + "px";
      div.style.minHeight = np.h + "px";
      // Stagger appearance per node within a level. Level L starts at
      // ~0.55 + L*0.9 seconds; nodes within the level cascade by 0.07s.
      const baseDelay = 0.6 + np.lvl * 0.9 + np.idx * 0.07;
      div.style.animationDelay = baseDelay + "s";

      const scoreFmt = (np.score >= 0 ? "+" : "") + np.score.toFixed(2);
      div.innerHTML = `
        ${np.kept ? `<div class="hrag-node-star">★</div>` : ""}
        <div class="hrag-node-label">${escapeHtml(np.label).slice(0, 70)}</div>
        <div class="hrag-node-meta">
          <span>${np.kept ? "kept" : "pruned"}</span>
          <span class="hrag-node-score">${scoreFmt}</span>
        </div>`;
      nodesEl.appendChild(div);
    });

    // ---- level-row labels -------------------------------------------
    // (Tiny "LEVEL N" chips on the left-hand side, animated in alongside
    // their row.)
    trace.forEach((level, lvlIdx) => {
      const lbl = document.createElement("div");
      lbl.className = "hrag-level-label";
      const consCount = (level.considered || []).length;
      const keptCount = (level.considered || []).filter(c => c.kept).length;
      lbl.textContent = `LEVEL ${level.depth ?? lvlIdx} · ${consCount} cons · ${keptCount} kept`;
      const labelY = QUERY_OFFSET_Y + 28 + (lvlIdx + 1) * ROW_H - 8;
      lbl.style.top = labelY + "px";
      lbl.style.setProperty("--label-delay", (0.5 + lvlIdx * 0.9) + "s");
      stageEl.appendChild(lbl);
    });
  }

  // ---- leaf doc cards (always rendered, regardless of stage layout) --
  function renderLeaves() {
    leavesEl.innerHTML = "";
    if (!leaves.length) return;
    // Append after all level animations have finished.
    const leavesDelayBase = 0.65 + (trace.length || 1) * 0.9 + 0.3;
    leaves.forEach((lf, i) => {
      const card = document.createElement("div");
      card.className = "hrag-leaf-card";
      card.style.setProperty("--leaf-delay", (leavesDelayBase + i * 0.12) + "s");
      const docCount = lf.doc_count || (lf.doc_titles || []).length;
      const titles = lf.doc_titles || [];
      const items = titles.map((t, j) => {
        const safe = escapeHtml(String(t)).slice(0, 64);
        const liDelay = leavesDelayBase + i * 0.12 + 0.30 + j * 0.05;
        return `<li style="--doc-delay:${liDelay}s" title="${escapeHtml(String(t))}">${safe}</li>`;
      }).join("");
      const more = docCount > titles.length
        ? `<div class="hrag-leaf-more">+ ${docCount - titles.length} more</div>`
        : "";
      card.innerHTML = `
        <div class="hrag-leaf-head">
          <div class="hrag-leaf-title">📂 ${escapeHtml(lf.label || "?")}</div>
          <div class="hrag-leaf-badge">${docCount} doc${docCount===1?"":"s"}</div>
        </div>
        <ul class="hrag-leaf-docs">${items}</ul>
        ${more}`;
      leavesEl.appendChild(card);
    });
  }

  // ---- note (warning) ----------------------------------------------
  if (DATA.note) {
    noteEl.textContent = "⚠ " + DATA.note;
    noteEl.style.display = "block";
  }

  // ---- sparkles -----------------------------------------------------
  function renderSparkles() {
    const N = 18;
    for (let i = 0; i < N; i++) {
      const sp = document.createElement("div");
      sp.className = "hrag-sparkle";
      sp.style.left = (Math.random() * 100) + "%";
      sp.style.bottom = "-10px";
      sp.style.setProperty("--sp-delay", (Math.random() * 6) + "s");
      sp.style.setProperty("--sp-dur", (5 + Math.random() * 4) + "s");
      sparklesEl.appendChild(sp);
    }
  }

  // ---- HTML escape helper ------------------------------------------
  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // ---- run ----------------------------------------------------------
  layout();
  renderLeaves();
  renderSparkles();
})();
"""
