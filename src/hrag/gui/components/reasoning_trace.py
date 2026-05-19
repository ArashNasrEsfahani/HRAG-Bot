"""Unified reasoning-trace component for the chat page.

Renders an animated, transparent walk-through of what the bot is doing on
every turn:

  1. Query bubble at the top.
  2. Intent verdict chip (e.g. "PERSONAL · 95% · fast_path").
  3. Typing-animated thinking narration explaining the routing choice
     ("Looks like you're asking about yourself. Let me check my memories.").
  4. Action panel — varies by intent:
        FACTUAL  → animated SVG taxonomy tree (curved Bezier edges, glowing
                   particles travelling along kept paths, doc-card reveal).
        PERSONAL → memory cards with score badges and snippet text.
        GREETING → short "skipping retrieval" pill.
        GENERAL  → "no corpus match — answering from general knowledge".
        UNCLEAR  → "asking for clarification".
  5. Final "generating answer" status line.

Implemented as one self-contained HTML document rendered via
``streamlit.components.v1.html(html, height=h)`` so we get full browser
capabilities (SVG, motion-path particles, CSS keyframes).
"""

from __future__ import annotations

import json
import secrets
from typing import Any

# Layout tunables (must match values inside the embedded JS) -----------------
_HEADER_PX = 84
_QUERY_PX = 68
_VERDICT_PX = 56
_THINKING_PX = 56
_LEVEL_PX = 120
_TREE_BASE_PX = 90        # query bubble + bottom pad inside the tree stage
_LEAF_PX = 220
_MEMORY_CARD_PX = 95
_SKIP_PX = 90
_FOOTER_PX = 56
_SHELL_PAD_PX = 56        # padding + margins around everything


def render(state: dict[str, Any]) -> tuple[str, int]:
    """Build the reasoning-trace HTML + recommended iframe height.

    ``state`` shape (any subset present; missing keys are tolerated)::

        {
            "query":        str,                            # original user message
            "intent":       "greeting"|"personal"|"factual"|"general"|"unclear",
            "confidence":   float,                          # 0..1
            "source":       "fast_path"|"llm"|"fallback",
            "swapped_from_factual": bool,                   # FACTUAL→GENERAL swap
            "swap_top_score":       Optional[float],
            "taxonomy":     Optional[dict],                 # describe_last_descend payload
            "memories":     Optional[list[dict]],           # personal_memories event
            "n_passages":   Optional[int],                  # retrieve.n_results
        }
    """
    intent = (state.get("intent") or "factual")
    taxonomy = state.get("taxonomy")
    memories = state.get("memories") or []
    swap = bool(state.get("swapped_from_factual"))

    # Compute iframe height for the embed.
    h = _SHELL_PAD_PX + _HEADER_PX + _QUERY_PX
    if state.get("intent"):
        h += _VERDICT_PX + _THINKING_PX
    if intent == "factual" and not swap and taxonomy:
        levels = len([lv for lv in (taxonomy.get("trace") or []) if (lv.get("considered") or [])])
        leaves = len(taxonomy.get("leaves") or [])
        h += _TREE_BASE_PX + max(1, levels) * _LEVEL_PX
        if leaves:
            h += _LEAF_PX + 24
    elif intent == "personal" and memories:
        # One row per memory card.
        h += len(memories) * _MEMORY_CARD_PX + 40
    else:
        h += _SKIP_PX
    h += _FOOTER_PX

    dom_id = f"hrag-rtrace-{secrets.token_hex(4)}"
    data_json = json.dumps(state, ensure_ascii=False, default=str)

    return _build_html(dom_id, data_json), max(h, 360)


# ---------------------------------------------------------------------------
# HTML / CSS / JS
# ---------------------------------------------------------------------------


def _build_html(dom_id: str, data_json: str) -> str:
    js = _JS_TEMPLATE.replace("__DOM_ID__", dom_id).replace(
        "__DATA_JSON__", data_json
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_CSS_BLOCK}</style></head>
<body>
<div id="{dom_id}" class="rt-shell">
  <div class="rt-header">
    <div class="rt-header-left">
      <span class="rt-leaf">🌿</span>
      <span class="rt-title">Reasoning Trace</span>
    </div>
    <div class="rt-intent-chip" data-role="intent-chip">
      <span data-role="intent-label">analyzing…</span>
      <span class="rt-intent-detail" data-role="intent-detail"></span>
    </div>
  </div>
  <div class="rt-body" data-role="body">
    <div class="rt-step rt-step-query" data-role="query"></div>
    <div class="rt-arrow rt-arrow-1" data-role="arrow1">↓</div>
    <div class="rt-step rt-step-verdict" data-role="verdict"></div>
    <div class="rt-arrow rt-arrow-2" data-role="arrow2">↓</div>
    <div class="rt-step rt-step-thinking" data-role="thinking"></div>
    <div class="rt-arrow rt-arrow-3" data-role="arrow3">↓</div>
    <div class="rt-action" data-role="action"></div>
    <div class="rt-arrow rt-arrow-4" data-role="arrow4">↓</div>
    <div class="rt-step rt-step-footer" data-role="footer">✨ generating answer…</div>
  </div>
  <div class="rt-sparkles" data-role="sparkles"></div>
</div>
<script>{js}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# CSS — emerald/mint palette with high-tech glow accents
# ---------------------------------------------------------------------------

_CSS_BLOCK = r"""
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: transparent;
  color: #064e3b;
}
.rt-shell {
  position: relative;
  background:
    radial-gradient(ellipse at top, rgba(16,185,129,0.10) 0%, transparent 65%),
    linear-gradient(180deg, #f0fdf4 0%, #ecfdf5 60%, #d1fae5 100%);
  border: 1px solid rgba(110,231,183,0.7);
  border-radius: 18px;
  padding: 14px 18px 18px;
  margin: 4px 0 12px;
  overflow: hidden;
  box-shadow:
    0 12px 36px rgba(16,185,129,0.12),
    inset 0 0 0 1px rgba(255,255,255,0.55);
}
.rt-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(90deg, rgba(16,185,129,0.05) 1px, transparent 1px) 0 0 / 36px 36px,
    linear-gradient(0deg,  rgba(16,185,129,0.05) 1px, transparent 1px) 0 0 / 36px 36px;
  mask-image: radial-gradient(ellipse at center, black 35%, transparent 85%);
  -webkit-mask-image: radial-gradient(ellipse at center, black 35%, transparent 85%);
  pointer-events: none;
  animation: rt-grid-shift 22s linear infinite;
}
@keyframes rt-grid-shift {
  from { background-position: 0 0, 0 0; }
  to   { background-position: 72px 72px, 72px 72px; }
}

/* --- Header --------------------------------------------------------------- */
.rt-header {
  position: relative;
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; flex-wrap: wrap;
  padding: 10px 14px;
  background: linear-gradient(135deg,
    rgba(187,247,208,0.6) 0%,
    rgba(167,243,208,0.5) 60%,
    rgba(110,231,183,0.4) 100%);
  border: 1px solid rgba(16,185,129,0.40);
  border-radius: 14px;
  margin-bottom: 14px;
  box-shadow: 0 4px 16px rgba(16,185,129,0.12);
  animation: rt-fade-down 0.5s 0.05s ease-out both;
}
@keyframes rt-fade-down {
  from { opacity: 0; transform: translateY(-8px); }
  to   { opacity: 1; transform: translateY(0); }
}
.rt-header-left {
  display: flex; align-items: center; gap: 9px;
  color: #065f46; font-weight: 700; font-size: 0.96rem;
}
.rt-leaf {
  font-size: 1.25rem;
  filter: drop-shadow(0 0 8px rgba(16,185,129,0.55));
  animation: rt-leaf-bob 4s ease-in-out infinite;
}
@keyframes rt-leaf-bob {
  0%, 100% { transform: translateY(0) rotate(0deg); }
  50%      { transform: translateY(-2px) rotate(-4deg); }
}
.rt-intent-chip {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(255,255,255,0.85);
  border: 1.5px solid rgba(16,185,129,0.55);
  border-radius: 999px;
  padding: 5px 12px;
  color: #065f46;
  font-size: 0.78rem; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase;
  box-shadow: 0 2px 10px rgba(16,185,129,0.18);
}
.rt-intent-chip .rt-intent-detail {
  font-weight: 600;
  font-family: ui-monospace, SF Mono, Menlo, monospace;
  font-size: 0.72rem;
  color: #047857;
  letter-spacing: 0;
  text-transform: none;
  background: rgba(16,185,129,0.15);
  padding: 1px 7px;
  border-radius: 999px;
}
/* Per-intent recolor of the chip */
.rt-intent-chip.intent-greeting { background: rgba(254,243,199,0.9); border-color: #fbbf24; color: #92400e; }
.rt-intent-chip.intent-greeting .rt-intent-detail { background: rgba(245,158,11,0.18); color: #92400e; }
.rt-intent-chip.intent-personal { background: rgba(243,244,246,0.95); border-color: #a3a3a3; color: #1f2937; }
.rt-intent-chip.intent-personal .rt-intent-detail { background: rgba(163,163,163,0.18); color: #1f2937; }
.rt-intent-chip.intent-factual  { background: rgba(209,250,229,0.95); border-color: #10b981; color: #065f46; }
.rt-intent-chip.intent-factual  .rt-intent-detail { background: rgba(16,185,129,0.20); color: #065f46; }
.rt-intent-chip.intent-general  { background: rgba(219,234,254,0.95); border-color: #60a5fa; color: #1e40af; }
.rt-intent-chip.intent-general  .rt-intent-detail { background: rgba(96,165,250,0.18); color: #1e40af; }
.rt-intent-chip.intent-unclear  { background: rgba(243,244,246,0.95); border-color: #9ca3af; color: #374151; }
.rt-intent-chip.intent-unclear  .rt-intent-detail { background: rgba(156,163,175,0.18); color: #374151; }

/* --- Body chain of steps -------------------------------------------------- */
.rt-body { position: relative; display: flex; flex-direction: column; align-items: stretch; gap: 4px; }
.rt-step, .rt-action {
  position: relative;
  padding: 10px 14px;
  border-radius: 12px;
  background: rgba(255,255,255,0.85);
  border: 1.5px solid rgba(110,231,183,0.55);
  color: #064e3b;
  font-size: 0.86rem;
  line-height: 1.45;
  box-shadow: 0 3px 12px rgba(16,185,129,0.08);
  animation: rt-step-in 0.45s var(--step-delay, 0s) cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes rt-step-in {
  from { opacity: 0; transform: translateY(-8px) scale(0.96); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
.rt-arrow {
  text-align: center;
  color: #10b981;
  font-size: 1.05rem;
  font-weight: 700;
  margin: -2px 0;
  animation: rt-arrow-pulse 0.4s var(--arrow-delay, 0s) ease-out both;
}
@keyframes rt-arrow-pulse {
  from { opacity: 0; transform: scaleY(0); }
  to   { opacity: 1; transform: scaleY(1); }
}

/* Query step — special gradient */
.rt-step-query {
  background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
  border-color: #6ee7b7;
  font-weight: 600;
}
.rt-step-query .rt-query-quote {
  color: #047857;
  font-style: italic;
  font-size: 0.95rem;
  font-weight: 700;
}
.rt-step-query .rt-query-prefix {
  color: #059669;
  font-weight: 700;
  margin-right: 6px;
}

/* Verdict step */
.rt-step-verdict {
  background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(236,253,245,0.95));
  border-color: #34d399;
  display: flex; align-items: center; gap: 9px;
}
.rt-verdict-icon { font-size: 1.05rem; filter: drop-shadow(0 0 6px rgba(16,185,129,0.55)); }
.rt-verdict-text { font-weight: 600; color: #065f46; }
.rt-verdict-text b { color: #064e3b; font-weight: 800; }

/* Thinking narration — typewriter cursor */
.rt-step-thinking {
  background: rgba(255,255,255,0.95);
  border: 1.5px dashed #6ee7b7;
  font-style: italic;
  color: #065f46;
  font-weight: 500;
}
.rt-step-thinking::after {
  content: "▍";
  display: inline-block;
  margin-left: 2px;
  color: #10b981;
  animation: rt-cursor-blink 1s steps(2, start) infinite;
}
@keyframes rt-cursor-blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* --- Action panel --------------------------------------------------------- */
.rt-action {
  padding: 14px;
  background: rgba(255,255,255,0.94);
  border: 1.5px solid #6ee7b7;
  box-shadow: 0 6px 20px rgba(16,185,129,0.15);
}

/* Skip-message variant (greeting / unclear / general / swap) */
.rt-action.skip {
  display: flex; align-items: center; gap: 12px;
  background: linear-gradient(135deg, rgba(236,253,245,0.95), rgba(209,250,229,0.95));
}
.rt-action.skip .skip-icon {
  font-size: 1.6rem;
  filter: drop-shadow(0 0 8px rgba(16,185,129,0.5));
}
.rt-action.skip .skip-text {
  color: #065f46;
  font-weight: 600;
  line-height: 1.45;
}
.rt-action.skip .skip-text b { color: #064e3b; font-weight: 800; }

/* Memory cards (personal intent) */
.rt-memory-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 10px;
  color: #3730a3; font-weight: 800; font-size: 0.92rem;
}
.rt-memory-head .rt-memory-count {
  background: rgba(129,140,248,0.18);
  color: #3730a3;
  font-size: 0.74rem; font-weight: 700;
  padding: 2px 9px; border-radius: 999px;
}
.rt-memory-list { display: flex; flex-direction: column; gap: 8px; }
.rt-memory-card {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 9px 12px;
  background: linear-gradient(180deg, #ffffff 0%, #f5f3ff 100%);
  border: 1.5px solid #c7d2fe;
  border-radius: 10px;
  box-shadow: 0 3px 10px rgba(99,102,241,0.10);
  animation: rt-mem-slide 0.5s var(--mem-delay, 0s) cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes rt-mem-slide {
  from { opacity: 0; transform: translateX(-10px); }
  to   { opacity: 1; transform: translateX(0); }
}
.rt-memory-card .rt-mem-icon { font-size: 1.05rem; margin-top: 1px; }
.rt-memory-card .rt-mem-body { flex: 1; min-width: 0; }
.rt-memory-card .rt-mem-meta {
  display: flex; align-items: center; gap: 8px;
  font-size: 0.7rem; color: #4338ca; font-weight: 700;
  margin-bottom: 3px;
}
.rt-memory-card .rt-mem-score {
  background: rgba(99,102,241,0.15);
  color: #3730a3;
  font-family: ui-monospace, SF Mono, Menlo, monospace;
  font-weight: 700;
  padding: 1px 7px; border-radius: 999px;
}
.rt-memory-card .rt-mem-text {
  color: #1e1b4b;
  font-size: 0.84rem;
  line-height: 1.4;
  word-wrap: break-word;
}

/* Tree (factual intent) — embedded SVG visualization */
.rt-tree {
  position: relative;
}
.rt-tree-stage {
  position: relative;
  width: 100%;
}
.rt-tree-svg {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  overflow: visible;
}
.rt-tree-nodes { position: relative; width: 100%; }
.rt-tree-stats {
  display: flex; gap: 8px; flex-wrap: wrap;
  margin-bottom: 12px;
}
.rt-tree-stat {
  background: rgba(16,185,129,0.15);
  color: #065f46;
  font-size: 0.72rem; font-weight: 700;
  padding: 3px 10px; border-radius: 999px;
}

.rt-query-bubble {
  position: absolute;
  display: inline-flex; align-items: center; gap: 7px;
  padding: 7px 14px;
  border-radius: 999px;
  background: linear-gradient(135deg, #34d399 0%, #10b981 60%, #059669 100%);
  color: #ecfdf5; font-weight: 700; font-size: 0.78rem;
  box-shadow:
    0 0 0 3px rgba(16,185,129,0.18),
    0 6px 22px rgba(16,185,129,0.35);
  transform: translateX(-50%);
  white-space: nowrap;
  animation: rt-bubble-pulse 2.5s ease-in-out infinite;
}
@keyframes rt-bubble-pulse {
  0%, 100% { box-shadow: 0 0 0 3px rgba(16,185,129,0.18), 0 6px 22px rgba(16,185,129,0.35); }
  50%      { box-shadow: 0 0 0 9px rgba(16,185,129,0.08), 0 6px 32px rgba(16,185,129,0.50); }
}

.rt-node {
  position: absolute;
  width: 170px; min-height: 56px;
  padding: 7px 10px;
  border-radius: 11px;
  background: rgba(255,255,255,0.93);
  border: 1.5px solid #a7f3d0;
  color: #064e3b;
  font-size: 0.78rem;
  display: flex; flex-direction: column; justify-content: center; gap: 3px;
  opacity: 0; transform: translateY(-10px) scale(0.5);
  transform-origin: center top;
  box-shadow: 0 3px 11px rgba(16,185,129,0.08);
  animation: rt-node-pop 0.55s cubic-bezier(.34,1.56,.64,1) forwards;
}
@keyframes rt-node-pop {
  0%   { opacity: 0; transform: translateY(-10px) scale(0.5); }
  60%  { opacity: 1; transform: translateY(0) scale(1.08); }
  100% { opacity: 1; transform: translateY(0) scale(1.0); }
}
.rt-node-label {
  font-weight: 700;
  overflow: hidden; text-overflow: ellipsis;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  line-height: 1.2;
}
.rt-node-meta {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.66rem; color: #047857;
}
.rt-node-score {
  background: rgba(16,185,129,0.15);
  color: #065f46;
  font-family: ui-monospace, SF Mono, Menlo, monospace;
  font-weight: 700;
  padding: 0 6px; border-radius: 999px;
}
.rt-node.kept {
  background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 55%, #6ee7b7 100%);
  border: 1.5px solid #10b981;
  box-shadow:
    0 0 18px rgba(16,185,129,0.55),
    inset 0 0 0 1px rgba(255,255,255,0.5);
  animation:
    rt-node-pop 0.55s cubic-bezier(.34,1.56,.64,1) forwards,
    rt-kept-glow 2.4s 0.6s ease-in-out infinite;
}
@keyframes rt-kept-glow {
  0%, 100% { box-shadow: 0 0 18px rgba(16,185,129,0.55), inset 0 0 0 1px rgba(255,255,255,0.5); }
  50%      { box-shadow: 0 0 32px rgba(16,185,129,0.85), inset 0 0 0 1px rgba(255,255,255,0.7); }
}
.rt-node-star {
  position: absolute;
  top: -8px; right: -7px;
  width: 20px; height: 20px;
  border-radius: 50%;
  background: linear-gradient(135deg, #fde68a, #f59e0b);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.66rem; color: #78350f;
  box-shadow: 0 0 8px rgba(245,158,11,0.65);
  animation: rt-star-spin 4s linear infinite;
}
@keyframes rt-star-spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
.rt-node.pruned {
  background: rgba(241,245,249,0.85);
  border: 1.5px dashed #cbd5e1;
  color: #64748b;
  animation:
    rt-node-pop 0.55s cubic-bezier(.34,1.56,.64,1) forwards,
    rt-prune 0.7s 0.85s cubic-bezier(.55,.06,.68,.19) forwards;
}
@keyframes rt-prune {
  0%   { opacity: 1; transform: translateY(0) scale(1.0) rotate(0deg); filter: grayscale(0); }
  100% { opacity: 0.36; transform: translateY(10px) scale(0.86) rotate(-3deg); filter: grayscale(0.6); }
}

.rt-edge {
  fill: none;
  stroke: url(#rtEdgeGradient);
  stroke-width: 2.2;
  stroke-linecap: round;
  filter: drop-shadow(0 0 4px rgba(16,185,129,0.45));
  stroke-dasharray: var(--edge-len, 200);
  stroke-dashoffset: var(--edge-len, 200);
  animation: rt-edge-draw 0.55s var(--edge-delay, 0s) ease-out forwards;
}
@keyframes rt-edge-draw { to { stroke-dashoffset: 0; } }
.rt-edge.pruned-edge {
  stroke: #cbd5e1;
  filter: none;
  animation:
    rt-edge-draw 0.55s var(--edge-delay, 0s) ease-out forwards,
    rt-edge-fade 0.6s calc(var(--edge-delay, 0s) + 0.7s) ease-out forwards;
}
@keyframes rt-edge-fade { to { opacity: 0.25; } }

.rt-edge-particle {
  fill: #34d399;
  filter: drop-shadow(0 0 6px #34d399);
  animation: rt-particle-travel var(--p-dur, 2.4s) var(--p-delay, 0s) linear infinite;
}
@keyframes rt-particle-travel {
  from { offset-distance: 0%; opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  to   { offset-distance: 100%; opacity: 0; }
}

.rt-leaves {
  margin-top: 18px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}
.rt-leaf-card {
  background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);
  border: 1.5px solid #6ee7b7;
  border-radius: 12px;
  padding: 10px 12px;
  box-shadow: 0 5px 18px rgba(16,185,129,0.15);
  position: relative;
  overflow: hidden;
  animation: rt-leaf-open 0.7s var(--leaf-delay, 0s) cubic-bezier(.34,1.56,.64,1) both;
}
.rt-leaf-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #34d399, #10b981, #059669);
  background-size: 200% 100%;
  animation: rt-leaf-shimmer 3.5s linear infinite;
}
@keyframes rt-leaf-shimmer {
  from { background-position: 200% 0; }
  to   { background-position: -200% 0; }
}
@keyframes rt-leaf-open {
  0%   { opacity: 0; transform: scale(0.5) rotate(-3deg); }
  60%  { opacity: 1; transform: scale(1.05) rotate(1deg); }
  100% { opacity: 1; transform: scale(1.0)  rotate(0deg); }
}
.rt-leaf-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 6px;
}
.rt-leaf-title {
  font-size: 0.88rem; font-weight: 800; color: #064e3b;
}
.rt-leaf-badge {
  background: rgba(16,185,129,0.20);
  color: #065f46;
  font-size: 0.68rem; font-weight: 700;
  padding: 2px 8px; border-radius: 999px;
}
.rt-leaf-docs { list-style: none; margin: 0; padding: 0; font-size: 0.74rem; }
.rt-leaf-docs li {
  padding: 3px 0 3px 16px;
  position: relative; color: #064e3b;
  font-family: ui-monospace, SF Mono, Menlo, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.rt-leaf-docs li::before { content: "📄"; position: absolute; left: 0; top: 3px; font-size: 0.75rem; }
.rt-leaf-more {
  margin-top: 4px;
  font-size: 0.7rem; color: #047857; font-style: italic;
}

/* --- Footer --------------------------------------------------------------- */
.rt-step-footer {
  background: linear-gradient(135deg, rgba(167,243,208,0.6), rgba(110,231,183,0.6));
  border-color: #34d399;
  font-weight: 700;
  text-align: center;
  color: #065f46;
  font-size: 0.86rem;
}

/* --- Sparkles ------------------------------------------------------------- */
.rt-sparkles {
  position: absolute; inset: 0; pointer-events: none; overflow: hidden;
}
.rt-sparkle {
  position: absolute;
  width: 3px; height: 3px;
  border-radius: 50%;
  background: radial-gradient(circle, #34d399 0%, transparent 70%);
  filter: blur(0.4px);
  opacity: 0;
  animation: rt-sparkle-float var(--sp-dur, 7s) var(--sp-delay, 0s) linear infinite;
}
@keyframes rt-sparkle-float {
  0%   { opacity: 0; transform: translateY(0) scale(0.6); }
  10%  { opacity: 0.8; }
  90%  { opacity: 0.8; }
  100% { opacity: 0; transform: translateY(-200px) scale(1.4); }
}
"""


# ---------------------------------------------------------------------------
# Embedded JS — renders progressively from state JSON
# ---------------------------------------------------------------------------

_JS_TEMPLATE = r"""
(function() {
  const STATE = __DATA_JSON__;
  const root = document.getElementById("__DOM_ID__");
  if (!root) return;

  const escapeHtml = s => String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");

  // --- Intent chip ----------------------------------------------------
  const intent = STATE.intent || "factual";
  const chip   = root.querySelector('[data-role="intent-chip"]');
  const chipLbl= root.querySelector('[data-role="intent-label"]');
  const chipDet= root.querySelector('[data-role="intent-detail"]');
  chip.classList.add("intent-" + intent);
  chipLbl.textContent = intent;
  if (typeof STATE.confidence === "number") {
    const conf = (STATE.confidence * 100).toFixed(0) + "%";
    const src = STATE.source || "?";
    chipDet.textContent = conf + " · " + src;
  } else {
    chipDet.textContent = "";
  }

  // --- Step 1: query bubble ------------------------------------------
  const qEl = root.querySelector('[data-role="query"]');
  qEl.innerHTML = `<span class="rt-query-prefix">💭 you asked:</span>`
    + `<span class="rt-query-quote">"${escapeHtml(STATE.query || "")}"</span>`;
  qEl.style.setProperty("--step-delay", "0.15s");

  // --- Step 2: intent verdict (typed-in icon + message) --------------
  const vEl = root.querySelector('[data-role="verdict"]');
  vEl.style.setProperty("--step-delay", "0.55s");
  const intentMeta = {
    greeting: {icon: "👋", label: "GREETING", color: "#92400e"},
    personal: {icon: "👤", label: "PERSONAL", color: "#3730a3"},
    factual:  {icon: "🔍", label: "FACTUAL",  color: "#065f46"},
    general:  {icon: "🌍", label: "GENERAL",  color: "#1e40af"},
    unclear:  {icon: "❓", label: "UNCLEAR",  color: "#374151"},
  }[intent] || {icon: "🔍", label: intent.toUpperCase(), color: "#065f46"};
  const confTxt = (typeof STATE.confidence === "number")
    ? ` <span class="rt-node-score" style="margin-left:6px">${(STATE.confidence*100).toFixed(0)}% · ${STATE.source||"?"}</span>`
    : "";
  vEl.innerHTML = `<span class="rt-verdict-icon">${intentMeta.icon}</span>`
    + `<span class="rt-verdict-text">Intent classified as <b style="color:${intentMeta.color}">${intentMeta.label}</b>${confTxt}</span>`;

  // --- Step 3: thinking narration ------------------------------------
  // Sentence picked per intent. Tries to feel like the bot is reasoning
  // out loud, not boilerplate. Falls back to a generic line.
  const thinkingByIntent = {
    greeting: "Looks like a quick hello — no need to search anything. I'll just reply.",
    personal: "This is about you. Let me check my memories before answering.",
    factual:  "Substantive question — let me dig through your document library.",
    general:  "Couldn't find anything useful in your library on that. I'll answer from general knowledge instead.",
    unclear:  "Hmm, I'm not sure what you mean. Let me ask for clarification.",
  };
  // Narration picks up the most specific story we can tell:
  //   1. Named-topic hit: name the topic we recognized.
  //   2. FACTUAL → GENERAL swap: explain why we abandoned retrieval.
  //   3. Default per-intent line.
  let thinkingText;
  if (STATE.source === "named_topic" && STATE.raw_label) {
    thinkingText = `I recognize "${STATE.raw_label}" from your library — going straight to your documents.`;
  } else if (STATE.swapped_from_factual) {
    const ts = (STATE.swap_top_score ?? 0).toFixed(2);
    thinkingText = `I searched your library, but the best match scored only ${ts} — that's too weak. Falling through to general knowledge.`;
  } else {
    thinkingText = thinkingByIntent[intent] || "Thinking…";
  }
  const tEl = root.querySelector('[data-role="thinking"]');
  tEl.style.setProperty("--step-delay", "1.0s");
  typeInto(tEl, thinkingText, 1.15);

  function typeInto(el, text, startDelay) {
    el.textContent = "";
    const speed = 16; // chars / sec multiplied; lower = faster
    const total = Math.min(2.5, 0.04 * text.length);
    const chars = text.split("");
    chars.forEach((ch, i) => {
      setTimeout(() => { el.textContent += ch; },
        startDelay*1000 + (i / chars.length) * total * 1000);
    });
  }

  // --- Action panel --------------------------------------------------
  const aEl = root.querySelector('[data-role="action"]');
  aEl.style.setProperty("--step-delay", "1.5s");

  if (intent === "factual" && !STATE.swapped_from_factual && STATE.taxonomy) {
    renderTree(aEl, STATE.taxonomy);
  } else if (intent === "personal" && (STATE.memories || []).length) {
    renderMemories(aEl, STATE.memories);
  } else {
    renderSkip(aEl, intent, STATE);
  }

  // --- Arrow delays --------------------------------------------------
  root.querySelector('[data-role="arrow1"]').style.setProperty("--arrow-delay", "0.40s");
  root.querySelector('[data-role="arrow2"]').style.setProperty("--arrow-delay", "0.85s");
  root.querySelector('[data-role="arrow3"]').style.setProperty("--arrow-delay", "1.30s");
  root.querySelector('[data-role="arrow4"]').style.setProperty("--arrow-delay", "1.95s");

  const fEl = root.querySelector('[data-role="footer"]');
  fEl.style.setProperty("--step-delay", "2.15s");

  // --- Sparkles ------------------------------------------------------
  const sparkles = root.querySelector('[data-role="sparkles"]');
  for (let i = 0; i < 16; i++) {
    const sp = document.createElement("div");
    sp.className = "rt-sparkle";
    sp.style.left = (Math.random()*100) + "%";
    sp.style.bottom = "-12px";
    sp.style.setProperty("--sp-delay", (Math.random()*7) + "s");
    sp.style.setProperty("--sp-dur", (5 + Math.random()*4) + "s");
    sparkles.appendChild(sp);
  }

  // ===================================================================
  // Action panel renderers
  // ===================================================================

  function renderSkip(el, kind, state) {
    el.classList.add("skip");
    const map = {
      greeting: {ic: "👋", text: "<b>Skipped retrieval.</b> Greeting doesn't need document context — I'll reply directly."},
      general:  {ic: "🌍", text: "<b>No corpus retrieval.</b> Answering from my general knowledge with a short note that it's not from your library."},
      unclear:  {ic: "❓", text: "<b>Skipping retrieval.</b> The question is ambiguous — I'll ask you for more detail."},
      personal: {ic: "👤", text: "<b>No memories on file.</b> I'll answer from your profile alone."},
      factual:  {ic: "🔍", text: "<b>Searching documents…</b>"},
    };
    const m = map[kind] || map.factual;
    let extra = "";
    if (state.swapped_from_factual && typeof state.swap_top_score === "number") {
      extra = ` <span class="rt-node-score" style="margin-left:6px">top score ${(state.swap_top_score).toFixed(2)}</span>`;
    }
    el.innerHTML = `<span class="skip-icon">${m.ic}</span><span class="skip-text">${m.text}${extra}</span>`;
  }

  function renderMemories(el, memories) {
    const items = memories.map((m, i) => {
      const title = escapeHtml(m.title || "memory");
      const text  = escapeHtml(m.text  || "");
      const score = (typeof m.score === "number") ? `+${m.score.toFixed(2)}` : "";
      return `<div class="rt-memory-card" style="--mem-delay:${(1.6 + i*0.15).toFixed(2)}s">
        <span class="rt-mem-icon">📝</span>
        <div class="rt-mem-body">
          <div class="rt-mem-meta">
            <span>${title}</span>
            ${score ? `<span class="rt-mem-score">${score}</span>` : ""}
          </div>
          <div class="rt-mem-text">${text}</div>
        </div>
      </div>`;
    }).join("");
    el.innerHTML = `<div class="rt-memory-head">
        💾 Retrieved from your memory
        <span class="rt-memory-count">${memories.length} memor${memories.length===1?"y":"ies"}</span>
      </div>
      <div class="rt-memory-list">${items}</div>`;
  }

  // --- Tree (factual) rendering --- adapted from taxonomy_tree.py ------
  function renderTree(el, payload) {
    el.classList.add("rt-tree");
    const stats = payload.stats || {};
    const totalDocs   = stats.total_docs || 0;
    const docsOpened  = stats.docs_opened || 0;
    const leavesPicked= stats.leaves_picked || (payload.leaves || []).length;
    const nodesCons   = stats.nodes_considered || 0;
    const pct = totalDocs > 0 ? Math.max(1, Math.round(docsOpened * 100 / totalDocs)) : 0;

    el.innerHTML = `
      <div class="rt-tree-stats">
        <span class="rt-tree-stat">🌲 ${leavesPicked} leaves picked</span>
        <span class="rt-tree-stat">${nodesCons} nodes considered</span>
        <span class="rt-tree-stat">${docsOpened} of ${totalDocs} docs (${pct}%)</span>
      </div>
      <div class="rt-tree-stage" data-role="stage">
        <svg class="rt-tree-svg" data-role="svg" xmlns="http://www.w3.org/2000/svg"
             preserveAspectRatio="xMidYMin meet"></svg>
        <div class="rt-tree-nodes" data-role="nodes"></div>
      </div>
      <div class="rt-leaves" data-role="leaves"></div>`;

    const stageEl = el.querySelector('[data-role="stage"]');
    const svgEl   = el.querySelector('[data-role="svg"]');
    const nodesEl = el.querySelector('[data-role="nodes"]');
    const leavesEl= el.querySelector('[data-role="leaves"]');

    const ROW_H = 110, NODE_W = 170, NODE_H = 56;
    const QUERY_Y = 30;

    const trace = (payload.trace || []).filter(l => (l.considered || []).length > 0);

    const ro = new ResizeObserver(() => layoutTree());
    ro.observe(el);

    let stageW = 0, stageH = 0;
    function layoutTree() {
      const rect = stageEl.getBoundingClientRect();
      stageW = rect.width || 720;
      stageH = QUERY_Y + 22 + (trace.length || 1) * ROW_H + 12;
      stageEl.style.height = stageH + "px";
      svgEl.setAttribute("viewBox", `0 0 ${stageW} ${stageH}`);
      svgEl.style.width = stageW + "px";
      svgEl.style.height = stageH + "px";

      const queryX = stageW / 2, queryY = QUERY_Y;
      const positions = [];

      trace.forEach((level, lvlIdx) => {
        const cons = level.considered || [];
        const rowY = QUERY_Y + 22 + (lvlIdx + 1) * ROW_H;
        const spacing = stageW / (cons.length + 1);
        cons.forEach((ns, idx) => {
          const cx = spacing * (idx + 1);
          const x = Math.max(NODE_W/2 + 6, Math.min(stageW - NODE_W/2 - 6, cx));
          positions.push({
            lvl: lvlIdx, idx, x, y: rowY, w: NODE_W, h: NODE_H,
            kept: !!ns.kept,
            label: ns.label || "?",
            score: typeof ns.score === "number" ? ns.score : 0,
          });
        });
      });

      const byLevel = {};
      positions.forEach(np => (byLevel[np.lvl] = byLevel[np.lvl] || []).push(np));

      const edges = [];
      (byLevel[0] || []).forEach((np, idx) => {
        edges.push({fx: queryX, fy: queryY + 12, tx: np.x, ty: np.y - np.h/2,
                    kept: np.kept, lvl: 0, idx});
      });
      for (let L = 0; L < trace.length - 1; L++) {
        const kept = (byLevel[L] || []).filter(n => n.kept);
        const next = byLevel[L+1] || [];
        kept.forEach(par => {
          next.forEach((np, idx) => {
            edges.push({fx: par.x, fy: par.y + par.h/2, tx: np.x, ty: np.y - np.h/2,
                        kept: np.kept, lvl: L+1, idx});
          });
        });
      }

      drawTree(positions, edges, queryX, queryY);
    }

    function curvedPath(x1, y1, x2, y2) {
      const dy = (y2 - y1);
      return `M ${x1} ${y1} C ${x1} ${y1 + dy*0.55}, ${x2} ${y2 - dy*0.55}, ${x2} ${y2}`;
    }

    function drawTree(positions, edges, queryX, queryY) {
      nodesEl.innerHTML = "";
      svgEl.innerHTML = `<defs>
        <linearGradient id="rtEdgeGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"  stop-color="#34d399"/>
          <stop offset="60%" stop-color="#10b981"/>
          <stop offset="100%" stop-color="#059669"/>
        </linearGradient>
      </defs>`;

      // Query bubble
      const qb = document.createElement("div");
      qb.className = "rt-query-bubble";
      qb.style.left = queryX + "px";
      qb.style.top  = (queryY - 14) + "px";
      qb.innerHTML = `🔎 <span>query embedding</span>`;
      nodesEl.appendChild(qb);

      // Edges (under nodes)
      edges.forEach(e => {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const d = curvedPath(e.fx, e.fy, e.tx, e.ty);
        const len = Math.hypot(e.tx-e.fx, e.ty-e.fy) * 1.18;
        path.setAttribute("d", d);
        path.setAttribute("class", e.kept ? "rt-edge" : "rt-edge pruned-edge");
        path.style.setProperty("--edge-len", len);
        const baseDelay = 1.55 + e.lvl * 0.9 + (e.idx * 0.04);
        path.style.setProperty("--edge-delay", baseDelay + "s");
        svgEl.appendChild(path);
        if (e.kept) {
          const particle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          particle.setAttribute("r", "3");
          particle.setAttribute("class", "rt-edge-particle");
          particle.style.offsetPath = `path("${d}")`;
          particle.style.webkitOffsetPath = `path("${d}")`;
          particle.style.setProperty("--p-delay", (baseDelay + 0.4) + "s");
          particle.style.setProperty("--p-dur", "2.4s");
          svgEl.appendChild(particle);
        }
      });

      // Nodes
      positions.forEach(np => {
        const div = document.createElement("div");
        div.className = "rt-node " + (np.kept ? "kept" : "pruned");
        div.style.left = (np.x - np.w/2) + "px";
        div.style.top  = (np.y - np.h/2) + "px";
        div.style.width = np.w + "px";
        div.style.minHeight = np.h + "px";
        const baseDelay = 1.6 + np.lvl * 0.9 + np.idx * 0.06;
        div.style.animationDelay = baseDelay + "s";
        const scoreFmt = (np.score >= 0 ? "+" : "") + np.score.toFixed(2);
        div.innerHTML = `
          ${np.kept ? `<div class="rt-node-star">★</div>` : ""}
          <div class="rt-node-label">${escapeHtml(np.label).slice(0, 70)}</div>
          <div class="rt-node-meta">
            <span>${np.kept ? "kept" : "pruned"}</span>
            <span class="rt-node-score">${scoreFmt}</span>
          </div>`;
        nodesEl.appendChild(div);
      });

      // Leaves
      leavesEl.innerHTML = "";
      const leaves = payload.leaves || [];
      if (!leaves.length) return;
      const baseLeafDelay = 1.7 + (trace.length || 1) * 0.9 + 0.3;
      leaves.forEach((lf, i) => {
        const card = document.createElement("div");
        card.className = "rt-leaf-card";
        card.style.setProperty("--leaf-delay", (baseLeafDelay + i * 0.12) + "s");
        const docCount = lf.doc_count || (lf.doc_titles || []).length;
        const titles = (lf.doc_titles || []).slice(0, 5);
        const items = titles.map(t => `<li title="${escapeHtml(String(t))}">${escapeHtml(String(t)).slice(0,64)}</li>`).join("");
        const more = docCount > titles.length
          ? `<div class="rt-leaf-more">+ ${docCount - titles.length} more</div>`
          : "";
        card.innerHTML = `
          <div class="rt-leaf-head">
            <div class="rt-leaf-title">📂 ${escapeHtml(lf.label || "?")}</div>
            <div class="rt-leaf-badge">${docCount} doc${docCount===1?"":"s"}</div>
          </div>
          <ul class="rt-leaf-docs">${items}</ul>
          ${more}`;
        leavesEl.appendChild(card);
      });
    }

    layoutTree();
  }
})();
"""
