"""Shared Streamlit-scoped state, CSS injection, and UI helpers for the GUI pages.

Streamlit reruns each page on every interaction, so anything heavy
(Orchestrator → embedder model load, Chroma client) MUST live behind
``st.cache_resource``. We also keep the active user_id in
``st.session_state`` so switching users on the Users page propagates
everywhere.

UI design notes
---------------
- Every page calls :func:`apply_chrome` once at the top. It injects the
  shared CSS (fonts, animations, source cards, sidebar gradient) and the
  active-user pill.
- :func:`page_header` renders the page title + optional "💡 Tips" panel.
- :func:`stream_chat_events` runs ``Orchestrator.chat(stream=True)`` in a
  background thread so the calling page can render token-by-token output
  and switch the status badge between Retrieve / Rerank / Write phases
  without freezing the UI.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import streamlit as st

from hrag.config import Config, load_config
from hrag.orchestrator import ChatResult, Orchestrator


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="🚀 Booting HRAG-Bot (one-time embedder load)…")
def get_orchestrator() -> Orchestrator:
    """Build the orchestrator once per Streamlit process.

    Cache key is empty so the same orchestrator survives across page
    navigations. To switch model/KG flags, use Streamlit's hamburger →
    "Clear cache" to force a rebuild.
    """
    cfg: Config = load_config()
    return Orchestrator(cfg)


# ---------------------------------------------------------------------------
# CSS / chrome
# ---------------------------------------------------------------------------


_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Vazirmatn:wght@400;500;600;700;800&display=swap');

:root {
  /* Editorial monochrome palette. The only "colour" is a champagne subtone
     used very sparingly — never as a background fill, only as a hairline
     accent on hover/focus/active. Everything else is calibrated grayscale. */
  --hrag-bg:         #0a0a0a;
  --hrag-bg-2:       #111111;
  --hrag-ink:        #e5e5e5;
  --hrag-ink-bright: #fafafa;
  --hrag-muted:      #9a9a9a;
  --hrag-muted-2:    #6b6b6b;
  --hrag-card:        rgba(255,255,255,0.025);
  --hrag-card-hover:  rgba(255,255,255,0.05);
  --hrag-border:        rgba(255,255,255,0.08);
  --hrag-border-strong: rgba(255,255,255,0.20);

  /* The single subtone — warm off-white champagne. Use only for hairline
     accents, never as a fill. Vary opacity for intensity. */
  --hrag-accent:    #e8dcc4;
  --hrag-accent-2:  #d4c5a0;
  --hrag-accent-3:  #b8a87a;

  /* Status hints — heavily desaturated for "classy" feel. */
  --hrag-good: #b8d4b8;
  --hrag-warn: #e8d4a8;
  --hrag-bad:  #d4b4b4;
  --hrag-info: #c0c8d0;

  --hrag-shadow-soft: 0 4px 14px rgba(0,0,0,0.35);
  --hrag-shadow-lift: 0 12px 32px rgba(0,0,0,0.55);
  --hrag-glow:        0 0 0 1px rgba(232,220,196,0.18), 0 6px 18px rgba(0,0,0,0.45);

  --hrag-font: 'Inter', 'Vazirmatn', system-ui, -apple-system, 'Segoe UI', Roboto, 'Tahoma', sans-serif;
  --hrag-font-fa: 'Vazirmatn', 'Inter', system-ui, 'Tahoma', sans-serif;
  --hrag-mono: 'JetBrains Mono', ui-monospace, 'SF Mono', 'Cascadia Code', monospace;
}

/* ----- Persian / RTL support -------------------------------------------- */
/* Streamlit doesn't auto-detect script direction. Add the `rtl` class via
   the .rtl helper (or apply [lang="fa"]) to a container to flip direction
   and switch to Vazirmatn. The chat page uses `:lang(fa)` heuristic via
   Unicode-range detection below for ad-hoc Persian text in messages. */
[dir="rtl"], .rtl, :lang(fa) {
  direction: rtl;
  text-align: right;
  font-family: var(--hrag-font-fa) !important;
}
[dir="rtl"] code, .rtl code, :lang(fa) code {
  direction: ltr;
  unicode-bidi: embed;
  font-family: var(--hrag-mono) !important;
}
/* Auto-detect Persian (and Arabic) Unicode in chat messages. */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  unicode-bidi: plaintext;
}

/* ----- background — neutral vignette, no aurora ------------------------- */
.stApp {
  background:
    radial-gradient(ellipse 90% 60% at 50% 0%, rgba(255,255,255,0.025), transparent 60%),
    var(--hrag-bg) !important;
  background-attachment: fixed !important;
  font-family: var(--hrag-font) !important;
}
[data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }

/* Reduce default top padding so the page header sits closer to the top. */
.block-container { padding-top: 1.2rem !important; max-width: 1400px; }

/* Typography polish */
* { font-family: var(--hrag-font); }
code, pre, kbd, [class*="stCode"] { font-family: var(--hrag-mono) !important; }
h1, h2, h3, h4 { letter-spacing: -0.018em; font-weight: 700; color: var(--hrag-ink-bright); }
h1 { font-weight: 800; }
p, li { color: var(--hrag-ink); }

/* Custom scrollbar — neutral grey */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.10);
  border-radius: 10px;
  border: 2px solid var(--hrag-bg);
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

/* ----- sidebar — flat charcoal with hairline border --------------------- */
section[data-testid="stSidebar"] {
  background: var(--hrag-bg-2);
  border-right: 1px solid var(--hrag-border);
}
section[data-testid="stSidebar"] * { color: var(--hrag-ink); }

/* Active-user pill — black card with champagne hairline */
.hrag-user-pill {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--hrag-border);
  border-radius: 12px;
  font-size: 0.85rem;
  margin-bottom: 12px;
  transition: border-color 0.25s ease;
}
.hrag-user-pill:hover { border-color: rgba(232,220,196,0.25); }
.hrag-user-pill .pill-avatar {
  width: 28px; height: 28px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  background: var(--hrag-ink-bright);
  font-weight: 700; color: var(--hrag-bg); font-size: 0.85rem;
  flex-shrink: 0;
}
.hrag-user-pill .pill-meta { display: flex; flex-direction: column; line-height: 1.15; }
.hrag-user-pill .pill-label { color: var(--hrag-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; }
.hrag-user-pill .pill-value { font-weight: 600; font-size: 0.95rem; }

/* ----- page header banner ------------------------------------------------ */
.hrag-page-header {
  position: relative;
  display: flex; align-items: center; gap: 16px;
  padding: 22px 26px;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--hrag-border);
  border-radius: 16px;
  margin-bottom: 18px;
  overflow: hidden;
  animation: hrag-slidein 0.4s cubic-bezier(0.22, 1, 0.36, 1);
  box-shadow: var(--hrag-shadow-soft);
}
.hrag-page-header::before {
  content: '';
  position: absolute; left: 0; top: 0; bottom: 0;
  width: 2px;
  background: linear-gradient(180deg, transparent, var(--hrag-accent), transparent);
  opacity: 0.5;
}
.hrag-page-header::after {
  content: '';
  position: absolute; top: 0; left: -100%;
  width: 40%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
  animation: hrag-shimmer 9s ease-in-out infinite;
  pointer-events: none;
}
.hrag-page-header .icon {
  font-size: 2.1rem;
  width: 58px; height: 58px;
  display: inline-flex; align-items: center; justify-content: center;
  background: rgba(255,255,255,0.025);
  border-radius: 12px;
  border: 1px solid var(--hrag-border);
  animation: hrag-float 5s ease-in-out infinite;
}
.hrag-page-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; color: var(--hrag-ink-bright); }
.hrag-page-header .subtitle { color: var(--hrag-muted); font-size: 0.95rem; margin-top: 4px; }

/* ----- LLM status badge (Chat page) — monochrome dots ------------------- */
.hrag-status-bar {
  display: flex; gap: 10px; align-items: center;
  padding: 8px 14px; border-radius: 10px;
  background: var(--hrag-card);
  border: 1px solid var(--hrag-border);
  font-size: 0.92rem;
  margin: 6px 0 8px;
  animation: hrag-fadein 0.25s ease-out;
}
.hrag-status-bar .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--hrag-ink);
  box-shadow: 0 0 8px rgba(255,255,255,0.25);
}
.hrag-status-bar.retrieve .dot { background: var(--hrag-info); box-shadow: 0 0 8px rgba(192,200,208,0.4); }
.hrag-status-bar.rerank   .dot { background: var(--hrag-warn); box-shadow: 0 0 8px rgba(232,212,168,0.4); }
.hrag-status-bar.write    .dot { background: var(--hrag-good); box-shadow: 0 0 8px rgba(184,212,184,0.4); }
.hrag-status-bar.done     .dot { background: var(--hrag-ink-bright); animation: none; }
.hrag-status-bar .dot { animation: hrag-pulse 1.4s infinite ease-in-out; }
.hrag-status-bar.done .dot { animation: none !important; }

.hrag-thinking-dots::after {
  content: '·';
  animation: hrag-dots 1.4s infinite steps(1, end);
  font-weight: bold; letter-spacing: 2px;
}

/* ----- source cards ----------------------------------------------------- */
.hrag-source-card {
  padding: 10px 14px;
  border-radius: 10px;
  background: var(--hrag-card);
  border-left: 2px solid rgba(255,255,255,0.20);
  margin: 6px 0;
  animation: hrag-fadein 0.25s ease-out;
  transition: background 0.18s ease, transform 0.15s ease, border-color 0.18s ease;
}
.hrag-source-card:hover {
  background: var(--hrag-card-hover);
  transform: translateX(2px);
  border-left-color: var(--hrag-accent);
}
.hrag-source-card .stitle { font-weight: 600; color: var(--hrag-ink-bright); }
.hrag-source-card .stags  { color: var(--hrag-muted); font-size: 0.82rem; }
.hrag-source-card.episodic  { border-left-color: rgba(212,197,160,0.45); }
.hrag-source-card.community { border-left-color: rgba(192,200,208,0.45); }

/* ----- buttons — black/white with champagne accent --------------------- */
button[kind="primary"] {
  background: var(--hrag-ink-bright) !important;
  color: var(--hrag-bg) !important;
  border: 1px solid var(--hrag-ink-bright) !important;
  font-weight: 600 !important;
  transition: transform 0.15s ease, background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease !important;
}
button[kind="primary"]:hover {
  transform: translateY(-1px);
  background: var(--hrag-accent) !important;
  color: var(--hrag-bg) !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.4);
}
button[kind="secondary"] {
  background: transparent !important;
  color: var(--hrag-ink) !important;
  border: 1px solid var(--hrag-border-strong) !important;
  transition: border-color 0.2s ease, color 0.2s ease, background 0.2s ease !important;
}
button[kind="secondary"]:hover {
  border-color: var(--hrag-accent) !important;
  color: var(--hrag-ink-bright) !important;
  background: rgba(255,255,255,0.03) !important;
}

/* ----- tip card --------------------------------------------------------- */
.hrag-tipcard {
  padding: 14px 18px;
  border-radius: 12px;
  background: rgba(255,255,255,0.015);
  border: 1px solid var(--hrag-border);
  border-left: 2px solid var(--hrag-accent);
  font-size: 0.88rem;
  margin-bottom: 12px;
}
.hrag-tipcard b { color: var(--hrag-ink-bright); letter-spacing: 0.02em; }
.hrag-tipcard ul { margin: 6px 0 0 18px; padding: 0; }
.hrag-tipcard li { margin: 3px 0; }

/* ----- metric cards (legacy st.metric) ---------------------------------- */
[data-testid="stMetric"] {
  background: var(--hrag-card);
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid var(--hrag-border);
  transition: transform 0.12s ease, border-color 0.15s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: var(--hrag-border-strong); }

/* ----- KPI card — minimal, monochrome ----------------------------------- */
.hrag-kpi {
  position: relative;
  padding: 18px 20px;
  border-radius: 14px;
  background: var(--hrag-card);
  border: 1px solid var(--hrag-border);
  transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1),
              border-color 0.25s ease, background 0.25s ease,
              box-shadow 0.25s ease;
  overflow: hidden;
  height: 100%;
  animation: hrag-fadein-up 0.5s ease-out backwards;
}
.hrag-kpi::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: 2px; background: rgba(255,255,255,0.15);
  transition: background 0.25s ease;
}
.hrag-kpi.accent::before { background: var(--hrag-accent); }
.hrag-kpi.good::before   { background: var(--hrag-good); }
.hrag-kpi.warn::before   { background: var(--hrag-warn); }
.hrag-kpi.bad::before    { background: var(--hrag-bad); }
.hrag-kpi.info::before   { background: var(--hrag-info); }
.hrag-kpi:hover {
  transform: translateY(-3px);
  border-color: var(--hrag-border-strong);
  background: var(--hrag-card-hover);
  box-shadow: var(--hrag-shadow-lift);
}
.hrag-kpi:hover::before { background: var(--hrag-accent); }
.hrag-kpi .kpi-row { display: flex; align-items: center; gap: 10px; }
.hrag-kpi .kpi-icon { font-size: 1.4rem; opacity: 0.85; transition: transform 0.3s ease; }
.hrag-kpi:hover .kpi-icon { transform: translateY(-1px); }
.hrag-kpi .kpi-label {
  color: var(--hrag-muted); font-size: 0.72rem;
  text-transform: uppercase; letter-spacing: 0.10em; font-weight: 600;
}
.hrag-kpi .kpi-value {
  font-size: 1.85rem; font-weight: 700; line-height: 1.1; margin-top: 6px;
  color: var(--hrag-ink-bright);
  font-feature-settings: 'tnum' 1, 'ss01' 1;
}
.hrag-kpi .kpi-sub { color: var(--hrag-muted); font-size: 0.78rem; margin-top: 3px; }

/* ----- chips — monochrome with subtle tonal differentiation ------------- */
.hrag-chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--hrag-border);
  color: var(--hrag-ink);
}
.hrag-chip.accent  { background: rgba(232,220,196,0.10); border-color: rgba(232,220,196,0.35); color: var(--hrag-accent); }
.hrag-chip.violet  { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.18); color: var(--hrag-ink); }
.hrag-chip.good    { background: rgba(184,212,184,0.10); border-color: rgba(184,212,184,0.35); color: var(--hrag-good); }
.hrag-chip.warn    { background: rgba(232,212,168,0.10); border-color: rgba(232,212,168,0.35); color: var(--hrag-warn); }
.hrag-chip.bad     { background: rgba(212,180,180,0.10); border-color: rgba(212,180,180,0.35); color: var(--hrag-bad); }
.hrag-chip.info    { background: rgba(192,200,208,0.10); border-color: rgba(192,200,208,0.35); color: var(--hrag-info); }
.hrag-chip.muted   { background: rgba(255,255,255,0.03); border-color: var(--hrag-border);     color: var(--hrag-muted); }

/* ----- empty state ------------------------------------------------------ */
.hrag-empty {
  text-align: center;
  padding: 40px 24px;
  border-radius: 14px;
  background: var(--hrag-card);
  border: 1px dashed var(--hrag-border-strong);
  margin: 12px 0;
}
.hrag-empty .empty-icon { font-size: 2.4rem; opacity: 0.65; }
.hrag-empty .empty-title { font-size: 1.05rem; font-weight: 700; margin-top: 8px; color: var(--hrag-ink-bright); }
.hrag-empty .empty-msg { color: var(--hrag-muted); font-size: 0.92rem; margin-top: 4px; max-width: 520px; margin-left: auto; margin-right: auto; }

/* ----- nav card — flat with champagne hairline on hover ----------------- */
.hrag-nav-card {
  position: relative;
  display: block;
  padding: 18px 20px;
  border-radius: 14px;
  background: var(--hrag-card);
  border: 1px solid var(--hrag-border);
  text-decoration: none !important;
  color: var(--hrag-ink) !important;
  transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1),
              border-color 0.25s ease, background 0.25s ease, box-shadow 0.25s ease;
  height: 100%;
}
.hrag-nav-card:hover {
  transform: translateY(-3px);
  border-color: var(--hrag-accent);
  background: var(--hrag-card-hover);
  box-shadow: var(--hrag-shadow-lift);
}
.hrag-nav-card .nav-icon { font-size: 1.6rem; opacity: 0.9; }
.hrag-nav-card .nav-title { font-weight: 700; font-size: 1.05rem; margin-top: 6px; color: var(--hrag-ink-bright); }
.hrag-nav-card .nav-desc { color: var(--hrag-muted); font-size: 0.86rem; margin-top: 4px; line-height: 1.4; }
.hrag-nav-card .nav-arrow { position: absolute; top: 16px; right: 18px; color: var(--hrag-muted); transition: transform 0.25s ease, color 0.25s ease; }
.hrag-nav-card:hover .nav-arrow { color: var(--hrag-accent); transform: translateX(3px); }

/* ----- entity / doc card (used in browse lists) ------------------------- */
.hrag-entity-card {
  position: relative;
  padding: 12px 14px;
  border-radius: 12px;
  background: var(--hrag-card);
  border: 1px solid var(--hrag-border);
  transition: transform 0.12s ease, border-color 0.15s ease, background 0.15s ease;
  margin: 6px 0;
}
.hrag-entity-card:hover { transform: translateY(-1px); border-color: var(--hrag-border-strong); background: var(--hrag-card-hover); }
.hrag-entity-card.active { border-color: var(--hrag-accent); box-shadow: 0 4px 16px rgba(0,0,0,0.4); }
.hrag-entity-card .ec-row { display: flex; align-items: center; gap: 10px; }
.hrag-entity-card .ec-icon { font-size: 1.4rem; flex-shrink: 0; }
.hrag-entity-card .ec-title { font-weight: 600; }
.hrag-entity-card .ec-meta  { color: var(--hrag-muted); font-size: 0.8rem; margin-top: 2px; }
.hrag-entity-card .ec-tags  { margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; }

/* ----- avatar (user initials) ------------------------------------------- */
.hrag-avatar {
  width: 36px; height: 36px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  background: var(--hrag-ink-bright);
  font-weight: 700; color: var(--hrag-bg); font-size: 0.95rem;
  flex-shrink: 0;
  border: 1px solid rgba(255,255,255,0.05);
}
.hrag-avatar.lg { width: 44px; height: 44px; font-size: 1.1rem; }

/* ----- chat message — slide-in + glass card + rich markdown ------------- */
[data-testid="stChatMessage"] {
  animation: hrag-slidein 0.35s cubic-bezier(0.22, 1, 0.36, 1);
  padding: 16px 20px !important;
  border-radius: 14px !important;
  margin-bottom: 10px !important;
  background: var(--hrag-card) !important;
  border: 1px solid var(--hrag-border);
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stChatMessage"]:hover {
  border-color: var(--hrag-border-strong);
  box-shadow: var(--hrag-shadow-soft);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  font-size: 0.96rem; line-height: 1.65;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { margin: 0.4em 0; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4 {
  margin: 0.7em 0 0.35em; font-weight: 700;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol {
  margin: 0.4em 0 0.6em 1.2em; padding-left: 0.5em;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li { margin: 0.18em 0; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote {
  margin: 0.5em 0; padding: 6px 14px;
  background: rgba(255,255,255,0.025);
  border-left: 2px solid var(--hrag-accent);
  border-radius: 4px;
  color: #e2e2e2; font-style: italic;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code:not(pre code) {
  padding: 1px 6px;
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--hrag-border);
  border-radius: 5px;
  font-family: var(--hrag-mono);
  font-size: 0.86em;
  color: var(--hrag-ink-bright);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
  margin: 0.6em 0;
  border-radius: 10px !important;
  background: rgba(0,0,0,0.35) !important;
  border: 1px solid var(--hrag-border) !important;
  padding: 12px 14px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  overflow-x: auto;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code {
  font-family: var(--hrag-mono) !important;
  font-size: 0.86em;
  color: #f8fafc;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.6em 0;
  font-size: 0.9em;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
  border: 1px solid var(--hrag-border);
  padding: 6px 10px;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
  background: rgba(255,255,255,0.04);
  font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--hrag-ink-bright);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] tr:nth-child(even) td {
  background: rgba(255,255,255,0.015);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {
  color: var(--hrag-ink-bright);
  text-decoration: none;
  border-bottom: 1px dashed rgba(255,255,255,0.30);
  transition: color 0.2s ease, border-color 0.2s ease;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a:hover {
  color: var(--hrag-accent);
  border-bottom-color: var(--hrag-accent);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] hr {
  border: 0;
  border-top: 1px solid var(--hrag-border);
  margin: 0.8em 0;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] img {
  max-width: 100%; border-radius: 8px;
}

/* ----- tabs ------------------------------------------------------------- */
button[data-baseweb="tab"] { font-weight: 600 !important; color: var(--hrag-muted) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--hrag-ink-bright) !important;
  border-bottom: 2px solid var(--hrag-accent) !important;
}

/* ----- animations ------------------------------------------------------- */
@keyframes hrag-pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50%      { transform: scale(1.4); opacity: 0.6; }
}
@keyframes hrag-fadein {
  from { opacity: 0; transform: translateY(2px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes hrag-fadein-up {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes hrag-slidein {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes hrag-dots {
  0%   { content: '·'; }
  33%  { content: '··'; }
  66%  { content: '···'; }
  100% { content: '·'; }
}
@keyframes hrag-aurora {
  0%   { transform: translate(0%, 0%) rotate(0deg); opacity: 1; }
  50%  { transform: translate(4%, -3%) rotate(2deg); opacity: 0.85; }
  100% { transform: translate(-3%, 4%) rotate(-2deg); opacity: 1; }
}
@keyframes hrag-shimmer {
  0%   { left: -100%; }
  60%  { left: 200%; }
  100% { left: 200%; }
}
@keyframes hrag-float {
  0%, 100% { transform: translateY(0px); }
  50%      { transform: translateY(-4px); }
}
@keyframes hrag-glow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(232,220,196,0.30); }
  50%      { box-shadow: 0 0 0 8px rgba(232,220,196,0.00); }
}
@keyframes hrag-skeleton {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes hrag-typewriter-cursor {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* Staggered fade-in for grouped children (e.g. KPI strip). */
.hrag-stagger > * { animation: hrag-fadein-up 0.5s ease-out backwards; }
.hrag-stagger > *:nth-child(1) { animation-delay: 0.02s; }
.hrag-stagger > *:nth-child(2) { animation-delay: 0.08s; }
.hrag-stagger > *:nth-child(3) { animation-delay: 0.14s; }
.hrag-stagger > *:nth-child(4) { animation-delay: 0.20s; }
.hrag-stagger > *:nth-child(5) { animation-delay: 0.26s; }
.hrag-stagger > *:nth-child(6) { animation-delay: 0.32s; }

/* Skeleton loader */
.hrag-skeleton {
  background: linear-gradient(90deg,
    rgba(255,255,255,0.04) 25%,
    rgba(255,255,255,0.10) 50%,
    rgba(255,255,255,0.04) 75%);
  background-size: 200% 100%;
  animation: hrag-skeleton 1.6s infinite ease-in-out;
  border-radius: 8px;
  height: 14px;
}

/* Glow halo for highlighted CTAs */
.hrag-glow-pulse { animation: hrag-glow 2s infinite; }

/* Streamlit input polish */
[data-testid="stTextInput"] input,
[data-testid="stChatInput"] textarea,
[data-baseweb="textarea"] textarea,
[data-baseweb="input"] input {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--hrag-border) !important;
  border-radius: 10px !important;
  color: var(--hrag-ink) !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stChatInput"] textarea:focus,
[data-baseweb="textarea"] textarea:focus,
[data-baseweb="input"] input:focus {
  border-color: rgba(232,220,196,0.45) !important;
  box-shadow: 0 0 0 3px rgba(232,220,196,0.08) !important;
  outline: none !important;
}

/* Expander polish */
[data-testid="stExpander"] {
  border-radius: 12px !important;
  border: 1px solid var(--hrag-border) !important;
  background: var(--hrag-card) !important;
  transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stExpander"]:hover { border-color: var(--hrag-border-strong) !important; }
[data-testid="stExpander"] summary { font-weight: 600 !important; }

/* Toast — sleeker */
[data-testid="stToast"] {
  background: rgba(20,20,20,0.95) !important;
  border: 1px solid var(--hrag-border-strong) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  box-shadow: var(--hrag-shadow-lift) !important;
  animation: hrag-slidein 0.3s cubic-bezier(0.22, 1, 0.36, 1) !important;
}

/* hide the default streamlit footer */
footer[class*="viewerBadge"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
"""


def apply_chrome(page_icon: str = "🧠", page_title: str = "HRAG-Bot") -> None:
    """Call once at the top of every page. Sets page config, injects CSS,
    and renders the sidebar user pill."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    _sidebar_user_pill()
    _sidebar_quickref()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def current_user_id(default: str = "default") -> str:
    """Active user id; initialises session_state on first read."""
    if "user_id" not in st.session_state:
        cfg = load_config()
        st.session_state["user_id"] = cfg.user.default_user_id or default
    return st.session_state["user_id"]


def set_user_id(user_id: str) -> None:
    st.session_state["user_id"] = user_id
    # Active chat session is per-user; reset when switching.
    st.session_state.pop("chat_session_id", None)
    st.session_state.pop("chat_history", None)


# ---------------------------------------------------------------------------
# Sidebar widgets
# ---------------------------------------------------------------------------


def _sidebar_user_pill() -> None:
    user = current_user_id()
    initial = (user[:1] or "?").upper()
    st.sidebar.markdown(
        f"<div class='hrag-user-pill'>"
        f"<div class='pill-avatar'>{initial}</div>"
        f"<div class='pill-meta'>"
        f"<span class='pill-label'>active user</span>"
        f"<span class='pill-value'>{user}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _sidebar_quickref() -> None:
    """Compact "what can I type?" cheat-sheet at the bottom of the sidebar."""
    with st.sidebar.expander("⌨️ Slash shortcuts", expanded=False):
        st.markdown(
            "Inside the chat box on the **💬 Chat** page:\n"
            "- `/remember <text>` — save a memory\n"
            "- `/recall <query>` — search memories only\n"
            "Otherwise just type a question; memories + docs compete in retrieval."
        )
    with st.sidebar.expander("💡 General tips", expanded=False):
        st.markdown(
            "- Switch user from the **👥 Users** page.\n"
            "- Bulk-import a folder of notes from **📚 Memories → Add → drop files**.\n"
            "- See per-message retrieval evidence by expanding the *Sources* block."
        )


# ---------------------------------------------------------------------------
# Page header + tips banner
# ---------------------------------------------------------------------------


def page_header(
    title: str,
    icon: str = "✨",
    subtitle: str = "",
    tips: Optional[list[str]] = None,
) -> None:
    """Render a polished page header banner with an optional tips card."""
    safe_title = title.replace("<", "&lt;")
    safe_subtitle = subtitle.replace("<", "&lt;") if subtitle else ""
    st.markdown(
        f"""
        <div class='hrag-page-header'>
          <div class='icon'>{icon}</div>
          <div>
            <h1>{safe_title}</h1>
            <div class='subtitle'>{safe_subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if tips:
        items = "".join(f"<li>{t}</li>" for t in tips)
        st.markdown(
            f"<div class='hrag-tipcard'><b>💡 Tips</b><ul>{items}</ul></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Streaming chat (threading + queue → status updates)
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    event: str
    payload: dict


class ChatStream:
    """Runs ``Orchestrator.chat(stream=True)`` in a daemon thread.

    The caller iterates ``events()`` for live progress; when the iterator
    finishes, ``.result`` is the final :class:`ChatResult` (or ``.error``
    is the exception). Token events are emitted as ``("generate_token", ...)``.
    """

    _END = "__hrag_end__"

    def __init__(
        self,
        orch: Orchestrator,
        question: str,
        user_id: str,
        session_id: Optional[str],
    ) -> None:
        self._orch = orch
        self._question = question
        self._user_id = user_id
        self._session_id = session_id
        self._queue: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.result: Optional[ChatResult] = None
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            self.result = self._orch.chat(
                self._question,
                user_id=self._user_id,
                session_id=self._session_id,
                progress=lambda e, p: self._queue.put((e, p)),
                stream=True,
            )
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
        finally:
            self._queue.put((self._END, {}))

    def start(self) -> "ChatStream":
        self._thread.start()
        return self

    def events(self) -> Iterator[StreamEvent]:
        while True:
            event, payload = self._queue.get()
            if event == self._END:
                break
            yield StreamEvent(event, payload)
        self._thread.join(timeout=1.0)


def stream_chat_events(
    orch: Orchestrator,
    question: str,
    user_id: str,
    session_id: Optional[str],
) -> ChatStream:
    """Convenience: build + start a ChatStream."""
    return ChatStream(orch, question, user_id, session_id).start()


# ---------------------------------------------------------------------------
# Source-card renderer
# ---------------------------------------------------------------------------


def render_source(idx: int, src) -> None:
    """Render a single RetrievalResult as a styled card."""
    chunk = src.chunk
    kind = "episodic" if chunk.source_type == "episodic" else "document"
    rr = (
        f"rerank={src.rerank_score:.2f}"
        if src.rerank_score is not None
        else f"score={src.score:.3f}"
    )
    title = (chunk.title or "Untitled").replace("<", "&lt;")
    section = (chunk.section or "").replace("<", "&lt;")
    tag_bits = [f"[{idx}]", title]
    if section:
        tag_bits.append(f"· {section}")
    tag_bits.append(f"· {kind}")
    tag_bits.append(f"· {rr}")
    st.markdown(
        f"<div class='hrag-source-card {kind}'>"
        f"<div class='stitle'>{' '.join(tag_bits)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.expander("show passage", expanded=False):
        st.code(chunk.text[:1200], language=None)


# ---------------------------------------------------------------------------
# Design-system primitives (chips, KPI cards, empty states, nav cards, …)
# ---------------------------------------------------------------------------


_CHIP_TONES = {"default", "accent", "violet", "good", "warn", "bad", "info", "muted"}


def chip(text: str, tone: str = "default") -> str:
    """Return HTML markup for an inline chip. Render via st.markdown(unsafe_allow_html=True)."""
    cls = tone if tone in _CHIP_TONES else "default"
    safe = str(text).replace("<", "&lt;")
    return f"<span class='hrag-chip {cls}'>{safe}</span>"


_POLARITY_TONE = {
    "fact": "info",
    "style": "violet",
    "like": "good",
    "dislike": "bad",
}


def polarity_chip(polarity: str) -> str:
    """Color-coded chip for the four preference polarities."""
    tone = _POLARITY_TONE.get(str(polarity).lower(), "muted")
    return chip(polarity, tone)


def kpi_card(
    container,
    label: str,
    value: object,
    icon: str = "",
    sub: str = "",
    tone: str = "accent",
) -> None:
    """Render a richer KPI card into a column / container.

    ``tone`` is one of: accent, good, warn, bad, info.
    """
    if tone not in {"accent", "good", "warn", "bad", "info"}:
        tone = "accent"
    icon_html = f"<span class='kpi-icon'>{icon}</span>" if icon else ""
    sub_html = (
        f"<div class='kpi-sub'>{str(sub).replace('<', '&lt;')}</div>" if sub else ""
    )
    safe_label = str(label).replace("<", "&lt;")
    container.markdown(
        f"<div class='hrag-kpi {tone}'>"
        f"<div class='kpi-row'>{icon_html}"
        f"<span class='kpi-label'>{safe_label}</span></div>"
        f"<div class='kpi-value'>{value}</div>"
        f"{sub_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def empty_state(
    icon: str,
    title: str,
    message: str = "",
    cta_label: Optional[str] = None,
    cta_page: Optional[str] = None,
) -> None:
    """Polished empty-state panel; optional CTA renders a Streamlit page_link below."""
    safe_title = title.replace("<", "&lt;")
    safe_msg = message.replace("<", "&lt;")
    st.markdown(
        f"<div class='hrag-empty'>"
        f"<div class='empty-icon'>{icon}</div>"
        f"<div class='empty-title'>{safe_title}</div>"
        f"<div class='empty-msg'>{safe_msg}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if cta_label and cta_page:
        # Center the CTA roughly with column spacers.
        left, mid, right = st.columns([2, 1, 2])
        with mid:
            try:
                st.page_link(cta_page, label=cta_label, icon="➡️")
            except Exception:  # noqa: BLE001
                # page_link can fail at module-load time outside a Streamlit run; ignore.
                pass


_DOC_TYPE_ICONS = {
    ".pdf": "📕",
    ".docx": "📘",
    ".doc": "📘",
    ".md": "📝",
    ".markdown": "📝",
    ".txt": "📄",
    ".rtf": "📄",
    ".html": "🌐",
    ".htm": "🌐",
}


def doc_type_icon(path: Optional[str]) -> str:
    """Return an emoji icon based on file extension. Falls back to 📎."""
    if not path:
        return "📎"
    p = str(path).lower()
    for ext, icon in _DOC_TYPE_ICONS.items():
        if p.endswith(ext):
            return icon
    return "📎"


def avatar_html(label: str, size: str = "") -> str:
    """Return HTML for a colored avatar circle using the first letter of ``label``."""
    initial = (str(label)[:1] or "?").upper()
    cls = "hrag-avatar lg" if size == "lg" else "hrag-avatar"
    return f"<div class='{cls}'>{initial}</div>"


def nav_card(container, icon: str, title: str, desc: str, page: str) -> None:
    """Render a clickable navigation card into a column.

    Because Streamlit doesn't yet support fully clickable card containers, we
    render the visual card with ``st.markdown`` (HTML) and stack a
    ``st.page_link`` below it that visually merges into the card — clicking
    anywhere on the link navigates. The card itself reflects hover state.
    """
    safe_title = title.replace("<", "&lt;")
    safe_desc = desc.replace("<", "&lt;")
    with container.container(border=False):
        st.markdown(
            f"<div class='hrag-nav-card'>"
            f"<span class='nav-arrow'>›</span>"
            f"<div class='nav-icon'>{icon}</div>"
            f"<div class='nav-title'>{safe_title}</div>"
            f"<div class='nav-desc'>{safe_desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        try:
            st.page_link(page, label=f"Open {title}", icon="➡️")
        except Exception:  # noqa: BLE001
            pass


def section_title(title: str, icon: str = "", caption: str = "") -> None:
    """Compact section header used between blocks on a page."""
    cap = (
        f"<div style='color:var(--hrag-muted);font-size:0.85rem;margin-top:-4px;'>"
        f"{caption.replace('<', '&lt;')}</div>"
        if caption
        else ""
    )
    icon_part = f"<span style='margin-right:6px;'>{icon}</span>" if icon else ""
    st.markdown(
        f"<h3 style='margin:18px 0 6px 0;'>{icon_part}{title.replace('<', '&lt;')}</h3>{cap}",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Backwards-compat re-exports (older pages imported these names)
# ---------------------------------------------------------------------------


def sidebar_user_pill() -> None:  # kept for any external callers
    _sidebar_user_pill()
