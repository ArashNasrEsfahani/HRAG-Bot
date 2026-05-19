// HRAG web UI — single-file vanilla JS chat client.
// Talks to /api/* on the same origin. SSE for streaming.

'use strict';

// ---------- markdown setup ----------
let _hljsEnabled = false;  // disabled during stream for perf; flipped on at end
marked.setOptions({
  gfm: true,
  breaks: true,
  highlight: (code, lang) => {
    if (!_hljsEnabled) return code;
    try {
      if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
      return hljs.highlightAuto(code).value;
    } catch { return code; }
  },
});

function renderMD(src, { withHighlight = false } = {}) {
  if (!src) return '';
  const prev = _hljsEnabled;
  _hljsEnabled = withHighlight;
  try {
    const html = marked.parse(src, { async: false });
    return DOMPurify.sanitize(html, { ADD_ATTR: ['target'] });
  } finally {
    _hljsEnabled = prev;
  }
}

// ChatGPT-style incremental markdown render during SSE streaming.
// Debounced to ~33ms so we never re-parse more than ~30 times/sec even
// when tokens arrive in a fast burst. Renders go through the same
// DOMPurify-sanitized path as the final render — no new XSS sink.
const STREAM_MD_DEBOUNCE_MS = 33;

function scheduleStreamingMdRender(els) {
  if (!els || !els.bubble) return;
  if (els.bubble._mdRenderPending) return;
  els.bubble._mdRenderPending = true;
  const now = performance.now();
  const elapsed = now - (els.bubble._mdLastRender || 0);
  const delay = Math.max(0, STREAM_MD_DEBOUNCE_MS - elapsed);
  els.bubble._mdRenderTimer = setTimeout(() => {
    els.bubble._mdRenderPending = false;
    els.bubble._mdRenderTimer = null;
    els.bubble._mdLastRender = performance.now();
    renderStreamingMd(els);
  }, delay);
}

function renderStreamingMd(els) {
  if (!els || !els.bubble) return;
  const raw = els.bubble._rawText || '';
  try {
    // No hljs during stream — keeps per-render cost low and matches the
    // existing pattern where withHighlight is only true on the final render.
    // Append an inline blinking caret span so the user sees streaming
    // progress even after innerHTML is replaced each tick. The trailing
    // <span class="md-cursor"></span> is stripped by the final-event
    // handler when it re-renders without it.
    els.bubble.innerHTML = renderMD(raw, { withHighlight: false }) +
      '<span class="md-cursor" aria-hidden="true"></span>';
  } catch (e) {
    // Fallback: dump the raw text so the user never sees a blank bubble.
    els.bubble.textContent = raw;
  }
}

function escapeHTML(s) {
  return String(s ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function attachCopyButtons(root) {
  for (const pre of root.querySelectorAll('pre')) {
    if (pre.querySelector('.copy-btn')) continue;
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.type = 'button';
    btn.textContent = 'copy';
    btn.addEventListener('click', async () => {
      const code = pre.querySelector('code');
      try {
        await navigator.clipboard.writeText(code?.innerText ?? pre.innerText);
        btn.textContent = 'copied';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('copied'); }, 1400);
      } catch { btn.textContent = 'failed'; }
    });
    pre.appendChild(btn);
  }
}

/** Detect text direction from a string. Returns 'rtl' if the sample is
 *  dominantly Persian/Arabic/Hebrew, 'ltr' if dominantly Latin, 'auto'
 *  for mixed/unknown — caller hands 'auto' to the browser so BiDi
 *  resolves per-paragraph. */
function detectDir(text) {
  if (!text) return 'auto';
  const sample = text.slice(0, 500);
  // Hebrew + Arabic + Persian + Arabic Supplement + N'Ko + Samaritan +
  // Mandaic + Arabic Extended-A + Arabic Presentation Forms A/B
  const rtlRe = /[֐-׿؀-ۿ܀-ݏݐ-ݿހ-޿߀-߿ࠀ-࠿ࡀ-࡟ࢠ-ࣿיִ-﷿ﹰ-﻿]/;
  const ltrRe = /[A-Za-z]/;
  const rtl = rtlRe.test(sample);
  const ltr = ltrRe.test(sample);
  if (rtl && !ltr) return 'rtl';
  if (ltr && !rtl) return 'ltr';
  return 'auto';  // mixed or unknown — let the browser decide
}

/** Mark a content-bearing element so the browser BiDi algorithm picks
 *  direction per first strong character, and apply `lang="fa"` only when
 *  the text leans RTL so the Persian font stack kicks in. Paragraph-level
 *  resolution comes from `unicode-bidi: plaintext` in styles.css. */
function maybeRTL(el, text) {
  // `dir="auto"` handles mixed English+Farsi paragraphs (e.g.
  // "Mahmoud monfared درباره چی میدونی") far better than a hard rtl/ltr flip.
  el.setAttribute('dir', 'auto');
  if (detectDir(text) === 'rtl') {
    el.setAttribute('lang', 'fa');
  } else {
    el.removeAttribute('lang');
  }
}

// ---------- DOM refs ----------
const $ = (id) => document.getElementById(id);
const app = $('app');
const messagesEl = $('messages');
const emptyState = $('empty-state');
const scrollEl = $('scroll');
const anchorEl = $('anchor');
const inputEl = $('input');
const sendBtn = $('send');
const composerEl = $('composer');
const sessionListEl = $('session-list');
const newChatBtn = $('new-chat');
const toggleSidebarBtn = $('toggle-sidebar');
const openSidebarBtn = $('open-sidebar');
const openSettingsBtn = $('open-settings');
const closeDrawerBtn = $('close-drawer');
const drawerScrim = $('drawer-scrim');
const userPillBtn = $('user-pill');

// settings drawer fields
const cfgProvider = $('cfg-provider');
const cfgModel = $('cfg-model');
const cfgNumCtx = $('cfg-num-ctx');
const cfgThink = $('cfg-think');
const cfgRetriever = $('cfg-retriever');
const cfgReranker = $('cfg-reranker');
const cfgRerankEnabled = $('cfg-rerank-enabled');
const cfgGate = $('cfg-gate');
const cfgClue = $('cfg-clue');
const cfgMst = $('cfg-mst');
const cfgMask = $('cfg-mask');
// Phase 6 settings
const cfgAdaptive = $('cfg-adaptive');
const cfgBias = $('cfg-bias');
const cfgBiasRow = $('cfg-bias-row');
const cfgKeepAlive = $('cfg-keep-alive');
const topkInputs = {
  greeting: $('topk-greeting'),
  personal: $('topk-personal'),
  factual:  $('topk-factual'),
  general:  $('topk-general'),
  unclear:  $('topk-unclear'),
};
const beVectorVal = $('be-vector-val');
const beKgVal = $('be-kg-val');
// Phase 7-A settings
const cfgMathMeta = $('cfg-math-meta');
const cfgFormula = $('cfg-formula');
const cfgFormulaRow = $('cfg-formula-row');
const cfgMathRerank = $('cfg-math-rerank');
const cfgFormulaTokens = $('cfg-formula-tokens');

// Phase 6 inline: num_keep
const cfgNumKeep = $('cfg-num-keep');

// Phase 6-B per-intent retriever selects
const adrSelects = {
  greeting: $('adr-greeting'),
  personal: $('adr-personal'),
  factual:  $('adr-factual'),
  general:  $('adr-general'),
  unclear:  $('adr-unclear'),
};

// Phase 7-B embedding model
const cfgEmbModel = $('cfg-embeddings-model');
const embCurrentVal = $('emb-current-val');

// Phase 7-C nougat
const cfgUseNougat = $('cfg-use-nougat');
const cfgNougatModel = $('cfg-nougat-model');
const nougatStatusBadge = $('nougat-status-badge');
const nougatStatusVal = $('nougat-status-val');

// topbar
const topbarModel = $('topbar-model');
const topbarRetriever = $('topbar-retriever');

// user pill
const avatarEl = $('avatar');
const userNameEl = $('user-name');

// sidebar nav + drawers
const openMemoriesBtn = $('open-memories');
const closeMemoriesBtn = $('close-memories');
const openDocsBtn = $('open-docs');
const closeDocsBtn = $('close-docs');
const memCountEl = $('mem-count');
const docCountEl = $('doc-count');
const memListEl = $('mem-list');
const memSearchEl = $('mem-search');
const memRefreshBtn = $('mem-refresh');
const memAddBtn = $('mem-add');
const docsListEl = $('docs-list');
const uploadZoneEl = $('upload-zone');
const uploadInputEl = $('upload-input');
const uploadBrowseBtn = $('upload-browse');
const uploadProgressEl = $('upload-progress');

// feedback drawer
const openFeedbackBtn = $('open-feedback');
const closeFeedbackBtn = $('close-feedback');
const refreshFeedbackBtn = $('refresh-feedback');
const fbCountEl = $('fb-count');
const fbUpNum = $('fb-up-num');
const fbDownNum = $('fb-down-num');
const fbTotalNum = $('fb-total-num');
const fbSessionsNum = $('fb-sessions-num');
const fbRatioUp = $('fb-ratio-up');
const fbRatioDown = $('fb-ratio-down');
const fbNegativesEl = $('fb-negatives');

// ---------- state ----------
const state = {
  sessionId: null,
  config: null,
  streaming: false,
  controller: null,  // AbortController for in-flight stream
  // Phase 8 inline review-card state
  currentTurnId: null,       // captured from the orchestrator's start event
  pendingReview: null,       // the ReviewRequired payload while the card is open
  reviewCardOpen: false,
  _streamEls: null,          // active turn's streaming DOM handles (used by the inline review card)
};

// ---------- theme ----------
function applyTheme(theme) {
  if (theme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  // swap the highlight.js stylesheet too
  const hljsLink = document.getElementById('hljs-style');
  if (hljsLink) {
    hljsLink.href = theme === 'light'
      ? 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.10.0/build/styles/github.min.css'
      : 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.10.0/build/styles/github-dark.min.css';
  }
  // swap the topbar icon (sun ↔ moon)
  const icon = document.getElementById('theme-icon');
  if (icon) {
    icon.innerHTML = theme === 'light'
      ? '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>'   // moon
      : '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>';  // sun
  }
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem('hrag-theme'); } catch {}
  const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  const theme = saved || (prefersLight ? 'light' : 'dark');
  applyTheme(theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  const next = current === 'light' ? 'dark' : 'light';
  applyTheme(next);
  try { localStorage.setItem('hrag-theme', next); } catch {}
}

initTheme();

// ---------- init ----------
(async function init() {
  // wire UI
  // Bilingual: textarea picks RTL or LTR per first-strong character so a
  // Farsi sentence shows the caret on the right, English on the left.
  inputEl.setAttribute('dir', 'auto');
  inputEl.addEventListener('input', autoResize);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  });
  composerEl.addEventListener('submit', (e) => { e.preventDefault(); submit(); });
  newChatBtn.addEventListener('click', () => startNew());
  toggleSidebarBtn.addEventListener('click', () => app.classList.toggle('collapsed'));
  openSidebarBtn.addEventListener('click', () => {
    app.classList.remove('collapsed');
    app.classList.add('sidebar-open');
  });
  const themeBtn = document.getElementById('theme-toggle');
  if (themeBtn) themeBtn.addEventListener('click', toggleTheme);
  openSettingsBtn.addEventListener('click', () => app.classList.add('drawer-open'));
  closeDrawerBtn.addEventListener('click', () => app.classList.remove('drawer-open'));
  drawerScrim.addEventListener('click', () => {
    app.classList.remove('drawer-open');
    app.classList.remove('memories-open');
    app.classList.remove('docs-open');
    app.classList.remove('feedback-open');
  });
  openMemoriesBtn.addEventListener('click', async () => {
    app.classList.add('memories-open');
    await loadMemories();
  });
  closeMemoriesBtn.addEventListener('click', () => app.classList.remove('memories-open'));
  openDocsBtn.addEventListener('click', async () => {
    app.classList.add('docs-open');
    await loadDocs();
  });
  closeDocsBtn.addEventListener('click', () => app.classList.remove('docs-open'));

  // Feedback drawer wiring
  if (openFeedbackBtn) {
    openFeedbackBtn.addEventListener('click', async () => {
      app.classList.add('feedback-open');
      if (!_feedbackCache) await loadFeedbackStats();
      else paintFeedbackStats(_feedbackCache);
    });
  }
  if (closeFeedbackBtn) {
    closeFeedbackBtn.addEventListener('click', () => app.classList.remove('feedback-open'));
  }
  if (refreshFeedbackBtn) {
    refreshFeedbackBtn.addEventListener('click', () => loadFeedbackStats());
  }

  // Upload zone wiring
  if (uploadZoneEl) {
    uploadBrowseBtn.addEventListener('click', () => uploadInputEl.click());
    uploadInputEl.addEventListener('change', (e) => handleUploadFiles(e.target.files));
    ['dragenter', 'dragover'].forEach(ev => {
      uploadZoneEl.addEventListener(ev, (e) => {
        e.preventDefault(); e.stopPropagation();
        uploadZoneEl.classList.add('drag-over');
      });
    });
    ['dragleave', 'drop'].forEach(ev => {
      uploadZoneEl.addEventListener(ev, (e) => {
        e.preventDefault(); e.stopPropagation();
        uploadZoneEl.classList.remove('drag-over');
      });
    });
    uploadZoneEl.addEventListener('drop', (e) => {
      const files = e.dataTransfer?.files;
      if (files && files.length) handleUploadFiles(files);
    });
  }
  memSearchEl.addEventListener('input', filterMemories);
  memRefreshBtn.addEventListener('click', () => loadMemories());
  memAddBtn.addEventListener('click', () => startMemoryAdd());
  userPillBtn.addEventListener('click', switchUserPrompt);

  // "Remember" composer button — saves the current input as an episodic memory.
  // If input is empty AND there's an active session with ≥ 2 messages,
  // it opens the Smart Remember modal instead.
  const rememberBtn = document.getElementById('remember-btn');
  if (rememberBtn) {
    rememberBtn.addEventListener('click', () => onRememberClick());
  }
  const rememberCaretBtn = document.getElementById('remember-caret-btn');
  if (rememberCaretBtn) {
    rememberCaretBtn.addEventListener('click', () => openSmartRememberModal());
  }
  // Alt+R = same as clicking the button (input contents decide).
  // Alt+Shift+R = always open the Smart Remember modal.
  document.addEventListener('keydown', (e) => {
    if (e.altKey && (e.key === 'r' || e.key === 'R')) {
      e.preventDefault();
      if (e.shiftKey) openSmartRememberModal();
      else onRememberClick();
    }
  });
  // Smart Remember modal handlers
  const remModal = document.getElementById('rem-modal');
  const remScrim = document.getElementById('rem-modal-scrim');
  const remClose = document.getElementById('rem-modal-close');
  const remCancel = document.getElementById('rem-modal-cancel');
  const remSave = document.getElementById('rem-modal-save');
  if (remClose) remClose.addEventListener('click', closeSmartRememberModal);
  if (remCancel) remCancel.addEventListener('click', closeSmartRememberModal);
  if (remScrim) remScrim.addEventListener('click', closeSmartRememberModal);
  if (remSave) remSave.addEventListener('click', saveSmartRememberSelected);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && remModal && !remModal.hidden) closeSmartRememberModal();
  });

  // suggestion buttons
  document.querySelectorAll('.suggestion').forEach(b => {
    b.addEventListener('click', () => { inputEl.value = b.textContent || ''; autoResize(); inputEl.focus(); });
  });

  // shortcuts
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault(); startNew(); inputEl.focus();
    }
    if (e.key === 'Escape') {
      app.classList.remove('drawer-open');
      app.classList.remove('memories-open');
      app.classList.remove('docs-open');
      app.classList.remove('feedback-open');
    }
  });

  // settings change → patch
  cfgModel.addEventListener('change', async () => {
    await patchConfig({ model: cfgModel.value });
    flashToast(`switched LLM → ${cfgModel.value}`);
  });
  cfgNumCtx.addEventListener('change', () => patchConfig({ num_ctx: parseInt(cfgNumCtx.value, 10) }));
  cfgThink.addEventListener('change', () => patchConfig({ think: cfgThink.checked }));
  cfgRetriever.addEventListener('change', async () => {
    await patchConfig({ retriever: cfgRetriever.value });
    flashToast(`retriever → ${cfgRetriever.value}`);
  });
  cfgReranker.addEventListener('change', async () => {
    await patchConfig({ reranker: cfgReranker.value });
    flashToast(`reranker → ${cfgReranker.value}`);
  });
  cfgRerankEnabled.addEventListener('change', () => patchConfig({ rerank_enabled: cfgRerankEnabled.checked }));
  cfgGate.addEventListener('change', () => patchConfig({ gate_enabled: cfgGate.checked }));
  cfgClue.addEventListener('change', () => patchConfig({ clue_enabled: cfgClue.checked }));
  cfgMst.addEventListener('change', () => patchConfig({ dialog_mst_enabled: cfgMst.checked }));
  cfgMask.addEventListener('change', () => patchConfig({ mask_uncertain: cfgMask.checked }));

  // ---- Phase 6 wiring ----
  cfgAdaptive.addEventListener('change', async () => {
    await patchConfig({ adaptive_enabled: cfgAdaptive.checked });
    flashToast(`adaptive retrieval ${cfgAdaptive.checked ? 'on' : 'off'}`);
    syncBiasEnabled();
  });
  cfgBias.addEventListener('change', async () => {
    if (cfgBias.disabled) return;
    await patchConfig({ adaptive_personal_episodic_bias: cfgBias.checked });
    flashToast(`episodic bias ${cfgBias.checked ? 'on' : 'off'}`);
  });
  // Save keep-alive on blur or Enter (NOT every keystroke)
  const saveKeepAlive = async () => {
    const v = cfgKeepAlive.value.trim();
    const prev = state.config?.llm?.keep_alive ?? '';
    if (!v || v === prev) return;
    await patchConfig({ keep_alive: v });
    flashToast(`keep-alive → ${v}`);
  };
  cfgKeepAlive.addEventListener('blur', saveKeepAlive);
  cfgKeepAlive.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); cfgKeepAlive.blur(); }
  });
  // Per-intent top_k: PATCH the full dict on blur / Enter of any input
  const saveTopK = async () => {
    const out = {};
    for (const [k, el] of Object.entries(topkInputs)) {
      const n = parseInt(el.value, 10);
      if (Number.isFinite(n) && n >= 0) out[k] = n;
    }
    if (!Object.keys(out).length) return;
    // Skip if unchanged
    const cur = state.config?.retrieval?.adaptive_top_k || {};
    let dirty = false;
    for (const [k, v] of Object.entries(out)) if (cur[k] !== v) { dirty = true; break; }
    if (!dirty) return;
    await patchConfig({ adaptive_top_k: out });
    flashToast('per-intent top_k saved');
  };
  for (const el of Object.values(topkInputs)) {
    el.addEventListener('blur', saveTopK);
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); el.blur(); }
    });
  }

  // ---- Phase 7-A wiring ----
  if (cfgMathMeta) {
    cfgMathMeta.addEventListener('change', async () => {
      await patchConfig({ math_meta_filter_enabled: cfgMathMeta.checked });
      flashToast(`math-meta filter ${cfgMathMeta.checked ? 'on' : 'off'}`);
      syncFormulaEnabled();
    });
  }
  if (cfgFormula) {
    cfgFormula.addEventListener('change', async () => {
      if (cfgFormula.disabled) return;
      await patchConfig({ formula_extraction_enabled: cfgFormula.checked });
      flashToast(`formula extraction ${cfgFormula.checked ? 'on' : 'off'}`);
    });
  }
  const saveMathRerank = async () => {
    const v = parseFloat(cfgMathRerank.value);
    if (!Number.isFinite(v)) return;
    const prev = state.config?.retrieval?.math_meta_rerank_threshold;
    if (prev != null && Math.abs(prev - v) < 1e-9) return;
    await patchConfig({ math_meta_rerank_threshold: v });
    flashToast(`math-meta rerank → ${v}`);
  };
  if (cfgMathRerank) {
    cfgMathRerank.addEventListener('blur', saveMathRerank);
    cfgMathRerank.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); cfgMathRerank.blur(); }
    });
  }
  const saveFormulaTokens = async () => {
    const v = parseInt(cfgFormulaTokens.value, 10);
    if (!Number.isFinite(v) || v <= 0) return;
    const prev = state.config?.formula_extraction?.max_tokens;
    if (prev === v) return;
    await patchConfig({ formula_extraction_max_tokens: v });
    flashToast(`extraction max_tokens → ${v}`);
  };
  if (cfgFormulaTokens) {
    cfgFormulaTokens.addEventListener('blur', saveFormulaTokens);
    cfgFormulaTokens.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); cfgFormulaTokens.blur(); }
    });
  }

  // ---- num_keep (inline with keep_alive) ----
  const saveNumKeep = async () => {
    const raw = cfgNumKeep.value.trim();
    const prev = state.config?.llm?.num_keep;
    let v;
    if (raw === '') {
      v = null;
    } else {
      const n = parseInt(raw, 10);
      if (!Number.isFinite(n) || n < 0) return;
      v = n;
    }
    if (v === prev) return;
    await patchConfig({ num_keep: v });
    flashToast(`num_keep → ${v == null ? '(auto)' : v}`);
  };
  if (cfgNumKeep) {
    cfgNumKeep.addEventListener('blur', saveNumKeep);
    cfgNumKeep.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); cfgNumKeep.blur(); }
    });
  }

  // ---- Phase 6-B per-intent retriever overrides ----
  for (const [intent, sel] of Object.entries(adrSelects)) {
    if (!sel) continue;
    sel.addEventListener('change', async () => {
      await patchConfig({ adaptive_retriever_per_intent: { [intent]: sel.value } });
      flashToast(`${intent} retriever → ${sel.value}`);
    });
  }

  // ---- Phase 7-B embedding model ----
  if (cfgEmbModel) {
    cfgEmbModel.addEventListener('change', async () => {
      const v = cfgEmbModel.value;
      if (!v) return;
      try {
        const r = await fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ embeddings_model: v }),
        });
        const body = await r.json();
        state.config = body;
        paintConfig(body);
        // Surface a server warning if present
        const warn = body?.warning || body?.warnings?.embeddings_model || '';
        flashToast(warn ? `embedding → ${v} (${warn})` : `embedding → ${v} (re-ingest required)`);
      } catch (e) {
        console.error('embeddings model patch failed', e);
        flashToast('embedding model update failed');
      }
    });
  }

  // ---- Phase 7-C nougat ----
  if (cfgUseNougat) {
    cfgUseNougat.addEventListener('change', async () => {
      await patchConfig({ use_nougat: cfgUseNougat.checked });
      flashToast(`nougat ${cfgUseNougat.checked ? 'on' : 'off'}`);
    });
  }
  const saveNougatModel = async () => {
    const v = cfgNougatModel.value.trim();
    const prev = state.config?.ingest?.nougat_model ?? '';
    if (!v || v === prev) return;
    await patchConfig({ nougat_model: v });
    flashToast(`nougat model → ${v}`);
  };
  if (cfgNougatModel) {
    cfgNougatModel.addEventListener('blur', saveNougatModel);
    cfgNougatModel.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); cfgNougatModel.blur(); }
    });
  }

  // Phase 8: wire up the inline review-card global keyboard shortcuts.
  initReviewCard();

  await loadConfig();
  await loadLLMModels();
  await loadEmbeddingSuggestions();
  await loadNougatStatus();
  await loadSessions();
  await refreshSidebarCounts();
})();

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 220) + 'px';
  sendBtn.disabled = inputEl.value.trim().length === 0 && !state.streaming;
}

// ---------- config ----------
async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    state.config = await r.json();
    paintConfig(state.config);
  } catch (e) { console.error('config load failed', e); }
}

function paintConfig(c) {
  cfgProvider.textContent = c.llm.provider;
  // cfgModel options are populated by loadLLMModels(); we just sync `value`.
  if (cfgModel.querySelector(`option[value="${CSS.escape(c.llm.model)}"]`)) {
    cfgModel.value = c.llm.model;
  } else if (c.llm.model) {
    // model not yet in dropdown (e.g. first paint before loadLLMModels) —
    // insert a temporary option so the value still reflects reality.
    const opt = document.createElement('option');
    opt.value = c.llm.model;
    opt.textContent = c.llm.model;
    cfgModel.appendChild(opt);
    cfgModel.value = c.llm.model;
  }
  if (c.llm.num_ctx != null) cfgNumCtx.value = String(c.llm.num_ctx);
  cfgThink.checked = !!c.llm.think;
  cfgRetriever.value = c.retrieval.retriever;
  cfgReranker.value = c.retrieval.reranker;
  cfgRerankEnabled.checked = !!c.retrieval.rerank_enabled;
  cfgGate.checked = !!c.compaction.gate_enabled;
  cfgClue.checked = !!c.compaction.clue_enabled;
  cfgMst.checked = !!c.compaction.dialog_mst_enabled;
  cfgMask.checked = !!c.compaction.mask_uncertain;

  // Phase 6 fields
  if (cfgAdaptive) cfgAdaptive.checked = !!c.retrieval?.adaptive_enabled;
  if (cfgBias) cfgBias.checked = !!c.retrieval?.adaptive_personal_episodic_bias;
  if (cfgKeepAlive && document.activeElement !== cfgKeepAlive) {
    cfgKeepAlive.value = c.llm?.keep_alive ?? '';
  }
  const topk = c.retrieval?.adaptive_top_k || {};
  for (const [k, el] of Object.entries(topkInputs)) {
    if (!el || document.activeElement === el) continue;
    if (topk[k] != null) el.value = String(topk[k]);
  }
  if (beVectorVal) beVectorVal.textContent = c.retrieval?.vector_backend || 'chroma';
  if (beKgVal) beKgVal.textContent = c.kg?.backend || 'networkx';
  syncBiasEnabled();

  // Phase 7-A fields
  if (cfgMathMeta) cfgMathMeta.checked = !!c.retrieval?.math_meta_filter_enabled;
  if (cfgFormula) cfgFormula.checked = !!c.formula_extraction?.enabled;
  if (cfgMathRerank && document.activeElement !== cfgMathRerank) {
    const v = c.retrieval?.math_meta_rerank_threshold;
    if (v != null) cfgMathRerank.value = String(v);
  }
  if (cfgFormulaTokens && document.activeElement !== cfgFormulaTokens) {
    const v = c.formula_extraction?.max_tokens;
    if (v != null) cfgFormulaTokens.value = String(v);
  }
  syncFormulaEnabled();

  // num_keep (Phase 6 inline)
  if (cfgNumKeep && document.activeElement !== cfgNumKeep) {
    const nk = c.llm?.num_keep;
    cfgNumKeep.value = (nk == null) ? '' : String(nk);
  }

  // Phase 6-B per-intent retriever overrides
  const adrMap = c.retrieval?.adaptive_retriever_per_intent || {};
  for (const [intent, sel] of Object.entries(adrSelects)) {
    if (!sel || document.activeElement === sel) continue;
    const v = adrMap[intent] || 'default';
    if (sel.querySelector(`option[value="${CSS.escape(v)}"]`)) {
      sel.value = v;
    } else {
      sel.value = 'default';
    }
  }

  // Phase 7-B embedding model (current chip)
  if (embCurrentVal) {
    const m = c.embeddings?.model || '—';
    const d = c.embeddings?.dim;
    embCurrentVal.textContent = (d != null) ? `${m} (${d}-d)` : m;
  }
  // Sync embedding dropdown selection if it's already populated
  if (cfgEmbModel && c.embeddings?.model && document.activeElement !== cfgEmbModel) {
    if (cfgEmbModel.querySelector(`option[value="${CSS.escape(c.embeddings.model)}"]`)) {
      cfgEmbModel.value = c.embeddings.model;
    }
  }

  // Phase 7-C nougat
  if (cfgUseNougat) cfgUseNougat.checked = !!c.ingest?.use_nougat;
  if (cfgNougatModel && document.activeElement !== cfgNougatModel) {
    cfgNougatModel.value = c.ingest?.nougat_model || '';
  }

  topbarModel.textContent = c.llm.model;
  topbarRetriever.textContent = c.retrieval.retriever;
  if (c.user_id) {
    userNameEl.textContent = c.user_id;
    avatarEl.textContent = (c.user_id[0] || '?').toUpperCase();
  }
}

function syncBiasEnabled() {
  if (!cfgBias || !cfgBiasRow) return;
  const on = !!cfgAdaptive?.checked;
  cfgBias.disabled = !on;
  cfgBiasRow.classList.toggle('disabled', !on);
  cfgBiasRow.title = on ? '' : 'Enable Adaptive retrieval first';
}

function syncFormulaEnabled() {
  if (!cfgFormula || !cfgFormulaRow) return;
  const on = !!cfgMathMeta?.checked;
  cfgFormula.disabled = !on;
  cfgFormulaRow.classList.toggle('disabled', !on);
  cfgFormulaRow.title = on ? '' : 'Requires math-meta filter to also be on';
}

async function loadLLMModels() {
  try {
    const r = await fetch('/api/llm/models');
    const data = await r.json();
    const models = data.models || [];
    // Build a fresh options list
    cfgModel.innerHTML = '';
    if (!models.length) {
      const opt = document.createElement('option');
      opt.textContent = state.config?.llm?.model || '(none installed)';
      cfgModel.appendChild(opt);
    } else {
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        const size = m.size_gb != null ? ` · ${m.size_gb.toFixed(1)} GB` : '';
        opt.textContent = `${m.name}${size}`;
        cfgModel.appendChild(opt);
      }
    }
    if (state.config?.llm?.model) cfgModel.value = state.config.llm.model;
  } catch (e) { console.error('llm models load failed', e); }
}

async function loadEmbeddingSuggestions() {
  if (!cfgEmbModel) return;
  try {
    const r = await fetch('/api/embeddings/suggested');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    const suggestions = data.suggestions || [];
    cfgEmbModel.innerHTML = '';
    if (!suggestions.length) {
      const opt = document.createElement('option');
      const cur = data.current || state.config?.embeddings?.model || '(none)';
      opt.value = cur;
      opt.textContent = cur;
      cfgEmbModel.appendChild(opt);
    } else {
      // Make sure the current model is always selectable, even if it's not
      // in the suggested list (insert at top).
      const current = data.current || state.config?.embeddings?.model || '';
      const seen = new Set();
      if (current && !suggestions.some(s => s.model === current)) {
        const opt = document.createElement('option');
        opt.value = current;
        opt.textContent = `${current} (current)`;
        cfgEmbModel.appendChild(opt);
        seen.add(current);
      }
      for (const s of suggestions) {
        if (seen.has(s.model)) continue;
        seen.add(s.model);
        const opt = document.createElement('option');
        opt.value = s.model;
        const dim = s.dim != null ? ` · ${s.dim}-d` : '';
        opt.textContent = `${s.label || s.model}${dim}`;
        cfgEmbModel.appendChild(opt);
      }
      if (current && cfgEmbModel.querySelector(`option[value="${CSS.escape(current)}"]`)) {
        cfgEmbModel.value = current;
      }
    }
    // Refresh the current-chip
    if (embCurrentVal) {
      const m = data.current || state.config?.embeddings?.model || '—';
      const d = data.current_dim ?? state.config?.embeddings?.dim;
      embCurrentVal.textContent = (d != null) ? `${m} (${d}-d)` : m;
    }
  } catch (e) {
    console.error('embedding suggestions load failed', e);
    // Fall back to whatever the config says
    if (cfgEmbModel.children.length === 0) {
      const cur = state.config?.embeddings?.model || '(unknown)';
      const opt = document.createElement('option');
      opt.value = cur;
      opt.textContent = cur;
      cfgEmbModel.appendChild(opt);
    }
  }
}

async function loadNougatStatus() {
  if (!nougatStatusBadge || !nougatStatusVal) return;
  try {
    const r = await fetch('/api/ingest/nougat_status');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    nougatStatusBadge.classList.remove('installed', 'missing');
    if (data.available) {
      nougatStatusBadge.classList.add('installed');
      nougatStatusVal.textContent = 'Nougat installed';
    } else {
      nougatStatusBadge.classList.add('missing');
      nougatStatusVal.textContent = 'Not installed — run `pip install nougat-ocr` (~800MB)';
    }
    // Also reflect the current `use_nougat` from server
    if (cfgUseNougat && data.use_nougat != null) {
      cfgUseNougat.checked = !!data.use_nougat;
    }
    if (cfgNougatModel && document.activeElement !== cfgNougatModel && data.model) {
      if (!cfgNougatModel.value) cfgNougatModel.value = data.model;
    }
  } catch (e) {
    console.error('nougat status load failed', e);
    nougatStatusBadge.classList.remove('installed', 'missing');
    nougatStatusBadge.classList.add('missing');
    nougatStatusVal.textContent = 'status unavailable';
  }
}

let _toastTimer = null;
function flashToast(msg) {
  let t = document.getElementById('inline-toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'inline-toast';
    t.className = 'inline-toast';
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add('visible');
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove('visible'), 1800);
}

async function patchConfig(patch) {
  try {
    const r = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    state.config = await r.json();
    paintConfig(state.config);
  } catch (e) { console.error('config patch failed', e); }
}

async function switchUserPrompt() {
  const next = prompt('Switch to user id:', state.config?.user_id || 'default');
  if (!next || !next.trim()) return;
  try {
    const r = await fetch('/api/users/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: next.trim() }),
    });
    if (!r.ok) throw new Error('switch failed');
    state.sessionId = null;
    resetMessages();
    await loadConfig();
    await loadSessions();
  } catch (e) { console.error(e); }
}

// ---------- sessions ----------
async function loadSessions() {
  try {
    const r = await fetch('/api/sessions');
    const items = await r.json();
    paintSessions(items);
  } catch (e) { console.error('sessions load failed', e); paintSessions([]); }
}

function paintSessions(items) {
  if (!items.length) {
    sessionListEl.innerHTML = '<div class="loading-row" style="color:var(--muted-2);font-size:0.85rem">No conversations yet.</div>';
    return;
  }
  sessionListEl.innerHTML = '';
  for (const s of items) {
    const btn = document.createElement('div');
    btn.className = 'session-item' + (s.session_id === state.sessionId ? ' active' : '');
    btn.title = s.title;
    btn.textContent = s.title;
    btn.addEventListener('click', () => loadSession(s.session_id));

    const del = document.createElement('button');
    del.className = 'delete';
    del.type = 'button';
    del.title = 'Delete';
    del.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14"/></svg>';
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('Delete this conversation?')) return;
      await fetch('/api/sessions/' + s.session_id, { method: 'DELETE' });
      if (s.session_id === state.sessionId) { state.sessionId = null; resetMessages(); }
      loadSessions();
    });
    btn.appendChild(del);
    sessionListEl.appendChild(btn);
  }
}

async function loadSession(sid) {
  if (state.streaming) abortStream();
  try {
    const [sessResp, fbResp] = await Promise.all([
      fetch('/api/sessions/' + sid),
      fetch('/api/feedback?session_id=' + encodeURIComponent(sid)),
    ]);
    const data = await sessResp.json();
    // Build a map of message_id → rating from existing feedback
    let fbMap = {};
    try {
      const fbItems = await fbResp.json();
      for (const fb of fbItems) fbMap[String(fb.message_id)] = fb.rating;
    } catch {}
    state.sessionId = sid;
    resetMessages();
    for (const m of data.messages) {
      const { wrap } = appendMessage(m.role, m.content, m.id) || {};
      if (wrap && m.role === 'assistant' && m.id != null) {
        const mid = String(m.id);
        if (fbMap[mid] !== undefined) {
          const fbBar = wrap.querySelector('.msg-feedback');
          if (fbBar?._applyRating) fbBar._applyRating(fbMap[mid]);
        }
      }
    }
    scrollToEnd(true);
    document.querySelectorAll('.session-item').forEach(el => el.classList.toggle('active', el.title.startsWith((data.messages.find(x => x.role === 'user')?.content || '').slice(0, 60))));
    // simpler: reload list to repaint active state
    loadSessions();
  } catch (e) { console.error(e); }
}

function startNew() {
  if (state.streaming) abortStream();
  state.sessionId = null;
  resetMessages();
  inputEl.focus();
  loadSessions();
}

// ---------- message rendering ----------
function resetMessages() {
  messagesEl.innerHTML = '';
  // Re-insert empty state
  const es = document.createElement('div');
  es.className = 'empty-state';
  es.innerHTML = `
    <div class="hero-mark">◆</div>
    <h1>How can I help today?</h1>
    <p class="hero-sub">Ask anything across your documents and memories.</p>
    <div class="suggestions">
      <button class="suggestion">Summarise the key ideas from my last paper.</button>
      <button class="suggestion">/remember Today I learned about RAGate.</button>
      <button class="suggestion">/recall what did I save about taxonomy?</button>
      <button class="suggestion">Compare HippoRAG and GraphRAG approaches.</button>
    </div>`;
  messagesEl.appendChild(es);
  es.querySelectorAll('.suggestion').forEach(b => b.addEventListener('click', () => {
    inputEl.value = b.textContent || ''; autoResize(); inputEl.focus();
  }));
}

function removeEmptyState() {
  const es = messagesEl.querySelector('.empty-state');
  if (es) es.remove();
}

function appendMessage(role, content, messageId = null) {
  removeEmptyState();
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  if (messageId != null) wrap.dataset.messageId = String(messageId);
  const head = document.createElement('div');
  head.className = 'msg-header';
  head.innerHTML = `<span class="role">${role === 'user' ? 'You' : 'HRAG'}</span>`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (role === 'user') {
    bubble.textContent = content;
    maybeRTL(bubble, content);
  } else {
    bubble.innerHTML = renderMD(content, { withHighlight: true });
    attachCopyButtons(bubble);
    maybeRTL(bubble, content);
    wrap.appendChild(head);
    wrap.appendChild(bubble);
    wrap.appendChild(buildFeedbackBar(messageId));
    messagesEl.appendChild(wrap);
    return { wrap, bubble };
  }
  wrap.appendChild(head);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  return { wrap, bubble };
}

// ---------- feedback bar ----------
function buildFeedbackBar(messageId = null) {
  const bar = document.createElement('div');
  bar.className = 'msg-feedback';

  const upBtn = document.createElement('button');
  upBtn.className = 'fb-btn fb-up';
  upBtn.type = 'button';
  upBtn.title = 'Helpful';
  upBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>';

  const downBtn = document.createElement('button');
  downBtn.className = 'fb-btn fb-down';
  downBtn.type = 'button';
  downBtn.title = 'Not helpful';
  downBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>';

  function applyRating(rating) {
    upBtn.classList.toggle('active', rating === 1);
    downBtn.classList.toggle('active', rating === -1);
  }

  async function postFeedback(rating) {
    // message_id stored on the wrapping .msg element
    const wrap = bar.closest('.msg');
    const mid = wrap?.dataset?.messageId;
    if (!mid) { flashToast('message id not available yet'); return; }
    try {
      const r = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message_id: mid, rating }),
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      applyRating(rating);
      flashToast(rating === 1 ? 'Marked helpful' : rating === -1 ? 'Marked unhelpful' : 'Feedback cleared');
    } catch (e) { flashToast('feedback error: ' + e.message); }
  }

  upBtn.addEventListener('click', () => {
    const isActive = upBtn.classList.contains('active');
    postFeedback(isActive ? 0 : 1);
  });
  downBtn.addEventListener('click', () => {
    const isActive = downBtn.classList.contains('active');
    postFeedback(isActive ? 0 : -1);
  });

  bar.appendChild(upBtn);
  bar.appendChild(downBtn);
  bar._applyRating = applyRating;
  return bar;
}

function appendStreaming() {
  removeEmptyState();
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';
  const head = document.createElement('div');
  head.className = 'msg-header';
  head.innerHTML = `<span class="role">HRAG</span>`;

  const phase = document.createElement('div');
  phase.className = 'phase phase-retrieve';
  phase.innerHTML = `<span class="pdot"></span><span class="ptext">thinking…</span>`;

  // Reasoning trace panel — collapsible, lives between phase and bubble
  const trace = document.createElement('details');
  trace.className = 'trace';
  trace.open = true;
  trace.innerHTML = `
    <summary>
      <span class="trace-arrow">›</span>
      <span class="trace-title">Reasoning</span>
      <span class="trace-meta" id="trace-meta"></span>
    </summary>
    <div class="trace-body"></div>
  `;
  trace._t0 = performance.now();
  trace._steps = [];
  trace._descend = null;
  trace._intent = null;
  trace._rerankN = 0;
  trace._rerankTotal = 0;
  // Phase 6 state
  trace._adaptive = null;       // { intent, top_k_vector, top_k_final }
  trace._skipped = null;        // { reason }
  trace._episodicBias = null;   // { episodic_count, total }
  // Phase 6-B state
  trace._retrieverPick = null;  // { intent, retriever, global }
  // Phase 7-A state
  trace._mathMeta = null;          // { query, where }
  trace._mathMetaFallback = null;  // { reason }
  trace._formulaExtract = null;    // { duration_s, chars }

  const bubble = document.createElement('div');
  bubble.className = 'bubble cursor';
  bubble._rawText = '';

  const fbBar = buildFeedbackBar(null);

  wrap.appendChild(head);
  wrap.appendChild(phase);
  wrap.appendChild(trace);
  wrap.appendChild(bubble);
  wrap.appendChild(fbBar);
  messagesEl.appendChild(wrap);
  return { wrap, head, phase, trace, bubble, fbBar };
}

// ---------- reasoning trace renderer ----------
// dtOverrideMs: when the payload carries the authoritative duration of the
// step (orchestrator's `duration_s`), pass it here so we don't infer from
// inter-event gaps (which mistakes start-events for completed steps).
function pushTraceStep(trace, label, detail, dtOverrideMs = null) {
  const t = performance.now();
  const inferred = trace._steps.length
    ? (t - trace._steps[trace._steps.length - 1].t)
    : (t - trace._t0);
  const dt = (dtOverrideMs != null) ? dtOverrideMs : inferred;
  trace._steps.push({ label, detail, t, dt });
  renderTrace(trace);
}

function renderTrace(trace) {
  const body = trace.querySelector('.trace-body');
  if (!body) return;
  let html = '';

  // Intent chip
  if (trace._intent) {
    const it = trace._intent;
    html += `<div class="trace-intent intent-${it.intent || 'unclear'}">
      <span class="ti-label">${escapeHTML(it.intent || 'unclear')}</span>
      <span class="ti-detail">conf ${(it.confidence ?? 0).toFixed(2)} · ${escapeHTML(it.source || '')}</span>
    </div>`;
  }

  // Phase 6 — adaptive top_k chip (sits right after intent)
  if (trace._adaptive) {
    const a = trace._adaptive;
    const skipped = (a.top_k_vector == null && a.top_k_final == null);
    const ks = skipped
      ? '(null, null)'
      : `(${a.top_k_vector ?? '·'}, ${a.top_k_final ?? '·'})`;
    html += `<div class="trace-adaptive${skipped ? ' skipped' : ''}">
      <span class="ta-label">adaptive</span>
      <span class="ta-intent">${escapeHTML(a.intent || '')}</span>
      <span class="ta-ks">${escapeHTML(ks)}</span>
    </div>`;
  }

  // Phase 6-B — adaptive-retriever-picked chip (right after adaptive top_k)
  if (trace._retrieverPick) {
    const rp = trace._retrieverPick;
    html += `<div class="trace-retriever-pick">
      <span class="tr-label">Retriever override</span>
      <span class="tr-intent">${escapeHTML(rp.intent || '')}</span>
      <span class="tr-arrow">→</span>
      <span class="tr-retriever">${escapeHTML(rp.retriever || '')}</span>
      <span class="tr-global">(global: ${escapeHTML(rp.global || '')})</span>
    </div>`;
  }

  // Phase 6 — retrieval-skipped chip (right after adaptive)
  if (trace._skipped) {
    html += `<div class="trace-skipped">
      <span class="ts-label">Retrieval skipped</span>
      <span class="ts-reason">${escapeHTML(trace._skipped.reason || '')}</span>
    </div>`;
  }

  // Phase 7-A — math-meta filter chip (sits near retrieval, before retrieve_done)
  if (trace._mathMeta) {
    let whereText = '';
    try { whereText = JSON.stringify(trace._mathMeta.where || {}); }
    catch { whereText = String(trace._mathMeta.where || ''); }
    html += `<div class="trace-math-meta">
      <span class="tm-label">Math-meta filter ON</span>
      <span class="tm-where">where=${escapeHTML(whereText)}</span>
    </div>`;
  }

  // Phase 7-A — math-meta fallback chip (when filtered retrieval was empty)
  if (trace._mathMetaFallback) {
    html += `<div class="trace-math-fallback">
      <span class="ts-label">Math-meta filter empty → retried unfiltered</span>
      <span class="ts-reason">${escapeHTML(trace._mathMetaFallback.reason || '')}</span>
    </div>`;
  }

  // Timeline
  if (trace._steps.length) {
    html += `<ol class="trace-steps">`;
    for (const s of trace._steps) {
      const ms = s.dt < 1000 ? `${Math.round(s.dt)}ms` : `${(s.dt / 1000).toFixed(2)}s`;
      const slowCls = s.dt >= 1000 ? ' slow' : '';
      html += `<li class="ts${slowCls}">
        <span class="tdot"></span>
        <span class="tlabel">${escapeHTML(s.label)}</span>
        ${s.detail ? `<span class="tdetail">${escapeHTML(s.detail)}</span>` : ''}
        <span class="ttime">${ms}</span>
      </li>`;
    }
    html += `</ol>`;
  }

  // Rerank counter (compact)
  if (trace._rerankTotal > 0) {
    html += `<div class="trace-rerank">
      <span class="rr-label">rerank</span>
      <span class="rr-bar"><span class="rr-fill" style="width:${Math.min(100, (trace._rerankN / trace._rerankTotal) * 100).toFixed(1)}%"></span></span>
      <span class="rr-count">${trace._rerankN}/${trace._rerankTotal}</span>
    </div>`;
  }

  // Phase 6 — episodic bias line (sits after retrieve results)
  if (trace._episodicBias) {
    const b = trace._episodicBias;
    html += `<div class="trace-episodic">
      Lifted <span class="te-count">${b.episodic_count ?? 0}</span>
      of <span class="te-count">${b.total ?? 0}</span> episodic memories
    </div>`;
  }

  // Phase 7-A — formula extraction card (sits after answer generation)
  if (trace._formulaExtract) {
    const f = trace._formulaExtract;
    const chars = f.chars ?? 0;
    const dur = typeof f.duration_s === 'number' ? f.duration_s.toFixed(1) : '?';
    html += `<div class="trace-formula">
      <span class="tf-icon">∑</span>
      <span class="tf-label">Formula extraction</span>
      <span class="tf-sep">·</span>
      <span class="tf-chars">${chars} chars</span>
      <span class="tf-sep">·</span>
      <span class="tf-dur">${dur}s</span>
    </div>`;
  }

  // Taxonomy descend tree
  if (trace._descend && (trace._descend.trace?.length || trace._descend.leaves?.length)) {
    html += renderDescend(trace._descend);
  }

  body.innerHTML = html;

  // meta in summary
  const totalMs = performance.now() - trace._t0;
  const meta = trace.querySelector('#trace-meta');
  if (meta) meta.textContent = `${(totalMs / 1000).toFixed(1)}s · ${trace._steps.length} steps`;
}

function renderDescend(d) {
  let html = `<div class="descend">
    <div class="descend-head">
      <span class="dh-title">Taxonomy descent</span>
      <span class="dh-stats" title="${d.stats?.docs_opened ?? 0} documents (each containing many chunks) opened out of ${d.stats?.total_docs ?? 0} in the corpus">opened ${d.stats?.docs_opened ?? 0}/${d.stats?.total_docs ?? 0} docs · ${d.stats?.leaves_picked ?? 0} leaves</span>
    </div>`;
  if (d.note) {
    html += `<div class="descend-note">${escapeHTML(d.note)}</div>`;
  }
  // levels
  for (let i = 0; i < (d.trace || []).length; i++) {
    const level = d.trace[i];
    html += `<div class="d-level">
      <div class="d-level-label">L${i}</div>
      <div class="d-nodes">`;
    const nodes = [...(level.kept || []), ...(level.pruned || [])];
    nodes.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    for (const n of nodes) {
      const isKept = (level.kept || []).some(k => k.node?.node_id === n.node?.node_id);
      const label = n.node?.label || n.label || '(unnamed)';
      const score = (n.score ?? 0).toFixed(2);
      html += `<span class="d-node ${isKept ? 'kept' : 'pruned'}" title="${escapeHTML(label)}">
        <span class="dn-label">${escapeHTML(label)}</span>
        <span class="dn-score">${score}</span>
      </span>`;
    }
    html += `</div></div>`;
  }
  // leaves
  if ((d.leaves || []).length) {
    html += `<div class="d-leaves">
      <div class="d-level-label">→ leaves</div>
      <div class="d-leaf-list">`;
    for (const lf of d.leaves) {
      html += `<div class="d-leaf">
        <span class="dl-label">${escapeHTML(lf.label)}</span>
        <span class="dl-score">${lf.score.toFixed(2)}</span>
        <span class="dl-count">${lf.doc_count} docs</span>
      </div>`;
    }
    html += `</div></div>`;
  }
  html += `</div>`;
  return html;
}

// ---------- streaming ----------
function abortStream() {
  if (state.controller) {
    try { state.controller.abort(); } catch {}
  }
  state.streaming = false;
  setSendStop(false);
}

function setSendStop(isStop) {
  if (isStop) {
    sendBtn.classList.add('stop');
    sendBtn.disabled = false;
    sendBtn.title = 'Stop';
    sendBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
  } else {
    sendBtn.classList.remove('stop');
    sendBtn.title = 'Send';
    sendBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M5 12l14-7-4 14-3-5-7-2z"/></svg>';
    sendBtn.disabled = inputEl.value.trim().length === 0;
  }
}

function scrollToEnd(force) {
  // user-friendly anchoring: only auto-scroll if already near the bottom,
  // unless `force` is set.
  const threshold = 120;
  const dist = scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight;
  if (force || dist < threshold) {
    scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: force ? 'auto' : 'smooth' });
  }
}

async function submit() {
  if (state.streaming) { abortStream(); return; }
  const msg = inputEl.value.trim();
  if (!msg) return;

  appendMessage('user', msg);
  inputEl.value = '';
  autoResize();
  scrollToEnd(true);

  // Phase 8: a new turn starts here — drop any stale review state.
  state.currentTurnId = null;
  state.pendingReview = null;

  const streamEls = appendStreaming();
  // Phase 8: the inline review card needs a handle to the active assistant
  // wrap so it can insert itself ABOVE the streaming bubble.
  state._streamEls = streamEls;
  state.streaming = true;
  setSendStop(true);

  const ctrl = new AbortController();
  state.controller = ctrl;

  let buffer = '';
  let finalSeen = false;

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({ message: msg, session_id: state.sessionId }),
      signal: ctrl.signal,
    });
    if (!r.ok || !r.body) throw new Error('HTTP ' + r.status);

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        handleSSE(chunk, streamEls);
        if (finalSeen) break;
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      streamEls.bubble.classList.remove('cursor');
      streamEls.phase.classList.remove('phase-retrieve', 'phase-rerank', 'phase-write');
      streamEls.phase.classList.add('phase-done');
      streamEls.phase.querySelector('.ptext').textContent = 'stopped';
    } else {
      streamEls.bubble.classList.remove('cursor');
      streamEls.bubble.innerHTML = `<div style="color:var(--bad)">Error: ${escapeHTML(e.message)}</div>`;
      streamEls.phase.remove();
    }
  } finally {
    state.streaming = false;
    setSendStop(false);
    state.controller = null;
    state._streamEls = null;
    // refresh sidebar to capture the new session
    loadSessions();
  }
}

function handleSSE(chunk, els) {
  // chunk = "event: foo\ndata: line1\ndata: line2"
  const lines = chunk.split('\n');
  let event = 'message';
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
  }
  if (dataLines.length === 0) return;
  const raw = dataLines.join('\n');
  let data;
  try { data = JSON.parse(raw); } catch { data = raw; }

  switch (event) {
    case 'open':
      setPhase(els, 'retrieve', 'retrieving…');
      break;

    case 'token': {
      const piece = typeof data === 'string' ? data : (data.token || data.content || '');
      if (!els.bubble._rawText) {
        // first token — flip phase to "write" and prep the bubble for
        // incremental markdown rendering (ChatGPT-style). The trailing
        // <span class="md-cursor"> is appended on every streaming render.
        setPhase(els, 'write', 'writing…');
        els.bubble.classList.remove('streaming-plain', 'cursor');
        els.bubble.innerHTML = '';
        els.bubble._rawText = '';
        maybeRTL(els.bubble, piece);
      }
      els.bubble._rawText += piece;
      scheduleStreamingMdRender(els);
      // Auto-scroll at most once per frame to keep the latest text in view.
      if (!els.bubble._scrollPending) {
        els.bubble._scrollPending = true;
        requestAnimationFrame(() => {
          els.bubble._scrollPending = false;
          scrollToEnd(false);
        });
      }
      break;
    }

    case 'progress': {
      // Generic event hub — fan out to phase chip + reasoning trace.
      const ev = data?.event || '';
      const payload = data?.payload || {};
      handleProgress(els, ev, payload);
      break;
    }

    case 'final': {
      const final = data;
      els.bubble._rawText = final.answer || els.bubble._rawText;
      // Cancel any pending streaming render so it can't clobber the final one.
      if (els.bubble._mdRenderTimer) {
        clearTimeout(els.bubble._mdRenderTimer);
        els.bubble._mdRenderTimer = null;
      }
      els.bubble._mdRenderPending = false;
      // Swap the incremental streaming markdown for the full final render
      // (with syntax highlighting + copy buttons) in a single repaint.
      els.bubble.classList.remove('streaming-plain', 'cursor');
      els.bubble._textNode = null;
      els.bubble.innerHTML = renderMD(els.bubble._rawText, { withHighlight: true });
      attachCopyButtons(els.bubble);
      maybeRTL(els.bubble, els.bubble._rawText);
      setPhase(els, 'done', 'done');
      // Store the message_id on the wrap so the feedback bar can POST it.
      if (final.message_id != null) {
        els.wrap.dataset.messageId = String(final.message_id);
      }
      // sources
      if (final.sources && final.sources.length) {
        els.wrap.appendChild(renderSources(final.sources));
      }
      if (final.session_id) state.sessionId = final.session_id;
      scrollToEnd(false);
      break;
    }

    case 'error':
      if (els.bubble._mdRenderTimer) {
        clearTimeout(els.bubble._mdRenderTimer);
        els.bubble._mdRenderTimer = null;
      }
      els.bubble._mdRenderPending = false;
      els.bubble.classList.remove('cursor');
      els.bubble.innerHTML = `<div style="color:var(--bad)">Error: ${escapeHTML(data?.message || 'unknown error')}</div>`;
      setPhase(els, 'done', 'error');
      break;

    case 'done': {
      if (els.bubble._mdRenderTimer) {
        clearTimeout(els.bubble._mdRenderTimer);
        els.bubble._mdRenderTimer = null;
      }
      els.bubble._mdRenderPending = false;
      els.bubble.classList.remove('cursor');
      // Strip any lingering streaming caret if `final` didn't fire.
      const lingeringCursor = els.bubble.querySelector('.md-cursor');
      if (lingeringCursor) lingeringCursor.remove();
      break;
    }

    // ---- Phase 8: human-in-the-loop review ----
    case 'review_required':
      handleReviewRequired(typeof data === 'object' ? data : {});
      break;
    case 'review_resolved': {
      // The modal has already closed (we close it on submit). This is a
      // confirmation echo from the orchestrator; surface a toast on timeout.
      const p = (typeof data === 'object' && data) || {};
      if (p.timed_out) flashToast('Review timed out — continuing with defaults');
      break;
    }
    case 'followups': {
      const p = (typeof data === 'object' && data) || {};
      if (p && Array.isArray(p.chips) && p.chips.length) {
        renderFollowups(els, p.chips);
      }
      break;
    }
  }
}

function handleProgress(els, ev, payload) {
  const trace = els.trace;
  switch (ev) {
    case 'start':
      // Phase 8: capture turn_id so the review modal can POST /resume.
      if (payload && payload.turn_id) state.currentTurnId = payload.turn_id;
      pushTraceStep(trace, 'start', '');
      setPhase(els, 'retrieve', 'thinking…');
      break;
    case 'review_warning':
      // Non-fatal warnings emitted by the interaction module (e.g. redescend
      // not supported by the active retriever). Surface as a quick toast.
      flashToast((payload && payload.message) || 'Review warning');
      pushTraceStep(trace, 'review-warn', (payload && payload.message) || '');
      break;
    case 'intent_check':
      trace._intent = payload;
      pushTraceStep(trace, 'intent', `${payload.intent} · ${(payload.confidence ?? 0).toFixed(2)}`);
      renderTrace(trace);
      break;
    case 'intent_route':
      pushTraceStep(trace, 'route', payload.intent || payload.route || '');
      break;
    case 'gate_check':
      pushTraceStep(trace, 'gate', payload.decision || '');
      setPhase(els, 'retrieve', `gate ${payload.decision || ''}`);
      break;
    case 'clue_generate':
      pushTraceStep(trace, 'clue', payload.clue ? `"${(payload.clue || '').slice(0, 60)}…"` : '');
      setPhase(els, 'retrieve', 'clue…');
      break;
    case 'dialog_compact':
      pushTraceStep(trace, 'compact dialog', `${payload.clusters || ''} clusters`);
      setPhase(els, 'retrieve', 'compact…');
      break;
    case 'router_classify':
      pushTraceStep(trace, 'router', payload.label || '');
      break;
    case 'taxonomy_descend':
      trace._descend = payload;
      pushTraceStep(trace, 'taxonomy', `opened ${payload?.stats?.docs_opened ?? 0}/${payload?.stats?.total_docs ?? 0} docs · ${payload?.stats?.leaves_picked ?? 0} leaves`);
      setPhase(els, 'retrieve', 'taxonomy descent');
      break;
    case 'retrieve': {
      const dur = payload.duration_s != null ? payload.duration_s * 1000 : null;
      pushTraceStep(trace, 'retrieve', `${payload.n_results ?? '?'} chunks`, dur);
      setPhase(els, 'rerank', 'retrieved');
      break;
    }
    case 'rerank_step':
      trace._rerankN = (payload.i ?? 0) + 1;
      trace._rerankTotal = payload.n ?? trace._rerankTotal;
      // throttle rerank-step renders to avoid flooding
      if (!trace._rerankPending) {
        trace._rerankPending = true;
        requestAnimationFrame(() => { trace._rerankPending = false; renderTrace(trace); });
      }
      setPhase(els, 'rerank', `rerank ${trace._rerankN}/${trace._rerankTotal}`);
      break;
    case 'organize_done':
      pushTraceStep(trace, 'organize', `${payload.kept ?? ''} kept`);
      break;
    case 'personal_memories':
      pushTraceStep(trace, 'memories', `${(payload.memories || []).length} hit`);
      break;
    case 'generate_start':
      // Don't push a step yet — the matching `generate` event fires when
      // streaming completes and carries the authoritative duration. We
      // remember the start time to compute that duration if the orchestrator
      // didn't include `duration_s` in the payload for some reason.
      trace._genStartT = performance.now();
      setPhase(els, 'write', 'writing…');
      break;
    case 'generate': {
      const dur = (payload.duration_s != null)
        ? payload.duration_s * 1000
        : (trace._genStartT != null ? performance.now() - trace._genStartT : null);
      const chars = payload.answer_chars ?? null;
      const detail = chars != null ? `${chars} chars` : '';
      pushTraceStep(trace, 'generate', detail, dur);
      break;
    }
    case 'uncertain_render':
      pushTraceStep(trace, 'uncertain', `${payload.count ?? 0} marker(s)`);
      break;
    case 'adaptive_top_k':
      trace._adaptive = payload;
      pushTraceStep(
        trace,
        'adaptive',
        `${payload.intent || ''} · vec=${payload.top_k_vector ?? 'null'} final=${payload.top_k_final ?? 'null'}`,
      );
      break;
    case 'adaptive_retriever_picked':
      trace._retrieverPick = payload;
      pushTraceStep(
        trace,
        'retriever override',
        `${payload.intent || ''} → ${payload.retriever || ''} (global: ${payload.global || ''})`,
      );
      break;
    case 'retrieval_skipped':
      trace._skipped = payload;
      pushTraceStep(trace, 'skipped', `retrieval (${payload.reason || ''})`);
      setPhase(els, 'write', 'no retrieval');
      break;
    case 'episodic_bias_applied':
      trace._episodicBias = payload;
      pushTraceStep(
        trace,
        'ep. bias',
        `${payload.episodic_count ?? 0}/${payload.total ?? 0} lifted`,
      );
      break;
    case 'math_meta_filter': {
      trace._mathMeta = payload;
      let whereText = '';
      try { whereText = JSON.stringify(payload.where || {}); }
      catch { whereText = String(payload.where || ''); }
      pushTraceStep(trace, 'math-meta', `filter ${whereText}`);
      setPhase(els, 'retrieve', 'math-meta filter');
      break;
    }
    case 'math_meta_filter_fallback':
      trace._mathMetaFallback = payload;
      pushTraceStep(trace, 'math-meta fallback', payload.reason || 'empty result');
      break;
    case 'formula_extract': {
      trace._formulaExtract = payload;
      const chars = payload.chars ?? 0;
      const dur = (payload.duration_s != null) ? payload.duration_s * 1000 : null;
      pushTraceStep(trace, 'formula extract', `${chars} chars`, dur);
      setPhase(els, 'write', 'formula extraction');
      break;
    }
    case 'done': {
      // The "done" payload carries total_s — show that as the trace meta
      // (already updated by renderTrace via _t0) and as the step detail.
      // Use dt-since-last for the actual gap so the timeline stays honest.
      pushTraceStep(trace, 'done', `total ${(payload.total_s ?? 0).toFixed(2)}s`);
      trace.open = false;  // auto-collapse on completion (still expandable)
      break;
    }
    default:
      // Unknown event — log a generic step so it shows up
      pushTraceStep(trace, ev, '');
  }
}

function setPhase(els, name, text) {
  if (!els.phase) return;
  els.phase.classList.remove('phase-retrieve', 'phase-rerank', 'phase-write', 'phase-done');
  els.phase.classList.add('phase-' + name);
  const t = els.phase.querySelector('.ptext');
  if (t) t.textContent = text;
}

// ---------- Phase 8: inline review card ----------
// The orchestrator may pause between retrieval and generation when one of the
// configured signals trips (low score floor, ambiguous top-2, cross-branch
// hits, intent unclear, …). It emits `review_required` with a full payload and
// blocks on the resume queue until the frontend POSTs back to
// /api/chat/turns/{turn_id}/resume.
//
// We render the review UI as an INLINE CARD inside the active assistant
// message wrap (above the streaming bubble), not as a centred modal. Once the
// user resolves it, the card collapses to a one-line "Reviewed: …" summary
// that stays as a thin separator above the answer.

const REVIEW_REASON_INFO = {
  score_floor:          { label: 'Low confidence',   tone: 'warn', hint: 'No strong matches in your docs' },
  ambiguity_delta:      { label: 'Ambiguous',        tone: 'warn', hint: 'Top candidates are tied' },
  branch_threshold:     { label: 'Cross-domain',     tone: 'info', hint: 'Hits span multiple branches' },
  intent_unclear:       { label: 'Intent unclear',   tone: 'warn', hint: "I'm not sure what you're asking" },
  router_ambiguous:     { label: 'Router uncertain', tone: 'info' },
  factual_general_swap: { label: 'Maybe off-corpus', tone: 'warn', hint: 'I might do better with general knowledge' },
  always_mode:          { label: 'Always-pause',     tone: 'info', hint: 'Step-through mode is on' },
};

function handleReviewRequired(payload) {
  state.pendingReview = payload || {};
  state.reviewCardOpen = true;
  if (payload && payload.turn_id && !state.currentTurnId) {
    state.currentTurnId = payload.turn_id;
  }

  // Find the active assistant message wrap — the streaming bubble belongs to it.
  // submit() exposes a `_streamEls` handle for the current chat turn.
  const wrap = state._streamEls && state._streamEls.wrap;
  if (!wrap) {
    console.warn('no _streamEls.wrap; cannot render inline review card');
    return;
  }

  const card = renderReviewCard(payload || {});
  // Place the card ABOVE the streaming bubble so when the answer streams,
  // the eventual collapsed summary stays above the answer text.
  const bubble = wrap.querySelector('.bubble');
  if (bubble && bubble.parentNode === wrap) {
    wrap.insertBefore(card, bubble);
  } else {
    wrap.appendChild(card);
  }
  scrollToEnd(false);

  // Focus the Continue button so Enter is the default decision
  setTimeout(() => {
    const btn = card.querySelector('.review-card-continue-btn');
    if (btn) btn.focus();
  }, 30);
}

function renderReviewCard(payload) {
  const card = document.createElement('div');
  card.className = 'review-card';
  card.dataset.turnId = payload.turn_id || '';

  // ----- Head row: title + reason chips on one line -----
  const head = document.createElement('div');
  head.className = 'review-card-head';
  const title = document.createElement('span');
  title.className = 'review-card-title';
  title.textContent = 'Quick check before I answer';
  head.appendChild(title);
  const reasonsBox = document.createElement('span');
  reasonsBox.className = 'review-card-reasons';
  (payload.reasons || []).forEach(r => {
    const info = REVIEW_REASON_INFO[r] || { label: r };
    const chip = document.createElement('span');
    chip.className = 'review-reason-chip';
    if (info.tone) chip.dataset.tone = info.tone;
    chip.textContent = info.label;
    if (info.hint) chip.title = info.hint;
    reasonsBox.appendChild(chip);
  });
  head.appendChild(reasonsBox);
  card.appendChild(head);

  // ----- Body -----
  const body = document.createElement('div');
  body.className = 'review-card-body';

  // Clue (editable single-line input)
  const clueRow = document.createElement('div');
  clueRow.className = 'review-card-clue-row';
  const clueInput = document.createElement('input');
  clueInput.type = 'text';
  clueInput.className = 'review-card-clue';
  clueInput.value = payload.clue || payload.original_question || '';
  clueInput.placeholder = payload.original_question || 'Edit if I misread your question.';
  clueInput.title = 'Edit if I misread your question. Press Enter to use this phrasing.';
  // Bilingual: let the browser BiDi-resolve direction from the clue text.
  clueInput.setAttribute('dir', 'auto');
  clueRow.appendChild(clueInput);
  body.appendChild(clueRow);

  // Original-question fallback line (muted, only if it differs from the clue)
  const orig = (payload.original_question || '').trim();
  const clue = (payload.clue || '').trim();
  if (orig && clue && orig !== clue) {
    const origLine = document.createElement('div');
    origLine.className = 'review-card-hint';
    origLine.style.opacity = '0.85';
    origLine.textContent = `Your question: "${orig}"`;
    body.appendChild(origLine);
  }

  // Rephrasings (inline chips)
  const rephrasings = (payload.rephrasings || []).filter(Boolean);
  if (rephrasings.length) {
    const r = document.createElement('div');
    r.className = 'review-card-rephrasings';
    rephrasings.forEach(p => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'review-rephrasing-chip';
      btn.textContent = p;
      btn.addEventListener('click', () => { clueInput.value = p; clueInput.focus(); });
      r.appendChild(btn);
    });
    body.appendChild(r);
  }

  // Taxonomy breadcrumbs (inline, click to redirect)
  const descend = payload.taxonomy_descend;
  if (descend && Array.isArray(descend.leaves) && descend.leaves.length) {
    const bc = document.createElement('div');
    bc.className = 'review-card-breadcrumb';
    const label = document.createElement('span');
    label.className = 'review-card-hint';
    label.style.opacity = '0.85';
    label.textContent = 'I walked:';
    bc.appendChild(label);
    descend.leaves.forEach((leaf, i) => {
      if (i > 0) {
        const sep = document.createElement('span');
        sep.className = 'review-breadcrumb-sep';
        sep.textContent = '·';
        bc.appendChild(sep);
      }
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'review-breadcrumb-node';
      const dc = (leaf.doc_count != null) ? ` (${leaf.doc_count})` : '';
      btn.textContent = `${leaf.label || leaf.node_id || '?'}${dc}`;
      btn.dataset.nodeId = leaf.node_id || '';
      btn.title = 'Click to redirect retrieval here';
      btn.addEventListener('click', () =>
        submitReviewDecision(card, { action: 'redescend', redirect_taxonomy_node_id: leaf.node_id }));
      bc.appendChild(btn);
    });
    body.appendChild(bc);
  }

  // Sources list (compact rows, max-height scroll)
  const sources = payload.sources || [];
  if (sources.length) {
    const srcBox = document.createElement('div');
    srcBox.className = 'review-card-sources';
    sources.forEach((s, i) => {
      const row = document.createElement('div');
      row.className = 'review-card-source';
      row.dataset.chunkId = s.chunk_id || '';
      row.dataset.docId = s.doc_id || '';
      row.dataset.index = String(i);

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'review-card-source-cb';
      cb.checked = true;

      const idx = document.createElement('span');
      idx.className = 'review-card-source-idx';
      idx.textContent = `[${i + 1}]`;

      const titleSpan = document.createElement('span');
      titleSpan.className = 'review-card-source-title';
      titleSpan.textContent = `${s.title || 'Untitled'}${s.section ? ' · ' + s.section : ''}`;
      titleSpan.setAttribute('dir', 'auto');

      const score = document.createElement('span');
      score.className = 'review-card-source-score';
      const rs = (typeof s.rerank_score === 'number') ? s.rerank_score.toFixed(2)
              : (typeof s.score === 'number' ? s.score.toFixed(3) : '—');
      score.textContent = rs;
      score.style.fontSize = '11px';
      score.style.color = 'var(--muted)';
      score.style.fontVariantNumeric = 'tabular-nums';

      const snip = document.createElement('div');
      snip.className = 'review-card-source-snippet';
      const snippetText = (s.snippet || (s.text || '').slice(0, 240)) || '';
      snip.textContent = snippetText.length > 120 ? snippetText.slice(0, 120) + '…' : snippetText;
      snip.dataset.full = snippetText;
      snip.setAttribute('dir', 'auto');
      snip.title = 'Click to expand';
      snip.addEventListener('click', () => {
        const expanded = snip.classList.toggle('expanded');
        snip.textContent = expanded ? (snip.dataset.full || '') :
          (snip.dataset.full.length > 120 ? snip.dataset.full.slice(0, 120) + '…' : snip.dataset.full);
      });

      const actions = document.createElement('span');
      actions.className = 'review-card-source-actions';
      const whyBtn = document.createElement('button');
      whyBtn.type = 'button'; whyBtn.textContent = 'why?';
      whyBtn.title = 'Explain why this source was selected';
      whyBtn.addEventListener('click', () => fetchWhySource(row, s, payload));
      const moreBtn = document.createElement('button');
      moreBtn.type = 'button'; moreBtn.textContent = '↗ more';
      moreBtn.title = 'Re-retrieve scoped to this document';
      moreBtn.addEventListener('click', () =>
        submitReviewDecision(card, { action: 'expand_doc', expand_from_doc_id: s.doc_id }));
      actions.appendChild(whyBtn);
      actions.appendChild(moreBtn);

      row.appendChild(cb);
      row.appendChild(idx);
      row.appendChild(titleSpan);
      row.appendChild(score);
      row.appendChild(snip);

      // Action buttons go on their own sub-row (spanning full grid width, right-aligned).
      const actRow = document.createElement('div');
      actRow.style.gridColumn = '1 / -1';
      actRow.style.display = 'flex';
      actRow.style.justifyContent = 'flex-end';
      actRow.appendChild(actions);
      row.appendChild(actRow);

      srcBox.appendChild(row);
    });
    body.appendChild(srcBox);
  }

  card.appendChild(body);

  // ----- Footer: toggles + actions -----
  const foot = document.createElement('div');
  foot.className = 'review-card-foot';
  const toggles = document.createElement('div');
  toggles.className = 'review-card-foot-toggles';

  const epiLabel = document.createElement('label');
  epiLabel.className = 'review-toggle';
  const epiCb = document.createElement('input');
  epiCb.type = 'checkbox'; epiCb.className = 'review-card-episodic';
  const epiText = document.createElement('span'); epiText.textContent = 'Include memories';
  epiLabel.appendChild(epiCb); epiLabel.appendChild(epiText);
  toggles.appendChild(epiLabel);

  const memLabel = document.createElement('label');
  memLabel.className = 'review-toggle';
  const memCb = document.createElement('input');
  memCb.type = 'checkbox'; memCb.className = 'review-card-remember';
  const memText = document.createElement('span'); memText.textContent = 'Remember choice';
  memLabel.appendChild(memCb); memLabel.appendChild(memText);
  toggles.appendChild(memLabel);
  foot.appendChild(toggles);

  const actions = document.createElement('div');
  actions.className = 'review-card-foot-actions';
  const mkBtn = (cls, label, action) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = cls; b.textContent = label;
    b.addEventListener('click', () => submitReviewDecision(card, { action }));
    return b;
  };
  actions.appendChild(mkBtn('ghost review-card-general-btn', 'General knowledge', 'general'));
  actions.appendChild(mkBtn('ghost review-card-clarify-btn', 'Clarify', 'clarify'));
  actions.appendChild(mkBtn('ghost review-card-abort-btn', 'Cancel', 'abort'));
  actions.appendChild(mkBtn('primary review-card-continue-btn', 'Continue ↵', 'continue'));
  foot.appendChild(actions);
  card.appendChild(foot);

  // Hint line
  const hint = document.createElement('div');
  hint.className = 'review-card-hint';
  hint.textContent = 'Enter accept · Esc cancel · 1-9 toggle source · E edit clue · G general';
  card.appendChild(hint);

  // Clue input: Enter submits, Esc cancels.
  clueInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitReviewDecision(card, { action: 'continue' });
    } else if (e.key === 'Escape') {
      e.preventDefault();
      submitReviewDecision(card, { action: 'abort' });
    }
  });

  return card;
}

async function submitReviewDecision(card, partial) {
  if (!card || card.dataset.resolved === '1') return;
  card.dataset.resolved = '1';
  const turnId = card.dataset.turnId
    || state.currentTurnId
    || (state.pendingReview && state.pendingReview.turn_id);
  if (!turnId) {
    console.warn('submitReviewDecision: no turn_id');
    card.remove();
    state.reviewCardOpen = false;
    state.pendingReview = null;
    return;
  }

  const epiEl = card.querySelector('.review-card-episodic');
  const memEl = card.querySelector('.review-card-remember');
  const clueEl = card.querySelector('.review-card-clue');

  const body = Object.assign({
    action: 'continue',
    selected_chunk_ids: [],
    rewritten_query: null,
    expand_from_doc_id: null,
    redirect_taxonomy_node_id: null,
    include_episodic: !!(epiEl && epiEl.checked),
    remember_choice: !!(memEl && memEl.checked),
  }, partial || {});

  // If the user edited the clue and clicked Continue, promote to rephrase.
  if (body.action === 'continue' && state.pendingReview) {
    const editedClue = ((clueEl && clueEl.value) || '').trim();
    const origClue = (state.pendingReview.clue || '').trim();
    const origQ = (state.pendingReview.original_question || '').trim();
    if (editedClue && editedClue !== origClue && editedClue !== origQ) {
      body.action = 'rephrase';
      body.rewritten_query = editedClue;
    }
  }

  // If the user kept Continue but unchecked some sources, switch to filter.
  if (body.action === 'continue') {
    const rows = Array.from(card.querySelectorAll('.review-card-source'));
    const checked = rows
      .filter(r => r.querySelector('.review-card-source-cb').checked)
      .map(r => r.dataset.chunkId)
      .filter(Boolean);
    const total = rows.map(r => r.dataset.chunkId).filter(Boolean);
    if (checked.length === 0 && total.length > 0) {
      body.action = 'general';
    } else if (checked.length < total.length) {
      body.action = 'filter';
      body.selected_chunk_ids = checked;
    }
  }

  // Collapse the card to a one-line summary above the streaming bubble.
  const summary = document.createElement('div');
  summary.className = 'review-card-resolved';
  summary.textContent = describeDecision(body, state.pendingReview);
  if (card.parentNode) {
    card.parentNode.replaceChild(summary, card);
  }
  state.reviewCardOpen = false;
  state.pendingReview = null;

  try {
    const r = await fetch(`/api/chat/turns/${encodeURIComponent(turnId)}/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      console.warn('resume failed', r.status);
      flashToast('Resume failed — please try again');
    }
  } catch (e) {
    console.error('resume error', e);
    flashToast('Network error during resume');
  }
}

function describeDecision(body, payload) {
  const action = body.action || 'continue';
  if (action === 'continue') return 'Reviewed: continuing with all sources';
  if (action === 'general')  return 'Reviewed: answering from general knowledge';
  if (action === 'filter')   return `Reviewed: filtered to ${body.selected_chunk_ids.length} source(s)`;
  if (action === 'rephrase') return `Reviewed: rephrased "${body.rewritten_query}"`;
  if (action === 'clarify')  return 'Reviewed: asking a clarifying question…';
  if (action === 'expand_doc')  return 'Reviewed: re-retrieving from one document';
  if (action === 'redescend')   return 'Reviewed: redirecting taxonomy descent';
  if (action === 'abort')    return 'Reviewed: cancelled';
  return `Reviewed: ${action}`;
}

async function fetchWhySource(row, src, payload) {
  let whyEl = row.querySelector('.review-card-why-text');
  if (whyEl) {
    whyEl.hidden = !whyEl.hidden;
    if (!whyEl.hidden && whyEl.dataset.fetched === '1') return;
  } else {
    whyEl = document.createElement('div');
    whyEl.className = 'review-card-why-text';
    whyEl.style.gridColumn = '1 / -1';
    row.appendChild(whyEl);
  }
  whyEl.hidden = false;
  whyEl.textContent = '…';
  try {
    const r = await fetch(`/api/chat/turns/${encodeURIComponent(payload.turn_id)}/why_source`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: payload.original_question || '',
        title: src.title || '',
        section: src.section || '',
        text: src.text || src.snippet || '',
      }),
    });
    const j = await r.json();
    whyEl.textContent = (j && j.explanation) || 'No explanation available.';
    whyEl.dataset.fetched = '1';
  } catch (e) {
    whyEl.textContent = 'Could not fetch explanation.';
  }
}

function renderFollowups(streamEls, chips) {
  if (!streamEls || !streamEls.wrap) return;
  let wrap = streamEls.wrap.querySelector('.followups');
  if (!wrap) {
    wrap = document.createElement('div');
    wrap.className = 'followups';
    streamEls.wrap.appendChild(wrap);
  }
  wrap.innerHTML = '';
  chips.slice(0, 3).forEach(text => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'followup-chip';
    btn.textContent = text;
    btn.addEventListener('click', () => {
      const inp = document.getElementById('input');
      inp.value = text;
      // Trigger the existing autoResize + send flow.
      inp.dispatchEvent(new Event('input', { bubbles: true }));
      document.getElementById('send').click();
    });
    wrap.appendChild(btn);
  });
}

function initReviewCard() {
  // The card itself is created dynamically, so we only wire the global
  // keyboard shortcuts. Per-card button handlers are attached in
  // renderReviewCard().
  document.addEventListener('keydown', (e) => {
    if (!state.reviewCardOpen) return;
    // Find the active (unresolved) inline review card.
    const card = document.querySelector('.review-card:not([data-resolved="1"])');
    if (!card) return;

    const tgt = e.target;
    const isTextField = tgt && (tgt.tagName === 'TEXTAREA' || tgt.tagName === 'INPUT');

    if (e.key === 'Enter' && !e.shiftKey && !isTextField) {
      e.preventDefault();
      submitReviewDecision(card, { action: 'continue' });
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      submitReviewDecision(card, { action: 'abort' });
      return;
    }
    if (!isTextField && /^[1-9]$/.test(e.key)) {
      const idx = parseInt(e.key, 10) - 1;
      const row = card.querySelectorAll('.review-card-source')[idx];
      if (row) {
        const cb = row.querySelector('.review-card-source-cb');
        if (cb) cb.checked = !cb.checked;
        e.preventDefault();
      }
      return;
    }
    if (!isTextField && (e.key === 'g' || e.key === 'G')) {
      e.preventDefault();
      submitReviewDecision(card, { action: 'general' });
      return;
    }
    if (!isTextField && (e.key === 'c' || e.key === 'C')) {
      e.preventDefault();
      submitReviewDecision(card, { action: 'clarify' });
      return;
    }
    if (!isTextField && (e.key === 'e' || e.key === 'E')) {
      e.preventDefault();
      const ce = card.querySelector('.review-card-clue');
      if (ce) ce.focus();
      return;
    }
    if (!isTextField && (e.key === 'r' || e.key === 'R')) {
      // Cycle focus through rephrasing chips.
      e.preventDefault();
      const chips = Array.from(card.querySelectorAll('.review-rephrasing-chip'));
      if (chips.length) {
        const idx = chips.findIndex(c => c === document.activeElement);
        const next = chips[(idx + 1) % chips.length];
        next.focus();
      }
    }
  });
}

// ---------- memories ----------
let _memoriesCache = [];

async function loadMemories() {
  memListEl.innerHTML = '<div class="loading-row"><span class="shimmer"></span></div>'.repeat(3);
  try {
    const r = await fetch('/api/memories?limit=200');
    const items = await r.json();
    _memoriesCache = items;
    paintMemories(items);
    memCountEl.textContent = String(items.length);
  } catch (e) {
    console.error('memories load failed', e);
    memListEl.innerHTML = '<div class="drawer-empty">Failed to load memories.</div>';
  }
}

function paintMemories(items) {
  if (!items.length) {
    memListEl.innerHTML = '<div class="drawer-empty"><div class="e-icon">📚</div>No memories yet. Try <code>/remember &lt;text&gt;</code> in the chat or click <b>+</b> above.</div>';
    return;
  }
  memListEl.innerHTML = '';
  for (const m of items) {
    memListEl.appendChild(buildMemoryCard(m));
  }
}

function buildMemoryCard(m) {
  const card = document.createElement('div');
  card.className = 'mem-card';
  card.dataset.memoryId = m.memory_id;

  const head = document.createElement('div');
  head.className = 'mem-head';

  const title = document.createElement('span');
  title.className = 'mem-title';
  title.textContent = m.title || `Memory ${m.memory_id?.slice(-8) ?? ''}`;

  const actions = document.createElement('span');
  actions.className = 'mem-actions';
  const editBtn = document.createElement('button');
  editBtn.className = 'mem-iconbtn';
  editBtn.title = 'Edit';
  editBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>';
  editBtn.addEventListener('click', () => startMemoryEdit(card, m));
  const delBtn = document.createElement('button');
  delBtn.className = 'mem-iconbtn mem-iconbtn-danger';
  delBtn.title = 'Delete';
  delBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14"/></svg>';
  delBtn.addEventListener('click', () => deleteMemory(m));
  actions.appendChild(editBtn);
  actions.appendChild(delBtn);

  const time = document.createElement('span');
  time.className = 'mem-time';
  time.textContent = (m.created_at || '').replace('T', ' ').slice(0, 16);

  head.appendChild(title);
  head.appendChild(time);
  head.appendChild(actions);

  const body = document.createElement('div');
  body.className = 'mem-body';
  body.textContent = m.text || '';
  maybeRTL(body, m.text || '');

  card.appendChild(head);
  card.appendChild(body);
  return card;
}

function startMemoryEdit(card, m) {
  // Replace the body with a textarea + save / cancel buttons.
  const body = card.querySelector('.mem-body');
  if (!body || card.classList.contains('editing')) return;
  card.classList.add('editing');
  const originalText = m.text || '';

  const ta = document.createElement('textarea');
  ta.className = 'mem-edit-textarea';
  ta.value = originalText;
  ta.rows = Math.min(12, Math.max(3, originalText.split('\n').length + 1));

  const actionsRow = document.createElement('div');
  actionsRow.className = 'mem-edit-actions';
  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'ghost-btn';
  cancelBtn.textContent = 'Cancel';
  const saveBtn = document.createElement('button');
  saveBtn.className = 'primary-btn';
  saveBtn.textContent = 'Save';
  actionsRow.appendChild(cancelBtn);
  actionsRow.appendChild(saveBtn);

  body.replaceWith(ta);
  card.appendChild(actionsRow);
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);

  cancelBtn.addEventListener('click', () => loadMemories());
  saveBtn.addEventListener('click', async () => {
    const newText = ta.value.trim();
    if (!newText) { flashToast('memory text required'); return; }
    if (newText === originalText) { loadMemories(); return; }
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';
    try {
      const r = await fetch('/api/memories/' + encodeURIComponent(m.memory_id), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newText, title: m.title || null }),
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      flashToast('memory saved');
      await loadMemories();
      refreshSidebarCounts();
    } catch (e) {
      flashToast('save failed: ' + e.message);
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
    }
  });
}

async function deleteMemory(m) {
  if (!confirm(`Delete memory "${m.title || m.memory_id?.slice(-8)}"?`)) return;
  try {
    const r = await fetch('/api/memories/' + encodeURIComponent(m.memory_id), { method: 'DELETE' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    flashToast('memory deleted');
    await loadMemories();
    refreshSidebarCounts();
  } catch (e) { flashToast('delete failed: ' + e.message); }
}

function startMemoryAdd() {
  // Insert a new card at the top in edit mode for a fresh memory.
  if (memListEl.querySelector('.mem-card.adding')) return;
  const card = document.createElement('div');
  card.className = 'mem-card adding';

  const head = document.createElement('div');
  head.className = 'mem-head';
  const title = document.createElement('span');
  title.className = 'mem-title';
  title.textContent = 'New memory';
  head.appendChild(title);
  card.appendChild(head);

  const ta = document.createElement('textarea');
  ta.className = 'mem-edit-textarea';
  ta.rows = 4;
  ta.placeholder = 'Type the memory text here…';
  card.appendChild(ta);

  const actionsRow = document.createElement('div');
  actionsRow.className = 'mem-edit-actions';
  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'ghost-btn';
  cancelBtn.textContent = 'Cancel';
  const saveBtn = document.createElement('button');
  saveBtn.className = 'primary-btn';
  saveBtn.textContent = 'Save';
  actionsRow.appendChild(cancelBtn);
  actionsRow.appendChild(saveBtn);
  card.appendChild(actionsRow);

  memListEl.insertBefore(card, memListEl.firstChild);
  ta.focus();

  cancelBtn.addEventListener('click', () => card.remove());
  saveBtn.addEventListener('click', async () => {
    const text = ta.value.trim();
    if (!text) { flashToast('memory text required'); return; }
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';
    try {
      // Background flow: returns immediately with a job_id, global job
      // bar shows the embed progress, list reloads via onJobFinished().
      const r = await fetch('/api/memories?background=true', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json().catch(() => ({}));
      const snippet = text.slice(0, 50) + (text.length > 50 ? '…' : '');
      if (data.background && data.job_id) {
        trackJob(data.job_id, 'memory_embed', `Embedding memory: "${snippet}"`);
        card.remove();
      } else {
        // server fell back to sync (e.g. older deploy); behave as before
        flashToast('memory added');
        await loadMemories();
        refreshSidebarCounts();
      }
    } catch (e) {
      flashToast('add failed: ' + e.message);
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
    }
  });
}

// One-click memory save from the chat composer. Takes whatever's in the
// input (or shows a small prompt if empty), POSTs to /api/memories, then
// clears the input. Triggered by the "Remember" button OR Alt+R.
async function rememberFromComposer() {
  let text = (inputEl.value || '').trim();
  if (!text) {
    text = (window.prompt('What should I remember?') || '').trim();
    if (!text) return;
  }
  const btn = document.getElementById('remember-btn');
  if (btn) {
    btn.classList.add('saving');
    btn.disabled = true;
  }
  try {
    const r = await fetch('/api/memories?background=true', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json().catch(() => ({}));
    // Clear only when the input was the source — preserve user-typed prompts.
    if ((inputEl.value || '').trim() === text) {
      inputEl.value = '';
      autoResize();
      // Re-disable the send button (textarea is now empty).
      sendBtn.disabled = true;
    }
    if (data.background && data.job_id) {
      const snippet = text.slice(0, 50) + (text.length > 50 ? '…' : '');
      trackJob(data.job_id, 'memory_embed', `Embedding memory: "${snippet}"`);
      flashToast('queued · embedding in background');
    } else {
      flashToast('remembered · ' + (data.memory_id ? data.memory_id.slice(0, 12) + '…' : 'saved'));
    }
    refreshSidebarCounts();
  } catch (e) {
    flashToast('remember failed: ' + (e.message || e));
  } finally {
    if (btn) {
      btn.classList.remove('saving');
      btn.disabled = false;
    }
  }
}

// Router for the Remember button: if input has text, save it directly;
// otherwise open the Smart Remember modal (requires an active session
// with ≥ 2 messages).
function onRememberClick() {
  const text = (inputEl.value || '').trim();
  if (text) {
    rememberFromComposer();
    return;
  }
  openSmartRememberModal();
}

// ---------- Smart Remember modal ----------
let _smartItems = [];     // [{text, category, confidence}]
let _smartLoading = false;
let _smartSaving = false;

function countDomMessages() {
  return document.querySelectorAll('.msg').length;
}

function openSmartRememberModal() {
  if (_smartLoading || _smartSaving) return;
  const modal = document.getElementById('rem-modal');
  const scrim = document.getElementById('rem-modal-scrim');
  const sub = document.getElementById('rem-modal-sub');
  const body = document.getElementById('rem-modal-body');
  const saveBtn = document.getElementById('rem-modal-save');

  const nMsgs = countDomMessages();
  if (!state.sessionId || nMsgs < 2) {
    // Show a small explanatory state instead of silently failing.
    modal.hidden = false;
    scrim.hidden = false;
    sub.textContent = state.sessionId
      ? `Need at least 2 messages — current session has ${nMsgs}.`
      : 'No active session yet. Send a message first, then try Smart Remember.';
    body.innerHTML = '<div class="rem-empty">Start a conversation, then come back to Smart Remember to extract memories from it.</div>';
    saveBtn.disabled = true;
    return;
  }

  modal.hidden = false;
  scrim.hidden = false;
  sub.textContent = `Analyzing ${nMsgs} messages…`;
  body.innerHTML = '<div class="rem-loading"><span class="rem-spinner" aria-hidden="true"></span> extracting candidate memories…</div>';
  saveBtn.disabled = true;
  _smartItems = [];
  _smartLoading = true;

  fetch('/api/memories/extract', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: state.sessionId }),
  }).then(async (r) => {
    _smartLoading = false;
    if (r.status === 404) {
      sub.textContent = 'Backend missing — try again later.';
      body.innerHTML = '<div class="rem-empty">POST /api/memories/extract isn\'t available on this server yet.</div>';
      return;
    }
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    const items = Array.isArray(data.items) ? data.items : [];
    _smartItems = items;
    const nTurns = data.n_turns_considered ?? nMsgs;
    if (!items.length) {
      sub.textContent = `Analyzed ${nTurns} messages — no new memories proposed.`;
      body.innerHTML = '<div class="rem-empty">Nothing surfaced. Try chatting a bit more, or use the Remember button on specific text.</div>';
      return;
    }
    sub.textContent = `Analyzed ${nTurns} messages · ${items.length} candidate${items.length === 1 ? '' : 's'}`;
    renderSmartRememberItems(items);
    updateSmartRememberSaveBtn();
  }).catch((e) => {
    _smartLoading = false;
    sub.textContent = 'Extraction failed.';
    body.innerHTML = `<div class="rem-empty">Error: ${escapeHTML(e.message || String(e))}</div>`;
  });
}

function renderSmartRememberItems(items) {
  const body = document.getElementById('rem-modal-body');
  body.innerHTML = '';
  items.forEach((it, idx) => {
    const row = document.createElement('div');
    row.className = 'rem-item';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'rem-item-cb';
    cb.dataset.idx = String(idx);
    const conf = typeof it.confidence === 'number' ? it.confidence : null;
    cb.checked = conf == null ? true : conf >= 0.7;
    cb.addEventListener('change', updateSmartRememberSaveBtn);

    const textBox = document.createElement('textarea');
    textBox.className = 'rem-item-text';
    textBox.rows = 1;
    textBox.value = it.text || '';
    textBox.addEventListener('input', () => {
      _smartItems[idx].text = textBox.value;
      // auto-resize
      textBox.style.height = 'auto';
      textBox.style.height = Math.min(textBox.scrollHeight, 140) + 'px';
    });
    // Initial autosize on next tick (textarea must be in DOM).
    setTimeout(() => {
      textBox.style.height = 'auto';
      textBox.style.height = Math.min(textBox.scrollHeight, 140) + 'px';
    }, 0);

    const meta = document.createElement('div');
    meta.className = 'rem-item-meta';
    const cat = it.category ? `<span class="rem-pill">${escapeHTML(String(it.category))}</span>` : '';
    const pct = conf == null ? '' : `<span class="rem-conf">${Math.round(conf * 100)}%</span>`;
    meta.innerHTML = `${cat}${pct}`;

    row.appendChild(cb);
    const col = document.createElement('div');
    col.className = 'rem-item-col';
    col.appendChild(textBox);
    col.appendChild(meta);
    row.appendChild(col);
    body.appendChild(row);
  });
}

function updateSmartRememberSaveBtn() {
  const saveBtn = document.getElementById('rem-modal-save');
  const n = document.querySelectorAll('.rem-item-cb:checked').length;
  saveBtn.disabled = n === 0 || _smartSaving;
  saveBtn.textContent = n > 0 ? `Save ${n} selected` : 'Save selected';
}

async function saveSmartRememberSelected() {
  if (_smartSaving) return;
  const checks = Array.from(document.querySelectorAll('.rem-item-cb:checked'));
  if (!checks.length) return;
  const items = checks.map(cb => {
    const idx = parseInt(cb.dataset.idx, 10);
    return _smartItems[idx];
  }).filter(Boolean);

  _smartSaving = true;
  const saveBtn = document.getElementById('rem-modal-save');
  const cancelBtn = document.getElementById('rem-modal-cancel');
  const sub = document.getElementById('rem-modal-sub');
  saveBtn.disabled = true;
  if (cancelBtn) cancelBtn.disabled = true;

  let saved = 0, failed = 0;
  for (let i = 0; i < items.length; i++) {
    sub.textContent = `Saving ${i + 1}/${items.length}…`;
    const text = (items[i].text || '').trim();
    try {
      const r = await fetch('/api/memories?background=true', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json().catch(() => ({}));
      if (data.background && data.job_id) {
        const snippet = text.slice(0, 50) + (text.length > 50 ? '…' : '');
        trackJob(data.job_id, 'memory_embed', `Embedding memory: "${snippet}"`);
      }
      saved++;
    } catch {
      failed++;
    }
  }
  _smartSaving = false;
  if (cancelBtn) cancelBtn.disabled = false;
  closeSmartRememberModal();
  if (failed && saved) flashToast(`Saved ${saved}, ${failed} failed`);
  else if (failed) flashToast(`Failed to save ${failed} ${failed === 1 ? 'memory' : 'memories'}`);
  else flashToast(`Saved ${saved} ${saved === 1 ? 'memory' : 'memories'}`);
  refreshSidebarCounts();
}

function closeSmartRememberModal() {
  if (_smartSaving) return;
  const modal = document.getElementById('rem-modal');
  const scrim = document.getElementById('rem-modal-scrim');
  if (modal) modal.hidden = true;
  if (scrim) scrim.hidden = true;
  _smartItems = [];
  _smartLoading = false;
}

function filterMemories() {
  const q = memSearchEl.value.trim().toLowerCase();
  if (!q) return paintMemories(_memoriesCache);
  const filtered = _memoriesCache.filter(m =>
    (m.title || '').toLowerCase().includes(q) ||
    (m.text || '').toLowerCase().includes(q)
  );
  paintMemories(filtered);
}

// ---------- documents ----------
async function loadDocs() {
  docsListEl.innerHTML = '<div class="loading-row"><span class="shimmer"></span></div>'.repeat(3);
  try {
    const r = await fetch('/api/docs');
    const items = await r.json();
    paintDocs(items);
    docCountEl.textContent = String(items.length);
  } catch (e) {
    console.error('docs load failed', e);
    docsListEl.innerHTML = '<div class="drawer-empty">Failed to load documents.</div>';
  }
}

function paintDocs(items) {
  if (!items.length) {
    docsListEl.innerHTML = '<div class="drawer-empty"><div class="e-icon">📄</div>No documents ingested.<br><span style="color:var(--muted-2)">Run <code>hrag ingest &lt;path&gt;</code></span></div>';
    return;
  }
  docsListEl.innerHTML = '';
  for (const d of items) {
    const card = document.createElement('div');
    card.className = 'doc-card';
    const head = document.createElement('div');
    head.className = 'doc-head';
    const title = document.createElement('span');
    title.className = 'doc-title';
    title.textContent = d.title || '(untitled)';
    const meta = document.createElement('span');
    meta.className = 'doc-meta';
    meta.textContent = `${d.n_chunks ?? 0} chunks`;
    head.appendChild(title);
    head.appendChild(meta);
    const path = document.createElement('div');
    path.className = 'doc-path';
    path.textContent = d.source_path || '';
    card.appendChild(head);
    card.appendChild(path);
    docsListEl.appendChild(card);
  }
}

// ---------- SOTA per-file ingest progress tracker ----------
// Drives the .job-drawer (bottom-right floating stack of .job-card elements).
// Public entry point is trackJob(jobId, kind, label, file?) — every upload
// + memory POST already calls it. Each job polls /api/jobs/{id} every 250 ms
// for tight liveness; the server-side `_flush_job_state` is throttled to
// ~5 writes/sec so we're not in lockstep.
//
// Card contents:
//   - filename, elapsed counter, ETA, throughput (c/s)
//   - 6-stage chip strip: Load → Chunk → Filter → Embed → Index → Assign
//   - two-layer bar: faint underlay (total %), bright fill (active-stage %)
//   - expandable detail: per-stage durations + last 12 log lines
//   - red error state with the exception message
//   - ✕ cancel button → POST /api/jobs/{id}/cancel
//
// After a job reaches a terminal state we keep the card visible for 4 s so
// the user sees the final ✓ / ✗ / cancelled flourish, then fade it out.

const _jobs = new Map();  // job_id → state object
const _jobPolling = new Set();  // re-entrancy guard
const _jobAnchor = () => document.getElementById('job-drawer');

const JOB_STAGES = ['load', 'chunk', 'filter', 'embed', 'index', 'assign'];
const JOB_STAGE_LABELS = {
  load: 'Load', chunk: 'Chunk', filter: 'Filter',
  embed: 'Embed', index: 'Index', assign: 'Assign',
};
const MEMORY_STAGES = ['embed', 'index'];

function trackJob(jobId, kind, label, file) {
  if (!jobId) return;
  if (_jobs.has(jobId)) {
    // Refresh label if the caller relabels an in-flight job.
    const cur = _jobs.get(jobId);
    if (label) cur.label = label;
    renderJobs();
    return;
  }
  _jobs.set(jobId, {
    jobId,
    kind: kind || 'job',
    label: label || (kind || 'job'),
    file: file || null,
    status: 'queued',
    progress: 0,
    total: 1,
    message: '',
    result: null,
    stages: {},
    current_stage: null,
    files: [],
    errors: [],
    startedAt: performance.now(),
    eta: null,
    throughput: null,
    recentSamples: [],
    detailOpen: false,
    logLines: [],
    _terminalAt: null,
  });
  renderJobs();
  if (!_jobPolling.has(jobId)) pollJobLoop(jobId);
}

async function pollJobLoop(jobId) {
  _jobPolling.add(jobId);
  const POLL_MS = 250;     // tight enough for SOTA feel
  // Worst case: 30 min at 250 ms = 7200 attempts. Most jobs finish in seconds.
  for (let attempt = 0; attempt < 7200; attempt++) {
    const j = _jobs.get(jobId);
    if (!j) { _jobPolling.delete(jobId); return; }
    try {
      const r = await fetch('/api/jobs/' + encodeURIComponent(jobId));
      if (r.ok) {
        const data = await r.json();
        applyJobUpdate(jobId, data);
      }
    } catch (e) { /* network blip — keep polling */ }
    const cur = _jobs.get(jobId);
    if (!cur) { _jobPolling.delete(jobId); return; }
    if (cur.status === 'done' || cur.status === 'failed' || cur.status === 'cancelled') {
      // Let the success/fail/cancelled state linger for 4s, then auto-fade.
      _jobPolling.delete(jobId);
      setTimeout(() => onJobFinished(jobId), 4000);
      return;
    }
    await new Promise(rs => setTimeout(rs, POLL_MS));
  }
  _jobPolling.delete(jobId);
}

function applyJobUpdate(jobId, data) {
  const j = _jobs.get(jobId);
  if (!j) return;
  j.status = data.status || j.status;
  j.progress = data.progress != null ? data.progress : j.progress;
  j.total = data.total != null ? data.total : (j.total || 1);
  j.message = data.message || j.message;
  // The server stores per-stage detail inside `result` (JSON).
  let parsed = data.result;
  if (typeof parsed === 'string') {
    try { parsed = JSON.parse(parsed); } catch { parsed = null; }
  }
  if (parsed && typeof parsed === 'object') {
    j.result = parsed;
    j.stages = parsed.stages || {};
    j.current_stage = parsed.current_stage || j.current_stage;
    j.files = parsed.files || [];
    j.errors = parsed.errors || [];
  }

  // Compute throughput + ETA from the embed stage if active.
  const embed = j.stages.embed;
  if (embed && embed.n_total > 0) {
    const sample = { ts: performance.now(), n_done: embed.n_done || 0 };
    j.recentSamples.push(sample);
    if (j.recentSamples.length > 12) j.recentSamples.shift();
    if (j.recentSamples.length >= 2) {
      const first = j.recentSamples[0];
      const last = j.recentSamples[j.recentSamples.length - 1];
      const dt_s = (last.ts - first.ts) / 1000;
      const dn = last.n_done - first.n_done;
      if (dt_s > 0.15) {
        j.throughput = dn / dt_s;
        const remaining = embed.n_total - embed.n_done;
        if (j.throughput > 0.01) j.eta = remaining / j.throughput;
      }
    }
  }

  // Append message to the log if it changed.
  if (j.message && (j.logLines.length === 0 || j.logLines[j.logLines.length-1] !== j.message)) {
    j.logLines.push(j.message);
    if (j.logLines.length > 30) j.logLines.shift();
  }

  renderJobs();
}

function renderJobs() {
  const drawer = _jobAnchor();
  if (!drawer) return;
  if (_jobs.size === 0) {
    drawer.hidden = true;
    drawer.innerHTML = '';
    return;
  }
  drawer.hidden = false;
  // Newest first; keep up to 5 visible, fold the rest behind a count badge.
  const ids = Array.from(_jobs.keys()).reverse();
  const visibleIds = ids.slice(0, 5);
  const hiddenCount = ids.length - visibleIds.length;

  // Index existing children by jobId so we can reuse them and avoid layout
  // thrash on every poll.
  const existing = new Map();
  for (const el of Array.from(drawer.children)) {
    const id = el.dataset && el.dataset.jobId;
    if (id) existing.set(id, el);
  }
  drawer.innerHTML = '';
  for (const id of visibleIds) {
    const j = _jobs.get(id);
    let card = existing.get(id);
    if (!card || !card.classList.contains('job-card')) {
      card = buildJobCard(id, j);
    } else {
      updateJobCard(card, id, j);
    }
    drawer.appendChild(card);
  }
  if (hiddenCount > 0) {
    const more = document.createElement('div');
    more.className = 'job-drawer-collapsed';
    more.textContent = `+ ${hiddenCount} more job${hiddenCount === 1 ? '' : 's'}`;
    drawer.appendChild(more);
  }
}

function _jobStagesFor(j) {
  return (j.kind === 'memory_embed') ? MEMORY_STAGES : JOB_STAGES;
}

function buildJobCard(jobId, j) {
  const card = document.createElement('div');
  card.className = 'job-card';
  card.dataset.jobId = jobId;
  card.dataset.status = j.status;

  card.innerHTML = `
    <div class="job-card-head">
      <div class="job-card-icon">⟳</div>
      <div class="job-card-title">
        <div class="job-card-name"></div>
        <div class="job-card-meta"></div>
      </div>
      <button class="job-card-cancel" type="button" title="Cancel">✕</button>
    </div>
    <div class="job-card-stages"></div>
    <div class="job-card-bar">
      <div class="job-card-bar-underlay"></div>
      <div class="job-card-bar-fill"></div>
    </div>
    <button class="job-card-detail-toggle" type="button" aria-expanded="false">Details</button>
    <div class="job-card-detail" hidden>
      <div class="job-card-detail-rows"></div>
      <div class="job-card-detail-log"></div>
    </div>
  `;

  // Cancel button — terminal states relabel it "Dismiss".
  const cancelBtn = card.querySelector('.job-card-cancel');
  cancelBtn.addEventListener('click', () => {
    const cur = _jobs.get(jobId);
    if (!cur) return;
    if (cur.status === 'running' || cur.status === 'queued') {
      cancelJob(jobId);
    } else {
      // Already terminal — just dismiss the card.
      _jobs.delete(jobId);
      renderJobs();
    }
  });

  // Detail toggle
  const toggle = card.querySelector('.job-card-detail-toggle');
  toggle.addEventListener('click', () => {
    const open = toggle.getAttribute('aria-expanded') === 'true';
    toggle.setAttribute('aria-expanded', open ? 'false' : 'true');
    card.querySelector('.job-card-detail').hidden = open;
    const cur = _jobs.get(jobId);
    if (cur) cur.detailOpen = !open;
  });

  updateJobCard(card, jobId, j);
  return card;
}

function updateJobCard(card, jobId, j) {
  card.dataset.status = j.status;
  const stages = _jobStagesFor(j);

  // Title + meta line.
  const nameEl = card.querySelector('.job-card-name');
  const fname = (j.files && j.files[0] && j.files[0].name) || j.label || j.kind;
  nameEl.textContent = fname;
  nameEl.title = fname;
  card.querySelector('.job-card-meta').textContent = formatJobMeta(j);

  // Cancel/dismiss label.
  const cancelBtn = card.querySelector('.job-card-cancel');
  if (j.status === 'done' || j.status === 'failed' || j.status === 'cancelled') {
    cancelBtn.textContent = '×';
    cancelBtn.title = 'Dismiss';
  } else {
    cancelBtn.textContent = '✕';
    cancelBtn.title = 'Cancel';
  }

  // Stage chips.
  const stagesEl = card.querySelector('.job-card-stages');
  stagesEl.innerHTML = '';
  for (const s of stages) {
    const chip = document.createElement('div');
    chip.className = 'job-stage';
    chip.dataset.stage = s;
    chip.textContent = JOB_STAGE_LABELS[s] || s;
    const data = j.stages[s];
    if (j.errors && j.errors.length && j.current_stage === s) {
      chip.classList.add('failed');
    } else if (data && data.ts_end && data.n_done >= data.n_total) {
      chip.classList.add('done');
    } else if (j.current_stage === s) {
      chip.classList.add('active');
    } else {
      chip.classList.add('pending');
    }
    stagesEl.appendChild(chip);
  }

  // Two-layer progress bar.
  const totalPct = computeTotalPct(j, stages);
  const stagePct = computeStagePct(j);
  card.querySelector('.job-card-bar-underlay').style.setProperty('--total-pct', `${totalPct}%`);
  card.querySelector('.job-card-bar-fill').style.setProperty('--stage-pct', `${stagePct}%`);

  // Detail panel — auto-open on failure.
  const detailEl = card.querySelector('.job-card-detail');
  const toggleEl = card.querySelector('.job-card-detail-toggle');
  const shouldOpen = j.detailOpen || j.status === 'failed';
  if (shouldOpen) {
    detailEl.hidden = false;
    toggleEl.setAttribute('aria-expanded', 'true');
    const rows = card.querySelector('.job-card-detail-rows');
    rows.innerHTML = '';
    for (const s of stages) {
      const data = j.stages[s] || {};
      const done = data.ts_end && data.n_done >= data.n_total;
      const active = j.current_stage === s && !done;
      const name = document.createElement('div');
      name.className = 'stage-name';
      name.textContent = JOB_STAGE_LABELS[s] || s;
      const status = document.createElement('div');
      status.className = done ? 'stage-done' : (active ? 'stage-active' : 'stage-pending');
      status.textContent = done ? '✓' : (active ? '…' : '');
      const duration = document.createElement('div');
      duration.className = done ? 'stage-done' : 'stage-pending';
      duration.textContent = (data.duration_s != null)
        ? `${Number(data.duration_s).toFixed(1)}s`
        : '';
      rows.appendChild(name);
      rows.appendChild(status);
      rows.appendChild(duration);
    }
    const log = card.querySelector('.job-card-detail-log');
    log.textContent = (j.logLines || []).slice(-12).join('\n');
  }

  // Error pane (single row, always at the bottom of the detail pane).
  let errEl = card.querySelector('.job-card-error');
  if (j.errors && j.errors.length) {
    if (!errEl) {
      errEl = document.createElement('div');
      errEl.className = 'job-card-error';
      detailEl.appendChild(errEl);
    }
    const last = j.errors[j.errors.length - 1];
    errEl.textContent = (last && last.error) ? last.error : 'Failed';
  } else if (errEl) {
    errEl.remove();
  }
}

function formatJobMeta(j) {
  const stage = j.current_stage || (j.status === 'queued' ? 'queued' : 'running');
  const data = j.stages[stage] || {};
  const parts = [];
  // Stage label
  parts.push(JOB_STAGE_LABELS[stage] || stage);
  // Counts (skip when n_total is 1 or missing — meaningless)
  if (data.n_total && data.n_total > 1) {
    parts.push(`${data.n_done || 0}/${data.n_total}`);
  }
  // Throughput + ETA (only meaningful for embed)
  if (j.throughput && stage === 'embed' && j.throughput > 0.05) {
    parts.push(`${j.throughput.toFixed(1)} c/s`);
  }
  if (j.eta != null && stage === 'embed' && j.eta > 0.5 && j.eta < 3600) {
    parts.push(`ETA ${formatDuration(j.eta)}`);
  }
  // Elapsed (always)
  const elapsed = (performance.now() - j.startedAt) / 1000;
  parts.push(`${formatDuration(elapsed)} elapsed`);
  // Terminal summary
  if (j.status === 'done') {
    const nc = (j.files && j.files[0] && j.files[0].n_chunks) || (j.result && j.result.n_chunks);
    if (nc) parts.push(`${nc} chunks`);
  } else if (j.status === 'cancelled') {
    parts.push('cancelled');
  } else if (j.status === 'failed') {
    parts.push('failed');
  }
  return parts.join(' · ');
}

function formatDuration(seconds) {
  if (seconds < 1) return `${Math.round(seconds*1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const m = Math.floor(seconds/60); const s = Math.floor(seconds % 60);
  return `${m}m${s}s`;
}

function computeTotalPct(j, stages) {
  // Each stage contributes 1/N once fully done; partial contribution for the
  // active stage based on its own n_done/n_total ratio.
  let cum = 0;
  for (const s of stages) {
    const d = j.stages[s];
    if (!d) continue;
    if (d.ts_end && d.n_done >= d.n_total) cum += 1;
    else if (j.current_stage === s) {
      cum += d.n_total ? Math.min(1, (d.n_done || 0) / d.n_total) : 0;
    }
  }
  const pct = (cum / stages.length) * 100;
  return Math.max(0, Math.min(100, pct));
}

function computeStagePct(j) {
  const d = j.stages[j.current_stage] || null;
  if (!d || !d.n_total) return 0;
  return Math.max(2, Math.min(100, ((d.n_done || 0) / d.n_total) * 100));
}

async function cancelJob(jobId) {
  try {
    await fetch('/api/jobs/' + encodeURIComponent(jobId) + '/cancel', { method: 'POST' });
  } catch (e) { /* defensive — UI already optimistic */ }
  const j = _jobs.get(jobId);
  if (j) {
    j.status = 'cancelled';
    renderJobs();
  }
}

function onJobFinished(jobId) {
  const card = document.querySelector(`.job-card[data-job-id="${jobId}"]`);
  const j = _jobs.get(jobId);

  // Per-kind post-completion side effects (parity with old onJobFinished).
  if (j && j.status === 'done' && j.kind === 'doc_ingest') {
    let res = j.result || {};
    if (typeof res === 'string') { try { res = JSON.parse(res); } catch { res = {}; } }
    if (res && res.taxonomy_mode === 'manual' && res.doc_id) {
      try { promptTaxonomyChoice(res.doc_id, res.title || j.label); } catch {}
    }
  }
  if (j && j.status === 'done' && j.kind === 'memory_embed') {
    try {
      const drawer = document.getElementById('memories-drawer');
      const visible = drawer && !drawer.classList.contains('hidden')
        && drawer.getAttribute('aria-hidden') !== 'true';
      if (visible && typeof loadMemories === 'function') loadMemories();
    } catch {}
    try { refreshSidebarCounts && refreshSidebarCounts(); } catch {}
  }

  if (card) {
    card.style.transition = 'opacity 300ms ease-out, transform 300ms ease-out';
    card.style.opacity = '0';
    card.style.transform = 'translateY(8px) scale(0.97)';
    setTimeout(() => {
      _jobs.delete(jobId);
      renderJobs();
    }, 320);
  } else {
    _jobs.delete(jobId);
    renderJobs();
  }
}

// Manual-mode banner shown on the Taxonomy page after a doc-ingest job
// completes with taxonomy_mode=manual. Switches to the Taxonomy view and
// drops a small inline pill ("<title> is unfiled. [Auto-assign]").
function promptTaxonomyChoice(docId, title) {
  if (!docId) return;
  // 1. Switch to the Taxonomy page (the existing handler reloads the tree).
  try {
    const navBtn = document.getElementById('open-taxonomy')
      || document.getElementById('nav-taxonomy')
      || document.querySelector('[data-nav="taxonomy"]');
    if (navBtn) navBtn.click();
  } catch { /* if taxonomy nav isn't present we still show the banner */ }

  // 2. Find a host element on the Taxonomy page to drop the banner into;
  //    fall back to <body> so the message never gets lost.
  const host = document.getElementById('taxonomy-page')
    || document.querySelector('.tax-toolbar')?.parentElement
    || document.body;
  if (!host) return;

  // Replace any prior banner for the same doc.
  const prev = host.querySelector(`.tax-manual-banner[data-doc="${CSS.escape(docId)}"]`);
  if (prev) prev.remove();

  const banner = document.createElement('div');
  banner.className = 'tax-manual-banner';
  banner.dataset.doc = docId;

  const msg = document.createElement('div');
  msg.className = 'tax-manual-banner-msg';
  const titleSafe = title || docId.slice(0, 16);
  msg.textContent = `"${titleSafe}" is unfiled. Drag it onto a node, or auto-assign it now.`;

  const autoBtn = document.createElement('button');
  autoBtn.type = 'button';
  autoBtn.textContent = 'Auto-assign';
  autoBtn.addEventListener('click', async () => {
    autoBtn.disabled = true;
    autoBtn.textContent = 'Assigning…';
    try {
      const r = await fetch(
        '/api/taxonomy/auto-assign/' + encodeURIComponent(docId),
        { method: 'POST' },
      );
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      flashToast && flashToast(
        data.assigned_node
          ? 'Filed under ' + data.assigned_node
          : 'Tree empty — still unfiled'
      );
      banner.remove();
      // Reload the tree so the doc moves out of the unfiled list.
      if (typeof loadTaxonomyTree === 'function') {
        try { await loadTaxonomyTree(); } catch { /* ignore */ }
      }
    } catch (e) {
      autoBtn.disabled = false;
      autoBtn.textContent = 'Auto-assign';
      flashToast && flashToast('auto-assign failed: ' + (e.message || e));
    }
  });

  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'tmb-close';
  closeBtn.setAttribute('aria-label', 'Dismiss');
  closeBtn.textContent = '×';
  closeBtn.addEventListener('click', () => banner.remove());

  banner.appendChild(msg);
  banner.appendChild(autoBtn);
  banner.appendChild(closeBtn);

  // Insert near the top of the host so it's the first thing the user sees.
  host.insertBefore(banner, host.firstChild);
}

// ---------- upload (background-job flow) ----------
// Reads the .upload-taxonomy-choice radio group to pick "auto" vs "manual".
// "manual" tells the server to skip the auto-assign hook so the doc lands
// in the unfiled list; trackJob() then routes the user to the Taxonomy
// page when the job finishes.
function _readTaxonomyMode() {
  try {
    const picked = document.querySelector(
      '#upload-taxonomy-choice input[name="taxmode"]:checked'
    );
    const v = picked ? picked.value : 'auto';
    return (v === 'manual' || v === 'skip') ? 'manual' : 'auto';
  } catch { return 'auto'; }
}

async function handleUploadFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const files = Array.from(fileList);
  uploadProgressEl.innerHTML = '';

  const taxMode = _readTaxonomyMode();
  // Kick off all uploads in parallel — server runs each as a daemon thread.
  await Promise.all(files.map(f => uploadOne(f, taxMode)));

  // refresh listings after all jobs settle
  await loadDocs();
  refreshSidebarCounts();
  uploadInputEl.value = '';  // allow re-uploading the same file
}

async function uploadOne(f, taxMode = 'auto') {
  const row = document.createElement('div');
  row.className = 'upload-row pending';
  row.innerHTML = `
    <span class="ur-name">${escapeHTML(f.name)}</span>
    <span class="ur-status">uploading…</span>
  `;
  uploadProgressEl.appendChild(row);
  const status = row.querySelector('.ur-status');

  try {
    const form = new FormData();
    form.append('file', f);
    const url = '/api/ingest?background=true&taxonomy_mode=' +
      encodeURIComponent(taxMode);
    const r = await fetch(url, { method: 'POST', body: form });
    if (!r.ok) throw new Error(await r.text() || ('HTTP ' + r.status));
    const { job_id } = await r.json();
    status.textContent = 'queued…';
    // Mirror the per-row spinner into the global job bar.
    trackJob(job_id, 'doc_ingest', `Ingesting: ${f.name}`);
    await pollJob(job_id, row);
  } catch (e) {
    row.classList.remove('pending');
    row.classList.add('failed');
    status.textContent = 'failed: ' + (e.message || 'error');
  }
}

async function pollJob(jobId, row) {
  const status = row.querySelector('.ur-status');
  // Poll at 750ms; total cap 10min. Stops early on terminal state.
  for (let i = 0; i < 800; i++) {
    await new Promise(r => setTimeout(r, 750));
    let job;
    try {
      const resp = await fetch('/api/jobs/' + jobId);
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      job = await resp.json();
    } catch (e) {
      status.textContent = 'poll failed';
      row.classList.remove('pending');
      row.classList.add('failed');
      return;
    }
    if (job.status === 'running' || job.status === 'queued') {
      status.textContent = job.message || job.status;
      continue;
    }
    if (job.status === 'done') {
      row.classList.remove('pending');
      row.classList.add('done');
      status.textContent = job.message || 'done';
      return;
    }
    if (job.status === 'failed') {
      row.classList.remove('pending');
      row.classList.add('failed');
      status.textContent = job.message || 'failed';
      return;
    }
  }
  status.textContent = 'timeout';
  row.classList.remove('pending');
  row.classList.add('failed');
}

// ---------- feedback drawer ----------
let _feedbackCache = null;

async function loadFeedbackStats() {
  if (!fbNegativesEl) return;
  try {
    const r = await fetch('/api/feedback/stats');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    _feedbackCache = data;
    paintFeedbackStats(data);
  } catch (e) {
    console.error('feedback stats load failed', e);
    fbNegativesEl.innerHTML = '<div class="drawer-empty">Failed to load feedback stats.</div>';
  }
}

function paintFeedbackStats(data) {
  const up = Number(data?.thumbs_up || 0);
  const down = Number(data?.thumbs_down || 0);
  const total = Number(data?.total ?? (up + down));
  const sessions = Number(data?.sessions || 0);
  if (fbUpNum) fbUpNum.textContent = String(up);
  if (fbDownNum) fbDownNum.textContent = String(down);
  if (fbTotalNum) fbTotalNum.textContent = String(total);
  if (fbSessionsNum) fbSessionsNum.textContent = String(sessions);
  if (fbCountEl) fbCountEl.textContent = String(total);
  // Ratio bar — never go to NaN/divide-by-zero
  const denom = (up + down) || 1;
  const upPct = up ? (up / denom) * 100 : 0;
  const downPct = down ? (down / denom) * 100 : 0;
  if (fbRatioUp) fbRatioUp.style.width = `${upPct.toFixed(1)}%`;
  if (fbRatioDown) fbRatioDown.style.width = `${downPct.toFixed(1)}%`;

  const negatives = Array.isArray(data?.top_negative) ? data.top_negative : [];
  if (!negatives.length) {
    fbNegativesEl.innerHTML = '<div class="drawer-empty"><div class="e-icon">👎</div>No 👎 yet. Rate some answers from the chat to see them here.</div>';
    return;
  }
  fbNegativesEl.innerHTML = '';
  for (const item of negatives) {
    const card = document.createElement('div');
    card.className = 'fb-neg-card';
    const q = (item.question || item.user_question || item.content || '(no question)').toString();
    const truncated = q.length > 80 ? q.slice(0, 80) + '…' : q;
    const sid = item.session_id || '(no session)';
    const when = item.created_at || item.time || item.timestamp || '';
    const whenStr = when ? String(when).replace('T', ' ').slice(0, 16) : '';

    const qEl = document.createElement('div');
    qEl.className = 'fb-neg-q';
    qEl.textContent = truncated;
    maybeRTL(qEl, q);

    const metaEl = document.createElement('div');
    metaEl.className = 'fb-neg-meta';
    const sidShort = String(sid).slice(-8);
    metaEl.textContent = whenStr ? `session ${sidShort} · ${whenStr}` : `session ${sidShort}`;

    card.appendChild(qEl);
    card.appendChild(metaEl);
    fbNegativesEl.appendChild(card);
  }
}

async function refreshSidebarCounts() {
  try {
    const [mems, docs, fbStats] = await Promise.all([
      fetch('/api/memories?limit=1000').then(r => r.json()).catch(() => []),
      fetch('/api/docs').then(r => r.json()).catch(() => []),
      fetch('/api/feedback/stats').then(r => r.ok ? r.json() : null).catch(() => null),
    ]);
    memCountEl.textContent = String(mems.length);
    docCountEl.textContent = String(docs.length);
    if (fbCountEl && fbStats) {
      const total = Number(fbStats.total ?? ((fbStats.thumbs_up || 0) + (fbStats.thumbs_down || 0)));
      fbCountEl.textContent = String(total);
    }
  } catch {}
}

function renderSources(sources) {
  const wrap = document.createElement('div');
  wrap.className = 'sources';
  const det = document.createElement('details');
  const sum = document.createElement('summary');
  sum.textContent = `Sources · ${sources.length}`;
  det.appendChild(sum);
  const list = document.createElement('div');
  list.className = 'source-list';
  for (let i = 0; i < sources.length; i++) {
    const s = sources[i];
    const item = document.createElement('div');
    item.className = 'source-item';
    const head = document.createElement('div');
    head.className = 'src-head';
    const title = document.createElement('span');
    title.className = 'src-title';
    title.textContent = `[${i + 1}] ${s.title}${s.section ? ' · ' + s.section : ''}`;
    const meta = document.createElement('span');
    meta.className = 'src-meta';
    const score = s.rerank_score != null ? `rerank ${s.rerank_score.toFixed(2)}` : (s.score != null ? `score ${s.score.toFixed(3)}` : '');
    meta.textContent = `${s.source_type || 'doc'} · ${score}`;
    head.appendChild(title); head.appendChild(meta);
    const snip = document.createElement('div');
    snip.className = 'src-snippet';
    snip.textContent = (s.text || '').slice(0, 300);
    item.appendChild(head);
    item.appendChild(snip);
    list.appendChild(item);
  }
  det.appendChild(list);
  wrap.appendChild(det);
  return wrap;
}

// ============================================================
// Taxonomy editor (full-screen, D3-driven)
// ============================================================

const taxNav = $('open-taxonomy');
const taxPage = $('taxonomy-page');
const taxBackBtn = $('tax-back');
const taxSvg = $('tax-svg');
const taxLinksLayer = $('tax-links-layer');
const taxNodesLayer = $('tax-nodes-layer');
const taxZoomLayer = $('tax-zoom-layer');
const taxCanvas = $('tax-canvas');
const taxEmpty = $('tax-empty');
const taxEmptyCta = $('tax-empty-cta');
const taxLoading = $('tax-loading');
const taxRecomputeBtn = $('tax-recompute-btn');
const taxAssignBtn = $('tax-assign-btn');
const taxOptionsBtn = $('tax-options-btn');
const taxOptionsPanel = $('tax-options');
const taxOptionsCloseBtn = $('tax-options-close');
const taxOptHint = $('tax-options-hint');
const taxNodesCountEl = $('tax-nodes-n');
const taxDocsCountEl = $('tax-docs-n');
const taxUnfiledCountEl = $('tax-unfiled-n');
const taxUnfiledPillEl = $('tax-unfiled-pill');
const taxNavCountEl = $('tax-count');

// Popover refs
const taxPopover = $('tax-popover');
const taxPopLabel = $('tax-pop-label');
const taxPopDesc = $('tax-pop-desc');
const taxPopClose = $('tax-pop-close');
const taxPopAdd = $('tax-pop-add');
const taxPopDel = $('tax-pop-del');
const taxPopDocs = $('tax-pop-docs');
const taxPopDocCount = $('tax-pop-doc-count');

// Modal refs
const taxModalScrim = $('tax-modal-scrim');
const taxModal = $('tax-modal');
const taxModalTitle = $('tax-modal-title');
const taxModalCloseBtn = $('tax-modal-close');
const taxModalSummary = $('tax-modal-summary');
const taxModalError = $('tax-modal-error');
const taxStagesEl = $('tax-stages');

// Option inputs
const taxOptInputs = {
  beam_width: $('tax-opt-beam'),
  max_depth: $('tax-opt-depth'),
  propose_sample_size: $('tax-opt-sample'),
  max_children_per_node: $('tax-opt-children'),
  min_top_score_floor: $('tax-opt-floor'),
  max_docs_pct: $('tax-opt-pct'),
};

const taxState = {
  treeRaw: null,        // raw API response
  rootHier: null,       // d3.hierarchy root
  nodeMap: new Map(),   // id → d3 node
  selectedId: null,
  zoomBehavior: null,
  activeStream: null,   // EventSource for SSE
  optionsOpen: false,
  lastLoadedAt: 0,
};

const TAX_CARD_W = 200;
const TAX_CARD_H = 56;
const TAX_GAP_X  = 80;
const TAX_GAP_Y  = 18;

// ---- routing ----
function openTaxonomyPage() {
  document.body.dataset.view = 'taxonomy';
  taxPage.setAttribute('aria-hidden', 'false');
  // close any drawers
  app.classList.remove('drawer-open');
  app.classList.remove('memories-open');
  app.classList.remove('docs-open');
  app.classList.remove('feedback-open');
  loadTaxonomyTree();
  syncTaxOptionsFromConfig();
}

function closeTaxonomyPage() {
  document.body.dataset.view = '';
  taxPage.setAttribute('aria-hidden', 'true');
  // close popover and options drawer if open
  taxPopover.hidden = true;
  taxOptionsPanel.setAttribute('aria-hidden', 'true');
  taxState.optionsOpen = false;
  closeDocPanel();
  // stop any live recompute stream
  closeTaxModal();
}

if (taxNav) taxNav.addEventListener('click', openTaxonomyPage);
if (taxBackBtn) taxBackBtn.addEventListener('click', closeTaxonomyPage);

// Doc-panel close button + Esc handling
const _docPanelCloseBtn = document.getElementById('doc-panel-close');
if (_docPanelCloseBtn) _docPanelCloseBtn.addEventListener('click', closeDocPanel);

// Esc priority: doc-panel > popover > options > modal > page
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (document.body.dataset.view !== 'taxonomy') return;
  const panel = document.getElementById('doc-panel');
  if (panel && panel.getAttribute('aria-hidden') === 'false') { closeDocPanel(); return; }
  if (!taxPopover.hidden) { taxPopover.hidden = true; return; }
  if (taxState.optionsOpen) { closeTaxOptions(); return; }
  if (!taxModal.hidden && !taxModalCloseBtn.hidden) { closeTaxModal(); return; }
  closeTaxonomyPage();
});

// SVG elements DON'T support the `hidden` IDL attribute the way HTMLElement
// does — assigning `svgEl.hidden = false` is a silent no-op on every browser.
// Use these helpers to toggle visibility correctly via the underlying HTML
// attribute, which IS picked up by the `.tax-svg[hidden]` CSS rule.
function setSvgHidden(el, hidden) {
  if (!el) return;
  if (hidden) el.setAttribute('hidden', '');
  else        el.removeAttribute('hidden');
}

// ---- load & render ----
async function loadTaxonomyTree() {
  taxLoading.hidden = false;
  taxEmpty.hidden = true;
  // NOTE: don't hide the SVG yet — d3.zoom needs the element to have real
  // dimensions when it binds, otherwise getBoundingClientRect returns 0×0
  // and the initial transform is wonky (content lands off-screen).
  try {
    const r = await fetch('/api/taxonomy/tree');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    taxState.treeRaw = data;
    taxState.lastLoadedAt = Date.now();
    paintTaxCounts(data);
    if (!data.root || (data.node_count ?? 0) === 0) {
      taxLoading.hidden = true;
      taxEmpty.hidden = false;
      setSvgHidden(taxSvg, true);
      return;
    }
    // Show the SVG BEFORE rendering so d3 sees correct dimensions.
    setSvgHidden(taxSvg, false);
    taxEmpty.hidden = true;
    taxLoading.hidden = true;
    renderTaxonomy(data.root);
    fitTaxonomyToView();
  } catch (e) {
    console.error('[taxonomy] tree load/render failed', e);
    taxLoading.hidden = true;
    taxEmpty.hidden = false;
    setSvgHidden(taxSvg, true);
    flashToast('Failed to load taxonomy: ' + (e && e.message ? e.message : e));
  }
}

// Compute a zoom transform that fits the laid-out tree inside the canvas
// at a comfortable scale, and apply it. The SVG viewBox is set 1:1 with the
// canvas pixel dimensions (see renderTaxonomy), so the d3.zoom transform
// operates directly in screen-pixel space and we don't have to compensate
// for SVG's auto-fit.
function fitTaxonomyToView() {
  try {
    if (!taxState.zoomBehavior || !taxSvg || !taxState.rootHier) return;
    const canvasRect = taxCanvas.getBoundingClientRect();
    const vpW = canvasRect.width, vpH = canvasRect.height;
    if (vpW < 10 || vpH < 10) return;
    // Bounding box of the laid-out tree, in node coordinates.
    // (d3.tree convention: x = row [vertical], y = col [horizontal].)
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    taxState.rootHier.each(n => {
      if (n.x < minX) minX = n.x;
      if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.y > maxY) maxY = n.y;
    });
    if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return;
    // Include the card's footprint in the bbox so the outermost cards
    // never clip against the canvas edge.
    const treeW = (maxY - minY) + TAX_CARD_W + 80;
    const treeH = (maxX - minX) + TAX_CARD_H + 60;
    // Fit-to-canvas scale, clamped so cards stay readable. 0.5 floor keeps
    // text legible; 1.2 ceiling avoids ugly upscaling on small trees.
    const fitScale = Math.min(vpW / treeW, vpH / treeH);
    const scale = Math.min(1.2, Math.max(0.5, fitScale));
    // Center the tree's bbox at the canvas midpoint.
    const cx = (minY + maxY) / 2;
    const cy = (minX + maxX) / 2;
    const tx = vpW / 2 - cx * scale;
    const ty = vpH / 2 - cy * scale;
    const t = d3.zoomIdentity.translate(tx, ty).scale(scale);
    d3.select(taxSvg).call(taxState.zoomBehavior.transform, t);
  } catch (err) {
    console.warn('[taxonomy] fit-to-view failed', err);
  }
}

function paintTaxCounts(data) {
  const n = data?.node_count ?? 0;
  const d = data?.doc_count ?? 0;
  const u = data?.unfiled_count ?? 0;
  if (taxNodesCountEl) taxNodesCountEl.textContent = String(n);
  if (taxDocsCountEl)  taxDocsCountEl.textContent  = String(d);
  if (taxUnfiledCountEl) taxUnfiledCountEl.textContent = String(u);
  if (taxNavCountEl) taxNavCountEl.textContent = String(n);
  if (taxAssignBtn) taxAssignBtn.disabled = u <= 0;
  if (taxUnfiledPillEl) taxUnfiledPillEl.classList.toggle('has-unfiled', u > 0);
}

function renderTaxonomy(rootData) {
  // Build d3.hierarchy
  const hier = d3.hierarchy(rootData, n => n.children || []);
  const treeLayout = d3.tree().nodeSize([TAX_CARD_H + TAX_GAP_Y, TAX_CARD_W + TAX_GAP_X]);
  treeLayout(hier);

  taxState.rootHier = hier;
  taxState.nodeMap = new Map();
  hier.each(n => taxState.nodeMap.set(n.data.id, n));

  // The SVG fills the canvas. We use a viewBox that matches the canvas's
  // pixel dimensions so the SVG coordinate system is identical to screen
  // pixels — no auto-scaling. d3.zoom owns ALL pan/zoom positioning; this
  // avoids the double-transform pitfall where SVG's preserveAspectRatio
  // shrinks the tree to a single sub-pixel speck (the bug visible in the
  // user's screenshot).
  const canvasRect = taxCanvas.getBoundingClientRect();
  const vbW = Math.max(canvasRect.width  || 800, 100);
  const vbH = Math.max(canvasRect.height || 600, 100);
  taxSvg.setAttribute('viewBox', `0 0 ${vbW} ${vbH}`);
  taxSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  taxSvg.setAttribute('width', '100%');
  taxSvg.setAttribute('height', '100%');

  // Zoom & pan
  const svgSel = d3.select(taxSvg);
  const zoomLayerSel = d3.select(taxZoomLayer);
  if (!taxState.zoomBehavior) {
    taxState.zoomBehavior = d3.zoom()
      .scaleExtent([0.2, 2.5])
      .filter(ev => {
        // disable zoom-pan when the user starts a node-drag
        if (ev.target && ev.target.closest && ev.target.closest('.tax-node')) return false;
        return !ev.button;
      })
      .on('zoom', (ev) => { zoomLayerSel.attr('transform', ev.transform); });
    svgSel.call(taxState.zoomBehavior);
  }

  // Edges
  const linkSel = d3.select(taxLinksLayer)
    .selectAll('path.tax-link')
    .data(hier.links(), d => d.target.data.id);
  linkSel.exit().remove();
  const linkEnter = linkSel.enter()
    .append('path')
    .attr('class', 'tax-link')
    .attr('fill', 'none');
  linkEnter.merge(linkSel)
    .transition().duration(400).ease(d3.easeCubicInOut)
    .attr('d', d => taxLinkPath(d.source, d.target));

  // Nodes
  const nodeSel = d3.select(taxNodesLayer)
    .selectAll('g.tax-node')
    .data(hier.descendants(), d => d.data.id);
  nodeSel.exit().remove();

  const nodeEnter = nodeSel.enter()
    .append('g')
    .attr('class', d => 'tax-node depth-' + d.depth)
    .attr('data-id', d => String(d.data?.id ?? ''))
    .attr('transform', d => `translate(${d.y - TAX_CARD_W / 2}, ${d.x - TAX_CARD_H / 2})`);

  nodeEnter.append('rect')
    .attr('class', 'tax-card')
    .attr('width', TAX_CARD_W)
    .attr('height', TAX_CARD_H)
    .attr('rx', 12)
    .attr('ry', 12);

  nodeEnter.append('text')
    .attr('class', 'tax-card-label')
    .attr('x', 14)
    .attr('y', 22);

  nodeEnter.append('text')
    .attr('class', 'tax-card-sub')
    .attr('x', 14)
    .attr('y', 42);

  nodeEnter.append('g')
    .attr('class', 'tax-card-badge')
    .attr('transform', `translate(${TAX_CARD_W - 36}, 8)`)
    .call(g => {
      g.append('rect')
        .attr('class', 'tax-badge-bg')
        .attr('width', 28).attr('height', 22).attr('rx', 11);
      g.append('text')
        .attr('class', 'tax-badge-text')
        .attr('x', 14).attr('y', 15)
        .attr('text-anchor', 'middle');
    });

  // Merge updates
  const merged = nodeEnter.merge(nodeSel);
  merged.attr('data-id', d => String(d.data?.id ?? ''));
  merged.transition().duration(400).ease(d3.easeCubicOut)
    .attr('transform', d => `translate(${d.y - TAX_CARD_W / 2}, ${d.x - TAX_CARD_H / 2})`);

  merged.select('text.tax-card-label')
    .text(d => truncateText(d.data.label || '(unlabeled)', 22));
  merged.select('text.tax-card-sub')
    .text(d => {
      const dc = d.data.doc_count ?? 0;
      const cc = (d.data.children || []).length;
      const parts = [];
      if (cc) parts.push(`${cc} children`);
      parts.push(`${dc} docs`);
      return parts.join(' · ');
    });
  merged.select('g.tax-card-badge text.tax-badge-text')
    .text(d => String(d.data.doc_count ?? 0));

  // Interactions
  merged.on('click', (ev, d) => {
    ev.stopPropagation();
    openNodePopover(d);
  });

  // Drag handler — wire on every render so new nodes get it
  merged.call(d3.drag()
    .clickDistance(4)
    .on('start', taxNodeDragStart)
    .on('drag',  taxNodeDragMove)
    .on('end',   taxNodeDragEnd)
  );
}

function taxLinkPath(src, tgt) {
  // d3.linkHorizontal expects {x, y} pairs; here the layout uses x=row, y=col
  // so swap into the canonical horizontal-tree orientation.
  const link = d3.linkHorizontal()
    .x(d => d.y)
    .y(d => d.x);
  return link({ source: src, target: tgt });
}

function truncateText(s, n) {
  s = String(s ?? '');
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + '…';
}

// ---- drag-and-drop reparent ----
let _taxDragGhost = null;
let _taxDragSourceId = null;
let _taxDragValidTargets = null;  // Set<string>
let _taxDragHoverId = null;

function taxNodeDragStart(ev, d) {
  if (!d || !d.parent) {
    // root — cannot reparent
    ev.sourceEvent?.preventDefault?.();
    return;
  }
  _taxDragSourceId = d.data.id;
  _taxDragValidTargets = computeValidTargets(d);
  // Mark all card outlines: valid vs invalid
  d3.select(taxNodesLayer).selectAll('g.tax-node')
    .classed('drag-valid',  n => _taxDragValidTargets.has(n.data.id) && n.data.id !== d.data.id)
    .classed('drag-invalid', n => !_taxDragValidTargets.has(n.data.id) && n.data.id !== d.data.id)
    .classed('drag-source', n => n.data.id === d.data.id);
  // Floating ghost (a div above SVG that follows pointer)
  _taxDragGhost = document.createElement('div');
  _taxDragGhost.className = 'tax-drag-ghost';
  _taxDragGhost.textContent = d.data.label || '(node)';
  taxCanvas.appendChild(_taxDragGhost);
  positionGhost(ev.sourceEvent);
}

function taxNodeDragMove(ev) {
  if (!_taxDragSourceId) return;
  positionGhost(ev.sourceEvent);
  // hover detection — find topmost .tax-node at pointer
  const el = pointerNodeUnder(ev.sourceEvent);
  const nid = el ? el.__data__?.data?.id : null;
  if (nid !== _taxDragHoverId) {
    _taxDragHoverId = nid;
    d3.select(taxNodesLayer).selectAll('g.tax-node')
      .classed('drag-hover', n => n.data.id === nid && _taxDragValidTargets.has(nid));
  }
}

async function taxNodeDragEnd(ev, d) {
  // capture before clearing
  const sourceId = _taxDragSourceId;
  const hoverId  = _taxDragHoverId;
  const valid    = _taxDragValidTargets;
  // cleanup visual state first
  if (_taxDragGhost) { _taxDragGhost.remove(); _taxDragGhost = null; }
  d3.select(taxNodesLayer).selectAll('g.tax-node')
    .classed('drag-valid', false)
    .classed('drag-invalid', false)
    .classed('drag-source', false)
    .classed('drag-hover', false);
  _taxDragSourceId = null;
  _taxDragValidTargets = null;
  _taxDragHoverId = null;
  if (!sourceId || !hoverId || hoverId === sourceId) return;
  if (!valid || !valid.has(hoverId)) {
    flashToast('cannot drop there (would create a cycle)');
    return;
  }
  if (hoverId === d.parent?.data?.id) {
    // no-op: dropped on current parent
    return;
  }
  try {
    const r = await fetch(`/api/taxonomy/nodes/${encodeURIComponent(sourceId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parent_id: hoverId }),
    });
    if (!r.ok) {
      const txt = await r.text().catch(() => '');
      throw new Error(txt || ('HTTP ' + r.status));
    }
    flashToast('moved node');
    await loadTaxonomyTree();
  } catch (e) {
    console.error('reparent failed', e);
    flashToast('move failed: ' + e.message);
  }
}

function computeValidTargets(srcNode) {
  // Any node that is NOT a descendant of srcNode (and not srcNode itself)
  // is a valid drop target. Root is included.
  const descendants = new Set();
  srcNode.each(n => descendants.add(n.data.id));
  const valid = new Set();
  taxState.nodeMap.forEach((n, id) => {
    if (!descendants.has(id)) valid.add(id);
  });
  return valid;
}

function positionGhost(srcEvent) {
  if (!_taxDragGhost || !srcEvent) return;
  const rect = taxCanvas.getBoundingClientRect();
  const x = (srcEvent.clientX ?? 0) - rect.left + 14;
  const y = (srcEvent.clientY ?? 0) - rect.top + 14;
  _taxDragGhost.style.left = x + 'px';
  _taxDragGhost.style.top = y + 'px';
}

function pointerNodeUnder(srcEvent) {
  if (!srcEvent) return null;
  const x = srcEvent.clientX, y = srcEvent.clientY;
  // temporarily hide ghost so it doesn't intercept
  let prev = '';
  if (_taxDragGhost) { prev = _taxDragGhost.style.pointerEvents; _taxDragGhost.style.pointerEvents = 'none'; }
  const target = document.elementFromPoint(x, y);
  if (_taxDragGhost) _taxDragGhost.style.pointerEvents = prev;
  if (!target) return null;
  const g = target.closest && target.closest('g.tax-node');
  return g || null;
}

// ---- node popover ----
let _popoverNodeId = null;
let _popoverDocs = [];

async function openNodePopover(node) {
  _popoverNodeId = node.data.id;
  taxState.selectedId = _popoverNodeId;
  taxPopLabel.value = node.data.label || '';
  taxPopDesc.value = node.data.description || '';
  // Position popover near the node card
  positionPopover(node);
  taxPopover.hidden = false;
  // Load docs assigned to this node
  taxPopDocs.innerHTML = '<div class="tax-pop-empty">loading…</div>';
  taxPopDocCount.textContent = '(…)';
  try {
    const r = await fetch(`/api/taxonomy/nodes/${encodeURIComponent(node.data.id)}/docs`);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    _popoverDocs = await parseDocsResponse(r);
    renderPopoverDocs();
  } catch (e) {
    console.error('node docs load failed', e);
    taxPopDocs.innerHTML = `<div class="tax-pop-empty">failed: ${escapeHTML(e.message)}</div>`;
    taxPopDocCount.textContent = '(0)';
  }
}

// /api/taxonomy/nodes/{id}/docs returns {node_id, label, docs: [...]}.
// Older code expected a bare array — accept either shape so a backend
// roll-back doesn't break the popover.
async function parseDocsResponse(resp) {
  const body = await resp.json();
  if (Array.isArray(body)) return body;
  if (body && Array.isArray(body.docs)) return body.docs;
  return [];
}

function positionPopover(node) {
  // Compute the node's screen position
  const svgRect = taxSvg.getBoundingClientRect();
  const canvasRect = taxCanvas.getBoundingClientRect();
  // Use bounding box of the SVG group
  const gEl = taxNodesLayer.querySelector(`g.tax-node[data-id="${CSS.escape(String(node.data.id))}"]`);
  let cx, cy;
  if (gEl) {
    const r = gEl.getBoundingClientRect();
    cx = r.right + 12 - canvasRect.left;
    cy = r.top - canvasRect.top;
  } else {
    cx = svgRect.width / 2;
    cy = svgRect.height / 2;
  }
  taxPopover.style.left = cx + 'px';
  taxPopover.style.top = cy + 'px';
}

// helper: is this an episodic memory id?
function isEpisodicDocId(docId) {
  return typeof docId === 'string' && docId.startsWith('episodic:');
}

function renderPopoverDocs() {
  taxPopDocCount.textContent = `(${_popoverDocs.length})`;
  taxPopDocs.innerHTML = '';

  if (!_popoverDocs.length) {
    const empty = document.createElement('div');
    empty.className = 'tax-pop-empty';
    empty.textContent = 'No docs assigned.';
    taxPopDocs.appendChild(empty);
  } else {
    for (const doc of _popoverDocs) {
      taxPopDocs.appendChild(buildPopoverDocChip(doc));
    }
  }

  // "+ Add" button + picker (always present)
  const addRow = document.createElement('div');
  addRow.className = 'tax-pop-add-row';
  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.className = 'tax-mini-btn tax-pop-add-doc-btn';
  addBtn.textContent = '+ Add doc';
  const picker = document.createElement('div');
  picker.className = 'tax-pop-picker';
  picker.hidden = true;
  addBtn.addEventListener('click', () => togglePopoverPicker(picker, addBtn));
  addRow.appendChild(addBtn);
  taxPopDocs.appendChild(addRow);
  taxPopDocs.appendChild(picker);
}

function buildPopoverDocChip(doc) {
  const episodic = isEpisodicDocId(doc.doc_id) || doc.source_type === 'episodic';
  const chip = document.createElement('div');
  chip.className = 'tax-doc-chip' + (episodic ? ' is-episodic' : '');
  chip.draggable = !episodic;
  chip.dataset.docId = doc.doc_id;
  chip.title = doc.title || doc.doc_id;

  const icon = document.createElement('span');
  icon.className = 'tax-doc-chip-icon';
  icon.textContent = episodic ? '🧠' : '📄';

  const titleEl = document.createElement('span');
  titleEl.className = 'tax-doc-chip-title';
  titleEl.textContent = truncateText(doc.title || doc.doc_id, 36);

  const actions = document.createElement('span');
  actions.className = 'tax-doc-chip-actions';

  const viewBtn = document.createElement('button');
  viewBtn.type = 'button';
  viewBtn.className = 'tax-doc-chip-btn';
  viewBtn.title = 'Preview';
  viewBtn.setAttribute('aria-label', 'Preview');
  viewBtn.innerHTML = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>';

  const delBtn = document.createElement('button');
  delBtn.type = 'button';
  delBtn.className = 'tax-doc-chip-btn tax-doc-chip-del';
  delBtn.title = 'Delete permanently';
  delBtn.setAttribute('aria-label', 'Delete');
  delBtn.innerHTML = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m-9 0v14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V6"/></svg>';

  actions.appendChild(viewBtn);
  if (episodic) {
    const editBtn = document.createElement('button');
    editBtn.type = 'button';
    editBtn.className = 'tax-doc-chip-btn';
    editBtn.title = 'Edit memory text';
    editBtn.setAttribute('aria-label', 'Edit memory text');
    editBtn.innerHTML = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/></svg>';
    editBtn.addEventListener('click', (ev) => { ev.stopPropagation(); startInlineMemoryEdit(chip, doc); });
    actions.appendChild(editBtn);
  }
  actions.appendChild(delBtn);

  chip.appendChild(icon);
  chip.appendChild(titleEl);
  chip.appendChild(actions);

  // click on title or icon → open preview (toggleable)
  const openPreview = (ev) => { ev.stopPropagation(); togglePopoverDocPreview(chip, doc); };
  titleEl.addEventListener('click', openPreview);
  icon.addEventListener('click', openPreview);
  viewBtn.addEventListener('click', openPreview);

  delBtn.addEventListener('click', (ev) => {
    ev.stopPropagation();
    confirmDeleteDoc(doc);
  });

  if (!episodic) {
    chip.addEventListener('dragstart', (ev) => {
      ev.dataTransfer.setData('text/x-tax-doc', String(doc.doc_id));
      ev.dataTransfer.effectAllowed = 'move';
      chip.classList.add('dragging');
    });
    chip.addEventListener('dragend', () => chip.classList.remove('dragging'));
  }
  return chip;
}

// Opens the right-side preview panel. Replaces the old in-popover drawer
// which was hard to read inside the cramped popover footprint.
async function togglePopoverDocPreview(chip, doc) {
  // Toggle: clicking the chip whose preview is currently open closes it.
  const panel = document.getElementById('doc-panel');
  if (panel && panel.dataset.docId === String(doc.doc_id) && panel.getAttribute('aria-hidden') === 'false') {
    closeDocPanel();
    return;
  }
  await openDocPanel(doc);
}

// Fetch all chunks for a doc and replace the preview with the full text.
// Each chunk is rendered as a card so section headers + chunk_index stay
// visible while scrolling through long documents.
async function loadAllChunks(docId, bodyEl, triggerBtn) {
  if (triggerBtn) {
    triggerBtn.disabled = true;
    triggerBtn.textContent = 'Loading…';
  }
  try {
    const r = await fetch('/api/documents/' + encodeURIComponent(docId) + '/chunks');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    bodyEl.innerHTML = '';
    const header = document.createElement('div');
    header.className = 'doc-panel-chunks-header';
    header.textContent = `${data.n_chunks} chunks`;
    bodyEl.appendChild(header);
    for (const c of data.chunks) {
      const card = document.createElement('div');
      card.className = 'doc-panel-chunk';
      const sectionLine = c.section || c.section_title || '';
      card.innerHTML =
        `<div class="doc-panel-chunk-head">` +
          `<span class="doc-panel-chunk-idx">#${c.chunk_index}</span>` +
          (sectionLine ? `<span class="doc-panel-chunk-section">${escapeHTML(sectionLine)}</span>` : '') +
          (c.token_count ? `<span class="doc-panel-chunk-tok">${c.token_count} tok</span>` : '') +
        `</div>` +
        `<div class="doc-panel-chunk-text"></div>`;
      card.querySelector('.doc-panel-chunk-text').textContent = c.text;
      bodyEl.appendChild(card);
    }
  } catch (e) {
    if (triggerBtn) {
      triggerBtn.disabled = false;
      triggerBtn.textContent = 'View all chunks';
    }
    flashToast('failed to load chunks: ' + (e.message || e));
  }
}

function closeDocPanel() {
  const panel = document.getElementById('doc-panel');
  if (!panel) return;
  panel.setAttribute('aria-hidden', 'true');
  panel.dataset.docId = '';
}

async function openDocPanel(doc) {
  const panel    = document.getElementById('doc-panel');
  const iconEl   = document.getElementById('doc-panel-icon');
  const titleEl  = document.getElementById('doc-panel-title');
  const metaEl   = document.getElementById('doc-panel-meta');
  const bodyEl   = document.getElementById('doc-panel-body');
  const footEl   = document.getElementById('doc-panel-foot');
  const delBtn   = document.getElementById('doc-panel-delete');
  if (!panel) return;

  const episodic = isEpisodicDocId(doc.doc_id);
  panel.dataset.docId = String(doc.doc_id);
  panel.setAttribute('aria-hidden', 'false');
  iconEl.textContent = episodic ? '🧠' : '📄';
  titleEl.textContent = doc.title || doc.doc_id;
  metaEl.innerHTML = '';
  bodyEl.innerHTML = '<div class="doc-panel-placeholder">loading…</div>';
  footEl.hidden = true;

  try {
    let metaRows = [];
    let preview = '';
    let nChunks = 0;
    if (episodic) {
      // /api/memories returns memory_id WITH the "episodic:<user>:" prefix —
      // compare against the full doc.doc_id, no slicing.
      const r = await fetch('/api/memories?limit=2000');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const items = await r.json();
      const m = items.find(x => x.memory_id === doc.doc_id);
      if (!m) {
        // Orphaned assignment — the memory was deleted but the taxonomy row
        // was left behind. Offer a one-click cleanup.
        titleEl.textContent = doc.title || doc.doc_id;
        metaRows.push(['kind', 'episodic memory']);
        metaRows.push(['status', 'orphaned — memory has been deleted']);
        metaEl.innerHTML = metaRows.map(([k, v]) =>
          `<div class="row"><span class="k">${escapeHTML(k)}</span><span class="v">${escapeHTML(v)}</span></div>`
        ).join('');
        bodyEl.innerHTML = '<div class="doc-panel-placeholder" style="color:var(--bad)">' +
          'This memory was deleted but its taxonomy assignment still exists.<br>' +
          'Use the Delete button below to remove the stale entry.</div>';
        footEl.hidden = false;
        document.getElementById('doc-panel-delete').replaceWith(
          document.getElementById('doc-panel-delete').cloneNode(true));
        document.getElementById('doc-panel-delete').addEventListener('click', async () => {
          // Remove the orphan assignment directly via move-doc to /dev/null
          // effect: we hit the memory DELETE which is idempotent and also
          // removes the assignment row.
          await fetch('/api/memories/' + encodeURIComponent(doc.doc_id), { method: 'DELETE' });
          flashToast('orphaned assignment cleaned up');
          closeDocPanel();
          await loadTaxonomyTree();
          if (_popoverNodeId) {
            try {
              const dr = await fetch(`/api/taxonomy/nodes/${encodeURIComponent(_popoverNodeId)}/docs`);
              if (dr.ok) { _popoverDocs = await parseDocsResponse(dr); renderPopoverDocs(); }
            } catch {}
          }
        });
        return;
      }
      titleEl.textContent = m.title || '(untitled memory)';
      metaRows.push(['kind', 'episodic memory']);
      metaRows.push(['id', doc.doc_id]);
      preview = m.text || '';
    } else {
      const r = await fetch('/api/documents/' + encodeURIComponent(doc.doc_id));
      if (r.status === 404) throw new Error('backend missing GET /api/documents/{id}');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const d = await r.json();
      titleEl.textContent = d.title || doc.doc_id;
      if (d.source_path) metaRows.push(['source', d.source_path]);
      if (d.source_type) metaRows.push(['type', d.source_type]);
      if (d.n_chunks != null) { metaRows.push(['chunks', String(d.n_chunks)]); nChunks = d.n_chunks; }
      if (d.ingested_at) metaRows.push(['ingested', d.ingested_at]);
      preview = d.preview || '';
    }
    metaEl.innerHTML = metaRows.map(([k, v]) =>
      `<div class="row"><span class="k">${escapeHTML(k)}</span><span class="v">${escapeHTML(v)}</span></div>`
    ).join('');
    // Render body: preview text + "View all N chunks" button when more exist.
    bodyEl.innerHTML = '';
    const pre = document.createElement('div');
    pre.className = 'doc-panel-preview';
    pre.textContent = preview || '(no preview available)';
    bodyEl.appendChild(pre);
    if (!episodic && nChunks > 3) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'doc-panel-more';
      btn.textContent = `View all ${nChunks} chunks`;
      btn.addEventListener('click', () => loadAllChunks(doc.doc_id, bodyEl, btn));
      bodyEl.appendChild(btn);
    }
    footEl.hidden = false;
    // Wire delete button. Single-shot listener so we don't stack handlers
    // when the panel is reused for a different doc.
    const onDel = () => {
      delBtn.removeEventListener('click', onDel);
      confirmDeleteDoc(doc).then(closeDocPanel);
    };
    delBtn.replaceWith(delBtn.cloneNode(true));
    document.getElementById('doc-panel-delete').addEventListener('click', () => {
      confirmDeleteDoc(doc).then(closeDocPanel);
    });
  } catch (e) {
    bodyEl.innerHTML = `<div class="doc-panel-placeholder" style="color:var(--bad)">preview failed: ${escapeHTML(e.message || String(e))}</div>`;
  }
}

function startInlineMemoryEdit(chip, doc) {
  if (!isEpisodicDocId(doc.doc_id)) return;
  const existing = chip.nextElementSibling;
  if (existing && existing.classList.contains('tax-doc-edit') && existing.dataset.docId === String(doc.doc_id)) {
    existing.remove();
    return;
  }
  taxPopDocs.querySelectorAll('.tax-doc-edit, .tax-doc-preview').forEach(el => el.remove());

  const memId = doc.doc_id.slice('episodic:'.length);
  const wrap = document.createElement('div');
  wrap.className = 'tax-doc-edit';
  wrap.dataset.docId = String(doc.doc_id);
  wrap.innerHTML = `
    <div class="tax-doc-edit-loading">loading…</div>
  `;
  chip.insertAdjacentElement('afterend', wrap);

  (async () => {
    let text = '';
    let title = doc.title || '';
    try {
      const r = await fetch('/api/memories?limit=1000');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const items = await r.json();
      const m = items.find(x => x.memory_id === memId);
      if (!m) throw new Error('memory not found');
      text = m.text || '';
      title = m.title || title;
    } catch (e) {
      wrap.innerHTML = `<div class="tax-doc-preview-err">load failed: ${escapeHTML(e.message || String(e))}</div>`;
      return;
    }
    wrap.innerHTML = `
      <textarea class="tax-doc-edit-area" rows="4"></textarea>
      <div class="tax-doc-edit-actions">
        <button type="button" class="tax-mini-btn tax-doc-edit-cancel">Cancel</button>
        <button type="button" class="tax-mini-btn tax-doc-edit-save">Save</button>
      </div>
    `;
    const ta = wrap.querySelector('.tax-doc-edit-area');
    ta.value = text;
    ta.focus();
    wrap.querySelector('.tax-doc-edit-cancel').addEventListener('click', () => wrap.remove());
    wrap.querySelector('.tax-doc-edit-save').addEventListener('click', async () => {
      const newText = ta.value.trim();
      if (!newText) { flashToast('memory text required'); return; }
      const saveBtn = wrap.querySelector('.tax-doc-edit-save');
      saveBtn.disabled = true; saveBtn.textContent = 'Saving…';
      try {
        const r = await fetch('/api/memories/' + encodeURIComponent(memId), {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: newText, title: title || null }),
        });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const body = await r.json().catch(() => ({}));
        // PUT may reissue memory_id; refresh popover docs.
        flashToast('memory saved');
        if (_popoverNodeId) {
          try {
            const dr = await fetch(`/api/taxonomy/nodes/${encodeURIComponent(_popoverNodeId)}/docs`);
            if (dr.ok) { _popoverDocs = await parseDocsResponse(dr); renderPopoverDocs(); }
          } catch {}
        }
        refreshSidebarCounts();
      } catch (e) {
        saveBtn.disabled = false; saveBtn.textContent = 'Save';
        flashToast('save failed: ' + (e.message || e));
      }
    });
  })();
}

async function confirmDeleteDoc(doc) {
  const title = doc.title || doc.doc_id;
  const ok = window.confirm(`Permanently delete "${title}"? This removes its chunks, embeddings, and KG entries.`);
  if (!ok) return;
  const episodic = isEpisodicDocId(doc.doc_id);
  try {
    let url;
    if (episodic) {
      const memId = doc.doc_id.slice('episodic:'.length);
      url = '/api/memories/' + encodeURIComponent(memId);
    } else {
      url = '/api/documents/' + encodeURIComponent(doc.doc_id);
    }
    const r = await fetch(url, { method: 'DELETE' });
    if (r.status === 404 && !episodic) {
      flashToast('backend missing — DELETE /api/documents/{id} not implemented yet');
      return;
    }
    if (r.status === 400 && episodic) {
      flashToast('cannot delete episodic via documents endpoint');
      return;
    }
    if (!r.ok) throw new Error('HTTP ' + r.status);
    flashToast('deleted "' + truncateText(title, 30) + '"');
    // refresh popover docs + tree counts
    if (_popoverNodeId) {
      try {
        const dr = await fetch(`/api/taxonomy/nodes/${encodeURIComponent(_popoverNodeId)}/docs`);
        if (dr.ok) { _popoverDocs = await parseDocsResponse(dr); renderPopoverDocs(); }
      } catch {}
    }
    await loadTaxonomyTree();
    refreshSidebarCounts();
  } catch (e) {
    flashToast('delete failed: ' + (e.message || e));
  }
}

async function togglePopoverPicker(picker, btn) {
  if (!picker.hidden) {
    picker.hidden = true;
    btn.classList.remove('active');
    return;
  }
  btn.classList.add('active');
  picker.hidden = false;
  picker.innerHTML = '<div class="tax-pop-empty">loading…</div>';

  // Build node_id → label map for "currently at" display
  const nodeLabel = (id) => {
    if (!id) return 'unfiled';
    const n = taxState.nodeMap.get(id);
    return (n && n.data && n.data.label) || id;
  };

  try {
    const r = await fetch('/api/docs');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const items = await r.json();
    // Filter: exclude docs already at this node; sort by ingested_at desc
    const filtered = (Array.isArray(items) ? items : []).filter(d => {
      const nid = d.node_id || null;
      return nid !== _popoverNodeId;
    });
    filtered.sort((a, b) => String(b.ingested_at || '').localeCompare(String(a.ingested_at || '')));
    if (!filtered.length) {
      picker.innerHTML = '<div class="tax-pop-empty">No other docs to add.</div>';
      return;
    }
    picker.innerHTML = '';
    const heading = document.createElement('div');
    heading.className = 'tax-pop-picker-head';
    heading.textContent = 'Add an existing doc/memory to this node';
    picker.appendChild(heading);
    for (const d of filtered) {
      const row = document.createElement('div');
      row.className = 'tax-pop-picker-row';
      const episodic = d.source_type === 'episodic' || isEpisodicDocId(d.doc_id);
      row.innerHTML = `
        <span class="tax-pop-picker-icon">${episodic ? '🧠' : '📄'}</span>
        <span class="tax-pop-picker-title">${escapeHTML(truncateText(d.title || d.doc_id, 44))}</span>
        <span class="tax-pop-picker-at">currently at: ${escapeHTML(nodeLabel(d.node_id))}</span>
      `;
      row.addEventListener('click', async () => {
        row.classList.add('busy');
        try {
          const rr = await fetch('/api/taxonomy/move-doc', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ doc_id: d.doc_id, node_id: _popoverNodeId }),
          });
          if (!rr.ok) throw new Error('HTTP ' + rr.status);
          flashToast('added · ' + truncateText(d.title || d.doc_id, 30));
          // refresh popover docs + tree
          if (_popoverNodeId) {
            const dr = await fetch(`/api/taxonomy/nodes/${encodeURIComponent(_popoverNodeId)}/docs`);
            if (dr.ok) { _popoverDocs = await parseDocsResponse(dr); renderPopoverDocs(); }
          }
          await loadTaxonomyTree();
        } catch (e) {
          row.classList.remove('busy');
          flashToast('add failed: ' + (e.message || e));
        }
      });
      picker.appendChild(row);
    }
  } catch (e) {
    picker.innerHTML = `<div class="tax-pop-empty">failed: ${escapeHTML(e.message || String(e))}</div>`;
  }
}

// Apply edits to label/description on blur
function attachPopoverEditors() {
  const saveLabel = async () => {
    if (!_popoverNodeId) return;
    const node = taxState.nodeMap.get(_popoverNodeId);
    if (!node) return;
    const v = taxPopLabel.value.trim();
    if (!v || v === (node.data.label || '')) return;
    try {
      const r = await fetch(`/api/taxonomy/nodes/${encodeURIComponent(_popoverNodeId)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: v }),
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      node.data.label = v;
      flashToast('renamed');
      // partial repaint: just the visible card text
      d3.select(taxNodesLayer).selectAll('g.tax-node')
        .filter(d => d.data.id === _popoverNodeId)
        .select('text.tax-card-label').text(truncateText(v, 22));
    } catch (e) { flashToast('rename failed: ' + e.message); }
  };
  const saveDesc = async () => {
    if (!_popoverNodeId) return;
    const node = taxState.nodeMap.get(_popoverNodeId);
    if (!node) return;
    const v = taxPopDesc.value;
    if (v === (node.data.description || '')) return;
    try {
      const r = await fetch(`/api/taxonomy/nodes/${encodeURIComponent(_popoverNodeId)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: v }),
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      node.data.description = v;
      flashToast('description saved');
    } catch (e) { flashToast('save failed: ' + e.message); }
  };
  taxPopLabel.addEventListener('blur', saveLabel);
  taxPopLabel.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); taxPopLabel.blur(); }
  });
  taxPopDesc.addEventListener('blur', saveDesc);
}

if (taxPopClose) taxPopClose.addEventListener('click', () => { taxPopover.hidden = true; });
if (taxPopAdd) taxPopAdd.addEventListener('click', async () => {
  if (!_popoverNodeId) return;
  const label = prompt('New child label:');
  if (!label || !label.trim()) return;
  try {
    const r = await fetch('/api/taxonomy/nodes', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: label.trim(), parent_id: _popoverNodeId, description: '' }),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    flashToast('child created');
    taxPopover.hidden = true;
    await loadTaxonomyTree();
  } catch (e) { flashToast('create failed: ' + e.message); }
});
if (taxPopDel) taxPopDel.addEventListener('click', async () => {
  if (!_popoverNodeId) return;
  const node = taxState.nodeMap.get(_popoverNodeId);
  const hasChildren = !!(node?.children?.length);
  const msg = hasChildren
    ? 'Delete this node? Its children will be reparented to its parent.'
    : 'Delete this node?';
  if (!confirm(msg)) return;
  try {
    let url = `/api/taxonomy/nodes/${encodeURIComponent(_popoverNodeId)}`;
    if (hasChildren && node.parent?.data?.id) {
      url += `?reparent_children_to=${encodeURIComponent(node.parent.data.id)}`;
    }
    const r = await fetch(url, { method: 'DELETE' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    flashToast('deleted');
    taxPopover.hidden = true;
    await loadTaxonomyTree();
  } catch (e) { flashToast('delete failed: ' + e.message); }
});

attachPopoverEditors();

// Dismiss popover when clicking outside it (but not when clicking another node)
document.addEventListener('mousedown', (e) => {
  if (document.body.dataset.view !== 'taxonomy') return;
  if (taxPopover.hidden) return;
  if (taxPopover.contains(e.target)) return;
  if (e.target.closest && e.target.closest('g.tax-node')) return;
  taxPopover.hidden = true;
});

// ---- doc chip → node drop ----
// node cards accept doc chips via native HTML5 drag/drop
document.addEventListener('dragover', (e) => {
  if (document.body.dataset.view !== 'taxonomy') return;
  const g = e.target.closest && e.target.closest('g.tax-node');
  if (!g) return;
  if (!e.dataTransfer.types.includes('text/x-tax-doc')) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  g.classList.add('doc-drop-hover');
});
document.addEventListener('dragleave', (e) => {
  const g = e.target.closest && e.target.closest('g.tax-node');
  if (g) g.classList.remove('doc-drop-hover');
});
document.addEventListener('drop', async (e) => {
  if (document.body.dataset.view !== 'taxonomy') return;
  const g = e.target.closest && e.target.closest('g.tax-node');
  if (!g) return;
  const docId = e.dataTransfer.getData('text/x-tax-doc');
  if (!docId) return;
  e.preventDefault();
  g.classList.remove('doc-drop-hover');
  const nodeId = g.__data__?.data?.id;
  if (!nodeId) return;
  try {
    const r = await fetch('/api/taxonomy/move-doc', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id: docId, node_id: nodeId }),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    flashToast('moved doc');
    // refresh popover docs if open
    if (_popoverNodeId) {
      try {
        const dr = await fetch(`/api/taxonomy/nodes/${encodeURIComponent(_popoverNodeId)}/docs`);
        if (dr.ok) { _popoverDocs = await parseDocsResponse(dr); renderPopoverDocs(); }
      } catch {}
    }
    await loadTaxonomyTree();
  } catch (err) {
    flashToast('move-doc failed: ' + err.message);
  }
});

// ---- options drawer ----
function openTaxOptions() {
  taxState.optionsOpen = true;
  taxOptionsPanel.setAttribute('aria-hidden', 'false');
}
function closeTaxOptions() {
  taxState.optionsOpen = false;
  taxOptionsPanel.setAttribute('aria-hidden', 'true');
}
if (taxOptionsBtn) taxOptionsBtn.addEventListener('click', () => {
  if (taxState.optionsOpen) closeTaxOptions(); else openTaxOptions();
});
if (taxOptionsCloseBtn) taxOptionsCloseBtn.addEventListener('click', closeTaxOptions);

function syncTaxOptionsFromConfig() {
  const t = state.config?.taxonomy || {};
  const set = (el, v) => {
    if (!el || document.activeElement === el) return;
    if (v != null && v !== '') el.value = String(v);
  };
  set(taxOptInputs.beam_width, t.beam_width);
  set(taxOptInputs.max_depth, t.max_depth);
  set(taxOptInputs.propose_sample_size, t.propose_sample_size);
  set(taxOptInputs.max_children_per_node, t.max_children_per_node);
  set(taxOptInputs.min_top_score_floor, t.min_top_score_floor);
  set(taxOptInputs.max_docs_pct, t.max_docs_pct);
}

async function saveTaxOption(key, raw) {
  let v;
  if (key === 'min_top_score_floor' || key === 'max_docs_pct') {
    v = parseFloat(raw);
  } else {
    v = parseInt(raw, 10);
  }
  if (!Number.isFinite(v)) return;
  try {
    const r = await fetch('/api/taxonomy/options', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: v }),
    });
    if (!r.ok) {
      if (r.status === 404) {
        taxOptHint.textContent = 'Backend missing /api/taxonomy/options — option saved client-side only.';
        taxOptHint.classList.add('warn');
        return;
      }
      throw new Error('HTTP ' + r.status);
    }
    const body = await r.json().catch(() => null);
    if (body && body.taxonomy) {
      state.config = { ...(state.config || {}), taxonomy: body.taxonomy };
    }
    taxOptHint.textContent = `saved · ${key} = ${v}`;
    taxOptHint.classList.remove('warn');
  } catch (e) {
    taxOptHint.textContent = `save failed: ${e.message}`;
    taxOptHint.classList.add('warn');
  }
}

for (const [key, el] of Object.entries(taxOptInputs)) {
  if (!el) continue;
  el.addEventListener('blur', () => saveTaxOption(key, el.value));
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); el.blur(); }
  });
}

// ---- Recompute modal & SSE ----
function openTaxModal(mode = 'recompute') {
  taxModal.hidden = false;
  taxModalScrim.hidden = false;
  taxModalCloseBtn.hidden = true;
  taxModalSummary.hidden = true;
  taxModalError.hidden = true;
  taxModalSummary.textContent = '';
  taxModalError.textContent = '';
  taxModalTitle.textContent = mode === 'assign'
    ? 'Assigning unfiled documents…'
    : 'Recomputing taxonomy…';
  // Reset stage states
  for (const li of taxStagesEl.querySelectorAll('.tax-stage')) {
    li.classList.remove('stage-active', 'stage-done', 'stage-skipped');
    const detail = li.querySelector('[data-role="detail"]');
    if (detail) detail.textContent = '';
    const barInner = li.querySelector('[data-role="bar"] > span');
    if (barInner) barInner.style.width = '0%';
  }
  // In assign mode, mark the doc-summary etc as skipped immediately
  if (mode === 'assign') {
    for (const s of ['doc_summary','embed_centroids','propose_tree','materialize']) {
      const li = taxStagesEl.querySelector(`.tax-stage[data-stage="${s}"]`);
      if (li) li.classList.add('stage-skipped');
    }
  }
  // Reset and start the live elapsed/ETA tracker.
  taxState.tickStartedAt = Date.now();
  taxState.tickLastI = 0;
  taxState.tickLastN = 0;
  if (taxState.tickInterval) clearInterval(taxState.tickInterval);
  paintTaxTick();
  taxState.tickInterval = setInterval(paintTaxTick, 500);
}

function closeTaxModal() {
  if (taxState.activeStream) {
    try { taxState.activeStream.close(); } catch {}
    taxState.activeStream = null;
  }
  if (taxState.tickInterval) {
    clearInterval(taxState.tickInterval);
    taxState.tickInterval = null;
  }
  taxModal.hidden = true;
  taxModalScrim.hidden = true;
}

function paintTaxTick() {
  const tickEl = document.getElementById('tax-modal-tick');
  if (!tickEl || !taxState.tickStartedAt) return;
  const elapsed = (Date.now() - taxState.tickStartedAt) / 1000;
  const i = taxState.tickLastI || 0;
  const n = taxState.tickLastN || 0;
  let etaStr = '';
  if (i > 0 && n > 0 && i < n) {
    const perItem = elapsed / i;
    const etaSec = perItem * (n - i);
    etaStr = ' · ETA ' + fmtDuration(etaSec);
  }
  tickEl.textContent = 'elapsed ' + fmtDuration(elapsed) + etaStr;
}

function fmtDuration(secs) {
  if (!Number.isFinite(secs) || secs < 0) return '–';
  if (secs < 60) return secs.toFixed(secs < 10 ? 1 : 0) + 's';
  const m = Math.floor(secs / 60);
  const s = Math.round(secs - m * 60);
  return m + 'm ' + s + 's';
}

if (taxModalCloseBtn) taxModalCloseBtn.addEventListener('click', closeTaxModal);
if (taxModalScrim) taxModalScrim.addEventListener('click', () => {
  if (!taxModalCloseBtn.hidden) closeTaxModal();
});

function markStage(name, status, detail) {
  const li = taxStagesEl.querySelector(`.tax-stage[data-stage="${name}"]`);
  if (!li) return;
  li.classList.remove('stage-active', 'stage-done');
  if (status === 'active') li.classList.add('stage-active');
  else if (status === 'done') li.classList.add('stage-done');
  if (detail != null) {
    const d = li.querySelector('[data-role="detail"]');
    if (d) d.textContent = String(detail);
  }
}

function streamRecompute(endpoint, mode) {
  openTaxModal(mode);
  if (taxState.activeStream) { try { taxState.activeStream.close(); } catch {} }
  // Use fetch + ReadableStream to support POST + SSE (EventSource is GET-only)
  const ctrl = new AbortController();
  taxState.activeStream = { close: () => ctrl.abort() };

  (async () => {
    try {
      const r = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Accept': 'text/event-stream' },
        signal: ctrl.signal,
      });
      if (!r.ok || !r.body) throw new Error('HTTP ' + r.status);
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // SSE frames are separated by blank lines
        let idx;
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          handleSSEFrame(frame, mode);
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') return;
      console.error('recompute stream error', e);
      showStageError(e.message || String(e));
    }
  })();
}

function handleSSEFrame(frame, mode) {
  // Each frame: `event: NAME\ndata: JSON`
  const lines = frame.split('\n');
  let eventName = 'message';
  let dataStr = '';
  for (const line of lines) {
    if (line.startsWith('event:')) eventName = line.slice(6).trim();
    else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
  }
  if (eventName !== 'stage' && eventName !== 'message') return;
  let data = {};
  try { data = JSON.parse(dataStr || '{}'); } catch { data = {}; }
  const stage = data.stage || data.name || eventName;
  applyStageEvent(stage, data, mode);
}

function applyStageEvent(stage, data, mode) {
  switch (stage) {
    case 'start':
      // overall start — no-op (modal already open)
      break;
    case 'doc_summary': {
      markStage('doc_summary', 'active');
      const i = data.i, n = data.n;
      const li = taxStagesEl.querySelector('.tax-stage[data-stage="doc_summary"]');
      if (li && i != null && n) {
        const d = li.querySelector('[data-role="detail"]');
        if (d) d.textContent = `${i}/${n} · ${truncateText(data.title || data.doc_id || '', 36)}`;
        const bar = li.querySelector('[data-role="bar"] > span');
        if (bar) bar.style.width = ((i / n) * 100).toFixed(1) + '%';
        // Feed the live elapsed/ETA tracker.
        taxState.tickLastI = i;
        taxState.tickLastN = n;
        paintTaxTick();
      }
      break;
    }
    case 'summaries_done':
      markStage('doc_summary', 'done', data.n != null ? `${data.n} summarised` : null);
      break;
    case 'embed_centroids_start':
      markStage('embed_centroids', 'active', data.n != null ? `${data.n} vectors` : null);
      break;
    case 'embed_centroids_done':
      markStage('embed_centroids', 'done', data.duration_s != null ? `${data.duration_s.toFixed(1)}s` : null);
      break;
    case 'propose_tree_start':
      markStage('propose_tree', 'active', 'LLM thinking…');
      break;
    case 'propose_tree_done':
      markStage('propose_tree', 'done',
        data.n_nodes != null ? `${data.n_nodes} nodes proposed` : null);
      break;
    case 'materialize_start':
      markStage('materialize', 'active');
      break;
    case 'materialize_done':
      markStage('materialize', 'done',
        data.n_nodes != null ? `${data.n_nodes} nodes written` : null);
      break;
    case 'assign_docs_start':
      markStage('assign_docs', 'active', data.n != null ? `${data.n} docs` : null);
      break;
    case 'assign_docs_done':
      markStage('assign_docs', 'done',
        data.n_assigned != null ? `${data.n_assigned} assigned` : null);
      break;
    case 'done': {
      // mark every stage that hasn't been touched as done
      for (const s of ['doc_summary','embed_centroids','propose_tree','materialize','assign_docs']) {
        const li = taxStagesEl.querySelector(`.tax-stage[data-stage="${s}"]`);
        if (!li) continue;
        if (li.classList.contains('stage-skipped')) continue;
        if (!li.classList.contains('stage-done')) li.classList.add('stage-done');
        li.classList.remove('stage-active');
      }
      const dur = data.total_duration_s;
      const nn = data.n_nodes;
      const nd = data.n_docs_assigned;
      const parts = [];
      if (nn != null) parts.push(`${nn} nodes`);
      if (nd != null) parts.push(`${nd} docs assigned`);
      if (dur != null) parts.push(`${Number(dur).toFixed(1)}s`);
      taxModalTitle.textContent = mode === 'assign' ? 'Assigned' : 'Done';
      taxModalSummary.textContent = parts.join(' · ');
      taxModalSummary.hidden = false;
      taxModalCloseBtn.hidden = false;
      // refresh the tree behind the modal
      loadTaxonomyTree();
      break;
    }
    case 'error':
      showStageError(data.message || data.error || 'unknown error');
      break;
    default:
      // ignore unknown
      break;
  }
}

function showStageError(msg) {
  // mark active stage as failed
  const active = taxStagesEl.querySelector('.tax-stage.stage-active');
  if (active) active.classList.remove('stage-active');
  taxModalError.textContent = msg;
  taxModalError.hidden = false;
  taxModalCloseBtn.hidden = false;
  taxModalTitle.textContent = 'Failed';
}

if (taxRecomputeBtn) {
  taxRecomputeBtn.addEventListener('click', () => streamRecompute('/api/taxonomy/recompute', 'recompute'));
}
if (taxEmptyCta) {
  taxEmptyCta.addEventListener('click', () => streamRecompute('/api/taxonomy/recompute', 'recompute'));
}
if (taxAssignBtn) {
  taxAssignBtn.addEventListener('click', () => {
    if (taxAssignBtn.disabled) return;
    streamRecompute('/api/taxonomy/assign-unfiled', 'assign');
  });
}

// (data-id attributes are set inside renderTaxonomy via the merged.attr below)
