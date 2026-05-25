# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Visibility — every long-running process must show progress

User invariant: any process expected to take more than ~10 seconds must surface progress in real time. No silent waits.

- **Foreground long commands** — wrap with a `rich.progress.Progress` bar driven by a known count (chunks, docs, communities, benchmark questions). When the count isn't knowable upfront, emit a per-item line with a running tally (`[i/n] processing X...`) and flush stdout (`print(..., flush=True)`).
- **Background commands** (Bash `run_in_background: true`) — set up a `Monitor` over the log file with `grep --line-buffered` filtering for per-item completion + failure signatures. Cover failure cases (Traceback, UnicodeError, "killed", non-zero exit), not only the happy path.
- **Subagents** — every subagent prompt must require a final structured report. Long-running agent work the user is waiting on should be foreground, or background with a Monitor on its output file.
- **Existing CLI commands lacking progress** are a bug. Fix the source. Per-item prints with `flush=True` are the floor; a Rich progress bar is the target.
- **Don't ask "should I proceed?" on long jobs without first showing a progress channel.** Pattern: arm the monitor → start the job → tell the user what to expect → the job streams.

This rule supersedes any agent-prompt template that omits progress hooks.

## Subagent dispatch policy

For any request that decomposes into 2+ work items, dispatch to subagents. Pick model per item by importance × difficulty:

- **Opus** — algorithmically tricky, security-sensitive, cross-cutting wiring, novel design with multiple invariants. Examples: graph store with synonym merging, MST + redundancy pruning, query router with RRF fusion, prompt design.
- **Sonnet** — well-scoped mechanical work, single-file modules following an existing pattern, test-only additions, config/schema/dep edits, factory registrations.

Fan out in waves: agents on disjoint files run in parallel (single message, multiple `Agent` calls); only serialize when later work depends on an earlier deliverable. After each wave, the main thread verifies — `pytest`, ruff, schema sanity. Don't shard work smaller than ~one file per agent. Trivial single edits: just do it inline.

## Project status (phase-by-phase summary)

**Phase 1** — walking-skeleton RAG (ingest → vector → rerank → answer). Complete.

**Phase 2** (KG triple extraction + PPR + GraphRAG communities + LLM router + KG2RAG MST organizer) — complete, 381 tests. Behind `kg.enabled`. Modules: `kg/builder.py` (TripleExtractor), `kg/store.py` (NetworkX MultiDiGraph, synonym merge cos≥0.8, SQLite mirror), `kg/ner.py` (SpacyNER default, LLMNER opt-in), `kg/ppr.py` (scipy power iteration), `kg/communities.py` (Leiden at 3 resolutions, Chroma `hrag_community_summaries` + SQLite mirror), `retrieval/{kg_ppr,community,router,mst}.py`.

**Phase 3 — Personalization** (memory layer + LLM-proposed taxonomy) — complete, 537 tests. Default retriever is `taxonomy` (falls back to vector when tree empty).
- Memory: `memory/profile.py` (ProfileStore — rendered as `{user_profile}` into every answer prompt), `memory/store.py` (EpisodicMemoryStore, `/remember` `/recall` `/forget`, episodic chunks live in `chunks` with `source_type='episodic'`), `memory/{auto_extract,extractor}.py`, `context/builder.py`.
- Taxonomy: `taxonomy/store.py` (packed-float32 centroids, beam descend), `taxonomy/builder.py` (parallel summaries → LLM tree with explicit `max_tokens=8192` + truncation-salvage), `taxonomy/assigner.py` (cosine descent + LLM tiebreak when top-2 < 0.05 apart), `retrieval/taxonomy.py` (exposes `describe_last_descend()`), tables `kg_taxonomy_{nodes,assignments,doc_meta}`, flag `taxonomy.include_episodic` (default true).

**Phase 4** (compaction & gating: RAGate, clue, dialog MST, `[UNCERTAIN]` masking) — complete, 691+ tests. All four behind `compaction.*` flags, default OFF.
- Modules: `gating/{gate,clue,uncertain}.py`, `context/dialog_mst.py`, `CompactionConfig`. `prompts/answer.md` Step 4 writes `[UNCERTAIN]` after unsupported sub-claims.
- Orchestrator integration order: dialog compaction (after history) → RAGate (FACTUAL only; SKIP forces `plan.scope="none"` + intent→GENERAL) → clue generation (replaces retrieval query) → `render_uncertain` / silent `strip_uncertain`.
- Progress events: `dialog_compact`, `gate_check`, `clue_generate`, `uncertain_render`.

**Phase 5** (web ergonomics + extensibility) — complete, 700 tests.
- Track A (web UX): memory CRUD `POST/PUT/DELETE /api/memories`, multipart `POST /api/ingest`, background jobs (`POST /api/ingest?background=true` → `GET /api/jobs/{id}`), `jobs` table.
- Track B (verification): `tests/benchmark/run_phase5_web.py` over TestClient.
- Track C (pluggable backends): `kg/backends/{base,networkx,neo4j}.py` (KGBackend Protocol, 18 methods), `retrieval/backends/{base,chroma,sqlite_vec}.py` (VectorBackend Protocol, 5 methods). Factories `KGStore.from_config` + `_build_vector_backend`.
- Track D (self-improvement loop): `feedback` table (`message_id, rating, note`), `POST/GET/DELETE /api/feedback`, 👍/👎 in UI, `hrag export-training-pairs` JSONL.
- Track E (equation-aware ingest): `ingest/math_detect.py` (`find_display_math_spans`, `has_math` for `$$...$$`, `\begin{equation|align|...}`, inline `$...$` / `\(...\)`), chunker boundary nudge so math never splits, `quality.py` carve-outs when `metadata.has_math`.

**Phase 6** (real backends + adaptive retrieval + Ollama warmth) — complete, 726 tests, 8/8 acceptance.
- A: `retrieval/backends/sqlite_vec.py` real impl (~390 LOC, `vec0` virtual table, `distance_metric=cosine` → Chroma-shape distances, where-compiler handles `$and`/`$or`/`$eq`/`$ne`/`$in`).
- B: `kg/backends/neo4j.py` real impl (~370 LOC, all 18 protocol methods, single `:Node`/`:LINK`, `__json_attrs` sidecar for non-primitive attrs, clean RuntimeError on missing URI/driver, zero import side effects).
- C (adaptive top_k per intent): `_adaptive_top_k` maps intent → `(top_k_vector, top_k_final)`; `(None, None)` skips retrieval (GREETING default). PERSONAL broadens `source_types` to `["document","episodic"]` and stable-sorts episodic to top. Events: `adaptive_top_k`, `retrieval_skipped`, `episodic_bias_applied`. Default off.
- D: `cfg.llm.keep_alive` threaded as top-level chat() kwarg. Default `"30m"`.

**Phase 7-A** (math handling: detector + filter + extraction) — complete, 755 tests, 5/5 acceptance. Triggered by failure: HRAG retrieved formula-free chunks for "give me some formulas hipporag uses" despite 72 chunks with Unicode math.
- Method 1: `has_unicode_math(text, min_signals=2)` over Greek/math-italic letters, operators (`∑∫∏√∞≤≥≠≈⟨⟩⊕⊗⋅`), sub/superscripts, equation density, function patterns. `has_math = _has_latex_math or has_unicode_math`. `scripts/backfill_has_math.py` tagged 63/1082 chunks.
- Method 2: `_expand_math_meta(query)` in `HeuristicRewriter` appends `"equation parameter θ Θ loss function ..."` to meta-queries. Cosine vs `𝑌=Θ(𝑞|𝜃)` jumps from ~0.10 to ~0.35.
- Method 3: `_is_math_meta_query` (pure regex) + `where={"has_math": True}` filter pushdown + lowered rerank threshold (`-10.0`) + empty-result fallback. Optional formula-extraction second LLM call against `prompts/extract_formulas.md`. Events: `math_meta_filter`, `math_meta_filter_fallback`, `formula_extract`.
- Retriever Protocol widened: `where: Optional[dict] = None`. Vector-backed retrievers thread through to `VectorStore.query`; `bm25`, `kg_ppr`, `community` accept and silently ignore.

**Phase 6+7 wrap-up** (5 deferred items) — complete, 797 tests, 19/19 acceptance.
- **6-B1**: `cfg.retrieval.adaptive_retriever_per_intent` (5 intents → retriever name or `"default"`). `Orchestrator._pick_retriever_for_intent` caches; missing-dep falls back. Event `adaptive_retriever_picked`.
- **6-B2**: `feedback_stats.py::feedback_summary(db)` shared by CLI + web. CLI: `hrag feedback-stats`, `hrag feedback-export`. API: `GET /api/feedback/stats`. GUI: Feedback drawer.
- **6-B3**: `cfg.llm.num_keep: Optional[int]` threaded into `options.num_keep` (NOT top level — Ollama silently ignores there).
- **7-B**: `EmbeddingsConfig.suggested_models` (all-mpnet, specter2, jina-v2, bge-small). `dimension_for_model()` helper. CLI: `hrag embeddings-list/current`. API: `GET /api/embeddings/suggested`.
- **7-C**: `ingest/nougat_loader.py` with deferred imports (zero side effects). `_load_pdf` dispatches to Nougat when `cfg.ingest.use_nougat=True` AND dep installed; silent PyMuPDF fallback. API: `GET /api/ingest/nougat_status`.

**Phase 8** (interactive retrieval review loop) — complete, 897 tests, 5/5 acceptance. Triggered by "stars and moon" off-corpus failure — HRAG paid for retrieval+reasoning+apology when negative rerank scores already proved off-corpus.

Pause between retrieval and answer generation. Orchestrator emits `review_required` SSE with sources, clue, taxonomy descend, 0–3 rephrasings; frontend renders modal; user POSTs decision to `/api/chat/turns/{id}/resume`. All 13 sub-features (A–M) ship behind `interaction.review_enabled` (default OFF).
- `interaction/store.py` (InteractionStore, daemon reaper, idempotent `submit_decision`, `wait_for_decision`).
- `interaction/review.py` (`should_pause()` 7 trigger heuristics, `build_review_payload()`, `generate_rephrasings()`, `maybe_pause()`).
- `InteractionConfig` (13 fields).
- `messages.metadata TEXT` column via idempotent ALTER TABLE.
- Orchestrator: long-lived `InteractionStore`; `turn_id` on `start` event; pause hook between `organize_done` and prompt; action dispatcher (`continue/filter/rephrase/general/clarify/expand_doc/redescend/abort`); FACTUAL→GENERAL silent swap now gated on `action=="continue"`; follow-ups between `generate` and `done`.
- Web: `POST /api/chat/turns/{id}/resume` (pydantic Literal validation, idempotent), `POST /api/chat/turns/{id}/why_source`, SSE relays `review_required`/`review_resolved`/`followups` as dedicated event types.
- Frontend: `#review-modal` with `.review-modal[hidden] { display: none !important; }` guard; auto-promotion (continue→filter/general/rephrase based on user edits); keyboard shortcuts (Enter/Esc/1-9/E/R/G/C).
- Prompts: `rephrase.md`, `clarify.md`, `followups.md`, `why_source.md`.

**Phase 8.1** — memory recall fix: `cfg.retrieval.always_include_episodic` (default True) includes episodic memories for ALL intents, not only PERSONAL. PERSONAL stable-sort preserved on top.

**Phase 9** (speed, observability & accuracy quick wins) — complete, 1051 tests, 17 tickets behind default-off flags (9.3, 9.4, 9.11, 9.12 default ON as pure-speed wins).

- **9.1** `tests/benchmark/run_latency.py` — per-stage TTLT harness over fixed 20-Q set; markdown + JSON output.
- **9.2** `retrieval.async_preflight_enabled` — gate/clue/intent futures via ThreadPoolExecutor; mutually exclusive with combined preflight (combined wins). Event `async_preflight`.
- **9.3** `embeddings.query_cache_enabled` (ON) + `query_cache_size` — per-session LRU on `embed_one`; ambient session id via contextvar.
- **9.4** `llm.warmup_on_init` (ON) + `llm.num_keep_auto` — one-token Ollama warm-up; optional auto-tune of `num_keep` from `answer.md` prefix.
- **9.5** `llm.anthropic_prompt_caching` — wraps system + last user message in `cache_control={"type":"ephemeral"}` when ≥1024 chars; no-op otherwise.
- **9.6** `compaction.combined_preflight_enabled` — one LLM call against `prompts/combined_preflight.md` returns `{intent, gate, clue}` JSON. Re-emits per-stage events with `source="combined"`.
- **9.7** `embeddings.embed_precision` — fp32 / fp16 / onnx_int8 backends; silent fallback to fp32.
- **9.8** `retrieval.rerank_quantize` — INT8 ONNX cross-encoder via `optimum`. ~2-3× rerank throughput on CPU.
- **9.9** `retrieval.rerank_fallback_telemetry_enabled` + `rerank_fallback_events` table — logs `{turn_id, query, dropped_chunk_ids}` when zero-filter trips. Surfaced via `feedback_summary()["rerank_fallback_count"]`.
- **9.10** `retrieval.first_token_latency_enabled` — wall-clock to first streamed token; persists at `messages.metadata.latency.first_token_ms`.
- **9.11** `retrieval.router_short_circuit` (ON) — `QueryRouter` skips RRF for clearly-routed `entity`/`global` queries.
- **9.12** `kg.dedup_enabled` (ON) + `kg_canonical_triples` table — cross-chunk triple dedup keyed on `(canonical_subject, relation, canonical_object)`; `freq` counter on re-sighting.
- **9.13** `compaction.context_compression_enabled` + `context_budget_chars` — when prompt exceeds budget, drops bottom-quartile passages by rerank score + summarises oldest half of history. Emits `context_compress`.
- **9.14** `ingest.contextual_retrieval_enabled` — **Anthropic Contextual Retrieval**. One LLM call per chunk at ingest prepends a 50–100-token context line to `chunk.embedding_text` (HeteRAG fusion target — NOT `chunk.text`); cached in `ingest_context_cache` keyed by `sha256(chunk_text||doc_hash||model_name)`. Modules: `ingest/contextual.py` (ContextualAugmenter, ThreadPoolExecutor, per-chunk SQLite cache, per-item flush + 3 progress events), `prompts/contextual_chunk.md`, `IngestConfig` 4 new fields. Pipeline step 2c between quality-filter and chunk upsert; skips episodic. Events: `contextual_augment_{start,chunk,done}`. Anthropic reports ~35% reduction in retrieval failures (~67% combined w/ BM25 + reranking).
- **9.15** `retrieval.feedback_reranking_enabled` + `feedback_reranking_{weight,alpha}` — EMA(👍−👎) per chunk nudges cross-encoder score; emits `feedback_rerank_applied`.
- **9.16** `retrieval.crag_enabled` + `crag_score_floor` + `crag_retry_top_k_multiplier` — automatic CRAG-style rewrite + widened-`top_k` retry when post-rerank top score below floor AND no Phase-8 review steered the turn. Cap one retry; emits `crag_reroute`.
- **9.17** `compaction.self_rag_enabled` + `self_rag_max_spans` — scans for `[UNCERTAIN]`, runs one retrieval pass per span (max N) against preceding sentence, appends `**Sources for uncertain claims:**`. Pure-function `extract_uncertain_spans()` in `gating/uncertain.py`.

**Deferred (still):** actually running Nougat over a corpus, swapping to math-aware embedder + re-ingest, feedback-loop fine-tuning training infra, cross-turn KV-cache reuse via Ollama prefix tokens beyond `num_keep`.

## Contracts — load-bearing invariants (must survive future phases)

These are numbered globally; phase numbering shows when they were introduced.

**Phase 3 (1–8):**
1. `documents.source_type` and `chunks.source_type` ('document'|'episodic') must stay. `taxonomy.include_episodic` reads them; new source_types must teach `taxonomy/builder.py::_list_docs`.
2. Taxonomy auto-assign hook in `IngestPipeline.ingest_document` ("5c" block) keeps running for every eligible ingest. New ingest features layer in BEFORE that block.
3. `kg_taxonomy_{nodes,assignments,doc_meta}` migrations stay `CREATE TABLE IF NOT EXISTS` and backward-compatible.
4. `Retriever.retrieve(query, user_id, top_k, source_types)` signature is locked. `TaxonomyRetriever` + `/recall` depend on passing `source_types`.
5. `LLMProvider.complete(prompt, *, temperature, max_tokens)` must keep accepting explicit `max_tokens`. Taxonomy builder relies on this — Ollama silently truncates long JSON when default.
6. `EmbeddingProvider.embed(texts)` returns L2-normalised vectors. `TaxonomyStore.beam_descend` treats cosine == dot product on the hot path.
7. `progress(event, payload)` keeps emitting `taxonomy_descend` (with `describe_last_descend()` payload) whenever active retriever has `name=="taxonomy"`.
8. `TaxonomyStore.clear(user_id)` is non-destructive to `kg_taxonomy_doc_meta` by default. Pass `wipe_doc_meta=True` for full reset (CLI `hrag taxonomy clear` does NOT).

**Phase 4 (9–11):**
9. The four events `gate_check`, `clue_generate`, `dialog_compact`, `uncertain_render` must keep firing when their `compaction.*` flags are enabled. GUI Compaction expander subscribes.
10. `render_uncertain` is idempotent — applying twice must not double-render.
11. Silent `strip_uncertain` (when `mask_uncertain=False`) must keep running so raw `[UNCERTAIN]` never leaks to end users.

**Phase 6 (12–15):**
12. The three events `adaptive_top_k`, `retrieval_skipped`, `episodic_bias_applied` must fire when `retrieval.adaptive_enabled` is true. Payload shapes documented in `orchestrator.py`'s top-of-file event list.
13. `_adaptive_top_k(cfg, intent)` is a pass-through when `retrieval.adaptive_enabled` is False — returns `(cfg.retrieval.top_k_vector, cfg.retrieval.top_k_final)` unconditionally.
14. `OllamaProvider._build_chat_kwargs` must keep emitting `keep_alive` as a TOP-LEVEL kwarg, NOT inside `options`. Burying it in `options` is silently ignored by Ollama.
15. `Neo4jBackend` and `SqliteVecBackend` constructors are side-effect-free (no driver/extension import at module load). Missing-backend errors fire on first method call, not on `import hrag`.

**Phase 7-A (16–19):**
16. `Retriever.retrieve(query, user_id, top_k, source_types, intent_hint, where)` — every retriever accepts `where`, even if ignoring it. Wrappers (`doc_scope`, `router`, `hybrid`) thread through.
17. The three events `math_meta_filter`, `math_meta_filter_fallback`, `formula_extract` must fire when their flags are true.
18. `scripts/backfill_has_math.py` is idempotent — running twice must not corrupt metadata.
19. `_is_math_meta_query` is pure regex (no LLM) — runs on every turn when flag is on, must stay fast and deterministic.

**Phase 6+7 wrap-up (20–24):**
20. `cfg.retrieval.adaptive_retriever_per_intent` value is `"default"` or one of `{vector, bm25, hybrid, kg_ppr, community, router, taxonomy}`. Web API validates in POST; do not loosen.
21. `feedback_summary(db)` return shape is stable: `{thumbs_up, thumbs_down, total, sessions, top_negative}` (also `rerank_fallback_count` from 9.9). CLI + `GET /api/feedback/stats` depend on it.
22. `OllamaProvider._build_options` emits `num_keep` INSIDE `options` (NOT top-level via `_build_chat_kwargs`). Top-level is silently ignored.
23. `hrag.ingest.nougat_loader` import is side-effect-free: zero `nougat`/`nougat_ocr` modules in `sys.modules` until a method is called.
24. `EmbeddingsConfig.suggested_models` is advisory metadata only — `SentenceTransformersProvider` still accepts any HF model id.

**Phase 8 (25–32):**
25. `cfg.interaction.review_enabled=False` (default) is a true no-op: zero new SSE events, LLM calls, DB writes. 829 pre-Phase-8 tests stay byte-identical pass.
26. `InteractionStore` TTL cleanup runs on a daemon thread, never blocks the orchestrator. `store.shutdown()` is called from `Orchestrator.close()`; double-shutdown must not raise.
27. The five public SSE event types — `review_required`, `review_resolved`, `followups`, `review_warning`, plus `turn_id` on `start` — are contract. Renaming breaks the frontend.
28. `POST /api/chat/turns/{id}/resume` is idempotent on any action — retry returns `{"accepted": false, "reason": "turn_not_found_or_already_decided"}`; do not raise 4xx/5xx for retry.
29. `messages.metadata` is JSON-encoded TEXT. Readers tolerate `NULL` and malformed JSON without raising. Column stays optional.
30. `should_pause()` is pure (no LLM, no DB, no progress callback). Non-determinism here would make pause untestable.
31. SSE relay routes `review_required` / `review_resolved` / `followups` as DEDICATED SSE event types (`event: review_required`), not nested under `event: progress`.
32. `orchestrator.chat()` emits `turn_id` on the `start` event payload. Without it the frontend can't POST resume.

**Phase 8.1 (33):**
33. `cfg.retrieval.always_include_episodic=True` (default) includes episodic for ALL intents, not only PERSONAL. PERSONAL stable-sort preserved on top. False reverts to per-intent strictness. Both "full" and "episodic" branches in `Orchestrator.chat()` honour it; surfaced via GET/POST `/api/config`.

**Phase 9.14 (34–37):**
34. `cfg.ingest.contextual_retrieval_enabled=False` (default) is a true no-op: zero new LLM calls, SQL queries, prompts loaded; `ContextualAugmenter` not even instantiated.
35. `ContextualAugmenter.augment_chunks` mutates ONLY `chunk.embedding_text`. `chunk.text` is read-only inside the augmenter — KG triples, taxonomy auto-assign, and answer prompt all consume `chunk.text`; contextual prefix must never leak (Phase-3 contract 1 preserved).
36. Cache `ingest_context_cache` keyed by `sha256(chunk_text||doc_hash||model_name)`. Re-ingesting unchanged doc under same LLM model MUST be free. Changing the LLM model invalidates cleanly.
37. Episodic memories (`doc.source_type=="episodic"`) are skipped by the augmenter. Check happens before instantiation.

**Phase 9 (38–44):**
38. `cfg.compaction.combined_preflight_enabled=True` requires all of `gate_enabled`/`clue_enabled`/`intent.enabled` also True; constructor refuses to build `CombinedPreflight` otherwise (orchestrator silently falls back). Preserves per-stage event invariant.
39. When `combined_preflight_enabled=True`, orchestrator MUST still emit `intent_check`, `gate_check`, `clue_generate`. Payloads carry `source="combined"`; shapes otherwise unchanged.
40. `rerank_fallback_events` is append-only. Wrap inserts in `try/except` so a tight DB lock cannot regress the chat path.
41. `messages.metadata.latency.first_token_ms` is the agreed key path. The latency harness reads it; renaming breaks the harness.
42. `extract_uncertain_spans(answer, max_chars_per_span)` is pure. Self-RAG pass runs it on raw answer BEFORE `render_uncertain`/`strip_uncertain`.
43. Phase 9 events `combined_preflight`, `async_preflight`, `context_compress`, `crag_reroute`, `self_rag`, `feedback_rerank_applied`, `rerank_fallback_logged` are contract (whichever subset their flags activate).
44. Every Phase 9 flag defaults OFF (or to a value byte-identical to pre-Phase-9). Exceptions (intentional defaults-ON, pure-speed): `embeddings.query_cache_enabled` (9.3), `llm.warmup_on_init` (9.4), `retrieval.router_short_circuit` (9.11), `kg.dedup_enabled` (9.12).

## Manual triggers

```bash
# Phase 2 (KG router) — wipe + re-ingest:
#   in config.yaml: kg.enabled: true  AND  retrieval.retriever: router
rm -rf data/store.sqlite data/chroma data/kg
hrag init
hrag ingest "D:/Selected Dynamic Papers" --recursive
# Or, if chunks exist and you only want the KG layer:
hrag rebuild-kg

# Phase 3 — taxonomy
hrag taxonomy build              # parallel summaries → LLM tree → materialize
hrag taxonomy show               # text outline
hrag taxonomy clear              # drop tree (preserves doc-meta cache)

# Phase 4 — compaction (enable per-session or via env):
HRAG_COMPACTION__GATE_ENABLED=true HRAG_COMPACTION__CLUE_ENABLED=true \
HRAG_COMPACTION__DIALOG_MST_ENABLED=true HRAG_COMPACTION__MASK_UNCERTAIN=true \
python tests/benchmark/run_phase4.py
# Or CLI flags:
hrag chat --gate --clue --mask-uncertain
```

Acceptance: ≥3/4 questions pass for Phase 4.

## Commands

```bash
# Install (editable)
python -m pip install -e .[dev]

# Init SQLite + chroma
hrag init

# Ingest
hrag ingest <path>              # single file
hrag ingest <dir> --recursive   # whole tree

# Chat (slash: /help /sources /status /exit /remember /recall /forget /profile)
hrag chat
hrag chat --fast                            # top_k=4, no rerank
hrag chat --retriever taxonomy              # default; or vector/bm25/hybrid/kg_ppr/community/router
hrag chat --no-rerank

# Memory
hrag remember "<text>"                      # save episodic inline
hrag remember <path>                        # bulk-import .md/.txt
hrag memory list
hrag memory extract --session <id>          # mine preferences

# Streamlit GUI
hrag gui

# Other
hrag list-docs

# Tests / lint
pytest -q
pytest tests/test_orchestrator.py -q
pytest -k "rerank" -q
ruff check src tests
```

Heavy-deps tests (chromadb / sentence-transformers / ollama) skip cleanly when those deps are absent — `tests/conftest.py` provides stubs.

## Configuration

`config.yaml` is canonical; loaded into pydantic models in `src/hrag/config.py`. Overrides:

1. **Env vars** prefixed `HRAG_` with `__` as section separator: `HRAG_LLM__MODEL=gemma:7b` overrides `llm.model`.
2. **Per-session CLI flags** on `hrag chat` (`--retriever`, `--reranker`, `--no-rerank`, `--fast`).

`Config.resolve(relative_path)` resolves storage paths against `project_root` (cwd at load time).

## Architecture

Modular stack — CLI · GUI · Orchestrator · Retrievers · Personalization · Storage · Providers. Every layer behind an interface so it can be swapped (Chroma↔sqlite-vec, NetworkX↔Neo4j). README.md has the diagram.

### Pipeline (`orchestrator.py`)

`Orchestrator.chat(question, user_id, session_id, progress, stream)`:
1. Ensure user; create session row if `session_id is None`.
2. Persist user message; load up to 10 prior messages via `_load_history` (excludes the message just inserted).
3. `retriever.retrieve(question, user_id, top_k=cfg.retrieval.top_k_vector)` → `list[RetrievalResult]`.
4. If reranker is set: rerank with `cfg.retrieval.rerank_threshold`, cap at `top_k_final`. **Critical fallback**: if rerank drops everything, fall back to unreranked top-k (else LLM answers "I couldn't find that" even when retrieval succeeded). Emits `fallback_used: True` in `rerank_done`.
5. Render `prompts/answer.md` (RAFT-style `##begin_quote##` CoT) with `{user_profile, conversation_history, retrieved_passages, question}`.
6. `llm.generate_stream` (stream=True) or `llm.complete`.
7. Persist assistant message + commit.

`progress` is `Optional[Callable[[str, dict], None]]` — events at top of `orchestrator.py`. CLI uses it for live Rich display.

### Pluggable layers (factories)

| Layer | Factory | Implementations |
|---|---|---|
| LLM | `providers/llm.py::get_llm_provider` | Ollama, OpenAI, Anthropic |
| Embeddings | `providers/embeddings.py::get_embedding_provider` | sentence-transformers, OpenAI |
| Retriever | `retrieval/factory.py::build_retriever` | Vector, BM25, Hybrid (RRF), KGPPR, Community, QueryRouter, Taxonomy (default) |
| Reranker | `retrieval/factory.py::build_reranker` | CrossEncoder (default), LLMReranker (per-chunk 0–3), BatchedLLMReranker |

Reranker thresholds differ: `cross_encoder` is a logit (default `-5.0`, permissive); `llm`/`batched_llm` is int 0–3 (use `2` if switching).

### HeteRAG dual-text

`types.Chunk` has two text fields:
- `embedding_text` — title/section prepended (HeteRAG fusion); embedder sees this.
- `text` — raw chunk; LLM sees this at generation.

Anything embedding a chunk uses `embedding_text`. Anything showing or LLM-feeding uses `text`. Don't conflate.

### Storage

- **SQLite** (`db/schema.sql`, `db/connection.py`): users, documents, chunks, sessions, messages, `preferences`, `kg_{nodes,edges,communities,triple_cache}`, `kg_taxonomy_{nodes,assignments,doc_meta}`, `jobs`, `feedback`, `kg_canonical_triples`, `rerank_fallback_events`, `ingest_context_cache`. FK on `chunks` is `doc_id` (not `document_id`).
- **ChromaDB** (`retrieval/vector.py::VectorStore`): vectors + metadata mirror. Community summaries live in `hrag_community_summaries`.
- **NetworkX**: MultiDiGraph (phrase + passage nodes); mirrored in SQLite.

`user_id` is plumbed through every table and every API even in single-user mode — don't strip it.

### Ingest pipeline (`ingest/pipeline.py`)

`load_document` (PyMuPDF / python-docx / markdown-it-py / plain text) → `chunk_document` (token-budgeted, structure-aware; HeteRAG fusion) → `filter_chunks` (`quality.py`) → embed in batches of 32 → upsert into SQLite + ChromaDB (`delete_doc` then `add_chunks` so reingest is idempotent).

### Chunk quality filter (`ingest/quality.py`)

Drops typically 70–85% of chunks on academic PDFs. Configured under `chunking.quality.*`. Pure-function module — no I/O. **Important**: each check matches the *entire* chunk against a pattern. A chunk that *starts* with `"250\n6We use..."` doesn't match `_RE_PAGE_ARTIFACT` because the regex requires the whole text to be a page number. If artifacts surface in retrieval, the PDF parser is gluing artifacts onto real content — tightening the filter alone won't fix it.

After tweaking filter settings, **re-ingest** — the index doesn't update retroactively. Reingest deletes old chunks/vectors first.

### Tests

`tests/conftest.py` stubs heavy deps (chromadb, sentence-transformers, ollama). Tests genuinely needing a heavy dep skip via `pytest.importorskip`. Follow the same pattern for new tests.

### Prompts

Templates live as `.md` files in `src/hrag/prompts/`, loaded with `Path(__file__).parent / "prompts" / "<name>.md"` and `.format(...)`. `pyproject.toml` packages them via `package-data`. Edit the `.md` file, not Python string literals.

## Conventions

- Type hints required (`from __future__ import annotations` everywhere); pydantic for config models, dataclasses for runtime value objects (`types.py`).
- Heavy-dep imports go inside functions/methods, not at module top, so import-time failures don't cascade.
- Provider/retriever/reranker classes carry a `name: str` class attribute for logging/UI.
- Database access goes through `Database` (`db/connection.py`); use `with self.db.conn:` for transaction blocks. Always `self.db.commit()` at the end of a write path.
