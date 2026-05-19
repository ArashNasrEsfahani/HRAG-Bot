# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Visibility — every long-running process must show progress

User invariant: any process expected to take more than ~10 seconds must surface progress in real time. No silent waits. Specifically:

- **Foreground long commands** — wrap with a `rich.progress.Progress` bar driven by a known count (chunks, docs, communities, benchmark questions). When the count isn't knowable upfront, emit a per-item line with a running tally (`[i/n] processing X...`) and flush stdout (`print(..., flush=True)`).
- **Background commands** (Bash `run_in_background: true`) — set up a `Monitor` over the log file with `grep --line-buffered` filtering for per-item completion + failure signatures. Cover failure cases (Traceback, UnicodeError, "killed", non-zero exit), not only the happy path — silence on crash is the bug.
- **Subagents** — every subagent prompt must require a final structured report with what it did. Long-running agent work that the user is waiting on should be foreground (user sees the agent's intermediate work) or background with a Monitor on its output file (only when the agent's output is mostly noise and the user only needs the summary).
- **Existing CLI commands lacking progress** (e.g. `hrag rebuild-kg`'s community-summarization phase printed nothing for 15 minutes) are a bug. Fix the source so the next run is observable. Per-item prints with `flush=True` are the floor; a Rich progress bar is the target.
- **Don't ask "should I proceed?" on long jobs without first showing the user a progress channel they can watch.** The right pattern is: arm the monitor → start the job → tell the user what to expect → the job streams.

This rule applies project-wide and supersedes any agent-prompt template that omits progress hooks.

## Subagent dispatch policy (default for every non-trivial request)

For any request that decomposes into 2+ work items, dispatch the work to subagents instead of doing it serially in the main thread. Pick the model per item by importance × difficulty:

- **Opus** — algorithmically tricky, security-sensitive, cross-cutting wiring, novel design with multiple invariants, anything where a wrong call ripples into several files. Examples in this codebase: graph store with synonym merging, MST + redundancy pruning, query router with RRF fusion, prompt design.
- **Sonnet** — well-scoped mechanical work, single-file modules following an existing pattern, test-only additions, config/schema/dep edits, factory registrations. Examples: new `Retriever` impl following `VectorRetriever` shape, NER wrapper, ingest pipeline hook, CLI subcommand.

Fan out in waves: agents on disjoint files run in parallel (single message with multiple `Agent` tool calls); only serialize when later work depends on an earlier deliverable. After each wave, the main thread (Opus 4.7) verifies — `pytest`, ruff, schema sanity — before dispatching the next wave. Coordination overhead is a real cost; don't shard work smaller than ~one file per agent.

When the request is a single trivial edit, just do it inline — no subagents.

## Project status

**Phase 1** (walking-skeleton RAG: ingest → vector retrieve → rerank → answer) — complete.

**Phase 2** (hierarchical retrieval: KG triple extraction + Personalized PageRank + GraphRAG community summaries + LLM-routed retriever + KG2RAG MST organizer) — **implementation complete**, 381 passing tests. Behind `kg.enabled` flag.

**Phase 2 modules:**
- `src/hrag/kg/builder.py` — `Triple` + `TripleExtractor` (concurrent OpenIE per chunk via `prompts/triple_extraction.md`)
- `src/hrag/kg/store.py` — `KGStore` (NetworkX `MultiDiGraph` with phrase + passage nodes, synonym merging via embedding cosine ≥ 0.8, idempotent per-doc upsert, SQLite mirror at `kg_nodes`/`kg_edges`)
- `src/hrag/kg/ner.py` — `SpacyNER` (default, lazy spaCy + regex fallback) and `LLMNER` (opt-in via `kg.ner: llm`)
- `src/hrag/kg/ppr.py` — pure-scipy `personalized_pagerank` (power iteration with dangling-node redistribution)
- `src/hrag/kg/communities.py` — Leiden detection (`leidenalg`+`python-igraph`) at three resolutions, concurrent summarization (`prompts/community_summary.md`), Chroma collection `hrag_community_summaries` + SQLite `kg_communities` mirror, `detect_and_summarize` one-call entry point
- `src/hrag/retrieval/kg_ppr.py` — `Retriever` impl: NER → seeds → PPR → passage hydration
- `src/hrag/retrieval/community.py` — `Retriever` impl over community summaries (returns `RetrievalResult` with `source_type="community"`)
- `src/hrag/retrieval/router.py` — `QueryRouter` 5-shot LLM classifier (`prompts/router.md`) routing to entity / global / cross_document / ambiguous, RRF-fusing the cross_document and ambiguous paths, with per-query cache and graceful degradation
- `src/hrag/retrieval/mst.py` — `MSTOrganizer` (KG2RAG redundancy filter + tree-ordering, no-op when KG empty)

**Phase 3 — Personalization** (per-user memory layer + hierarchical document taxonomy) — **complete**, 537 passing tests. Default retriever is now `taxonomy` (falls back to vector when the tree is empty).

Memory sublayer (preferences + episodic notes + auto-extraction):
- `src/hrag/memory/profile.py` — `ProfileStore` (preferences table; rendered verbatim into every answer prompt as `{user_profile}`).
- `src/hrag/memory/store.py` — `EpisodicMemoryStore` (`/remember`, `/recall`, `/forget`; episodic chunks live in `chunks` with `source_type='episodic'` and compete with documents at retrieval).
- `src/hrag/memory/auto_extract.py` — `SessionAutoExtractor` (opt-in via `memory.auto_extract`; one extra LLM call on session close, daemon thread).
- `src/hrag/memory/extractor.py` — `PreferenceExtractor` (uses `prompts/preference_extract.md`).
- `src/hrag/context/builder.py` — `ContextBuilder` (assembles the `{user_profile}` block from `ProfileStore`).

Taxonomy sublayer (LLM-proposed, user-editable category tree over docs + memories; default retriever):
- `src/hrag/taxonomy/store.py` — `TaxonomyStore` (CRUD, packed-float32 centroids, in-memory beam descend with cached node map).
- `src/hrag/taxonomy/builder.py` — `TaxonomyBuilder` (parallel doc summaries → batched centroid embed → LLM tree proposal with explicit `max_tokens=8192` + truncation-salvage → materialize → overflow assignment → recompute centroids).
- `src/hrag/taxonomy/assigner.py` — `DocAssigner` (greedy cosine descent + LLM tiebreak when top-2 scores < 0.05 apart; used by the ingest auto-assign hook and the GUI's "Assign all unfiled" button).
- `src/hrag/retrieval/taxonomy.py` — `TaxonomyRetriever` (beam descend → over-fetch from Chroma → filter to leaf docs → hydrate; falls back to `taxonomy.fallback_retriever` when tree is empty). Exposes `describe_last_descend()` for the GUI's tree-navigation visual.
- `src/hrag/prompts/taxonomy_*.md` — `taxonomy_doc_summary.md`, `taxonomy_propose.md`, `taxonomy_relabel.md`, `taxonomy_route_tiebreak.md`.
- `src/hrag/gui/pages/8_Taxonomy.py` — graphviz tree + per-node edit panel (label, description, move, add child, delete, unfile).
- Tables: `kg_taxonomy_nodes`, `kg_taxonomy_assignments`, `kg_taxonomy_doc_meta` (the doc-meta cache survives `clear()` by default; pass `wipe_doc_meta=True` for a full reset).
- `taxonomy.include_episodic` (default `true`) — episodic memories are filed under the same tree as documents.

**Phase 4** (compaction & gating: RAGate, clue generation, dialog MST, `[UNCERTAIN]` masking) — **implementation complete**, 691+ passing tests. All four features behind `compaction.*` flags, default OFF.

**Phase 4 modules:**
- `src/hrag/gating/gate.py` — `RAGate` (single LLM call against `prompts/gate.md`, fail-open on garbled output)
- `src/hrag/gating/clue.py` — `ClueGenerator` (MemoRAG-style hypothesis against `prompts/clue.md`, fallback to question on empty/error)
- `src/hrag/gating/uncertain.py` — `render_uncertain` / `strip_uncertain` (pure regex post-processor for `[UNCERTAIN]` tokens)
- `src/hrag/context/dialog_mst.py` — `DialogMSTCompactor` (greedy cosine clustering + per-cluster summarization via `prompts/dialog_summary.md`)
- `src/hrag/config.py::CompactionConfig` — toggles for all four features
- `prompts/answer.md` — Step 4 instructs the LLM to write `[UNCERTAIN]` after unsupported sub-claims

**Orchestrator integration points (in `chat()` pipeline order):**
1. After history load → dialog compaction
2. After intent classification → RAGate (FACTUAL only; SKIP forces `plan.scope = "none"` and re-routes intent to GENERAL)
3. After query rewrite → clue generation (replaces retrieval query; original question still feeds answer prompt)
4. After answer generation → `render_uncertain` (when `mask_uncertain=True`) else silent `strip_uncertain`

New progress events: `dialog_compact`, `gate_check`, `clue_generate`, `uncertain_render`.

**Phase 5 — Web ergonomics + extensibility** — **complete**, 700 passing tests.

**Phase 5 modules:**
- **Track A (web UX)** —
  - `src/hrag/web/app.py` adds memory CRUD (`POST` / `PUT` / `DELETE /api/memories`), multipart upload (`POST /api/ingest`), background ingest jobs (`POST /api/ingest?background=true` → `GET /api/jobs/{id}`).
  - `src/hrag/web/static/{index.html,app.js,styles.css}` — inline memory edit/add cards, drag-and-drop upload zone with per-file progress bar, job-polling flow.
  - `jobs` table in `src/hrag/db/schema.sql` (job_id, kind, status, progress, total, message, result).
- **Track B (verification)** — `tests/benchmark/run_phase5_web.py` exercises the FastAPI layer over `TestClient`: config roundtrip, SSE event order, session continuity, hot-swap, memory CRUD.
- **Track C (pluggable backends)** —
  - `src/hrag/kg/backends/{base,networkx,neo4j}.py` defines `KGBackend` Protocol (18 methods) + NetworkX (default) + Neo4j stub. `KGStore.from_config` factory swaps by `kg.backend`.
  - `src/hrag/retrieval/backends/{base,chroma,sqlite_vec}.py` defines `VectorBackend` Protocol (5 methods) + Chroma (default) + sqlite-vec stub. `_build_vector_backend` factory in `orchestrator.py`.
- **Track D (self-improvement loop)** —
  - `feedback` table in `schema.sql` (`message_id`, `rating`, `note`).
  - `src/hrag/web/app.py` adds `POST/GET/DELETE /api/feedback`.
  - Web UI shows 👍/👎 buttons under assistant messages, pre-marked on session replay.
  - `hrag export-training-pairs --out pairs.jsonl` exports JSONL for downstream fine-tuning.
- **Track E (equation-aware ingest)** —
  - `src/hrag/ingest/math_detect.py` (new) — `find_display_math_spans`, `has_math` detectors covering `$$...$$`, `\begin{equation|align|eqnarray|gather}` (with `*`), and inline `$...$` / `\(...\)`.
  - `chunker.py` — boundary nudge so display-math blocks never split; oversized equations emit as a single over-budget chunk rather than getting shredded.
  - `quality.py` — `min_alpha_ratio` and `min_tokens` carve-outs when `metadata.has_math` is `True`.
  - PyMuPDF caveat (no LaTeX reconstruction from PDFs) documented in `loaders.py`.

**Closed in Phase 5 (originally scoped for it):** streaming refinements (web SPA replaced Streamlit with SSE + plain-text fast-path), fine-tuning hooks (feedback table + JSONL export), optional Neo4j / sqlite-vec backends (protocols + stubs ready), equation-aware ingest.

**Deferred (not in Phase 5):** real Neo4j / sqlite-vec implementations (stubs raise `NotImplementedError`), adaptive retrieval per intent, Ollama prompt-caching across turns. → All three landed in Phase 6.

**Phase 6 — Backends + adaptive retrieval + Ollama warmth** — **complete**, 726 passing tests, 8/8 acceptance benchmark.

**Phase 6 modules:**
- **Track A (sqlite-vec real backend)** — `src/hrag/retrieval/backends/sqlite_vec.py` (~390 LOC) replaces the stub with a real `vec0` virtual-table impl + typed `chunks_meta` mirror for filter pushdown. Uses `vec0(... distance_metric=cosine)` so distances arrive in Chroma's `1 − cos_sim` shape directly. Where-compiler handles flat equality plus `$and` / `$or` / `$eq` / `$ne` / `$in`. Optional dep group `sqlite-vec` in `pyproject.toml`; `tests/test_sqlite_vec_backend.py` has 9 tests (skip cleanly without the extension).
- **Track B (Neo4j real backend)** — `src/hrag/kg/backends/neo4j.py` (~370 LOC) implements all 18 protocol methods against parameterised Cypher. Single label `:Node`, single rel `:LINK`; multi-edges keyed by `uuid4().hex`. Non-primitive attrs serialised into a `__json_attrs` sidecar property. `Neo4jBackend()` raises a clear `RuntimeError` when no URI / driver is configured; class import has zero side effects. Optional dep group `neo4j` in `pyproject.toml`; `tests/test_neo4j_backend.py` has 13 tests, double-guarded (`importorskip("neo4j")` + `NEO4J_URI` env check) so CI stays green without a server.
- **Track C (Adaptive retrieval per intent)** — `src/hrag/orchestrator.py::_adaptive_top_k` resolver maps each intent to a `(top_k_vector, top_k_final)` pair; `(None, None)` signals "skip retrieval entirely" (greeting = 0 by default). GREETING short-circuits the gate / clue / retriever / reranker / organizer. PERSONAL broadens `source_types` to `["document", "episodic"]` and stable-sorts episodic chunks to the top. Three new progress events: `adaptive_top_k`, `retrieval_skipped`, `episodic_bias_applied`. Off by default (`retrieval.adaptive_enabled: false`); 12 tests in `tests/test_adaptive_retrieval.py`.
- **Track D (Ollama keep-alive)** — `src/hrag/providers/llm.py` threads `cfg.llm.keep_alive` into the chat() top-level kwarg (not inside `options`). Default `"30m"` so the model stays resident through a typical multi-turn conversation; `None` defers to Ollama's own 5-minute default; `"-1s"` never unloads. 5 tests in `tests/test_keep_alive.py`.

**Phase 6 contracts (must survive Phase 7):**

12. The three Phase-6 progress events (`adaptive_top_k`, `retrieval_skipped`, `episodic_bias_applied`) must keep firing when `retrieval.adaptive_enabled` is true. The benchmark and any future GUI trace panel depend on the payload shape documented in `orchestrator.py`'s top-of-file event list.
13. `_adaptive_top_k(cfg, intent)` must remain a pass-through when `retrieval.adaptive_enabled` is False — returning `(cfg.retrieval.top_k_vector, cfg.retrieval.top_k_final)` unconditionally. The default-off invariant (Phase 6 Q8) is load-bearing for everyone running an older `config.yaml`.
14. `OllamaProvider._build_chat_kwargs` must keep emitting `keep_alive` as a top-level kwarg, not inside `options`. Burying it inside `options` is silently ignored by the Ollama server — the model will appear to unload normally after 5 minutes regardless of the config.
15. The `Neo4jBackend` and `SqliteVecBackend` classes must keep their constructor side-effect-free (no driver / extension import at module load). Selecting a backend the user hasn't installed should fail at the first method call, not on `import hrag`.

**Deferred (not in Phase 6):** cross-turn KV-cache reuse on Ollama (keep-alive keeps the model loaded; getting Ollama to recognise the shared prefix between turns needs research into `num_keep` / context-priming), retriever-level adaptation (Phase 6 only varies top_k, not retriever choice, by intent), self-improvement via the feedback table (Phase 5 collects 👍 / 👎; mining + fine-tuning is a future phase).

**Phase 7-A — Math handling: detector + filter + extraction** — **complete**, 755 passing tests, 5/5 acceptance benchmark.

Triggered by a real user failure: HRAG retrieved three formula-free chunks for *"give me some formulas hipporag uses"* and reported "no formulas in the passages", despite 72 chunks in the live index containing Unicode math glyphs. Three Explore agents diagnosed every link in the retrieval chain (ingest, embed, query, rerank, answer) as math-blind. Phase 7-A ships three complementary fixes that all land together.

**Phase 7-A modules:**
- **Method 1 (Unicode-glyph `has_math` detector + backfill)** — `src/hrag/ingest/math_detect.py` adds `has_unicode_math(text, min_signals=2)`. Signals: Greek/mathematical-italic letters (`α β γ δ ε θ λ μ ν π ρ σ τ φ χ ψ ω Γ Δ Θ Λ Π Σ Φ Ψ Ω ∂ ∇` + `U+1D400..U+1D7FF`), math operators (`∑ ∫ ∏ √ ∞ ≤ ≥ ≠ ≈ ⟨ ⟩ ⊕ ⊗ ⋅`), sub/superscripts, equation density (2+ `=` in 200 chars), function-of-variable patterns. At least 2 must hit. `has_math = _has_latex_math(text) or has_unicode_math(text)`. The chunker already calls `has_math` on raw text, so no chunker change. One-shot backfill at `scripts/backfill_has_math.py` walks SQLite + ChromaDB (256-id batches, idempotent); tagged **63 of 1082 chunks** in the live corpus. 7 tests in `tests/test_math_detect_unicode.py`.
- **Method 2 (math-meta query expansion)** — `src/hrag/retrieval/query_rewriter.py` adds `_RE_MATH_META` + `_expand_math_meta(query)` that appends `"equation parameter θ Θ loss function objective gradient ∑ ∫ derivation variable"` to meta-queries. Called inside `HeuristicRewriter.rewrite` after follow-up rewriting. `prompts/query_rewrite.md` gets an LLM-path rule + worked example. Cosine similarity between a meta-query and a chunk like `𝑌= Θ(𝑞|𝜃)` jumps from ~0.10 to ~0.35. 8 tests in `tests/test_math_meta_rewrite.py`.
- **Method 3 (orchestrator filter + formula-extraction pass)** — `src/hrag/orchestrator.py::_is_math_meta_query` detects content-type queries; when `retrieval.math_meta_filter_enabled=true` AND the query matches, passes `where={"has_math": True}` into `retriever.retrieve(...)` AND lowers the rerank threshold to `retrieval.math_meta_rerank_threshold` (default `-10.0`). Empty-result fallback retries unfiltered. When `formula_extraction.enabled=true` AND results exist, a second LLM call against `prompts/extract_formulas.md` extracts every equation verbatim and the result is appended as `**Extracted formulas:**` to the answer. Three new progress events: `math_meta_filter`, `math_meta_filter_fallback`, `formula_extract`. 10 tests in `tests/test_math_meta_orchestrator.py`.
- **Retriever Protocol widened** — `src/hrag/retrieval/base.py` adds `where: Optional[dict] = None`. Vector-backed retrievers (`vector`, `hybrid`, `router`, `taxonomy`, `doc_scope`) thread it through to `VectorStore.query`. `bm25`, `kg_ppr`, `community` accept and silently ignore (documented). Test doubles in 6 test files were widened to accept the kwarg.

**Phase 7-A contracts (must survive Phase 7-B / 7-C):**

16. `Retriever.retrieve(query, user_id, top_k, source_types, intent_hint, where)` — every concrete retriever must accept the `where` kwarg, even if it ignores it. Wrappers (`doc_scope`, `router`, `hybrid`) must keep threading it through. Phase 7-A Q4 depends on this.
17. The three Phase-7-A progress events must keep firing when `retrieval.math_meta_filter_enabled` / `formula_extraction.enabled` are true. The benchmark and any future GUI trace panel will subscribe.
18. `scripts/backfill_has_math.py` must remain idempotent — running it twice on the same DB MUST NOT corrupt metadata. Anyone re-ingesting a single document and then re-running the backfill should converge.
19. `_is_math_meta_query` must remain a pure regex pass (no LLM) so it stays fast and deterministic. The orchestrator runs it on every turn when the flag is on.

**Deferred (not in Phase 7-A):** math-aware embedding model swap (Method 4 — replace `all-mpnet-base-v2` with `allenai/specter2` or `jina-embeddings-v2`, requires full re-embedding), Nougat PDF→LaTeX re-ingest (Method 5 — 800MB model, GPU, full re-ingest, ~85% formula fidelity), plus everything previously deferred from Phase 6. → **All five landed in the Phase 6 + 7 wrap-up below.**

**Phase 6 + 7 wrap-up — five deferred items completed** — **complete**, 797 passing tests, 19/19 acceptance (8/8 Phase 6 + 5/5 Phase 7-A + 6/6 new).

Five parallel implementation agents shipped the deferred items in one wave, then two more wired them into the web GUI:

- **6-B1 — Per-intent retriever override** — new `cfg.retrieval.adaptive_retriever_per_intent` (5 intents → retriever name or `"default"`). `Orchestrator._pick_retriever_for_intent` resolves and caches alternative retrievers; missing-dep cases silently fall back to the global with a logger warning. New event `adaptive_retriever_picked` (payload `{intent, retriever, global}`). 8 tests in `tests/test_per_intent_retriever.py`.
- **6-B2 — Feedback analytics CLI + API** — new pure-SQL helper `src/hrag/feedback_stats.py::feedback_summary(db)` shared by CLI and web. CLI: `hrag feedback-stats` + `hrag feedback-export --rating up|down --out pairs.jsonl`. API: `GET /api/feedback/stats`. GUI: new Feedback drawer alongside Memories / Documents with ratio bar + top-negatives list. 8 tests.
- **6-B3 — Ollama `num_keep` plumbing** — new `cfg.llm.num_keep: Optional[int]`. Threads into `options.num_keep` on chat() (NOT the top level — Ollama would silently ignore that placement). Default `None`. Best-effort cross-turn KV-cache priming. 5 tests in `tests/test_num_keep.py`.
- **7-B — Math-aware embedder selector** — curated `EmbeddingsConfig.suggested_models` list (4 entries: all-mpnet, specter2, jina-v2, bge-small). Pure helper `dimension_for_model()` for client-side dim validation. CLI: `hrag embeddings-list` + `hrag embeddings-current`. API: `GET /api/embeddings/suggested`. GUI: dropdown with "requires re-ingest" warning. 4 tests in `tests/test_embeddings_selector.py`.
- **7-C — Nougat PDF loader scaffold** — new `src/hrag/ingest/nougat_loader.py` with deferred imports (zero side effects on `import hrag`). `_load_pdf` dispatches to Nougat when `cfg.ingest.use_nougat=True` AND optional dep installed; silent PyMuPDF fallback otherwise. Optional `nougat` dep group in `pyproject.toml`. API: `GET /api/ingest/nougat_status`. GUI: toggle + availability badge. 5 tests.

**Phase 6 + 7 wrap-up contracts (must survive Phase 8+):**

20. `cfg.retrieval.adaptive_retriever_per_intent` value must always be either `"default"` or one of `{vector, bm25, hybrid, kg_ppr, community, router, taxonomy}`. The web API validates this in POST; do not loosen it.
21. `feedback_summary(db)` must keep its return shape stable (`{thumbs_up, thumbs_down, total, sessions, top_negative}`) — both the CLI command and `GET /api/feedback/stats` depend on it.
22. `OllamaProvider._build_options` must keep emitting `num_keep` inside `options` (NOT at the top level via `_build_chat_kwargs`). Putting it at the top level is silently ignored by the Ollama HTTP API.
23. `hrag.ingest.nougat_loader` import must remain side-effect-free: zero `nougat` / `nougat_ocr` modules pulled into `sys.modules` until a method is actually called. The Phase 6+7 wrap-up Q5 test verifies this property.
24. `EmbeddingsConfig.suggested_models` is purely advisory metadata — the SentenceTransformersProvider still accepts any HF model id. Adding or removing entries from the list does not break existing configs.

**Deferred to Phase 8+ (still deferred, requires real usage / infra):** actually running Nougat over a corpus (scaffold present, model download is opt-in), actually swapping to a math-aware embedder + re-ingest (selector present, user controls when), feedback-loop fine-tuning (analytics now expose data, but training infra is a Phase 8 lift), cross-turn KV-cache reuse via Ollama prefix tokens (research-heavy beyond `num_keep`).

## Phase 4 / 5 must preserve

When implementing later phases, do NOT break the following Phase-3 contracts. Each is load-bearing for the memory or taxonomy layer.

1. `documents.source_type` and `chunks.source_type` ('document' | 'episodic') must stay. `taxonomy.include_episodic` reads them; if you add a new source_type, teach the taxonomy builder to handle it (`src/hrag/taxonomy/builder.py::_list_docs`).
2. The taxonomy auto-assign hook in `IngestPipeline.ingest_document` (the "5c" block) must keep running for every eligible ingest. New ingest features (e.g. equation-aware chunker) layer in **before** that block; they do not skip it.
3. `kg_taxonomy_nodes` / `kg_taxonomy_assignments` / `kg_taxonomy_doc_meta` migrations stay `CREATE TABLE IF NOT EXISTS` and backward-compatible. Never drop a column the builder reads.
4. `Retriever.retrieve(query, user_id, top_k, source_types)` signature is locked. `TaxonomyRetriever` and the `/recall` path depend on `source_types` being passable through; do not strip it when wrapping.
5. `LLMProvider.complete(prompt, *, temperature, max_tokens)` must keep accepting an explicit `max_tokens`. The taxonomy builder relies on this — Ollama silently truncates long JSON when `max_tokens` is left to its default; don't reintroduce that bug by removing the keyword.
6. `EmbeddingProvider.embed(texts)` must keep returning L2-normalised vectors. `TaxonomyStore.beam_descend` treats cosine == dot product on the hot path.
7. The orchestrator's `progress(event, payload)` channel must keep emitting `taxonomy_descend` (with the `describe_last_descend()` payload) whenever the active retriever has `name == "taxonomy"`. The Chat page's **🌳 Tree navigation** expander and the Taxonomy page both subscribe; silently dropping the event regresses the visual.
8. `TaxonomyStore.clear(user_id)` is **non-destructive** to `kg_taxonomy_doc_meta` by default. Tree rebuilds rely on the summary + centroid cache surviving the wipe. Pass `wipe_doc_meta=True` only when you mean a full reset (the `hrag taxonomy clear` CLI does NOT pass it — it preserves the cache).

**Phase 4 contracts (must survive Phase 5):**

9. The four progress events `gate_check`, `clue_generate`, `dialog_compact`, `uncertain_render` must keep firing when their respective `compaction.*` flags are enabled. The GUI's Compaction expander subscribes to all four; silently dropping any event regresses the visual.
10. `render_uncertain` must remain idempotent — applying it twice to the same answer must not double-render or corrupt the output.
11. The orchestrator's silent `strip_uncertain` path (when `mask_uncertain=False`) must continue running so raw `[UNCERTAIN]` tokens never leak to end users regardless of the flag state.

## Phase 2 acceptance — manual trigger

```bash
# 1. Flip the toggles (or set HRAG_KG__ENABLED=true HRAG_RETRIEVAL__RETRIEVER=router)
#    in config.yaml: kg.enabled: true  AND  retrieval.retriever: router
# 2. Wipe the index and re-ingest (chunks + KG + communities all rebuilt)
rm -rf data/store.sqlite data/chroma data/kg
hrag init
hrag ingest "D:/Selected Dynamic Papers" --recursive
# 3. Re-run the Phase 1 benchmark using the project's benchmark skill — q4 should PASS
# 4. Or, if chunks already exist and you only want to (re)build the KG layer:
hrag rebuild-kg
```

## Phase 3 (Taxonomy) — manual trigger

```bash
# Ensure taxonomy.enabled: true and retrieval.retriever: taxonomy in config.yaml
# (both are defaults). Then build the tree from the current corpus:
hrag taxonomy build              # parallel summaries → LLM tree proposal → materialize
hrag taxonomy show               # print the tree as a text outline
hrag taxonomy clear              # drop the tree (preserves the doc-meta cache)

# New documents ingested via `hrag ingest` are auto-filed under the tree
# via DocAssigner. Edit the tree from the GUI: `hrag gui` → 🌳 Taxonomy page.
```

## Phase 4 (Compaction & Gating) — manual trigger

```bash
# Ensure the Phase 2/3 corpus is already ingested.
# Enable all four flags via environment variables (or edit config.yaml's
# compaction: block):
HRAG_COMPACTION__GATE_ENABLED=true \
HRAG_COMPACTION__CLUE_ENABLED=true \
HRAG_COMPACTION__DIALOG_MST_ENABLED=true \
HRAG_COMPACTION__MASK_UNCERTAIN=true \
python tests/benchmark/run_phase4.py

# Or toggle per-session via CLI flags (each flag is independent):
hrag chat --gate --clue --mask-uncertain
hrag chat --no-gate --clue            # clue only
hrag chat --gate --no-clue            # gate only
```

Acceptance: >= 3/4 acceptance questions pass.

## Commands

```bash
# Install (editable)
python -m pip install -e .[dev]

# Initialize SQLite + chroma directories
hrag init

# Ingest documents
hrag ingest <path>              # single file
hrag ingest <dir> --recursive   # whole tree

# Chat REPL (slash commands: /help, /sources, /status, /exit,
# /remember, /recall, /forget, /profile)
hrag chat
hrag chat --fast                            # top_k=4, no rerank
hrag chat --retriever taxonomy              # default; or vector / bm25 / hybrid / kg_ppr / community / router
hrag chat --no-rerank

# Phase 3 — taxonomy (LLM-proposed category tree over docs + memories)
hrag taxonomy build                         # build/rebuild the tree
hrag taxonomy show                          # print the tree as a text outline
hrag taxonomy clear                         # drop the tree (preserves doc-meta cache)

# Phase 3 — memory
hrag remember "<text>"                      # save an episodic memory inline
hrag remember <path>                        # bulk-import .md / .txt files
hrag memory list                            # show recent episodic memories
hrag memory extract --session <id>          # mine preferences from a session

# Streamlit GUI (dashboard with Chat · Memories · 🌳 Taxonomy · ...)
hrag gui

# Other
hrag list-docs

# Tests
pytest -q                                   # all tests
pytest tests/test_orchestrator.py -q        # one file
pytest -k "rerank" -q                       # by name pattern

# Lint
ruff check src tests
```

Heavy-deps tests (chromadb / sentence-transformers / ollama) skip cleanly when those deps are absent — `tests/conftest.py` provides stubs so lightweight tests run anywhere.

## Configuration

`config.yaml` at the repo root is the canonical config; values are loaded into pydantic models in `src/hrag/config.py`. Two override mechanisms:

1. **Environment variables** prefixed `HRAG_` with `__` as the section separator: `HRAG_LLM__MODEL=gemma:7b` overrides `llm.model`.
2. **Per-session CLI flags** on `hrag chat` (`--retriever`, `--reranker`, `--no-rerank`, `--fast`) override config for that REPL only.

`Config.resolve(relative_path)` resolves storage paths against `project_root` (cwd at load time).

## Architecture

Modular stack — CLI · GUI · Orchestrator · Retrievers · Personalization · Storage · Providers. Every layer is one small Python module behind an interface so layers can be swapped (e.g. ChromaDB → sqlite-vec, NetworkX → Neo4j). The architecture diagram in README.md is the canonical map.

### Pipeline (orchestrator.py)

`Orchestrator.chat(question, user_id, session_id, progress, stream)` runs:
1. Ensure user; create session row if `session_id is None`.
2. Persist the user message; load up to 10 prior messages via `_load_history` (excludes the message just inserted).
3. `retriever.retrieve(question, user_id, top_k=cfg.retrieval.top_k_vector)` — returns `list[RetrievalResult]`.
4. If reranker is set: rerank with `cfg.retrieval.rerank_threshold`, capped at `cfg.retrieval.top_k_final`. **Critical fallback**: if rerank drops everything, fall back to unreranked top-k (otherwise the LLM answers "I couldn't find that" even when retrieval succeeded). Emits `fallback_used: True` in the `rerank_done` progress event.
5. Render `prompts/answer.md` (RAFT-style `##begin_quote##` CoT) with `{user_profile, conversation_history, retrieved_passages, question}`.
6. `llm.generate_stream` (when `stream=True`) or `llm.complete` (one-shot).
7. Persist assistant message and commit.

`progress` is an optional `Callable[[str, dict], None]` — events documented at the top of `orchestrator.py`. The CLI uses this for the live Rich progress display.

### Pluggable layers (factory pattern)

| Layer | Factory | Implementations |
|---|---|---|
| LLM | `providers/llm.py::get_llm_provider` | `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider` |
| Embeddings | `providers/embeddings.py::get_embedding_provider` | sentence-transformers, OpenAI |
| Retriever | `retrieval/factory.py::build_retriever` | `VectorRetriever`, `BM25Retriever`, `HybridRetriever` (RRF), `KGPPRRetriever`, `CommunityRetriever`, `QueryRouter`, `TaxonomyRetriever` (default) |
| Reranker | `retrieval/factory.py::build_reranker` | `CrossEncoderReranker` (default), `LLMReranker` (per-chunk 0–3), `BatchedLLMReranker` (single LLM call) |

Reranker threshold semantics differ by reranker: `cross_encoder` is a logit (default `-5.0`, permissive), `llm`/`batched_llm` is an int 0–3 (use `2` if switching).

### HeteRAG dual-text on Chunk

`types.Chunk` has two text fields:
- `embedding_text` — title/section prepended (HeteRAG metadata fusion); what the embedder sees.
- `text` — raw chunk; what the LLM sees at generation time.

Anything that embeds a chunk must use `embedding_text`. Anything that shows or feeds it to the LLM uses `text`. Don't conflate.

### Storage

- **SQLite** (`db/schema.sql`, `db/connection.py`): users, documents, chunks, sessions, messages, `preferences` (Phase 3 memory), `kg_nodes`, `kg_edges`, `kg_communities`, `kg_triple_cache` (Phase 2), `kg_taxonomy_nodes`, `kg_taxonomy_assignments`, `kg_taxonomy_doc_meta` (Phase 3 taxonomy). Foreign key column on `chunks` is `doc_id` (not `document_id`).
- **ChromaDB** (`retrieval/vector.py::VectorStore`): vectors only; metadata mirror of chunk fields lives here too. Community summaries (Phase 2) live in the `hrag_community_summaries` collection alongside.
- **NetworkX** (Phase 2): MultiDiGraph holding phrase + passage nodes; mirrored in SQLite.

`user_id` is plumbed through every table and every API even in single-user mode — don't strip it when adding features.

### Ingest pipeline (`ingest/pipeline.py`)

`load_document` (`loaders.py`: PDF via PyMuPDF, DOCX via python-docx, MD via markdown-it-py, plain text) → `chunk_document` (token-budgeted, structure-aware; HeteRAG fusion happens here) → `filter_chunks` (`quality.py`) → embed in batches of 32 → upsert into SQLite and ChromaDB (`delete_doc` then `add_chunks` so reingest is idempotent).

### Chunk quality filter (`ingest/quality.py`)

Runs at ingest time, drops typically 70–85% of chunks on academic PDFs. Configured under `chunking.quality.*`. Pure-function module — no I/O. **Important**: each filter check matches the *entire* chunk against a pattern. A chunk that *starts* with `"250\n6We use..."` (page number followed by content) does NOT match `_RE_PAGE_ARTIFACT` because the regex requires the whole text to be a page number. If you're seeing artifacts in retrieval, the filter probably isn't broken — the PDF parser is gluing artifacts onto real content, and tightening the filter alone won't fix it.

After tweaking filter settings, **you must re-ingest** — the index doesn't update retroactively. Reingest deletes old chunks/vectors first.

### Tests

`tests/conftest.py` stubs out heavy deps (chromadb, sentence-transformers, ollama) so unit tests run in CI without them. Tests that genuinely need a heavy dep skip via `pytest.importorskip`. When adding a test that exercises a new external lib, follow the same pattern.

### Prompts

All prompt templates live as `.md` files in `src/hrag/prompts/` and are loaded with `Path(__file__).parent / "prompts" / "<name>.md"` and `.format(...)`. The setuptools config in `pyproject.toml` packages them via `package-data`. Keep prompt edits in the `.md` file, not in Python string literals.

## Conventions

- Type hints required (`from __future__ import annotations` everywhere); `pydantic` for config models, `dataclasses` for runtime value objects (`types.py`).
- Heavy-dep imports go inside functions/methods, not at module top, so import-time failures don't cascade.
- Provider/retriever/reranker classes carry a `name: str` class attribute for logging/UI.
- Database access goes through `Database` (`db/connection.py`); use `with self.db.conn:` for transaction blocks. Always `self.db.commit()` at the end of a write path.
