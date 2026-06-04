# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

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

- **Opus** — algorithmically tricky, security-sensitive, cross-cutting wiring, novel design with multiple invariants (graph store with synonym merging, MST + redundancy pruning, query router with RRF fusion, prompt design).
- **Sonnet** — well-scoped mechanical work, single-file modules following an existing pattern, test-only additions, config/schema/dep edits, factory registrations.

Fan out in waves: agents on disjoint files run in parallel (single message, multiple `Agent` calls); serialize only when later work depends on an earlier deliverable. After each wave the main thread verifies — `pytest`, ruff, schema sanity. Don't shard smaller than ~one file per agent. Trivial single edits: just do it inline.

## Project status (phases 1–12 complete; ~1051+ tests)

Every phase's feature sits behind a config flag (defaults noted). Default retriever is `taxonomy` (falls back to `vector` when the tree is empty). Heavy-dep tests skip cleanly via `tests/conftest.py` stubs.

**P1** — walking-skeleton RAG (ingest → vector → rerank → answer).

**P2** (`kg.enabled`) — KG triples + PPR + GraphRAG communities + LLM router + KG2RAG MST. `kg/{builder,store,ner,ppr,communities}.py` (NetworkX MultiDiGraph, synonym merge cos≥0.8, SQLite mirror; SpacyNER default, LLMNER opt-in; scipy PPR; Leiden ×3 resolutions → Chroma `hrag_community_summaries` + SQLite mirror), `retrieval/{kg_ppr,community,router,mst}.py`.

**P3 — Personalization** — memory layer + LLM-proposed taxonomy.
- Memory: `memory/profile.py` (ProfileStore → `{user_profile}` in every answer prompt), `memory/store.py` (EpisodicMemoryStore; `/remember /recall /forget`; episodic chunks in `chunks` with `source_type='episodic'`), `memory/{auto_extract,extractor}.py`, `context/builder.py`.
- Taxonomy: `taxonomy/{store,builder,assigner}.py` (packed-float32 centroids + beam descend; parallel summaries → LLM tree `max_tokens=8192` + truncation-salvage; cosine descent + LLM tiebreak when top-2 <0.05), `retrieval/taxonomy.py` (`describe_last_descend()`), tables `kg_taxonomy_{nodes,assignments,doc_meta}`, flag `taxonomy.include_episodic` (default true).

**P4** (`compaction.*`, default OFF) — RAGate + clue + dialog MST + `[UNCERTAIN]` masking. `gating/{gate,clue,uncertain}.py`, `context/dialog_mst.py`. Order: dialog compaction → RAGate (FACTUAL only; SKIP → `plan.scope="none"`, intent→GENERAL) → clue (replaces retrieval query) → `render_uncertain`/silent `strip_uncertain`. Events `dialog_compact gate_check clue_generate uncertain_render`.

**P5** (web ergonomics + extensibility) —
- Web UX: memory CRUD, multipart `POST /api/ingest`, background jobs (`?background=true` → `GET /api/jobs/{id}`, `jobs` table).
- Pluggable backends: `kg/backends/{base,networkx,neo4j}.py` (KGBackend Protocol, 18 methods), `retrieval/backends/{base,chroma,sqlite_vec}.py` (VectorBackend Protocol). Factories `KGStore.from_config`, `_build_vector_backend`.
- Self-improvement: `feedback` table, `POST/GET/DELETE /api/feedback`, 👍/👎 UI, `hrag export-training-pairs` JSONL.
- Equation-aware ingest: `ingest/math_detect.py`, chunker boundary nudge so math never splits, `quality.py` carve-outs on `metadata.has_math`.

**P6** —
- sqlite_vec real impl (`vec0` virtual table, cosine → Chroma-shape distances, where-compiler `$and/$or/$eq/$ne/$in`); neo4j real impl (18 methods, side-effect-free).
- Adaptive top_k per intent (`retrieval.adaptive_enabled`, default off): `_adaptive_top_k` maps intent → `(top_k_vector, top_k_final)`; `(None,None)` skips retrieval; PERSONAL broadens `source_types` + stable-sorts episodic to top. Events `adaptive_top_k retrieval_skipped episodic_bias_applied`.
- `cfg.llm.keep_alive` as top-level `chat()` kwarg (default `"30m"`).

**P6+7 wrap-up** — `retrieval.adaptive_retriever_per_intent` (intent→retriever; event `adaptive_retriever_picked`); `feedback_stats.py::feedback_summary(db)` shared CLI+web + `GET /api/feedback/stats`; `cfg.llm.num_keep` → `options.num_keep`; `EmbeddingsConfig.suggested_models` + `dimension_for_model()`; `ingest/nougat_loader.py` deferred imports, dispatch when `cfg.ingest.use_nougat` + dep present (silent PyMuPDF fallback).

**P7-A** (math handling) — `has_unicode_math()`; `has_math = _has_latex_math or has_unicode_math`; `_expand_math_meta` in HeuristicRewriter; `_is_math_meta_query` (pure regex) + `where={"has_math":True}` pushdown + lowered rerank threshold + empty-result fallback + optional `prompts/extract_formulas.md`. Events `math_meta_filter math_meta_filter_fallback formula_extract`. Retriever Protocol gains `where`.

**P8** (`interaction.review_enabled`, default OFF) — interactive review loop: pause between retrieval and answer. `interaction/{store,review}.py`; emits `review_required` SSE (sources, clue, descend, 0–3 rephrasings) → frontend `#review-modal` → `POST /api/chat/turns/{id}/resume` (idempotent). Actions `continue/filter/rephrase/general/clarify/expand_doc/redescend/abort`. Prompts `rephrase clarify followups why_source`.

**P8.1** — `retrieval.always_include_episodic` (default True): episodic for ALL intents, PERSONAL stable-sort preserved on top.

**P9** (speed / observability / accuracy; flags default OFF except 9.3/9.4/9.11/9.12 ON):
9.1 `run_latency.py` harness · 9.2 `async_preflight_enabled` · 9.3 `embeddings.query_cache_enabled`(ON)+`query_cache_size` · 9.4 `llm.warmup_on_init`(ON)+`num_keep_auto` · 9.5 `llm.anthropic_prompt_caching` (≥1024 chars) · 9.6 `compaction.combined_preflight_enabled` (`prompts/combined_preflight.md` → `{intent,gate,clue}`, re-emits per-stage with `source="combined"`) · 9.7 `embeddings.embed_precision` · 9.8 `retrieval.rerank_quantize` · 9.9 `rerank_fallback_telemetry_enabled`+`rerank_fallback_events` · 9.10 `first_token_latency_enabled` (`messages.metadata.latency.first_token_ms`) · 9.11 `router_short_circuit`(ON) · 9.12 `kg.dedup_enabled`(ON)+`kg_canonical_triples` · 9.13 `context_compression_enabled`+`context_budget_chars` · 9.14 `ingest.contextual_retrieval_enabled` (Anthropic Contextual Retrieval — per-chunk LLM context line prepended to `embedding_text`; `ingest/contextual.py`, cache `ingest_context_cache`, `prompts/contextual_chunk.md`) · 9.15 `feedback_reranking_enabled` · 9.16 `crag_enabled`+`crag_score_floor`+`crag_retry_top_k_multiplier` · 9.17 `compaction.self_rag_enabled` (`extract_uncertain_spans()`).

**P10** (embedding speed; flip via `config.yaml`) — `providers/embeddings.py` rewritten.
- Track A (no re-ingest): ST native `backend="onnx"`, six `precision` modes via `_resolve_precision(cfg)` (`fp32`|`fp16`|`bf16`|`onnx`|`onnx_int8`|`openvino`), each silent fp32 fallback; new `embed_batch_size`(32), `embed_onnx_cache_dir`, `embed_onnx_optimization`("O3").
- Track B (re-ingest): `BAAI/bge-small-en-v1.5` (384-dim) in `suggested_models`; dim-mismatch guard `Orchestrator._check_embedding_dim_match()` via `VectorBackend.dim()`.
- Track C: `Model2VecProvider` (lazy `model2vec`, explicit L2-norm). Dep groups `openvino`, `model2vec`; `quantize` = `onnxruntime`.

**P11** (`retrieval.reflection_mode` = `off`|`regex`|`hybrid`, default hybrid; live-switchable via `GET/POST /api/config`) — reflective/opinion personal questions ("what do you think about me?"): broaden retrieval with profile terms + render `prompts/answer_personal_reflect.md` (registry `render_personal_reflect`). Detection is two-tier asymmetric: `intent.is_reflective_strict()` (high precision, EN + ZWNJ-FA + Finglish; only it or the LLM judge may coerce non-PERSONAL→PERSONAL), `is_reflective_query()` (recall tier, within already-PERSONAL turns), `has_self_reference()` (gates the LLM call). Hybrid LLM signal from `combined_preflight.reflective` or lazy `ReflectiveClassifier` (`prompts/reflective_check.md`). Event `reflective_check {mode,strict,loose,llm,reflective}`.

**P12** (bilingual GUI + hybrid-keyword taxonomy) —
- GUI (`web/static/*`): KaTeX math on completed messages only; `:lang(fa)` Persian typography (Vazirmatn, RTL); per-message actions (copy-md/regenerate/edit-resend); `decorateBubble()` (callouts, code chips, `.table-wrap`, `linkCitations`); `renderDescend` with matched-keyword chips. Surfaces: taxonomy keyword editor (`PUT /api/taxonomy/nodes/{id}`, `POST /api/taxonomy/keywords`), Profile drawer (`/api/profile`, `POST /api/memories/extract`), Graph page (D3, `GET /api/kg/graph|stats`), feedback export.
- Taxonomy hybrid routing: per-node `keywords` (additive JSON TEXT col). Gen = LLM-in-propose (`taxonomy_propose.md`) + local backfill (`taxonomy/keywords.py`, pure stdlib YAKE-style, EN/FA + ZWNJ). `beam_descend` blends `cosine + keyword_weight*overlap` (no-op until keyworded); `DocAssigner._keyword_tiebreak`. In-memory node cache (TTL 30s, invalidated on `_commit`). `TaxonomyConfig`: `keyword_routing_enabled`(ON), `keyword_weight=0.2`, `keywords_per_node=8`, `keyword_tiebreak_skip`, `cache_tree_in_memory`. CLI `hrag taxonomy keywords [--force]`.

**Deferred (not blockers):** running Nougat over a real corpus; feedback-loop fine-tuning training infra; cross-turn KV-cache reuse via Ollama prefix tokens beyond `num_keep`.

## Contracts — load-bearing invariants (must survive future phases)

Numbered globally; the phase tag shows when each was introduced.

**P3 (1–8):**
1. `documents/chunks.source_type` ('document'|'episodic') stays; new source_types must teach `taxonomy/builder.py::_list_docs`.
2. Taxonomy auto-assign hook in `IngestPipeline.ingest_document` ("5c" block) runs for every eligible ingest; new ingest features layer in BEFORE it.
3. `kg_taxonomy_*` migrations stay `CREATE TABLE IF NOT EXISTS` and back-compatible.
4. `Retriever.retrieve(query, user_id, top_k, source_types)` signature is locked (`/recall` depends on `source_types`).
5. `LLMProvider.complete(prompt, *, temperature, max_tokens)` keeps accepting explicit `max_tokens` (Ollama silently truncates long JSON otherwise).
6. `EmbeddingProvider.embed()` returns L2-normalised vectors (`beam_descend` treats cosine == dot product).
7. `progress` keeps emitting `taxonomy_descend` (with `describe_last_descend()`) whenever the active retriever is `name=="taxonomy"`.
8. `TaxonomyStore.clear(user_id)` is non-destructive to `kg_taxonomy_doc_meta` unless `wipe_doc_meta=True` (CLI `taxonomy clear` does NOT wipe).

**P4 (9–11):**
9. Events `gate_check clue_generate dialog_compact uncertain_render` keep firing when their `compaction.*` flags are on.
10. `render_uncertain` is idempotent (applying twice must not double-render).
11. Silent `strip_uncertain` (when `mask_uncertain=False`) must keep running so raw `[UNCERTAIN]` never leaks.

**P6 (12–15):**
12. Events `adaptive_top_k retrieval_skipped episodic_bias_applied` fire when `retrieval.adaptive_enabled` is true.
13. `_adaptive_top_k(cfg, intent)` is a pass-through when `adaptive_enabled` is False — returns `(top_k_vector, top_k_final)` unconditionally.
14. `OllamaProvider._build_chat_kwargs` emits `keep_alive` as a TOP-LEVEL kwarg, NOT inside `options` (Ollama ignores it there).
15. `Neo4jBackend`/`SqliteVecBackend` constructors are side-effect-free; missing-backend errors fire on first method call, not on `import hrag`.

**P7-A (16–19):**
16. `Retriever.retrieve(query, user_id, top_k, source_types, intent_hint, where)` — every retriever accepts `where` (even if ignored); wrappers (`doc_scope`, `router`, `hybrid`) thread it through.
17. Events `math_meta_filter math_meta_filter_fallback formula_extract` fire when their flags are on.
18. `scripts/backfill_has_math.py` is idempotent.
19. `_is_math_meta_query` is pure regex (no LLM) — fast and deterministic on every turn.

**P6+7 wrap-up (20–24):**
20. `adaptive_retriever_per_intent` value is `"default"` or one of `{vector,bm25,hybrid,kg_ppr,community,router,taxonomy}` (web API validates; do not loosen).
21. `feedback_summary(db)` shape is stable: `{thumbs_up, thumbs_down, total, sessions, top_negative}` (+ `rerank_fallback_count` from 9.9).
22. `OllamaProvider._build_options` emits `num_keep` INSIDE `options` (top-level is ignored).
23. `hrag.ingest.nougat_loader` import is side-effect-free (zero `nougat`/`nougat_ocr` in `sys.modules` until a method is called).
24. `EmbeddingsConfig.suggested_models` is advisory only — the provider still accepts any HF model id.

**P8 (25–32):**
25. `interaction.review_enabled=False` (default) is a true no-op: zero new SSE events, LLM calls, DB writes.
26. `InteractionStore` TTL cleanup runs on a daemon thread, never blocks the orchestrator; `store.shutdown()` (from `Orchestrator.close()`) double-shutdown-safe.
27. SSE event types `review_required review_resolved followups review_warning` + `turn_id` on `start` are contract (renaming breaks the frontend).
28. `POST /api/chat/turns/{id}/resume` is idempotent on any action — retry returns `{"accepted": false, "reason": "turn_not_found_or_already_decided"}`, never 4xx/5xx.
29. `messages.metadata` is JSON-encoded TEXT; readers tolerate NULL and malformed JSON; column stays optional.
30. `should_pause()` is pure (no LLM, DB, or progress callback).
31. SSE relay routes `review_required`/`review_resolved`/`followups` as DEDICATED event types, not nested under `event: progress`.
32. `chat()` emits `turn_id` on the `start` payload (frontend needs it to POST resume).

**P8.1 (33):**
33. `always_include_episodic=True` (default) includes episodic for ALL intents (PERSONAL stable-sort on top); False reverts to per-intent strictness; surfaced via GET/POST `/api/config`.

**P9.14 (34–37):**
34. `ingest.contextual_retrieval_enabled=False` (default) is a true no-op: `ContextualAugmenter` not even instantiated.
35. `ContextualAugmenter.augment_chunks` mutates ONLY `chunk.embedding_text`; `chunk.text` is read-only (KG triples, taxonomy, answer prompt consume `text`; the prefix must never leak — preserves contract 1).
36. Cache `ingest_context_cache` keyed by `sha256(chunk_text||doc_hash||model_name)` — re-ingesting unchanged doc under same LLM is free; changing the model invalidates cleanly.
37. Episodic memories (`source_type=="episodic"`) are skipped by the augmenter (check before instantiation).

**P9 (38–44):**
38. `combined_preflight_enabled=True` requires `gate_enabled`/`clue_enabled`/`intent.enabled` all True; constructor refuses otherwise (orchestrator silently falls back).
39. With combined preflight on, orchestrator STILL emits `intent_check gate_check clue_generate` (payloads carry `source="combined"`, shapes unchanged).
40. `rerank_fallback_events` is append-only; inserts wrapped in try/except so a DB lock can't regress the chat path.
41. `messages.metadata.latency.first_token_ms` is the agreed key path (latency harness reads it).
42. `extract_uncertain_spans(answer, max_chars_per_span)` is pure; Self-RAG runs it on the raw answer BEFORE `render_uncertain`/`strip_uncertain`.
43. Phase-9 events `combined_preflight async_preflight context_compress crag_reroute self_rag feedback_rerank_applied rerank_fallback_logged` are contract (whichever subset their flags activate).
44. Every Phase-9 flag defaults OFF except the pure-speed ON exceptions: `query_cache_enabled`(9.3), `warmup_on_init`(9.4), `router_short_circuit`(9.11), `kg.dedup_enabled`(9.12).

**P10 (45–49):**
45. Every `EmbeddingProvider.embed()` returns L2-normalised vectors — ST uses `normalize_embeddings=True`; `Model2VecProvider` does explicit `vecs / clip(norm, 1e-9)` (model2vec isn't unit-norm). Protects contract 6.
46. `_resolve_precision(cfg)` is the single source of truth — reads `cfg.precision` first, falls back to `cfg.embed_precision` (9.7, kept for back-compat), defaults `"fp32"`.
47. Every ST precision branch silently falls back to `fp32` + `logger.warning` when its optional dep is missing; the orchestrator never crashes on a precision-load failure (`self._backend` reflects the loaded mode).
48. `_check_embedding_dim_match()` raises `RuntimeError` ONLY on a confirmed mismatch in a populated collection. Empty (`dim() is None`) → no-op; backend access raising → `logger.warning` only. `VectorBackend.dim()` keeps returning `Optional[int]`.
49. `Model2VecProvider` and the `openvino`/`onnx` ST branches keep imports lazy (zero `model2vec`/`openvino`/`optimum` in `sys.modules` at `import hrag`); selecting an uninstalled backend fails at provider construction with a clear error.

**P11 (50–53):**
50. `reflection_mode == "off"` is a true no-op (no coercion, no synthesis prompt, `ReflectiveClassifier` not constructed). `"regex"` is pure-function only — never calls the LLM judge.
51. `is_reflective_strict()`, `is_reflective_query()`, `has_self_reference()` are pure. Only `is_reflective_strict()` (or the hybrid LLM signal) may COERCE non-PERSONAL→PERSONAL; the loose recall tier must NOT coerce (protects `"describe me a function"`).
52. `PreflightDecision.reflective` is `Optional[bool]` — `None` (field omitted → fall back to the standalone judge) is distinct from `False`; `_extract_json` tolerates extra/missing keys.
53. `reflective_check` fires whenever `reflection_mode != "off"` with `{mode,strict,loose,llm,reflective}`. The reflective render path consumes `chunk.text`, never the profile-augmented retrieval query (preserves contracts 1 / 35).

**P12 (54–57):**
54. `kg_taxonomy_nodes.keywords` is additive JSON TEXT; `_row_to_node` tolerates a missing column / NULL / malformed JSON → `[]`; migration is guarded `ALTER TABLE ... ADD COLUMN`.
55. `beam_descend` with `keyword_weight == 0` (or no `query_keywords`, or an unkeyworded tree) is byte-identical to pre-P12 dense-only descent. `keyword_routing_enabled` defaults ON but is inert until a tree is keyworded.
56. `taxonomy/keywords.py` (`tokenize`, `extract_keywords`, `keyword_overlap`) is pure, bilingual (EN + ZWNJ-FA); `keyword_overlap` returns 0.0 when either side is empty.
57. The node cache is invalidated on every node-mutating write (all commit through `TaxonomyStore._commit`) and self-expires after `cache_ttl_s`; `cache_tree_in_memory=false` sets TTL 0 (disabled). Cached reads == fresh reads.

## Manual triggers

```bash
# Phase 2 (KG router) — config.yaml: kg.enabled: true AND retrieval.retriever: router
rm -rf data/store.sqlite data/chroma data/kg
hrag init
hrag ingest "D:/Selected Dynamic Papers" --recursive
hrag rebuild-kg            # or, if chunks already exist, build only the KG layer

# Phase 3 — taxonomy
hrag taxonomy build        # parallel summaries → LLM tree → materialize
hrag taxonomy show         # text outline (lists keywords)
hrag taxonomy clear        # drop tree (preserves doc-meta cache)

# Phase 4 — compaction (env or CLI flags)
hrag chat --gate --clue --mask-uncertain
```

## Commands

```bash
python -m pip install -e .[dev]      # editable install
hrag init                            # SQLite + chroma

hrag ingest <path>                   # single file
hrag ingest <dir> --recursive        # whole tree

hrag chat                            # slash: /help /sources /status /exit /remember /recall /forget /profile
hrag chat --fast                     # top_k=4, no rerank
hrag chat --retriever taxonomy       # default; or vector/bm25/hybrid/kg_ppr/community/router
hrag chat --no-rerank

hrag remember "<text>" | <path>      # save / bulk-import episodic
hrag memory list
hrag memory extract --session <id>   # mine preferences

hrag web                             # FastAPI SPA
hrag list-docs

pytest -q                            # full suite
pytest tests/test_orchestrator.py -q
pytest -k "rerank" -q
ruff check src tests
```

## Configuration

`config.yaml` is canonical; loaded into pydantic models in `src/hrag/config.py`. Overrides:
1. **Env vars** prefixed `HRAG_` with `__` as section separator: `HRAG_LLM__MODEL=gemma:7b`.
2. **Per-session CLI flags** on `hrag chat` (`--retriever`, `--reranker`, `--no-rerank`, `--fast`).

`Config.resolve(relative_path)` resolves storage paths against `project_root` (cwd at load time).

## Architecture

Modular stack — CLI · Web SPA · Orchestrator · Retrievers · Personalization · Storage · Providers. Every layer sits behind an interface so it can be swapped (Chroma↔sqlite-vec, NetworkX↔Neo4j). README.md has the diagram.

### Pipeline (`orchestrator.py`)

`Orchestrator.chat(question, user_id, session_id, progress, stream)`:
1. Ensure user; create session row if `session_id is None`.
2. Persist user message; `_load_history` loads up to 10 prior messages (excludes the just-inserted one).
3. `retriever.retrieve(question, user_id, top_k=cfg.retrieval.top_k_vector)` → `list[RetrievalResult]`.
4. If a reranker is set: rerank with `cfg.retrieval.rerank_threshold`, cap at `top_k_final`. **Critical fallback**: if rerank drops everything, fall back to unreranked top-k (else the LLM answers "I couldn't find that" even when retrieval succeeded). Emits `fallback_used: True` in `rerank_done`.
5. Render `prompts/answer.md` (RAFT-style `##begin_quote##` CoT) with `{user_profile, conversation_history, retrieved_passages, question}`.
6. `llm.generate_stream` (stream) or `llm.complete`.
7. Persist assistant message + commit.

`progress` is `Optional[Callable[[str, dict], None]]` — event list at the top of `orchestrator.py`.

### Pluggable layers (factories)

| Layer | Factory | Implementations |
|---|---|---|
| LLM | `providers/llm.py::get_llm_provider` | Ollama, OpenAI, Anthropic |
| Embeddings | `providers/embeddings.py::get_embedding_provider` | sentence-transformers, OpenAI, model2vec |
| Retriever | `retrieval/factory.py::build_retriever` | Vector, BM25, Hybrid (RRF), KGPPR, Community, QueryRouter, Taxonomy (default) |
| Reranker | `retrieval/factory.py::build_reranker` | CrossEncoder (default), LLMReranker (0–3), BatchedLLMReranker |

Reranker thresholds differ: `cross_encoder` is a logit (default `-5.0`, permissive); `llm`/`batched_llm` is int 0–3 (use `2` if switching).

### HeteRAG dual-text

`types.Chunk` carries two text fields:
- `embedding_text` — title/section prepended (HeteRAG fusion); the embedder sees this.
- `text` — raw chunk; the LLM sees this at generation.

Anything embedding a chunk uses `embedding_text`. Anything showing or LLM-feeding uses `text`. Don't conflate them.

### Storage

- **SQLite** (`db/schema.sql`, `db/connection.py`): users, documents, chunks, sessions, messages, `preferences`, `kg_{nodes,edges,communities,triple_cache}`, `kg_taxonomy_{nodes,assignments,doc_meta}`, `jobs`, `feedback`, `kg_canonical_triples`, `rerank_fallback_events`, `ingest_context_cache`. FK on `chunks` is `doc_id` (not `document_id`).
- **ChromaDB** (`retrieval/vector.py::VectorStore`): vectors + metadata mirror. Community summaries in `hrag_community_summaries`.
- **NetworkX**: MultiDiGraph (phrase + passage nodes), mirrored in SQLite.

`user_id` is plumbed through every table and API even in single-user mode — don't strip it.

### Ingest pipeline (`ingest/pipeline.py`)

`load_document` (PyMuPDF / python-docx / markdown-it-py / plain) → `chunk_document` (token-budgeted, structure-aware; HeteRAG fusion) → `filter_chunks` (`quality.py`) → embed in batches of `embed_batch_size` (32) → upsert into SQLite + ChromaDB (`delete_doc` then `add_chunks`, so reingest is idempotent).

### Chunk quality filter (`ingest/quality.py`)

Drops typically 70–85% of chunks on academic PDFs. Configured under `chunking.quality.*`; pure-function, no I/O. **Each check matches the *entire* chunk** — a chunk starting `"250\n6We use..."` doesn't match `_RE_PAGE_ARTIFACT` (the regex requires the whole text to be a page number). If artifacts surface in retrieval, the PDF parser is gluing artifacts onto real content; tightening the filter alone won't fix it. After tweaking settings, **re-ingest** (the index doesn't update retroactively).

### Prompts

Templates are `.md` files in `src/hrag/prompts/`, loaded via `Path(__file__).parent / "prompts" / "<name>.md"` + `.format(...)` and packaged by `pyproject.toml` package-data. Edit the `.md` file, not Python string literals.

## Conventions

- Type hints required (`from __future__ import annotations` everywhere); pydantic for config models, dataclasses for runtime value objects (`types.py`).
- Heavy-dep imports go inside functions/methods, not module top, so import-time failures don't cascade.
- Provider/retriever/reranker classes carry a `name: str` class attribute for logging/UI.
- DB access goes through `Database` (`db/connection.py`); use `with self.db.conn:` for transactions and `self.db.commit()` at the end of every write path.
