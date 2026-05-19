# Phase 3 Acceptance Benchmark Results

Run timestamp: 2026-05-15T21:21:48.076033+00:00
Total wall time: 3.32s
Pass rate: **16/16**

## Config snapshot

| key | value |
|---|---|
| embedder | FakeEmbedder (hash-based, deterministic, 384-dim) |
| llm | FakeLLM (canned responses, no Ollama needed) |
| vector_store | real ChromaDB in per-test tmp dir |
| rerank_enabled | False |
| doc_scope_enabled | False |
| kg_enabled | varied |
| notes | Phase 3 is mostly integration-level around storage/retrieval glue. … |

## Summary

| id | category | result | time (s) | message |
|---|---|---|---|---|
| t01_remember_writes | memory_write | PASS | 0.895 | memory_id=episodic:default:3548a56…, 1 chunk persisted |
| t15_remember_latency_under_100ms | performance | PASS | 0.082 | hot-path /remember took 8.3 ms (budget 100 ms) |
| t16_bulk_200_under_10s | performance | PASS | 1.412 | 200/200 saved in 1.38s (145 notes/sec, budget 10s) |
| t02_recall_returns_relevant | memory_read | PASS | 0.089 | target in top-3 hits; all source_type=episodic |
| t03_memory_competes_with_docs | memory_read | PASS | 0.073 | top-k contains both source_types (['document', 'episodic']); memory episodic:default… surfaced |
| t04_profile_upsert_idempotent | profile | PASS | 0.038 | pref_id stable across upsert; value updated to 'scientist' |
| t05_profile_render_grouped | profile | PASS | 0.034 | groups present; min_confidence + max_items both honoured |
| t06_context_builder_into_prompt | profile | PASS | 0.033 | rendered profile lines present in answer prompt |
| t07_forget_tombstones | forget | PASS | 0.061 | excluded=1 set; list/count both hide it |
| t08_forget_by_query | forget | PASS | 0.099 | target returned; 3 candidate(s) |
| t09_user_scoping | scoping | PASS | 0.070 | per-user counts + retrieve scoping clean |
| t10_bulk_import_iter | bulk_import | PASS | 0.094 | extracted 6 items; all persisted |
| t11_extractor_robust_json | extraction | PASS | 0.000 | fenced ✓, prose-prefix ✓, malformed → [] ✓ |
| t12_auto_extractor_min_conf | extraction | PASS | 0.033 | only above-threshold candidate upserted; topics=['lang'] |
| t13_kg_skip_episodic | guards | PASS | 0.229 | KG extraction skipped for source_type='episodic' |
| t14_quality_skip_episodic | guards | PASS | 0.077 | episodic survived; document control filtered (expected) |

## Per-test detail

### t01_remember_writes — memory_write

**Summary:** EpisodicMemoryStore.add() persists one chunk with source_type='episodic' and returns 'episodic:<uid>:<hex>'.

**Result:** PASS in 0.895s

**Message:** memory_id=episodic:default:3548a56…, 1 chunk persisted

*Notes: Sentinel against /remember silently dropping writes.*

### t15_remember_latency_under_100ms — performance

**Summary:** Single /remember (FakeEmbedder) completes under 100 ms wall.

**Result:** PASS in 0.082s

**Message:** hot-path /remember took 8.3 ms (budget 100 ms)

*Notes: User invariant: /remember must feel instantaneous.*

### t16_bulk_200_under_10s — performance

**Summary:** add_batch over 200 short notes (FakeEmbedder) finishes under 10 s wall.

**Result:** PASS in 1.412s

**Message:** 200/200 saved in 1.38s (145 notes/sec, budget 10s)

*Notes: Stretch target — proves the bulk path scales without a per-note bottleneck.*

### t02_recall_returns_relevant — memory_read

**Summary:** After 3 memories ingested, /recall on a topical query returns the on-topic chunk in the top result.

**Result:** PASS in 0.089s

**Message:** target in top-3 hits; all source_type=episodic

*Notes: Validates the source_types='episodic' filter at retrieval time.*

### t03_memory_competes_with_docs — memory_read

**Summary:** With source_types=None, an episodic memory matching the query out-ranks unrelated document chunks in the top-k.

**Result:** PASS in 0.073s

**Message:** top-k contains both source_types (['document', 'episodic']); memory episodic:default… surfaced

*Notes: Default behaviour: memories compete in the same top-k as docs.*

### t04_profile_upsert_idempotent — profile

**Summary:** Upsert on (user_id, topic, polarity) updates the existing row instead of creating a duplicate.

**Result:** PASS in 0.038s

**Message:** pref_id stable across upsert; value updated to 'scientist'

*Notes: Backed by the UNIQUE index in db/migrations.py.*

### t05_profile_render_grouped — profile

**Summary:** ProfileStore.render groups prefs by polarity (Facts / Style / Likes / Dislikes), drops below min_confidence, caps at max_items.

**Result:** PASS in 0.034s

**Message:** groups present; min_confidence + max_items both honoured

*Notes: Output goes verbatim into the answer prompt — keep the grouping stable.*

### t06_context_builder_into_prompt — profile

**Summary:** Orchestrator.chat() renders the profile into the answer prompt at the {user_profile} placeholder (no longer the empty string).

**Result:** PASS in 0.033s

**Message:** rendered profile lines present in answer prompt

*Notes: End-to-end glue from preferences table to LLM-facing prompt.*

### t07_forget_tombstones — forget

**Summary:** EpisodicMemoryStore.forget flips chunks.excluded=1 and the memory no longer appears in list_recent.

**Result:** PASS in 0.061s

**Message:** excluded=1 set; list/count both hide it

*Notes: Tombstone semantics must hold across read paths.*

### t08_forget_by_query — forget

**Summary:** forget_by_query returns chunk_ids of semantically-matching memories.

**Result:** PASS in 0.099s

**Message:** target returned; 3 candidate(s)

*Notes: /forget by free-text relies on this for confirmation flow.*

### t09_user_scoping — scoping

**Summary:** Memories under user 'alice' do not surface in count/list/recall for user 'bob' and vice versa.

**Result:** PASS in 0.070s

**Message:** per-user counts + retrieve scoping clean

*Notes: Critical for multi-user safety.*

### t10_bulk_import_iter — bulk_import

**Summary:** _iter_memory_texts_from_path expands a temp folder of .md and .txt into multiple items; add_batch persists them.

**Result:** PASS in 0.094s

**Message:** extracted 6 items; all persisted

*Notes: The 'remember a lot of documents fast' code path.*

### t11_extractor_robust_json — extraction

**Summary:** PreferenceExtractor parses markdown-fenced JSON and prose-prefixed JSON without crashing.

**Result:** PASS in 0.000s

**Message:** fenced ✓, prose-prefix ✓, malformed → [] ✓

*Notes: Small local LLMs frequently wrap JSON in ```json fences.*

### t12_auto_extractor_min_conf — extraction

**Summary:** SessionAutoExtractor does NOT upsert candidates below auto_extract_min_confidence.

**Result:** PASS in 0.033s

**Message:** only above-threshold candidate upserted; topics=['lang']

*Notes: Prevents low-signal noise from polluting the profile.*

### t13_kg_skip_episodic — guards

**Summary:** Ingesting a Document(source_type='episodic') with kg.enabled=true does NOT instantiate TripleExtractor.

**Result:** PASS in 0.229s

**Message:** KG extraction skipped for source_type='episodic'

*Notes: Keeps /remember under 100 ms even on a KG-enabled corpus.*

### t14_quality_skip_episodic — guards

**Summary:** A 3-token episodic memory survives the chunker even with chunking.quality.enabled=true (min_tokens=30 doesn't apply).

**Result:** PASS in 0.077s

**Message:** episodic survived; document control filtered (expected)

*Notes: Otherwise short notes like 'Postgres > MySQL' would be silently dropped at ingest.*

---

Run with: `python tests/benchmark/run_phase3.py`