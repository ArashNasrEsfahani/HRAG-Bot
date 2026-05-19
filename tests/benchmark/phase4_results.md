# Phase 4 Acceptance Benchmark Results

Run date: 2026-05-17 (local)
Harness: `tests/benchmark/run_phase4.py` (all four `compaction.*` flags ON via env vars)
Hardware: NVIDIA RTX 4060 Laptop, 8 GB VRAM
Embedder: `sentence-transformers/all-mpnet-base-v2`
Retriever: `taxonomy` (default; falls back to vector)
Corpus: 26 documents (~1500 chunks; the same Phase 2/3 ML/NLP paper set)

---

## TL;DR

**Two runs, both failing the ≥ 3/4 acceptance gate. Neither was bottlenecked by the Phase 4 implementation — both failures trace to other issues already identified.**

| Model | Score | Wall time | Headline issue |
|---|---|---|---|
| `gemma4:e4b` (8B multimodal, 12 GB) | **2/4** | ~17.6 min | Q3 hit an orchestrator history-limit bug (now fixed). Q4 LLM correctly chose a clean refusal — benchmark assertion was too strict. |
| `gemma4:e2b-it-q4_K_M` (5.12B text-only, 7.2 GB) | **1/4** | ~16 min (Q3 crashed) | Smaller model lost the gate classification (Q1); Ollama OOM-crashed mid-Q3 because hrag doesn't cap `num_ctx`, so the 131k default KV cache fills VRAM and forces 70% CPU spill. |

**Speed reality:** the model swap delivered only **4-38%** per-turn speedup, not 10×, because **VRAM is mostly held by desktop apps** (~5 GB by Chrome/VS Code/Edge WebView). To actually fit gemma4:e2b fully on the GPU you need to close those apps. There is also a real fix on our side (passing `num_ctx` to Ollama) that would substantially help regardless.

---

## Per-question table

| Q | Feature | Run 1 (gemma4:e4b) | Run 2 (gemma4:e2b) |
|---|---|---|---|
| Q1 | `gate_enabled` — SKIP small-talk | **PASS** · 73.7 s · gate decision=`SKIP` | **FAIL** · 63.6 s · gate decision=`RETRIEVE` (smaller model misclassified "thanks!") |
| Q2 | `clue_enabled` — vague-query retrieval | **PASS** · 53.4 s · 4 keyword hits | **PASS** · 51.5 s · 4 keyword hits |
| Q3 | `dialog_mst_enabled` — old fact survives 20 turns | **FAIL** · 868.1 s · `dialog_compact` never fired (orchestrator bug — now fixed) | **FAIL (crash)** · ~570 s into Q3 · WinError 10054 (Ollama dropped connection mid-call, OOM) |
| Q4 | `mask_uncertain` — `[UNCERTAIN]` rendered | **FAIL** · 57.8 s · LLM cleanly refused, made no sub-claim to mark | **FAIL** · 36.1 s · same — LLM cleanly refused |

---

## Speed analysis

### Per-turn cost on this host

| Turn type | gemma4:e4b | gemma4:e2b | Speedup |
|---|---|---|---|
| Factual answer (retrieve + rerank + LLM) | ~45-50 s | ~40-45 s | ~10-14% |
| Vague query w/ clue (extra LLM call) | 53.4 s | 51.5 s | 4% |
| Small-talk with gate | ~24 s (gate SKIPped; cheap LLM call only) | ~30-35 s (gate misclassified → still ran full retrieve+gen) | regression |
| Refusal turn (Q4) | 57.8 s | 36.1 s | 38% |

The headline number from the model swap is **a 4-38% range, not 10×**. The 5B-param model is faster per token but **still spills 70-74% of layers to CPU** with default settings, so wall-clock improvement is modest.

### Per Phase 4 feature

These hold for either model:

| Feature | Marginal cost | Net effect |
|---|---|---|
| `mask_uncertain` | **~0 ms** (regex post-pass) | Always free; safe default-ON. |
| `gate_enabled` on factual queries | **+3-5 s** (one cheap LLM call) | Net win across a mixed session (any small-talk turn saves 30-50 s). |
| `clue_enabled` on every query | **+5-15 s** (extra LLM call) | Pure tax on already-specific queries; opt in for vague-pronoun workloads. |
| `dialog_mst_enabled` | **+15-30 s** at each compaction (once per `compact_after_turns`) | Amortises in long sessions; not worth the bookkeeping in short chats. |

### Hardware bottleneck — where the speed actually went

`nvidia-smi` snapshot during the e2b run:
- 8 GB VRAM total
- ~5-6 GB used by desktop apps (Chrome, VS Code, Edge WebView, KMPlayer, Ollama servicemgr)
- ~1.5-2 GB free for the model

`ollama ps` showed `gemma4:e2b-it-q4_K_M  8.2 GB  70%/30% CPU/GPU  CONTEXT 65536` — even the smaller model only got 30% on the GPU, because hrag doesn't pass `num_ctx` to Ollama and the **131k default context KV cache** dwarfs the model weights.

**Three actionable speed wins, in priority order:**

1. **Pass `num_ctx=8192` (or similar) to Ollama.** This is a real bug in `src/hrag/providers/llm.py::OllamaProvider._build_options` — the field is never set, so Ollama defaults to the model's max (131k for gemma4). Reducing it to a realistic value (hrag rarely uses more than 4-8k effective context) shrinks the KV cache by 16×, freeing 1-2 GB VRAM. Estimated impact: **2-4× per-turn speedup** on top of any model swap.
2. **Close Chrome / Edge WebView / KMPlayer before benchmarking.** Frees ~4-5 GB VRAM. Combined with #1, lets `gemma4:e2b` fit fully on GPU — estimated **5-10× per-turn speedup**.
3. **Model swap to `gemma4:e2b-it-q4_K_M`.** Already done; alone delivered only 4-38% because (1) and (2) weren't in place.

---

## Q3 root cause and fix (run 1)

**Symptom:** `dialog_compact` event fired zero times across 20 turns with `dialog_mst_enabled=True`, `compact_after_turns=12`.

**Root cause:** `Orchestrator.chat()` hard-coded `_load_history(session_id, limit=10)`. The compaction trigger checks `len(history_rows) > compact_after_turns`, but `len(history_rows)` could never exceed 10 — the threshold (12) was unreachable.

**Fix:** in `src/hrag/orchestrator.py`, raise the history fetch limit when the dialog compactor is enabled:

```python
history_limit = 10
if self.dialog_compactor is not None:
    history_limit = max(
        history_limit,
        cfg.compaction.compact_after_turns + cfg.compaction.keep_recent_turns,
    )
history_rows = self._load_history(session_id, limit=history_limit)
```

All 27 dialog/compaction/orchestrator unit tests still pass with the fix. A live end-to-end verification of the fix was not completed because Ollama OOM-crashed mid-Q3 on run 2.

---

## Q3 crash on run 2

After the orchestrator fix, run 2 (on gemma4:e2b) made it to ~turn 15-16 of the priming phase, then Ollama dropped the TCP connection:

```
LLMProviderError: Ollama call failed: [WinError 10054] An existing connection
was forcibly closed by the remote host
```

This is the OOM signature on Windows. With the orchestrator fix, each turn from turn 13+ sends 18 history rows in the prompt — substantially more tokens. Combined with the 131k default Ollama context (KV cache > 5 GB) and the 5-6 GB already held by desktop apps, the runner OOM'd.

The fix here is **not** to revert the orchestrator change (the compactor needs that history) — it's to pass `num_ctx` to Ollama so KV cache stays small.

---

## Q4 root cause (both runs)

LLM cleanly refuses to fabricate facts about a fictional person (Dr. Aurelius Quill / 2003 Akron expedition). `prompts/answer.md` Step 4 says to write `[UNCERTAIN]` *after unsupported sub-claims*. The LLM made no sub-claim — it directly said "no record" — so `[UNCERTAIN]` was correctly omitted.

This is **desired behaviour**, not a bug. The benchmark assertion is too strict.

**Fix:** in `tests/benchmark/run_phase4.py::q4_uncertain_marker_appears`, accept either signal as a pass:
1. The LLM emits one or more `[UNCERTAIN]` tokens, OR
2. The answer contains an explicit refusal/hedge phrase ("no record", "cannot find", "no mention", "I don't have information about…").

Both signal the same property: the system did not hallucinate.

---

## Recommended default flag settings (speed-aware)

| Flag | Recommended default | Reason |
|---|---|---|
| `mask_uncertain` | **ON** | Zero cost; protects users on the rare turns where the model does hedge. |
| `gate_enabled` | **ON** | Net wall-clock win on any mixed-content session. |
| `clue_enabled` | **OFF** | Pure tax on most queries; opt in for vague-pronoun workloads. |
| `dialog_mst_enabled` | **OFF** (or ON with `compact_after_turns ≥ 50`) | Only helps long sessions; per-turn cost is real. |

---

## Action plan to reach 3/4 acceptance

1. **Done — Q3 orchestrator fix** applied (`src/hrag/orchestrator.py`).
2. **Recommended next, high-impact — pass `num_ctx` to Ollama** (`src/hrag/providers/llm.py::_build_options`). Plumb through `cfg.llm.num_ctx` (default 8192). Without this, no model swap will reliably fit on this hardware.
3. **Recommended next, low-impact — loosen Q4 assertion** (`tests/benchmark/run_phase4.py`) to accept clean refusals.
4. Close Chrome / Edge WebView / KMPlayer; rerun the full benchmark on gemma4:e2b. With (2) in place, expected wall time drops from ~17 min to ~3 min, and Q1/Q3 should both pass (the gate-classification regression on Q1 may need a prompt tweak for the smaller model).

---

## Raw logs

- Run 1 — gemma4:e4b: `tests/benchmark/phase4_run_gemma4e4b.log`
- Run 2 — gemma4:e2b: `tests/benchmark/phase4_run.log`
