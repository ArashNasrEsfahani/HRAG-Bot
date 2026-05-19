---
name: benchmark
description: Use this skill when the user asks to create, design, run, or verify a benchmark — including phrases like "create a benchmark", "run a benchmark", "verify <feature/phase> goals are met", "evaluation suite", "test the system end-to-end", or any multi-question harness that checks system behavior against expected outcomes. The skill enforces a two-phase workflow (design → run) and a visibility-first execution model so the user always sees per-test progress instead of a silent black box.
---

# Benchmark skill (project-scoped to HRAG-Bot)

## Why this exists

Benchmarks tend to be slow — LLM calls, integration tests, multi-step evaluations. When run silently in the background, the user has no signal about whether the system is making progress, which test is hanging, or how partial results are tracking. **Background execution hides the very signal the user wants.** This skill imposes a visibility-first method.

If the user previously got a benchmark run that produced a wall of text only after several minutes of silence, that was a violation of this skill. Don't repeat it.

## Two-phase workflow

Always split a benchmark into **design** and **run**. Mixing them in one agent or one script makes the design invisible and untestable.

### Phase A — Design

Produces: a structured spec file at `tests/benchmark/<name>.yaml`.

1. **Read the source material first** — PDFs, code, specs, the live SQLite index, whatever ground truth exists. Never write benchmark questions from memory; pattern-matching from training data is unreliable and the user can't audit it.
2. **Output a YAML spec** (preferred over JSON because comments help the user audit). Schema:

   ```yaml
   benchmark_version: 1
   description: <one line of what this benchmark verifies>
   questions:
     - id: q1
       category: <tag>           # e.g. factual_single_source, multi_hop, out_of_corpus
       question: "<text>"        # or `turns: [...]` for multi-turn
       expected_substrings:      # case-insensitive substring checks
         - "..."
       expected_source_doc: ...  # nullable; for systems that retrieve
       notes: "<rationale; helps future-you debug a regression>"
   ```

3. Pick robust expected strings. Prefer named entities, technical terms, exact numbers, hyphenated proper names. Avoid function words ("the", "is", "with") — they match by accident and give false confidence. A failing check should mean a real regression, not a paraphrase.
4. Cover edge cases. At minimum include:
   - one out-of-corpus / out-of-spec negative test (the system should refuse, not hallucinate)
   - one multi-turn case if the system has conversational state (tests query rewriting, session memory)
5. Keep the set tight. 5–10 questions for a quick sanity check, not 100. Long benchmarks tempt you to background them, which defeats the purpose.

### Phase B — Build the runner

Produces: a runner script at `tests/benchmark/run_<name>.py` and a results file written by it.

The runner MUST:

1. Print one line as each test STARTS:

   ```
   [3/6 ▶] q3 — multi_hop: running...
   ```

2. Print one line as each test COMPLETES with a pass/fail glyph and a one-line reason if failed:

   ```
   [3/6 ✓] q3 — PASS in 51.2s (substrings 3/3, citation ✓, raft ✓)
   [3/6 ✗] q3 — FAIL in 51.2s (missing substrings: ['damping factor'])
   ```

3. Flush every print — `print(..., flush=True)` in Python, or `sys.stdout.reconfigure(line_buffering=True)` once at the top. Rich `Console` is line-buffered by default but verify.
4. Show a running `[N/total]` counter so the user always knows where they are.
5. Capture per-test timing and a running total wall time.
6. Print a summary table to stdout at the end, in ADDITION to writing a Markdown report at `tests/benchmark/<name>_results.md`.
7. Log a config snapshot at the top of the results file — model name, top_k, threshold, dataset version — for reproducibility.

### Phase C — Run it

- Default: foreground execution via Bash/PowerShell, so the user sees every line as it streams. This is the whole point.
- Background only when ALL of these hold:
  1. The user explicitly asked for background.
  2. The run is expected to exceed ~15 minutes.
  3. You set up `Monitor` with a line-buffered script that emits ONE notification per test event so the chat still gets per-test progress.
- When delegating to a sub-agent: the agent's runner script must still print live progress, even if the agent itself runs in background. The progress lines will land in the agent's transcript and surface in the final report.

## Project-specific notes (HRAG-Bot)

- Chat calls take 50–100 seconds each on the local Ollama `gemma4:e4b`. A 6-question benchmark with one 2-turn follow-up is 7 chat calls = 6–14 minutes total. Plan for that.
- The Orchestrator API is `Orchestrator(cfg).chat(question, user_id, session_id, progress, stream)` returning `ChatResult(answer, session_id, sources, prompt)`. See `src/hrag/orchestrator.py` for the full surface and the list of progress events (`query_rewrite`, `rerank_done`, etc.).
- For multi-turn questions, chain calls with `result.session_id` so the orchestrator's history machinery and query-rewriter fire correctly.
- Use the progress callback to capture `query_rewrite` and `rerank_done` events — they're necessary to score "rewriter fired on follow-up" and "rerank fallback avoided" dimensions.
- Always set `user_id="benchmark"` (or similar) so benchmark sessions are visible separately in the SQLite `sessions` table.
- The current benchmark spec lives at `tests/benchmark/phase1.yaml` and runner at `tests/benchmark/run_phase1.py` — use those as references for future benchmarks.

## When delegating to sub-agents

If you split design + run across sub-agents, the runner agent's brief must explicitly require:

- The progression-line format above (start line + completion line per test, with `[N/total]`, glyph, reason).
- Flushed stdout.
- A summary table printed inline (not just written to file).
- A statement that the script must be re-runnable interactively so the user can run it themselves later and watch live.

Example brief snippet to include verbatim:

> The runner must print `[N/total ▶] <id> — <category>: running...` when a test starts and `[N/total ✓|✗] <id> — PASS|FAIL in <s>s (<reason>)` when it completes. Use `print(..., flush=True)`. Print a summary table to stdout at the end in addition to writing the Markdown report.

## Anti-patterns to avoid

- ❌ `run_in_background: true` with no progress events. User gets nothing for 10 minutes, then a wall of text.
- ❌ Buffered stdout. User sees nothing until the process exits.
- ❌ `try/except` swallowing failures. Surface them inline with a clear ✗ marker and a one-line reason.
- ❌ "I'll just use `pytest -q`." Pytest's quiet mode hides per-test progress and isn't a benchmark format. Build a custom runner.
- ❌ Discarding the runner after the benchmark passes. Keep it — it's now your regression suite.
- ❌ Hardcoding the spec into Python. The YAML separation lets the user edit questions without touching the runner.

## Output checklist

A complete benchmark deliverable is:

- [ ] `tests/benchmark/<name>.yaml` — questions and acceptance criteria, with comments.
- [ ] `tests/benchmark/run_<name>.py` — runner with live progress and summary table.
- [ ] `tests/benchmark/<name>_results.md` — summary table + per-question detail + config snapshot.
- [ ] One-line CLI invocation documented at the top of the results file (e.g. `python tests\benchmark\run_<name>.py`).

If any of these is missing, the benchmark isn't done.
