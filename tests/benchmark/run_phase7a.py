"""Phase 7-A — Math-handling acceptance benchmark.

Five questions that exercise the Phase 7-A trio (Unicode-glyph math detector,
math-meta query expansion, orchestrator filter + formula-extraction pass):

Q1 — Unicode-glyph detector recognises the surviving HippoRAG formula chunk.
Q2 — Live SQLite backfill: ≥ 10 chunks tagged with `has_math=True`.
Q3 — Math-meta query expansion appends math-vocabulary tokens.
Q4 — Orchestrator math-meta filter passes `where={"has_math": True}` and
     emits `math_meta_filter`; greeting-like queries do NOT trigger it.
Q5 — Formula-extraction LLM pass appends `**Extracted formulas:**` to the
     answer and emits `formula_extract`.

Usage (from project root):
    python tests/benchmark/run_phase7a.py
    python tests/benchmark/run_phase7a.py --json out.json

The whole suite runs in-process; no Ollama, no Chroma server required.
Q2 needs the live `data/store.sqlite` (post-backfill); if the DB is empty
or absent the test SKIPs cleanly.

Acceptance: >= 4/5 PASS.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

console = Console()

_SKIP = "SKIP"


# ---------------------------------------------------------------------------
# Inline fake LLM — matches the surface Orchestrator touches, no network.
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Deterministic LLM stub. The extraction pass needs a recognisable reply
    so the benchmark can verify the answer actually grew."""

    name = "fake"
    _extract_marker = "- 𝑌= Θ(𝑞| 𝜃)  — generation process of LLM"

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        if "Intent Classification" in prompt or "Output (one word only)" in prompt:
            return "factual"
        if "Formula extraction" in prompt or "formula, equation" in prompt:
            return self._extract_marker
        return "stub answer with no math."

    def generate(self, request):
        from hrag.types import GenerationResponse
        prompt = " ".join(m.content for m in request.messages)
        return GenerationResponse(text=self.complete(prompt), raw=None)

    def generate_stream(self, request):
        yield self.generate(request).text


# ---------------------------------------------------------------------------
# Q1 — Unicode-glyph detector
# ---------------------------------------------------------------------------


def q1_unicode_detector(tmp_dir: Path) -> tuple[bool | str, str]:
    """Pure prose → False. HippoRAG formula chunk → True. min_signals filter
    actually filters."""
    from hrag.ingest.math_detect import has_math, has_unicode_math

    prose = "The quick brown fox jumps over the lazy dog. " * 5
    hippo = (
        "The generation process of a LLM Θ(·) can be succinctly represented "
        "as 𝑌= Θ(𝑞| 𝜃), where 𝑞 denotes the input query, 𝑌 is the generated "
        "response, and 𝜃 represents the model's parameter."
    )
    sum_expr = "loss = ∑ x_i² / N"
    only_equals = "a=1 b=2 c=3 d=4"

    cases = [
        ("prose", prose, False),
        ("hippo formula chunk", hippo, True),
        ("loss=∑ x_i² / N", sum_expr, True),
        ("a=1 b=2 only", only_equals, False),
    ]
    rows = []
    for name, text, expected in cases:
        got = has_math(text)
        rows.append((name, expected, got))
        if got != expected:
            return False, f"{name}: expected {expected}, got {got}"

    # min_signals filter — high threshold suppresses borderline match
    if has_unicode_math("Θ alone", min_signals=2):
        return False, "single Greek glyph passed min_signals=2"
    if not has_unicode_math(hippo, min_signals=2):
        return False, "hippo chunk failed min_signals=2 (regression)"

    print("  [Q1] detector mappings:", flush=True)
    for n, exp, got in rows:
        glyph = "✓" if exp == got else "✗"
        print(f"        {glyph} {n:<28s} → {got!s}", flush=True)

    return True, f"detector: 4/4 cases correct, min_signals filter works"


# ---------------------------------------------------------------------------
# Q2 — Backfill effect on live DB
# ---------------------------------------------------------------------------


def q2_backfill_live_db(tmp_dir: Path) -> tuple[bool | str, str]:
    """Live SQLite must have ≥ 10 chunks tagged has_math=True after backfill."""
    db_path = _repo_root / "data" / "store.sqlite"
    if not db_path.exists():
        return _SKIP, f"no live DB at {db_path}; run `hrag init && hrag ingest` first"

    conn = sqlite3.connect(str(db_path))
    try:
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_math = conn.execute(
            "SELECT COUNT(*) FROM chunks "
            "WHERE json_extract(metadata, '$.has_math') = 1"
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"  [Q2] total chunks={total}  has_math=True={n_math}", flush=True)

    if total == 0:
        return _SKIP, "live DB empty; run `hrag ingest` first"
    if n_math < 10:
        return (
            False,
            f"only {n_math} of {total} chunks tagged; expected ≥ 10. "
            f"Run `python scripts/backfill_has_math.py` to backfill."
        )
    return True, f"backfill: {n_math} of {total} chunks tagged has_math=True"


# ---------------------------------------------------------------------------
# Q3 — Math-meta query expansion
# ---------------------------------------------------------------------------


def q3_query_expansion(tmp_dir: Path) -> tuple[bool | str, str]:
    """_expand_math_meta appends math-vocabulary tokens for meta-queries
    and is a no-op for non-meta queries."""
    from hrag.retrieval.query_rewriter import _expand_math_meta

    cases = [
        ("plain factual",        "what is hipporag?",          False),
        ("math meta — formula",  "give me some formulas",      True),
        ("math meta — equation", "what equations does it use", True),
        ("math meta — case",     "FORMULAS used",              True),
    ]
    rows = []
    for name, q, should_expand in cases:
        out = _expand_math_meta(q)
        expanded = out != q
        rows.append((name, should_expand, expanded, out))
        if expanded != should_expand:
            return False, f"{name}: expansion mismatch (got expanded={expanded})"

    # On expansion, several math-vocabulary tokens must appear.
    sample = _expand_math_meta("show me the math formulas")
    must_have = ("equation", "θ", "loss", "gradient", "∑")
    missing = [t for t in must_have if t not in sample]
    if missing:
        return False, f"expansion missing tokens: {missing}; got {sample!r}"

    print("  [Q3] expansion cases:", flush=True)
    for n, exp_e, got_e, _ in rows:
        glyph = "✓" if exp_e == got_e else "✗"
        print(f"        {glyph} {n:<24s} expand={got_e}", flush=True)
    print(f"  [Q3] sample expansion: {sample!r}", flush=True)

    return True, "expansion: 4/4 cases correct; math vocabulary tokens present"


# ---------------------------------------------------------------------------
# Q4 — Orchestrator math-meta filter + event
# ---------------------------------------------------------------------------


def _make_orch(tmp_dir: Path, *, math_meta=False, extract=False):
    """Build an Orchestrator with stub LLM + spy retriever + factual classifier."""
    from hrag.config import Config, EmbeddingsConfig, LLMConfig, StorageConfig
    from hrag.intent import Intent, IntentVerdict
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None

    cfg = Config(
        llm=LLMConfig(provider="ollama", model="test-model"),
        embeddings=EmbeddingsConfig(
            provider="sentence-transformers",
            model="sentence-transformers/all-mpnet-base-v2",
            dim=384,
        ),
        storage=StorageConfig(
            sqlite_path=str(tmp_dir / "store.sqlite"),
            chroma_path=str(tmp_dir / "chroma"),
            kg_path=str(tmp_dir / "kg"),
            data_root=str(tmp_dir / "data"),
        ),
    )
    cfg.project_root = tmp_dir
    cfg.retrieval.rerank_enabled = False
    cfg.retrieval.math_meta_filter_enabled = math_meta
    cfg.formula_extraction.enabled = extract

    from hrag.orchestrator import Orchestrator
    orch = Orchestrator(cfg)
    orch.llm = _FakeLLM()
    if orch.gate is not None:
        orch.gate.llm = orch.llm
    if orch.clue is not None:
        orch.clue.llm = orch.llm

    class _FactualClassifier:
        def classify(self, text, **kwargs):
            return IntentVerdict(
                intent=Intent.FACTUAL,
                confidence=1.0,
                source="test",
                raw_label="factual",
            )

    orch.intent_classifier = _FactualClassifier()  # type: ignore[assignment]
    return orch, cfg


def q4_orchestrator_filter(tmp_dir: Path) -> tuple[bool | str, str]:
    """Math-meta query → retriever called with where={'has_math': True} and
    math_meta_filter event fires. Non-meta query → no where filter."""
    from hrag.types import Chunk, RetrievalResult
    import hrag.db.connection as _conn_mod

    def mk_chunk(cid: str) -> Chunk:
        return Chunk(
            chunk_id=cid, doc_id="d", user_id="default",
            text=f"chunk {cid} with 𝑌= Θ(𝑞| 𝜃)",
            embedding_text=f"chunk {cid}",
            source_type="document",
        )

    # Spy that returns one result so we don't trip the FACTUAL→GENERAL swap.
    seeded = [RetrievalResult(chunk=mk_chunk("a"), score=0.9)]

    class _Spy:
        name = "spy"

        def __init__(self):
            self.calls = []

        def retrieve(self, query, user_id, top_k=10, source_types=None,
                     intent_hint=None, where=None):
            self.calls.append({"query": query, "where": where})
            return list(seeded)

    # --- math-meta ON, query IS math-meta ---
    persist = tmp_dir / "q4a"
    persist.mkdir(parents=True, exist_ok=True)
    orch, _ = _make_orch(persist, math_meta=True, extract=False)
    spy_a = _Spy()
    orch.retriever = spy_a
    events_a = []
    try:
        orch.chat("give me some formulas", user_id="default",
                  progress=lambda n, p: events_a.append((n, p)))
    finally:
        orch.close()
        _conn_mod._db_singleton = None

    filter_events = [p for n, p in events_a if n == "math_meta_filter"]
    if not filter_events:
        return False, "math_meta query did not emit 'math_meta_filter'"
    if not spy_a.calls or spy_a.calls[0]["where"] != {"has_math": True}:
        return False, f"math_meta query did not pass where filter; got {spy_a.calls!r}"

    # --- math-meta ON, query is NOT math-meta ---
    persist = tmp_dir / "q4b"
    persist.mkdir(parents=True, exist_ok=True)
    orch2, _ = _make_orch(persist, math_meta=True, extract=False)
    spy_b = _Spy()
    orch2.retriever = spy_b
    events_b = []
    try:
        orch2.chat("what is hipporag?", user_id="default",
                   progress=lambda n, p: events_b.append((n, p)))
    finally:
        orch2.close()
        _conn_mod._db_singleton = None

    filter_events_b = [p for n, p in events_b if n == "math_meta_filter"]
    if filter_events_b:
        return False, "non-meta query unexpectedly emitted 'math_meta_filter'"
    if spy_b.calls and spy_b.calls[0]["where"]:
        return False, f"non-meta query had where filter set: {spy_b.calls!r}"

    print(f"  [Q4] math_meta path: where={filter_events[0]['where']}", flush=True)
    print(f"  [Q4] non-meta path: where=None (correctly suppressed)", flush=True)

    return True, "math_meta filter fires on meta queries, suppressed on non-meta"


# ---------------------------------------------------------------------------
# Q5 — Formula-extraction pass
# ---------------------------------------------------------------------------


def q5_formula_extraction(tmp_dir: Path) -> tuple[bool | str, str]:
    """formula_extraction.enabled + math-meta query: extraction event fires
    and 'Extracted formulas:' appears in the assistant message."""
    from hrag.types import Chunk, RetrievalResult
    import hrag.db.connection as _conn_mod

    def mk_chunk(cid: str) -> Chunk:
        return Chunk(
            chunk_id=cid, doc_id="d", user_id="default",
            text=f"formula chunk {cid}: 𝑌= Θ(𝑞| 𝜃)",
            embedding_text=f"chunk {cid}",
            source_type="document",
        )

    seeded = [RetrievalResult(chunk=mk_chunk("c1"), score=0.9)]

    class _Spy:
        name = "spy"

        def retrieve(self, query, user_id, top_k=10, source_types=None,
                     intent_hint=None, where=None):
            return list(seeded)

    persist = tmp_dir / "q5"
    persist.mkdir(parents=True, exist_ok=True)
    orch, _ = _make_orch(persist, math_meta=True, extract=True)
    orch.retriever = _Spy()
    events = []
    result = None
    try:
        result = orch.chat(
            "give me some formulas this uses",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _conn_mod._db_singleton = None

    extract_events = [p for n, p in events if n == "formula_extract"]
    if not extract_events:
        return False, "formula_extract event did not fire"

    answer = result.answer if result is not None else ""
    if "Extracted formulas:" not in answer:
        # Some implementations may append after streaming; check the DB
        # representation too.
        return False, (
            f"answer missing 'Extracted formulas:' block; got: {answer!r:.200}"
        )

    print(f"  [Q5] formula_extract event chars={extract_events[0]['chars']}", flush=True)
    print(f"  [Q5] answer snippet: ...{answer[-150:]!r}", flush=True)

    return True, "formula_extract event + 'Extracted formulas:' in answer"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None, help="Write JSON output.")
    args = parser.parse_args()

    print("Phase 7-A — Math-handling acceptance benchmark", flush=True)
    print("=" * 60, flush=True)

    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="hrag_phase7a_"))
    print(f"Tmp dir: {tmp_dir}", flush=True)

    suites: list[tuple[str, Callable[[Path], tuple[bool | str, str]]]] = [
        ("Q1 — Unicode-glyph math detector", q1_unicode_detector),
        ("Q2 — Live SQLite backfill effect", q2_backfill_live_db),
        ("Q3 — Math-meta query expansion", q3_query_expansion),
        ("Q4 — Orchestrator math-meta filter + event", q4_orchestrator_filter),
        ("Q5 — Formula-extraction LLM pass", q5_formula_extraction),
    ]

    passed = 0
    skipped = 0
    failed = 0
    results: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        console=console,
        transient=False,
    ) as prog:
        task = prog.add_task("Phase 7-A benchmark", total=len(suites))

        for label, fn in suites:
            prog.update(task, description=f"[bold]{label}[/]")
            console.print(f"\n[bold]{label}[/bold]")
            t0 = time.time()
            try:
                outcome, msg = fn(tmp_dir)
            except Exception as exc:  # noqa: BLE001
                import traceback
                outcome = False
                msg = f"exception: {type(exc).__name__}: {exc}"
                console.print(f"  [red]EXCEPTION[/] — {msg}")
                console.print(traceback.format_exc())
            dur = time.time() - t0

            if outcome is _SKIP:
                skipped += 1
                console.print(f"  [yellow]SKIP[/] — {msg} ({dur:.2f}s)")
                status = "SKIP"
            elif outcome:
                passed += 1
                console.print(f"  [green]PASS[/] — {msg} ({dur:.2f}s)")
                status = "PASS"
            else:
                failed += 1
                console.print(f"  [red]FAIL[/] — {msg} ({dur:.2f}s)")
                status = "FAIL"

            results.append({
                "label": label,
                "status": status,
                "message": msg,
                "duration_s": round(dur, 3),
            })
            prog.advance(task)

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Phase 7-A Benchmark Summary[/bold]")
    console.print("=" * 60)
    for r in results:
        glyph = {"PASS": "[green]PASS[/]", "FAIL": "[red]FAIL[/]", "SKIP": "[yellow]SKIP[/]"}[r["status"]]
        console.print(f"  {glyph}  {r['label']}")
        if r["status"] != "PASS":
            console.print(f"       {r['message']}")

    total = len(suites)
    console.print(
        f"\n[bold]Score: {passed}/{total} "
        f"({skipped} skipped, {failed} failed)[/bold]"
    )

    accept = passed >= 4 or (passed + skipped >= 4 and failed == 0)
    if accept:
        console.print("[green bold]ACCEPTED (>= 4/5 pass or skip)[/]")
    else:
        console.print("[red bold]FAILED — need >= 4/5[/]")

    if args.json:
        payload = {
            "phase": "7-A",
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": total,
            "accepted": accept,
            "questions": results,
            "timestamp": time.time(),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"\n[dim]wrote JSON: {args.json}[/]")

    return 0 if accept else 1


if __name__ == "__main__":
    sys.exit(main())
