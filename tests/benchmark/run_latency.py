"""Phase 9.1 — Latency harness.

Runs a fixed 20-question set through ``Orchestrator.chat()`` and reports a
per-stage wall-clock breakdown so every other Phase-9 ticket can prove it
moved the needle.

Two modes:

* ``--mode synthetic`` (default) — replaces ``orch.llm`` and ``orch.retriever``
  with latency-simulating stubs.  Deterministic; runs in CI without Ollama
  or a populated corpus.  The numbers are not absolute Ollama latencies —
  they are stable baselines whose *relative* movement reflects orchestrator
  pipeline changes (async wiring, caches, short-circuits, batching).
* ``--mode live`` — real ``Orchestrator`` against the configured providers.
  Requires Ollama to be running and a populated ChromaDB.  Skips gracefully
  on missing infrastructure.

Output:

* Markdown table to stdout (median + p95 per stage, plus per-category split).
* ``--json <path>`` writes the full per-question detail for diffing across
  runs (``--diff before.json after.json``).

Per-stage timing strategy:

The orchestrator already emits ``duration_s`` on most progress events
(``retrieve``, ``rerank_done``, ``gate_check``, ``clue_generate``,
``generate``, ``formula_extract``, ``dialog_compact``, ``done``).  Those
are authoritative.  For stages without an explicit ``duration_s`` (intent,
organize, TTFT) the harness derives them from event-arrival timestamps
captured in the progress callback.  TTFT becomes first-class once Phase
9.10 lands; until then it is the gap between ``generate_start`` and the
first ``generate_token``.

Usage::

    python tests/benchmark/run_latency.py
    python tests/benchmark/run_latency.py --json runs/baseline.json
    python tests/benchmark/run_latency.py --mode live
    python tests/benchmark/run_latency.py --diff runs/baseline.json runs/after_9_2.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# sys.path setup — must happen before any hrag import
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass


# ---------------------------------------------------------------------------
# Synthetic-mode latency profile (milliseconds).
# Modeled after gemma4:e2b on an RTX 4060 laptop.  Tweak ``_SCALE`` to compress
# the run time if iterating fast; relative shape stays the same.
# ---------------------------------------------------------------------------

_SCALE = 0.10  # 10× compression: a "real" 8 s generate becomes 800 ms in CI
_RNG = random.Random(20260523)

_PROFILE = {
    "intent_ms":       (200, 400),
    "gate_ms":         (300, 500),
    "clue_ms":         (800, 1200),
    "rewrite_ms":      (800, 1200),
    "rephrase_ms":     (600, 900),
    "ttft_ms":         (600, 1200),
    "tok_ms":          (25, 50),     # per generated token
    "embed_query_ms":  (40, 100),
    "retrieve_ms":     (20, 60),
    "rerank_ms":       (250, 400),
    "formula_ms":      (700, 1000),
}


def _sleep(stage: str, scale: float = _SCALE) -> float:
    lo, hi = _PROFILE[stage]
    ms = _RNG.uniform(lo, hi) * scale
    time.sleep(ms / 1000.0)
    return ms


# ---------------------------------------------------------------------------
# Question set — 20 questions across 5 categories.
# ---------------------------------------------------------------------------

_QUESTIONS: list[dict[str, Any]] = [
    # --- greeting (skips retrieval entirely under adaptive_top_k) ---
    {"q": "hi",                                                          "cat": "greeting",        "tokens": 30},
    {"q": "hello there",                                                 "cat": "greeting",        "tokens": 30},
    {"q": "thanks!",                                                     "cat": "greeting",        "tokens": 20},
    {"q": "good morning",                                                "cat": "greeting",        "tokens": 30},
    # --- personal (broadens to episodic memory) ---
    {"q": "what was that thing I mentioned about retrieval last week?",  "cat": "personal",        "tokens": 80},
    {"q": "remind me what my preferred chunk size is",                   "cat": "personal",        "tokens": 60},
    {"q": "what topics am I working on right now?",                      "cat": "personal",        "tokens": 80},
    {"q": "did I ever tell you my embedding model preference?",          "cat": "personal",        "tokens": 60},
    # --- factual short (single-hop lookup) ---
    {"q": "what is HippoRAG?",                                           "cat": "factual_short",   "tokens": 120},
    {"q": "define personalized pagerank",                                "cat": "factual_short",   "tokens": 120},
    {"q": "what is RAGate?",                                             "cat": "factual_short",   "tokens": 100},
    {"q": "how does Leiden community detection work?",                   "cat": "factual_short",   "tokens": 150},
    # --- factual long / compound (RAG-Fusion territory) ---
    {"q": "compare PPR and BM25 for retrieval and explain tradeoffs",    "cat": "factual_long",    "tokens": 250},
    {"q": "how does GraphRAG differ from vanilla RAG and HippoRAG?",     "cat": "factual_long",    "tokens": 300},
    {"q": "walk through KG triple extraction and synonym merging",       "cat": "factual_long",    "tokens": 250},
    {"q": "explain query routing across entity/global/cross_document",   "cat": "factual_long",    "tokens": 280},
    # --- math-meta (Phase 7-A formula extraction path) ---
    {"q": "give me some formulas hipporag uses",                         "cat": "math_meta",       "tokens": 180},
    {"q": "what equations define the PPR objective?",                    "cat": "math_meta",       "tokens": 180},
    # --- off-corpus (low confidence; CRAG / Phase-8 territory) ---
    {"q": "what's the weather in Tehran?",                               "cat": "off_corpus",      "tokens": 60},
    {"q": "who won the 2025 world series?",                              "cat": "off_corpus",      "tokens": 60},
]


# ---------------------------------------------------------------------------
# Stubs for synthetic mode
# ---------------------------------------------------------------------------


class _LatencyLLM:
    """Scripted LLM that sleeps to simulate realistic Ollama latencies.

    Dispatches on prompt content to pick the right sleep profile (intent
    classifier, gate, clue, rephrase, generate).  generate_stream emits one
    token at a time with per-token latency, so TTFT is meaningful.
    """

    name = "latency-stub"

    def __init__(self, tokens_for_question: dict[str, int]):
        self._tokens_for_q = tokens_for_question
        self._active_question: Optional[str] = None

    def set_active_question(self, q: str) -> None:
        self._active_question = q

    # --- classify the prompt to pick a sleep profile ---
    @staticmethod
    def _classify(prompt: str) -> str:
        p = prompt.lower()
        # Phase 9.6 — combined preflight prompt contains all three of "intent",
        # "gate", "clue" plus a JSON example. Match it FIRST so we don't fall
        # through to the per-stage matchers (which would return non-JSON and
        # force the orchestrator into the fallback 3-call path).
        if "pre-retrieval triage" in p and "return only a json object" in p:
            return "combined"
        if "intent classification" in p or "output (one word only)" in p:
            return "intent"
        if "decision: retrieve" in p or "ragate" in p or ("decide" in p and "retrieve" in p):
            return "gate"
        if "clue" in p and "hypothesis" in p:
            return "clue"
        if "rephras" in p or "alternative phrasing" in p:
            return "rephrase"
        if "follow-up questions" in p:
            return "rephrase"
        if "formula extraction" in p or ("extract" in p and "formula" in p):
            return "formula"
        if "rewrite" in p and "query" in p:
            return "rewrite"
        return "answer"

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        kind = self._classify(prompt if isinstance(prompt, str) else str(prompt))
        if kind == "combined":
            # One sleep covers the merged intent+gate+clue call. Models the
            # Phase 9.6 latency win: one round-trip instead of three.
            _sleep("intent_ms")  # cheapest of the three; combined is bounded by it
            return (
                '{"intent": "factual", "gate": "RETRIEVE", '
                '"clue": "A hypothetical answer about the topic."}'
            )
        if kind == "intent":
            _sleep("intent_ms")
            return "factual"
        if kind == "gate":
            _sleep("gate_ms")
            return "RETRIEVE"
        if kind == "clue":
            _sleep("clue_ms")
            return "A hypothetical answer about the topic."
        if kind == "rephrase":
            _sleep("rephrase_ms")
            return "1. Alternative one\n2. Alternative two"
        if kind == "formula":
            _sleep("formula_ms")
            return "- Y = Θ(q|θ)\n- L = Σ x²/N"
        if kind == "rewrite":
            _sleep("rewrite_ms")
            return prompt if isinstance(prompt, str) else "rewritten query"
        # default — answer body (one-shot)
        _sleep("ttft_ms")
        n = self._tokens_for_q.get(self._active_question or "", 120)
        for _ in range(n):
            _sleep("tok_ms")
        return "This is a synthetic answer." * 4

    def generate(self, request):
        from hrag.types import GenerationResponse
        # answer-time: ignore system/messages distinction, simulate TTFT + tokens
        _sleep("ttft_ms")
        n = self._tokens_for_q.get(self._active_question or "", 120)
        for _ in range(n):
            _sleep("tok_ms")
        return GenerationResponse(text="This is a synthetic answer." * 4, raw=None)

    def generate_stream(self, request):
        # TTFT
        _sleep("ttft_ms")
        n = self._tokens_for_q.get(self._active_question or "", 120)
        yield "First "
        for _ in range(n - 1):
            _sleep("tok_ms")
            yield "tok "


class _LatencyRetriever:
    """Returns canned RetrievalResults after a sleep that scales with top_k."""

    name = "latency-stub-retriever"

    def __init__(self, n_results: int = 8):
        self._n = n_results

    def retrieve(self, query, user_id, top_k=10, source_types=None,
                 intent_hint=None, where=None):
        _sleep("embed_query_ms")
        _sleep("retrieve_ms")
        from hrag.types import Chunk, RetrievalResult
        out = []
        for i in range(min(top_k, self._n)):
            chunk = Chunk(
                chunk_id=f"c{i}",
                doc_id=f"d{i}",
                user_id=user_id,
                text=f"Synthetic chunk {i} text body for query: {query[:50]}",
                embedding_text=f"Title-{i} | Section-{i} | chunk {i}",
                title=f"Title-{i}",
                section=f"Section-{i}",
            )
            out.append(RetrievalResult(
                chunk=chunk,
                score=0.9 - i * 0.05,
                rerank_score=2.0 - i * 0.3,
            ))
        return out


class _LatencyReranker:
    """Sleeps then returns results scored linearly.

    Mirrors the real CrossEncoderReranker signature:
    ``rerank(query, results, *, threshold=0.0, top_k=None, progress=None)``.
    """

    name = "latency-stub-reranker"
    threshold = -100.0  # don't filter anything

    def rerank(self, query, results, *, threshold=0.0, top_k=None, progress=None,
               **_kwargs):
        _sleep("rerank_ms")
        out = list(results)
        n = len(out)
        for i, r in enumerate(out):
            score = 2.0 - i * 0.3
            try:
                r.rerank_score = score
            except Exception:
                pass
            if progress is not None:
                try:
                    progress(i + 1, n, score)
                except Exception:
                    pass
        # Apply threshold + top_k (matches CrossEncoderReranker.rerank shape)
        out = [r for r in out if (getattr(r, "rerank_score", 0.0) or 0.0) >= threshold]
        if top_k:
            out = out[:top_k]
        return out


# ---------------------------------------------------------------------------
# Orchestrator builders
# ---------------------------------------------------------------------------


def _reset_db() -> None:
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None


def _build_synthetic_orch(tmp_dir: Path):
    """Build an Orchestrator with latency-simulating LLM + retriever stubs.

    Mirrors the pattern used by run_phase8.py: real ``Orchestrator`` over a
    temp DB, then replace .llm / .retriever / .reranker after construction.
    """
    import os
    from hrag.config import Config, EmbeddingsConfig, LLMConfig, StorageConfig, _apply_env_overrides

    _reset_db()
    base_kwargs = {
        "llm": LLMConfig(provider="ollama", model="test-model"),
        "embeddings": EmbeddingsConfig(
            provider="sentence-transformers",
            model="sentence-transformers/all-mpnet-base-v2",
            dim=384,
        ),
        "storage": StorageConfig(
            sqlite_path=str(tmp_dir / "store.sqlite"),
            chroma_path=str(tmp_dir / "chroma"),
            kg_path=str(tmp_dir / "kg"),
            data_root=str(tmp_dir / "data"),
        ),
    }
    # Honour HRAG_*__* env overrides so the harness can toggle Phase 9 flags
    # without a config.yaml file. We merge the defaults first, then apply env
    # overrides into the resulting dict.
    raw = {k: v.model_dump() for k, v in base_kwargs.items()}
    raw = _apply_env_overrides(raw)
    cfg = Config(**raw)
    cfg.project_root = tmp_dir
    cfg.retrieval.rerank_enabled = True
    # Disable the LLM intent classifier; harness simulates it inside the LLM stub
    # by responding to its prompt.  Leaving cfg.intent.enabled=True lets us
    # measure the cost of the intent-check round-trip in synthetic mode too.

    from hrag.orchestrator import Orchestrator
    orch = Orchestrator(cfg)

    tokens_for_q = {q["q"]: q["tokens"] for q in _QUESTIONS}
    llm = _LatencyLLM(tokens_for_q)
    orch.llm = llm
    if orch.gate is not None:
        orch.gate.llm = llm
    if orch.clue is not None:
        orch.clue.llm = llm
    if getattr(orch, "combined_preflight", None) is not None:
        orch.combined_preflight.llm = llm
    # Sub-components capture their own LLM reference at __init__; patch them
    # so no path leaks through to the real Ollama provider during synthetic runs.
    if getattr(orch, "intent_classifier", None) is not None:
        try:
            orch.intent_classifier._llm = llm
        except Exception:
            pass
    for attr in ("_llm", "llm"):
        if hasattr(orch.query_rewriter, attr):
            try:
                setattr(orch.query_rewriter, attr, llm)
            except Exception:
                pass

    orch.retriever = _LatencyRetriever(n_results=8)
    orch.reranker = _LatencyReranker()

    return orch, cfg, llm


def _build_live_orch():
    """Build the default Orchestrator from config.yaml.  No stubbing.

    Returns ``(orch, cfg, None)``.  Caller must verify Ollama + corpus are
    actually reachable; we just hand back whatever the user has configured.
    """
    from hrag.config import load_config
    from hrag.orchestrator import Orchestrator

    cfg = load_config(_repo_root / "config.yaml")
    orch = Orchestrator(cfg)
    return orch, cfg, None


# ---------------------------------------------------------------------------
# Per-question runner — captures events with timestamps, derives per-stage ms
# ---------------------------------------------------------------------------


@dataclass
class StageTimes:
    """Per-question stage breakdown in milliseconds."""

    rewrite: Optional[float] = None
    intent: Optional[float] = None
    gate: Optional[float] = None
    clue: Optional[float] = None
    retrieve: Optional[float] = None
    rerank: Optional[float] = None
    organize: Optional[float] = None
    ttft: Optional[float] = None
    generate: Optional[float] = None
    formula: Optional[float] = None
    total: Optional[float] = None
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "events" and v is not None}
        d["n_events"] = len(self.events)
        return d


def _run_one(orch, llm_stub, q: dict[str, Any]) -> StageTimes:
    """Run a single question through orch.chat() and return derived stage times.

    ``llm_stub`` is None in live mode; otherwise the _LatencyLLM stub whose
    set_active_question hook tells it how many tokens to emit.
    """
    times = StageTimes()
    t_anchor = time.monotonic()
    t_seen: dict[str, float] = {}

    def _cb(name: str, payload: dict) -> None:
        now = time.monotonic()
        times.events.append({"name": name, "t_rel_ms": (now - t_anchor) * 1000, "payload": payload})
        t_seen.setdefault(name, now)

        # Authoritative duration_s from payload, where the orchestrator emits it
        if name == "retrieve" and "duration_s" in payload:
            times.retrieve = payload["duration_s"] * 1000
        elif name == "rerank_done" and "duration_s" in payload:
            times.rerank = payload["duration_s"] * 1000
        elif name == "gate_check" and "duration_s" in payload:
            times.gate = payload["duration_s"] * 1000
        elif name == "clue_generate" and "duration_s" in payload:
            times.clue = payload["duration_s"] * 1000
        elif name == "generate" and "duration_s" in payload:
            times.generate = payload["duration_s"] * 1000
        elif name == "formula_extract" and "duration_s" in payload:
            times.formula = payload["duration_s"] * 1000
        elif name == "done" and "total_s" in payload:
            times.total = payload["total_s"] * 1000

    if llm_stub is not None:
        llm_stub.set_active_question(q["q"])

    try:
        orch.chat(q["q"], user_id="default", progress=_cb, stream=True)
    except Exception as exc:  # noqa: BLE001
        times.events.append({"name": "_exception", "t_rel_ms": (time.monotonic() - t_anchor) * 1000,
                             "payload": {"error": f"{type(exc).__name__}: {exc}"}})

    # Derive stages without authoritative duration_s from event timestamps.
    # ``query_rewrite`` fires only when actual rewriting happens, so anchor
    # intent on whichever pre-classify event is present.
    if "query_rewrite" in t_seen and "start" in t_seen:
        times.rewrite = (t_seen["query_rewrite"] - t_seen["start"]) * 1000
    if "intent_check" in t_seen:
        anchor = t_seen.get("query_rewrite", t_seen.get("start"))
        if anchor is not None:
            times.intent = (t_seen["intent_check"] - anchor) * 1000
    if "organize_done" in t_seen and "rerank_done" in t_seen:
        times.organize = (t_seen["organize_done"] - t_seen["rerank_done"]) * 1000
    elif "organize_done" in t_seen and "retrieve" in t_seen:
        times.organize = (t_seen["organize_done"] - t_seen["retrieve"]) * 1000

    # TTFT: gap between generate_start and the first generate_token
    gs = t_seen.get("generate_start")
    first_tok = None
    for ev in times.events:
        if ev["name"] == "generate_token":
            first_tok = t_anchor + ev["t_rel_ms"] / 1000
            break
    if gs is not None and first_tok is not None:
        times.ttft = (first_tok - gs) * 1000

    return times


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


_STAGES = ["rewrite", "intent", "gate", "clue", "retrieve", "rerank",
           "organize", "ttft", "generate", "formula", "total"]


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _aggregate(rows: list[tuple[dict, StageTimes]]) -> dict[str, dict[str, float]]:
    """Return ``{stage: {median, p95, mean, count}}`` over all rows."""
    out: dict[str, dict[str, float]] = {}
    for stage in _STAGES:
        vals = [getattr(t, stage) for _, t in rows if getattr(t, stage) is not None]
        if not vals:
            continue
        out[stage] = {
            "median": statistics.median(vals),
            "p95": _percentile(vals, 0.95),
            "mean": statistics.mean(vals),
            "count": len(vals),
        }
    return out


def _aggregate_by_category(rows: list[tuple[dict, StageTimes]]) -> dict[str, dict[str, float]]:
    """Return ``{cat: {median_total, n}}`` keyed by question category."""
    by_cat: dict[str, list[float]] = {}
    for q, t in rows:
        if t.total is not None:
            by_cat.setdefault(q["cat"], []).append(t.total)
    return {
        cat: {"median_total_ms": statistics.median(vals), "n": len(vals)}
        for cat, vals in by_cat.items()
    }


def _render_markdown(agg: dict[str, dict[str, float]],
                     by_cat: dict[str, dict[str, float]],
                     n_questions: int,
                     mode: str) -> str:
    lines = []
    lines.append(f"# HRAG latency harness — mode={mode}, n={n_questions}")
    lines.append("")
    lines.append("## Per-stage timing (ms)")
    lines.append("")
    lines.append("| stage     | median |    p95 |   mean |  n |")
    lines.append("|-----------|-------:|-------:|-------:|---:|")
    for stage in _STAGES:
        if stage not in agg:
            continue
        v = agg[stage]
        lines.append(f"| {stage:<9} | {v['median']:6.1f} | {v['p95']:6.1f} | {v['mean']:6.1f} | {int(v['count']):>2} |")
    lines.append("")
    lines.append("## Median total by category")
    lines.append("")
    lines.append("| category       |  median total (ms) | n |")
    lines.append("|----------------|-------------------:|--:|")
    for cat, v in sorted(by_cat.items()):
        lines.append(f"| {cat:<14} | {v['median_total_ms']:18.1f} | {int(v['n']):>1} |")
    lines.append("")
    return "\n".join(lines)


def _render_diff(a: dict, b: dict) -> str:
    """Compare two JSON dumps from prior runs and render a delta table."""
    lines = ["# Latency diff (baseline → after)", ""]
    lines.append("| stage     | baseline median | after median |    Δ ms |   Δ %  |")
    lines.append("|-----------|----------------:|-------------:|--------:|-------:|")
    agg_a = a.get("aggregate", {})
    agg_b = b.get("aggregate", {})
    for stage in _STAGES:
        if stage not in agg_a or stage not in agg_b:
            continue
        ma = agg_a[stage]["median"]
        mb = agg_b[stage]["median"]
        d = mb - ma
        pct = (d / ma * 100) if ma > 0 else 0
        arrow = "🟢" if d < -1 else ("🔴" if d > 1 else "  ")
        lines.append(f"| {stage:<9} | {ma:15.1f} | {mb:12.1f} | {d:+7.1f} | {pct:+5.1f}% {arrow}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    global _SCALE  # noqa: PLW0603

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "live"), default="synthetic")
    parser.add_argument("--json", type=str, default=None, help="Write per-question JSON dump")
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Print a delta table for two prior --json runs and exit")
    parser.add_argument("--n", type=int, default=len(_QUESTIONS),
                        help=f"Run only the first N questions (default {len(_QUESTIONS)})")
    parser.add_argument("--scale", type=float, default=_SCALE,
                        help=f"Synthetic time compression (default {_SCALE})")
    args = parser.parse_args()

    if args.diff:
        before = json.loads(Path(args.diff[0]).read_text(encoding="utf-8"))
        after = json.loads(Path(args.diff[1]).read_text(encoding="utf-8"))
        print(_render_diff(before, after))
        return 0

    _SCALE = args.scale

    print(f"HRAG latency harness — mode={args.mode}, n={args.n}", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="hrag_latency_"))
    print(f"Tmp dir: {tmp_dir}", flush=True)

    if args.mode == "synthetic":
        orch, cfg, llm_stub = _build_synthetic_orch(tmp_dir)
    else:
        try:
            orch, cfg, llm_stub = _build_live_orch()
        except Exception as exc:  # noqa: BLE001
            print(f"live mode failed to construct orchestrator: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            return 2

    questions = _QUESTIONS[: args.n]
    rows: list[tuple[dict, StageTimes]] = []
    t0 = time.monotonic()

    try:
        for i, q in enumerate(questions, 1):
            label = f"[{i:>2}/{len(questions)}] {q['cat']:<14}"
            print(f"{label} {q['q'][:60]}", flush=True)
            try:
                times = _run_one(orch, llm_stub, q)
            except Exception as exc:  # noqa: BLE001
                print(f"  EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
                print(traceback.format_exc(), flush=True)
                times = StageTimes()
            rows.append((q, times))
            tot = f"{times.total:.0f}ms" if times.total is not None else "—"
            ttft = f"{times.ttft:.0f}ms" if times.ttft is not None else "—"
            # Surface any exception captured inside _run_one so silent failures
            # aren't hidden behind a missing total.
            errs = [e["payload"].get("error", "") for e in times.events
                    if e["name"] == "_exception"]
            err_suffix = f"  ERR={errs[0]}" if errs else ""
            print(f"      total={tot}  ttft={ttft}{err_suffix}", flush=True)
    finally:
        try:
            orch.close()
        except Exception:
            pass
        _reset_db()

    wall = time.monotonic() - t0
    print("", flush=True)
    print(f"Completed {len(rows)} questions in {wall:.1f}s wall clock", flush=True)
    print("", flush=True)

    agg = _aggregate(rows)
    by_cat = _aggregate_by_category(rows)
    md = _render_markdown(agg, by_cat, len(rows), args.mode)
    print(md)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.mode,
            "scale": args.scale,
            "n_questions": len(rows),
            "wall_s": wall,
            "aggregate": agg,
            "by_category": by_cat,
            "questions": [
                {
                    "q": q["q"],
                    "cat": q["cat"],
                    "tokens": q["tokens"],
                    "stages": t.to_dict(),
                }
                for q, t in rows
            ],
            "timestamp": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote JSON: {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
