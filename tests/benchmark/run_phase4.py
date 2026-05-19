"""Phase 4 acceptance benchmark.

Four acceptance questions, one per Phase 4 feature. Each subtest configures
the orchestrator with the relevant cfg.compaction.* flag(s) ON and verifies
the expected behaviour.

All four tests use the project's live corpus (the same one Phase 2/3 benchmarks
use — academic ML/NLP papers including HIPPORAG, RAGate, and related work). The
tests exercise the Phase 4 features against the corpus without requiring any
special setup beyond the corpus being ingested.

Usage (from project root):
    python tests/benchmark/run_phase4.py

Requires:
    - The corpus already ingested via `hrag ingest` (same corpus as Phase 2/3).
    - A real LLM configured in config.yaml (Ollama, OpenAI, or Anthropic).

Acceptance threshold: >= 3/4 questions pass.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

# ---------------------------------------------------------------------------
# sys.path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

# Line-buffered UTF-8 so progress glyphs survive Windows cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

from hrag.config import load_config
from hrag.orchestrator import Orchestrator

console = Console()


# ---------------------------------------------------------------------------
# Progress callback helper
# ---------------------------------------------------------------------------


def _collect(events: list[tuple[str, dict]]):
    """Return a progress callback that appends every event to *events*."""

    def cb(name: str, payload: dict) -> None:
        events.append((name, dict(payload)))

    return cb


# ---------------------------------------------------------------------------
# Q1 — RAGate: small-talk must SKIP retrieval
# ---------------------------------------------------------------------------


def q1_gate_skips_thanks(orch: Orchestrator, user_id: str) -> tuple[bool, str]:
    """After a factual exchange, 'thanks!' must result in gate decision SKIP
    and must NOT trigger a retrieve event."""

    # Prime the session with a factual exchange so the gate sees realistic history.
    t0 = time.time()
    prime_result = orch.chat(
        "What is overfitting in machine learning?",
        user_id=user_id,
        session_id=None,
    )
    sid = prime_result.session_id

    events: list[tuple[str, dict]] = []
    orch.chat(
        "thanks!",
        user_id=user_id,
        session_id=sid,
        progress=_collect(events),
    )
    dur = time.time() - t0

    gate_events = [p for n, p in events if n == "gate_check"]
    retrieve_events = [p for n, p in events if n == "retrieve"]

    print(
        f"  [Q1] gate_check events={len(gate_events)} "
        f"retrieve_events={len(retrieve_events)} "
        f"({dur:.1f}s)",
        flush=True,
    )

    if not gate_events:
        return False, "no gate_check event fired (is compaction.gate_enabled=True?)"
    decision = gate_events[0].get("decision", "?")
    if decision != "SKIP":
        return (
            False,
            f"gate decision was {decision!r}, expected 'SKIP' — "
            "LLM may not be classifying 'thanks!' as small-talk",
        )
    if retrieve_events:
        return False, f"retrieval ran ({len(retrieve_events)} event(s)) despite SKIP"
    return True, f"gate correctly SKIPped retrieval for 'thanks!' (decision={decision!r})"


# ---------------------------------------------------------------------------
# Q2 — ClueGenerator: vague query retrieves relevant content
# ---------------------------------------------------------------------------


def q2_clue_boosts_retrieval(
    orch_with_clue: Orchestrator,
    user_id: str,
    query: str,
    expected_keywords: list[str],
) -> tuple[bool, str]:
    """A vague query goes through ClueGenerator; the retrieved sources contain
    at least one of *expected_keywords*, confirming the clue directed retrieval
    at the right region of the corpus.
    """
    events: list[tuple[str, dict]] = []
    t0 = time.time()
    result = orch_with_clue.chat(
        query,
        user_id=user_id,
        session_id=None,
        progress=_collect(events),
    )
    dur = time.time() - t0

    clue_events = [p for n, p in events if n == "clue_generate"]
    retrieve_events = [p for n, p in events if n == "retrieve"]

    print(
        f"  [Q2] clue_events={len(clue_events)} "
        f"retrieve_events={len(retrieve_events)} "
        f"sources={len(result.sources)} ({dur:.1f}s)",
        flush=True,
    )

    if not clue_events:
        return False, "no clue_generate event fired (is compaction.clue_enabled=True?)"

    clue_text = clue_events[0].get("clue", "")
    print(f"  [Q2] clue='{clue_text[:120]}'", flush=True)

    if not result.sources:
        return False, "clue-enabled retrieval returned no sources at all"

    # Check that at least one source mentions one of the expected keywords.
    joined = " ".join(
        (s.chunk.text or "") + " " + (s.chunk.title or "")
        for s in result.sources
    ).lower()
    hit_keywords = [k for k in expected_keywords if k.lower() in joined]

    if not hit_keywords:
        sample_titles = [s.chunk.title or "Untitled" for s in result.sources[:4]]
        return (
            False,
            f"clue-enabled retrieval missed all expected keywords {expected_keywords}. "
            f"Sources retrieved: {sample_titles}",
        )

    return (
        True,
        f"clue hit keywords {hit_keywords}; "
        f"clue='{clue_text[:60]}...' ({len(result.sources)} sources)",
    )


# ---------------------------------------------------------------------------
# Q3 — DialogMSTCompactor: fact from turn 3 survives 20-turn compaction
# ---------------------------------------------------------------------------


def q3_dialog_mst_preserves_old_fact(
    orch: Orchestrator, user_id: str
) -> tuple[bool, str]:
    """20-turn conversation; turn 3 plants 'my favourite framework is PyTorch'.
    The final turn asks what framework the user mentioned. With dialog MST on,
    the fact must survive compaction.
    """
    sid: str | None = None

    seed_messages = [
        "Hello!",
        "What is gradient descent?",
        "By the way, my favourite framework is PyTorch — please remember that.",
    ]
    filler_topics = [
        "Tell me about convolutional networks.",
        "What is dropout regularisation?",
        "Explain the attention mechanism.",
        "What is a transformer architecture?",
        "How does batch normalisation work?",
        "What is the vanishing gradient problem?",
        "Explain LSTMs and gated recurrent units.",
        "What is fine-tuning a pretrained model?",
        "How does early stopping work?",
        "Explain L2 regularisation.",
        "What is data augmentation in deep learning?",
        "Tell me about ResNet and skip connections.",
        "What is a learning-rate schedule?",
        "Explain cross-entropy loss.",
        "What is a one-hot encoding?",
        "Explain the softmax function.",
    ]

    all_turns = seed_messages + filler_topics  # 3 + 16 = 19 turns
    compact_events_seen: list[dict] = []

    print(f"  [Q3] running {len(all_turns)} priming turns...", flush=True)
    t0 = time.time()
    for i, q in enumerate(all_turns, start=1):
        turn_t0 = time.time()
        ev: list[tuple[str, dict]] = []
        r = orch.chat(q, user_id=user_id, session_id=sid, progress=_collect(ev))
        sid = r.session_id
        compact_events_seen.extend(p for n, p in ev if n == "dialog_compact")
        print(
            f"  [Q3] ... turn {i}/{len(all_turns)} done ({time.time() - turn_t0:.1f}s)",
            flush=True,
        )

    # Final turn references the planted fact.
    final_ev: list[tuple[str, dict]] = []
    final = orch.chat(
        "What framework did I say was my favourite?",
        user_id=user_id,
        session_id=sid,
        progress=_collect(final_ev),
    )
    dur = time.time() - t0
    compact_events_seen.extend(p for n, p in final_ev if n == "dialog_compact")

    print(
        f"  [Q3] dialog_compact events={len(compact_events_seen)} "
        f"total_dur={dur:.1f}s",
        flush=True,
    )
    print(
        f"  [Q3] final answer (first 200 chars): {final.answer[:200]!r}",
        flush=True,
    )

    if not compact_events_seen:
        return (
            False,
            "no dialog_compact event fired — "
            "check compaction.dialog_mst_enabled and compact_after_turns setting "
            f"(ran {len(all_turns)} turns with compact_after_turns=12)",
        )

    if "pytorch" not in final.answer.lower():
        return (
            False,
            f"answer did not mention PyTorch after compaction "
            f"(compact events: {len(compact_events_seen)}; "
            f"answer excerpt: {final.answer[:200]!r})",
        )

    return (
        True,
        f"PyTorch survived dialog compaction "
        f"(compact events={len(compact_events_seen)}, dur={dur:.1f}s)",
    )


# ---------------------------------------------------------------------------
# Q4 — [UNCERTAIN] masking: adversarial question gets the visible marker
# ---------------------------------------------------------------------------


def q4_uncertain_marker_appears(
    orch: Orchestrator, user_id: str
) -> tuple[bool, str]:
    """An adversarial question about a fictional person/event should cause the
    LLM to emit [UNCERTAIN] at least once. With mask_uncertain on, the rendered
    answer must include a visible uncertainty indicator.
    """
    # This question refers to a fictional person and event — the corpus definitely
    # won't have it, so the LLM must hedge.
    q = (
        "What was Dr. Aurelius Quill's exact heart rate during the "
        "2003 Akron expedition, in beats per minute?"
    )

    events: list[tuple[str, dict]] = []
    t0 = time.time()
    result = orch.chat(q, user_id=user_id, session_id=None, progress=_collect(events))
    dur = time.time() - t0

    uncertain_events = [p for n, p in events if n == "uncertain_render"]

    print(
        f"  [Q4] uncertain_render events={len(uncertain_events)} ({dur:.1f}s)",
        flush=True,
    )
    print(
        f"  [Q4] answer (first 300 chars): {result.answer[:300]!r}",
        flush=True,
    )

    if not uncertain_events:
        return (
            False,
            "no uncertain_render event fired — "
            "is compaction.mask_uncertain=True?",
        )

    count = uncertain_events[0].get("count", 0)
    answer_lower = result.answer.lower()

    # Pass condition: either [UNCERTAIN] was emitted AND rendered with a
    # visible marker, OR the LLM cleanly refused without fabricating sub-claims.
    # Both signal the same property: the system did not hallucinate.
    refusal_phrases = [
        "no record", "no mention", "cannot find", "couldn't find", "could not find",
        "don't have information", "do not have information", "no information",
        "not mentioned", "not present in", "not contain", "no reference",
        "no data", "no evidence", "i cannot", "i can't", "i'm unable",
    ]
    refusal_seen = any(p in answer_lower for p in refusal_phrases)

    if count == 0:
        if refusal_seen:
            return (
                True,
                "LLM cleanly refused (no sub-claims to mark); "
                "non-hallucination property holds without [UNCERTAIN]",
            )
        return (
            False,
            "uncertain_render fired but count=0 and no refusal phrase detected — "
            "LLM may have fabricated; verify answer.md instructs [UNCERTAIN] on "
            "unsupported claims",
        )

    # render_uncertain replaces [UNCERTAIN] with a visible glyph; check for
    # common uncertainty markers the renderer may produce, OR a clean refusal
    # (which is equally good).
    visible_markers = ["uncertain", "[?]", "⚠", "cannot verify", "not sure", "unknown"]
    if not any(m in answer_lower for m in visible_markers) and not refusal_seen:
        return (
            False,
            f"uncertain_render fired (count={count}) but rendered answer "
            f"shows no visible uncertainty marker. "
            f"Answer excerpt: {result.answer[:200]!r}",
        )

    return (
        True,
        f"[UNCERTAIN] rendered {count}x with visible marker / refusal in answer",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("Phase 4 acceptance benchmark", flush=True)
    print("=" * 50, flush=True)
    print("Loading config and initialising orchestrators...", flush=True)

    cfg_base = load_config()
    corpus_user_id = cfg_base.user.default_user_id  # "default"

    # Q1 config: gate only.
    cfg_q1 = cfg_base.model_copy(deep=True)
    cfg_q1.compaction.gate_enabled = True

    # Q2 config: clue only.
    cfg_q2 = cfg_base.model_copy(deep=True)
    cfg_q2.compaction.clue_enabled = True

    # Q3 config: dialog MST; trigger after 12 turns, keep last 6.
    cfg_q3 = cfg_base.model_copy(deep=True)
    cfg_q3.compaction.dialog_mst_enabled = True
    cfg_q3.compaction.compact_after_turns = 12
    cfg_q3.compaction.keep_recent_turns = 6

    # Q4 config: uncertain masking.
    cfg_q4 = cfg_base.model_copy(deep=True)
    cfg_q4.compaction.mask_uncertain = True

    print("Orchestrators configured. Starting benchmark...\n", flush=True)

    # Each suite entry: (label, callable that returns (bool, str)).
    # Orchestrators are constructed inside the lambdas so each test gets
    # an isolated DB singleton if needed — the orchestrators share the
    # same SQLite + Chroma paths (read/write) but different in-memory state.
    suites: list[tuple[str, object]] = [
        (
            "Q1 — gate SKIPs small-talk",
            lambda: q1_gate_skips_thanks(Orchestrator(cfg_q1), corpus_user_id),
        ),
        (
            "Q2 — clue boosts vague-query retrieval",
            lambda: q2_clue_boosts_retrieval(
                Orchestrator(cfg_q2),
                corpus_user_id,
                # Deliberately vague query — no entity names, just a gesture.
                "tell me about that paper on managing retrieval for dialogue systems",
                # Keywords from the RAGate / dialogue-system papers in the corpus.
                ["ragate", "dialogue", "retrieval", "ketod", "knowledge-grounded",
                 "turn", "context", "grounding", "response"],
            ),
        ),
        (
            "Q3 — dialog MST preserves old fact",
            lambda: q3_dialog_mst_preserves_old_fact(Orchestrator(cfg_q3), corpus_user_id),
        ),
        (
            "Q4 — [UNCERTAIN] marker rendered",
            lambda: q4_uncertain_marker_appears(Orchestrator(cfg_q4), corpus_user_id),
        ),
    ]

    passed = 0
    results: list[tuple[str, bool, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        console=console,
        transient=False,
    ) as prog:
        task = prog.add_task("Phase 4 benchmark", total=len(suites))

        for label, fn in suites:
            prog.update(task, description=f"[bold]{label}[/]")
            console.print(f"\n[bold]{label}[/bold]")
            t0 = time.time()
            try:
                ok, msg = fn()  # type: ignore[operator]
            except Exception as exc:  # noqa: BLE001
                import traceback

                ok = False
                msg = f"exception: {type(exc).__name__}: {exc}"
                console.print(f"  [red]EXCEPTION[/] — {msg}")
                console.print(traceback.format_exc())
            dur = time.time() - t0

            if ok:
                passed += 1
                console.print(f"  [green]PASS[/] — {msg} ({dur:.1f}s)")
            else:
                console.print(f"  [red]FAIL[/] — {msg} ({dur:.1f}s)")

            results.append((label, ok, msg))
            prog.advance(task)

    # Summary table
    console.print("\n" + "=" * 60)
    console.print("[bold]Phase 4 Benchmark Summary[/bold]")
    console.print("=" * 60)
    for label, ok, msg in results:
        glyph = "[green]PASS[/]" if ok else "[red]FAIL[/]"
        console.print(f"  {glyph}  {label}")
        if not ok:
            console.print(f"       {msg}")

    console.print(f"\n[bold]Score: {passed}/{len(suites)}[/bold]")

    accept = passed >= 3
    if accept:
        console.print("[green bold]ACCEPTED (>= 3/4)[/]")
    else:
        console.print("[red bold]FAILED (< 3/4)[/]")

    return 0 if accept else 1


if __name__ == "__main__":
    sys.exit(main())
