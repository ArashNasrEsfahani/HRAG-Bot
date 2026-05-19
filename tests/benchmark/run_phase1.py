"""Phase 1 benchmark runner for the Hierarchical RAG chatbot.

Usage (from project root):
    python tests/benchmark/run_phase1.py

Produces:
    tests/benchmark/phase1_results.md  -- Markdown report
    stdout                             -- summary table
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import yaml

from hrag.config import load_config
from hrag.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_RE_CITATION = re.compile(r"\[Source\s+\d+\]")


def has_citation(text: str) -> bool:
    return bool(_RE_CITATION.search(text))


def has_raft_blocks(text: str) -> bool:
    return "Reasoning:" in text and "Answer:" in text


def check_substrings(text: str, substrings: list[str]) -> tuple[bool, list[str]]:
    """Return (all_found, missing_list)."""
    missing = [s for s in substrings if s.lower() not in text.lower()]
    return (len(missing) == 0), missing


# ---------------------------------------------------------------------------
# Score a single question result
# ---------------------------------------------------------------------------

def score_result(
    q: dict,
    answer: str,
    sources: list[Any],
    events: list[tuple[str, dict]],
    category: str,
) -> dict:
    """Return a dict with all dimension booleans and aggregates."""

    scores: dict[str, Any] = {}

    # -- passed_substrings --
    substrings = q.get("expected_substrings", [])
    if substrings:
        passed_sub, missing = check_substrings(answer, substrings)
    else:
        passed_sub, missing = True, []
    scores["passed_substrings"] = passed_sub
    scores["missing_substrings"] = missing

    # -- has_citation --
    scores["has_citation"] = has_citation(answer)

    # -- has_raft_blocks --
    scores["has_raft_blocks"] = has_raft_blocks(answer)

    # -- source_doc_match --
    expected_doc = q.get("expected_source_doc")
    if expected_doc is None:
        scores["source_doc_match"] = True
    else:
        titles = [r.chunk.title for r in sources] if sources else []
        scores["source_doc_match"] = any(expected_doc in t for t in titles)

    # -- rerank_fallback_avoided --
    fallback_used = any(
        payload.get("fallback_used", False)
        for event_name, payload in events
        if event_name == "rerank_done"
    )
    scores["rerank_fallback_avoided"] = not fallback_used

    # -- rewriter_fired_on_followup (only meaningful for conversational_followup) --
    if category == "conversational_followup":
        rewriter_fired = any(evt == "query_rewrite" for evt, _ in events)
        scores["rewriter_fired_on_followup"] = rewriter_fired
    else:
        scores["rewriter_fired_on_followup"] = True  # N/A → pass by convention

    # -- no_hallucination (only meaningful for out_of_corpus) --
    if category == "out_of_corpus":
        scores["no_hallucination"] = "couldn't find" in answer.lower()
    else:
        scores["no_hallucination"] = True  # N/A → pass by convention

    # -- aggregate --
    applicable = [
        scores["passed_substrings"],
        scores["has_citation"],
        scores["has_raft_blocks"],
        scores["source_doc_match"],
        scores["rerank_fallback_avoided"],
        scores["rewriter_fired_on_followup"],
        scores["no_hallucination"],
    ]
    scores["passed_overall"] = all(applicable)

    return scores


# ---------------------------------------------------------------------------
# Per-turn scorer for conversational_followup
# ---------------------------------------------------------------------------

def score_followup(
    q: dict,
    turn1_answer: str,
    turn2_answer: str,
    sources1: list[Any],
    sources2: list[Any],
    events1: list[tuple[str, dict]],
    events2: list[tuple[str, dict]],
) -> dict:
    """Score a two-turn conversational followup question."""

    subs_per_turn = q.get("expected_substrings_per_turn", [[], []])
    sub1 = subs_per_turn[0] if len(subs_per_turn) > 0 else []
    sub2 = subs_per_turn[1] if len(subs_per_turn) > 1 else []

    passed_sub1, missing1 = check_substrings(turn1_answer, sub1) if sub1 else (True, [])
    passed_sub2, missing2 = check_substrings(turn2_answer, sub2) if sub2 else (True, [])
    passed_substrings = passed_sub1 and passed_sub2
    missing_all = [f"T1:{m}" for m in missing1] + [f"T2:{m}" for m in missing2]

    all_events = events1 + events2
    all_sources = sources2  # final turn sources most relevant

    # Check rewriter fired on turn 2 (the follow-up turn)
    rewriter_fired = any(evt == "query_rewrite" for evt, _ in events2)

    expected_doc = q.get("expected_source_doc")
    if expected_doc is None:
        source_doc_match = True
    else:
        titles = [r.chunk.title for r in all_sources] if all_sources else []
        source_doc_match = any(expected_doc in t for t in titles)

    fallback_used = any(
        payload.get("fallback_used", False)
        for event_name, payload in all_events
        if event_name == "rerank_done"
    )

    scores = {
        "passed_substrings": passed_substrings,
        "missing_substrings": missing_all,
        "has_citation": has_citation(turn1_answer) and has_citation(turn2_answer),
        "has_raft_blocks": has_raft_blocks(turn1_answer) and has_raft_blocks(turn2_answer),
        "source_doc_match": source_doc_match,
        "rerank_fallback_avoided": not fallback_used,
        "rewriter_fired_on_followup": rewriter_fired,
        "no_hallucination": True,  # not out_of_corpus
    }
    scores["passed_overall"] = all([
        scores["passed_substrings"],
        scores["has_citation"],
        scores["has_raft_blocks"],
        scores["source_doc_match"],
        scores["rerank_fallback_avoided"],
        scores["rewriter_fired_on_followup"],
        scores["no_hallucination"],
    ])
    return scores


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    benchmark_path = Path(__file__).parent / "phase1.yaml"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench = yaml.safe_load(f)

    questions = bench["questions"]

    # Build ONE orchestrator for the entire run
    print("Loading config and initialising orchestrator …")
    cfg = load_config()
    orch = Orchestrator(cfg)
    print("Orchestrator ready.\n")

    run_start = time.time()
    results_data: list[dict] = []

    # Determine which user_id holds the corpus data.
    # The ingest pipeline always writes under cfg.user.default_user_id ("default").
    # Using any other user_id would yield empty retrieval results.
    corpus_user_id = cfg.user.default_user_id  # "default"

    for q in questions:
        qid = q["id"]
        category = q["category"]
        print(f"Running {qid} [{category}] …", flush=True)

        if category == "conversational_followup":
            turns = q["turns"]

            # Turn 1
            events1: list[tuple[str, dict]] = []
            def _cb1(evt: str, payload: dict, _ev=events1) -> None:
                _ev.append((evt, payload))

            t1_start = time.time()
            r1 = orch.chat(turns[0], user_id=corpus_user_id, session_id=None,
                           progress=_cb1, stream=False)
            t1_dur = time.time() - t1_start
            print(f"  Turn 1 done in {t1_dur:.1f}s", flush=True)

            # Turn 2 — same session
            events2: list[tuple[str, dict]] = []
            def _cb2(evt: str, payload: dict, _ev=events2) -> None:
                _ev.append((evt, payload))

            t2_start = time.time()
            r2 = orch.chat(turns[1], user_id=corpus_user_id,
                           session_id=r1.session_id,
                           progress=_cb2, stream=False)
            t2_dur = time.time() - t2_start
            print(f"  Turn 2 done in {t2_dur:.1f}s", flush=True)

            scores = score_followup(
                q,
                r1.answer, r2.answer,
                r1.sources, r2.sources,
                events1, events2,
            )
            results_data.append({
                "q": q,
                "category": category,
                "answer": f"[T1] {r1.answer}\n\n[T2] {r2.answer}",
                "answer_t1": r1.answer,
                "answer_t2": r2.answer,
                "sources": r2.sources,
                "events": events1 + events2,
                "events1": events1,
                "events2": events2,
                "scores": scores,
                "duration_s": t1_dur + t2_dur,
                "duration_per_turn": [t1_dur, t2_dur],
            })
        else:
            events: list[tuple[str, dict]] = []
            def _cb(evt: str, payload: dict, _ev=events) -> None:
                _ev.append((evt, payload))

            t_start = time.time()
            r = orch.chat(q["question"], user_id=corpus_user_id,
                          session_id=None, progress=_cb, stream=False)
            dur = time.time() - t_start
            print(f"  Done in {dur:.1f}s", flush=True)

            scores = score_result(q, r.answer, r.sources, events, category)
            results_data.append({
                "q": q,
                "category": category,
                "answer": r.answer,
                "sources": r.sources,
                "events": events,
                "scores": scores,
                "duration_s": dur,
            })

    total_wall = time.time() - run_start
    passed_count = sum(1 for d in results_data if d["scores"]["passed_overall"])

    # -----------------------------------------------------------------------
    # Print summary table to stdout
    # -----------------------------------------------------------------------
    def _bool(v: bool) -> str:
        return "PASS" if v else "FAIL"

    header = (
        "| id | category | overall | substrings | citation | raft | src match "
        "| no fallback | rewriter | no halluc | time (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"

    print("\n# Phase 1 Benchmark Results")
    print(f"Pass rate: {passed_count}/6\n")
    print(header)
    print(sep)
    for d in results_data:
        s = d["scores"]
        row = (
            f"| {d['q']['id']} "
            f"| {d['category']} "
            f"| {_bool(s['passed_overall'])} "
            f"| {_bool(s['passed_substrings'])} "
            f"| {_bool(s['has_citation'])} "
            f"| {_bool(s['has_raft_blocks'])} "
            f"| {_bool(s['source_doc_match'])} "
            f"| {_bool(s['rerank_fallback_avoided'])} "
            f"| {_bool(s['rewriter_fired_on_followup'])} "
            f"| {_bool(s['no_hallucination'])} "
            f"| {d['duration_s']:.1f} |"
        )
        print(row)

    print(f"\nTotal wall time: {total_wall:.1f}s")

    # -----------------------------------------------------------------------
    # Build Markdown report
    # -----------------------------------------------------------------------
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    md_lines: list[str] = [
        "# Phase 1 Benchmark Results",
        "",
        f"Run timestamp: {ts}",
        f"Total wall time: {total_wall:.1f}s",
        f"Pass rate: {passed_count}/6 questions",
        "",
        "## Summary table",
        "",
        header,
        sep,
    ]
    for d in results_data:
        s = d["scores"]
        md_lines.append(
            f"| {d['q']['id']} "
            f"| {d['category']} "
            f"| {_bool(s['passed_overall'])} "
            f"| {_bool(s['passed_substrings'])} "
            f"| {_bool(s['has_citation'])} "
            f"| {_bool(s['has_raft_blocks'])} "
            f"| {_bool(s['source_doc_match'])} "
            f"| {_bool(s['rerank_fallback_avoided'])} "
            f"| {_bool(s['rewriter_fired_on_followup'])} "
            f"| {_bool(s['no_hallucination'])} "
            f"| {d['duration_s']:.1f} |"
        )

    md_lines += ["", "## Per-question detail", ""]

    for d in results_data:
        q = d["q"]
        s = d["scores"]
        cat = d["category"]
        qid = q["id"]

        md_lines.append(f"### {qid} — {cat}")
        md_lines.append("")

        if cat == "conversational_followup":
            turns = q.get("turns", [])
            md_lines.append(f"**Turn 1:** {turns[0] if turns else ''}")
            md_lines.append(f"**Turn 2:** {turns[1] if len(turns) > 1 else ''}")
        else:
            md_lines.append(f"**Question:** {q.get('question', '')}")

        # Answer excerpt
        answer_text = d["answer"]
        excerpt = answer_text[:400].replace("\n", " ")
        md_lines.append(f"**Answer (excerpt, first 400 chars):** {excerpt}")
        md_lines.append("")

        # Sources
        srcs = d["sources"]
        if srcs:
            src_lines = [
                f"  - [{i+1}] {r.chunk.title or 'Untitled'} / {r.chunk.section or 'N/A'}"
                for i, r in enumerate(srcs[:6])
            ]
            md_lines.append("**Sources returned:**")
            md_lines.extend(src_lines)
        else:
            md_lines.append("**Sources returned:** (none)")
        md_lines.append("")

        # Events of interest
        rerank_events = [(ev, pl) for ev, pl in d["events"] if ev == "rerank_done"]
        if rerank_events:
            for _, pl in rerank_events:
                if pl.get("fallback_used"):
                    md_lines.append(f"**WARNING: rerank fallback used** — kept {pl['kept']} chunks")
        rewrite_events = [(ev, pl) for ev, pl in d["events"] if ev == "query_rewrite"]
        if rewrite_events:
            for _, pl in rewrite_events:
                md_lines.append(f"**Query rewrite:** `{pl['original']}` → `{pl['rewritten']}`")
        md_lines.append("")

        # Missing substrings
        if s["missing_substrings"]:
            md_lines.append(f"**Missing substrings:** {s['missing_substrings']}")
        else:
            md_lines.append("**Missing substrings:** (none)")
        md_lines.append("")

        # Dimension breakdown
        md_lines.append("**Dimension scores:**")
        md_lines.append(f"- substrings: {_bool(s['passed_substrings'])}")
        md_lines.append(f"- citation: {_bool(s['has_citation'])}")
        md_lines.append(f"- raft_blocks: {_bool(s['has_raft_blocks'])}")
        md_lines.append(f"- source_doc_match: {_bool(s['source_doc_match'])}")
        md_lines.append(f"- rerank_fallback_avoided: {_bool(s['rerank_fallback_avoided'])}")
        md_lines.append(f"- rewriter_fired_on_followup: {_bool(s['rewriter_fired_on_followup'])}")
        md_lines.append(f"- no_hallucination: {_bool(s['no_hallucination'])}")
        md_lines.append("")
        md_lines.append(f"**Score: {'PASS' if s['passed_overall'] else 'FAIL'}**")
        md_lines.append("")
        if q.get("notes"):
            md_lines.append(f"**Notes:** {q['notes'].strip()}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    report_path = Path(__file__).parent / "phase1_results.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    run_benchmark()
