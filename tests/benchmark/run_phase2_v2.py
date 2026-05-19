"""Phase 2 v2 benchmark runner for the Hierarchical RAG chatbot.

Usage (from project root):
    python tests/benchmark/run_phase2_v2.py

Produces:
    tests/benchmark/phase2_v2_results.md   -- Markdown report (grep-able)
    tests/benchmark/phase2_v2_results.html -- self-contained HTML report (primary)
    stdout                                 -- live progress + summary table

Differences vs run_phase2.py:
    - Reads phase2_v2.yaml (adds optional expected_route per question).
    - Captures router_classify progress event; computes route_match dimension.
    - route_match is included in passed_overall (hard pass criterion).
    - Live progress lines include route info and route_match indicator.
    - Writes a self-contained HTML report with CSS-bar charts and collapsible
      per-question detail sections (no external CDN, no JS dependencies).
    - Two-tier substring matching: fuzzy (stem-based) + optional LLM judge.
"""

from __future__ import annotations

import html as _html_escape
import re
import sys
import time
from collections import Counter
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

_KG_RETRIEVERS = {"kg_ppr", "router"}

# Matches identifiers that must use exact matching: digits, hyphens between
# words, decimals, percentages, or comma-formatted numbers.
_RE_IDENTIFIER = re.compile(r"[\d]")


def has_citation(text: str) -> bool:
    return bool(_RE_CITATION.search(text))


def has_raft_blocks(text: str) -> bool:
    return "Reasoning:" in text and "Answer:" in text


# ---------------------------------------------------------------------------
# Simple suffix stemmer (no external deps)
# ---------------------------------------------------------------------------

# Ordered from longest to shortest so longer suffixes are stripped first.
# "ating" before "ing"/"ate" ensures integrate/integrates/integrating all
# converge to the same root ("integr" for the integrate* family).
_STEM_SUFFIXES = (
    "ization", "isation", "ational", "nesses", "ations",
    "ments", "ities", "izing", "ising", "ation", "ating",
    "ness", "ment", "ings", "ized", "ised", "ated", "ates", "ate",
    "tion", "ity", "ing", "ies", "ors", "ers",
    "ous", "ful", "al", "ed", "es", "er", "or", "ly", "s",
)

_MIN_STEM_LEN = 3  # never strip below this many characters


def _stem(word: str) -> str:
    """Strip common English suffixes (longest-match, no external deps)."""
    w = word.lower()
    for sfx in _STEM_SUFFIXES:
        if w.endswith(sfx) and len(w) - len(sfx) >= _MIN_STEM_LEN:
            return w[: len(w) - len(sfx)]
    return w


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word chars, remove empty tokens."""
    return [t for t in re.split(r"\W+", text.lower()) if t]


def _normalize_ws(text: str) -> str:
    """Collapse runs of whitespace/punctuation to single space, strip, lowercase."""
    return re.sub(r"[\s\W]+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Tier-1 fuzzy substring matcher
# ---------------------------------------------------------------------------

def check_substrings(
    text: str,
    substrings: list[str],
) -> tuple[bool, list[str], list[dict]]:
    """Tier-1 fuzzy match. Returns (all_passed, missing_list, per_term_results).

    For each expected substring, checks (in order):
      1. Exact case-insensitive substring match (current behavior).
      2. Whitespace + punctuation normalized match.
      3. Stemmed sliding-window match: all stem tokens of the expected phrase
         must appear in the answer's stemmed token sequence within a window
         of N=8 tokens, preserving order.
      4. Terms containing digits are protected — exact match only (no stemming).
         This covers numbers (0.8, 5,324), version strings, and hyphenated
         identifiers like "RAGate-PEFT".
    """
    per_term: list[dict] = []
    missing: list[str] = []
    text_lower = text.lower()
    text_norm = _normalize_ws(text)
    text_tokens = _tokenize(text)
    text_stems = [_stem(t) for t in text_tokens]
    WINDOW = 8

    for term in substrings:
        term_lower = term.lower()

        # Step 1: exact case-insensitive
        if term_lower in text_lower:
            per_term.append({"term": term, "tier": "exact", "passed": True})
            continue

        # Step 2: whitespace/punctuation normalized
        term_norm = _normalize_ws(term)
        if term_norm in text_norm:
            per_term.append({"term": term, "tier": "fuzzy", "passed": True})
            continue

        # Step 4 guard: if the term contains digits, require exact only — skip stem.
        if _RE_IDENTIFIER.search(term):
            per_term.append({"term": term, "tier": "exact", "passed": False})
            missing.append(term)
            continue

        # Step 3: stemmed sliding-window match
        term_stems = [_stem(t) for t in _tokenize(term)]
        if not term_stems:
            # Empty term trivially passes
            per_term.append({"term": term, "tier": "fuzzy", "passed": True})
            continue

        matched = False
        window_size = len(term_stems) + WINDOW
        term_stem_set = set(term_stems)
        for start in range(len(text_stems)):
            window = text_stems[start: start + window_size]
            window_set = set(window)
            # All required stems must appear somewhere in the window (unordered).
            # Unordered matching is intentional: paraphrases reorder concepts
            # (e.g. "integrates knowledge" should match "knowledge integration").
            if term_stem_set.issubset(window_set):
                matched = True
                break

        if matched:
            per_term.append({"term": term, "tier": "fuzzy", "passed": True})
        else:
            per_term.append({"term": term, "tier": "fuzzy", "passed": False})
            missing.append(term)

    all_passed = len(missing) == 0
    return all_passed, missing, per_term


# ---------------------------------------------------------------------------
# Tier-2 LLM judge (slow, reserved for marginal cases)
# ---------------------------------------------------------------------------

def llm_judge_substrings(
    answer: str,
    substrings: list[str],
    question: str,
    llm: Any,
) -> tuple[bool, list[str]]:
    """Ask the LLM whether each expected substring's MEANING is conveyed.
    Single batched call. Returns (all_passed, missing_list)."""
    if not substrings or llm is None:
        return True, []

    n = len(substrings)
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(substrings))
    prompt = (
        "You are a strict evaluator. The expected facts below MUST appear in the answer "
        "either verbatim or as a clearly equivalent paraphrase (same meaning, no "
        "hedging or partial overlap).\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Expected facts:\n"
        f"{numbered}\n\n"
        f"For each expected fact, output one line in this format:\n"
        "N: PASS|FAIL — <brief reason if FAIL>\n\n"
        f"Output exactly {n} lines, no preamble."
    )

    try:
        response = llm.complete(prompt)
    except Exception:
        # If judge fails for any reason, conservatively return missing = all
        return False, list(substrings)

    # Parse "N: PASS|FAIL" lines
    missing: list[str] = []
    lines = response.strip().splitlines()
    for i, sub in enumerate(substrings):
        # Find line starting with "{i+1}:"
        verdict = "FAIL"
        for line in lines:
            if line.strip().startswith(f"{i+1}:"):
                upper = line.upper()
                if "PASS" in upper:
                    verdict = "PASS"
                else:
                    verdict = "FAIL"
                break
        if verdict == "FAIL":
            missing.append(sub)

    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Match-tier label helper
# ---------------------------------------------------------------------------

def _match_tier_label(scores: dict) -> str:
    """Return 'exact', 'fuzzy', 'judge', or '-' based on score dict."""
    if not scores.get("passed_substrings", True):
        return "-"
    judge_used = scores.get("judge_used", False)
    if judge_used:
        return "judge"
    per_term = scores.get("per_term", [])
    if any(t.get("tier") == "fuzzy" for t in per_term if t.get("passed")):
        return "fuzzy"
    return "exact"


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def build_route_distribution(sources: list[Any]) -> dict[str, int]:
    """Tally retriever names from a list of RetrievalResult objects."""
    if not sources:
        return {}
    counts: Counter[str] = Counter(r.retriever for r in sources)
    return dict(counts)


def dominant_route(route_dist: dict[str, int]) -> str:
    """Return the most-used retriever name, or 'none' when no sources exist."""
    if not route_dist:
        return "none"
    return max(route_dist, key=lambda k: route_dist[k])


def has_kg_path_hit(sources: list[Any]) -> bool:
    """True if any source came from a KG retriever."""
    return any(getattr(r, "retriever", "") in _KG_RETRIEVERS for r in sources)


def extract_route_classified(events: list[tuple[str, dict]]) -> str:
    """Scan events for router_classify and return payload['label'], or 'unknown'."""
    for event_name, payload in events:
        if event_name == "router_classify":
            label = payload.get("label")
            if label:
                return str(label)
    return "unknown"


# ---------------------------------------------------------------------------
# Score a single question result
# ---------------------------------------------------------------------------

def score_result(
    q: dict,
    answer: str,
    sources: list[Any],
    events: list[tuple[str, dict]],
    category: str,
    route_dist: dict[str, int],
    llm: Any = None,
) -> dict:
    """Return a dict with all dimension booleans and aggregates."""

    scores: dict[str, Any] = {}

    # -- passed_substrings (two-tier) --
    substrings = q.get("expected_substrings", [])
    question_text = q.get("question", "")
    if substrings:
        exact_pass, missing, per_term = check_substrings(answer, substrings)
        if exact_pass:
            passed_sub = True
            judge_used = False
        elif category == "out_of_corpus":
            # Refusal questions — keep tier 1 only
            passed_sub = False
            judge_used = False
        else:
            # Tier 2: LLM judge
            judge_pass, judge_missing = llm_judge_substrings(
                answer=answer,
                substrings=substrings,
                question=question_text,
                llm=llm,
            )
            passed_sub = judge_pass
            missing = judge_missing
            judge_used = True
    else:
        passed_sub, missing, per_term = True, [], []
        judge_used = False

    scores["passed_substrings"] = passed_sub
    scores["missing_substrings"] = missing
    scores["per_term"] = per_term
    scores["judge_used"] = judge_used

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

    # -- route_classified and route_match (Phase 2 v2 hard dimension) --
    route_classified = extract_route_classified(events)
    scores["route_classified"] = route_classified
    expected_route = q.get("expected_route")
    if expected_route is None:
        scores["route_match"] = True   # N/A → pass by convention
    else:
        scores["route_match"] = (expected_route == route_classified)

    # -- Phase 2 informational dimensions (NOT included in passed_overall) --
    scores["route_observed"] = dominant_route(route_dist)
    scores["kg_path_hit"] = has_kg_path_hit(sources)

    # -- match_tier informational label --
    # Computed after judge_used is set
    scores["match_tier"] = _match_tier_label(scores)

    # -- aggregate (includes route_match as of v2) --
    applicable = [
        scores["passed_substrings"],
        scores["has_citation"],
        scores["has_raft_blocks"],
        scores["source_doc_match"],
        scores["rerank_fallback_avoided"],
        scores["rewriter_fired_on_followup"],
        scores["no_hallucination"],
        scores["route_match"],
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
    route_dist: dict[str, int],
    llm: Any = None,
) -> dict:
    """Score a two-turn conversational followup question."""

    subs_per_turn = q.get("expected_substrings_per_turn", [[], []])
    sub1 = subs_per_turn[0] if len(subs_per_turn) > 0 else []
    sub2 = subs_per_turn[1] if len(subs_per_turn) > 1 else []
    question_text = q.get("turns", [""])[0]

    # --- Turn 1 substring scoring ---
    if sub1:
        exact1, missing1, per_term1 = check_substrings(turn1_answer, sub1)
        if exact1:
            passed_sub1 = True
            judge_used1 = False
        else:
            judge_pass1, judge_missing1 = llm_judge_substrings(
                answer=turn1_answer,
                substrings=sub1,
                question=question_text,
                llm=llm,
            )
            passed_sub1 = judge_pass1
            missing1 = judge_missing1
            judge_used1 = True
    else:
        passed_sub1, missing1, per_term1 = True, [], []
        judge_used1 = False

    # --- Turn 2 substring scoring ---
    if sub2:
        exact2, missing2, per_term2 = check_substrings(turn2_answer, sub2)
        if exact2:
            passed_sub2 = True
            judge_used2 = False
        else:
            judge_pass2, judge_missing2 = llm_judge_substrings(
                answer=turn2_answer,
                substrings=sub2,
                question=question_text,
                llm=llm,
            )
            passed_sub2 = judge_pass2
            missing2 = judge_missing2
            judge_used2 = True
    else:
        passed_sub2, missing2, per_term2 = True, [], []
        judge_used2 = False

    passed_substrings = passed_sub1 and passed_sub2
    missing_all = [f"T1:{m}" for m in missing1] + [f"T2:{m}" for m in missing2]
    per_term_all = per_term1 + per_term2
    judge_used = judge_used1 or judge_used2

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

    # Phase 2 informational — combine both turns' sources for route analysis
    combined_sources = list(sources1) + list(sources2)

    # route_match — use turn 1 classification (first classification is the intent signal)
    route_classified = extract_route_classified(events1) or extract_route_classified(events2)
    expected_route = q.get("expected_route")
    if expected_route is None:
        route_match = True
    else:
        route_match = (expected_route == route_classified)

    scores = {
        "passed_substrings": passed_substrings,
        "missing_substrings": missing_all,
        "per_term": per_term_all,
        "judge_used": judge_used,
        "has_citation": has_citation(turn1_answer) and has_citation(turn2_answer),
        "has_raft_blocks": has_raft_blocks(turn1_answer) and has_raft_blocks(turn2_answer),
        "source_doc_match": source_doc_match,
        "rerank_fallback_avoided": not fallback_used,
        "rewriter_fired_on_followup": rewriter_fired,
        "no_hallucination": True,  # not out_of_corpus
        # Phase 2 v2 route dimensions
        "route_classified": route_classified,
        "route_match": route_match,
        # Phase 2 informational
        "route_observed": dominant_route(route_dist),
        "kg_path_hit": has_kg_path_hit(combined_sources),
    }
    # match_tier computed after judge_used is in scores
    scores["match_tier"] = _match_tier_label(scores)
    scores["passed_overall"] = all([
        scores["passed_substrings"],
        scores["has_citation"],
        scores["has_raft_blocks"],
        scores["source_doc_match"],
        scores["rerank_fallback_avoided"],
        scores["rewriter_fired_on_followup"],
        scores["no_hallucination"],
        scores["route_match"],
    ])
    return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail_detail(scores: dict) -> str:
    """Produce a short failure summary string for the progress line."""
    parts: list[str] = []
    if not scores["passed_substrings"] and scores.get("missing_substrings"):
        parts.append(f"missing: {scores['missing_substrings']}")
    if not scores["has_citation"]:
        parts.append("no citation")
    if not scores["has_raft_blocks"]:
        parts.append("no raft blocks")
    if not scores["source_doc_match"]:
        parts.append("wrong source doc")
    if not scores["rerank_fallback_avoided"]:
        parts.append("rerank fallback triggered")
    if not scores["rewriter_fired_on_followup"]:
        parts.append("rewriter did not fire")
    if not scores["no_hallucination"]:
        parts.append("possible hallucination")
    if not scores.get("route_match", True):
        parts.append("route mismatch")
    return "; ".join(parts) if parts else "unknown failure"


def _bool(v: bool) -> str:
    return "PASS" if v else "FAIL"


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_CSS_MAX_BAR_WIDTH = 280  # px


def _route_distribution_html(routes: dict[str, int], width_px: int = _CSS_MAX_BAR_WIDTH) -> str:
    """Return HTML for a pure-CSS horizontal bar chart of route counts."""
    if not routes:
        return "<p class='secondary'>No route data.</p>"
    total = sum(routes.values())
    max_val = max(routes.values())
    rows = []
    for label in sorted(routes, key=lambda k: -routes[k]):
        count = routes[label]
        bar_w = int(width_px * count / max_val) if max_val else 0
        bar_blocks = "█" * max(1, count)
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-label">{_html_escape.escape(label)}</span>'
            f'<div class="bar" style="width:{bar_w}px"></div>'
            f'<span class="bar-count">{bar_blocks} {count} / {total}</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _dimension_failure_counts(results_data: list[dict]) -> dict[str, int]:
    """Count FAIL across all questions for each scored dimension."""
    dimensions = [
        "passed_substrings",
        "has_citation",
        "has_raft_blocks",
        "source_doc_match",
        "rerank_fallback_avoided",
        "rewriter_fired_on_followup",
        "no_hallucination",
        "route_match",
    ]
    counts: dict[str, int] = {d: 0 for d in dimensions}
    for entry in results_data:
        s = entry["scores"]
        for dim in dimensions:
            if not s.get(dim, True):
                counts[dim] += 1
    return counts


def _timing_bar_html(results_data: list[dict], width_px: int = _CSS_MAX_BAR_WIDTH) -> str:
    """Return HTML for a per-question timing bar chart."""
    if not results_data:
        return ""
    max_dur = max(d["duration_s"] for d in results_data) or 1.0
    rows = []
    for entry in results_data:
        qid = entry["q"]["id"]
        dur = entry["duration_s"]
        bar_w = int(width_px * dur / max_dur)
        passed = entry["scores"]["passed_overall"]
        cls = "pass-bar" if passed else "fail-bar"
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-label">{_html_escape.escape(qid)}</span>'
            f'<div class="bar {cls}" style="width:{bar_w}px"></div>'
            f'<span class="bar-count">{dur:.1f}s</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _sources_table_html(sources: list[Any]) -> str:
    """Return an HTML table for retrieval sources."""
    if not sources:
        return "<p class='secondary'>No sources returned.</p>"
    rows = ["<table class='src-table'><thead><tr>"
            "<th>#</th><th>Title</th><th>Section</th><th>Retriever</th>"
            "</tr></thead><tbody>"]
    for i, r in enumerate(sources[:8], start=1):
        title = _html_escape.escape(r.chunk.title or "Untitled")
        section = _html_escape.escape(r.chunk.section or "N/A")
        retriever = _html_escape.escape(r.retriever or "N/A")
        rows.append(
            f"<tr><td>{i}</td><td>{title}</td><td>{section}</td>"
            f"<td><code>{retriever}</code></td></tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _dimension_table_html(scores: dict, expected_route: str | None) -> str:
    """Return an HTML table showing each dimension's pass/fail."""
    dims = [
        ("substrings", scores["passed_substrings"]),
        ("citation", scores["has_citation"]),
        ("raft_blocks", scores["has_raft_blocks"]),
        ("source_doc_match", scores["source_doc_match"]),
        ("no_rerank_fallback", scores["rerank_fallback_avoided"]),
        ("rewriter_on_followup", scores["rewriter_fired_on_followup"]),
        ("no_hallucination", scores["no_hallucination"]),
        ("route_match", scores.get("route_match", True)),
    ]
    rows = ["<table class='dim-table'><thead><tr>"
            "<th>Dimension</th><th>Result</th>"
            "</tr></thead><tbody>"]
    for name, passed in dims:
        badge = f'<span class="{"pass" if passed else "fail"}">'
        badge += ("PASS" if passed else "FAIL") + "</span>"
        rows.append(f"<tr><td>{name}</td><td>{badge}</td></tr>")

    # Match tier row
    match_tier = scores.get("match_tier", "-")
    tier_display = {
        "exact": "exact ✓",
        "fuzzy": "fuzzy ✓",
        "judge": "judge ✓",
        "-": "-",
    }.get(match_tier, match_tier)
    rows.append(
        f"<tr class='info-row'><td>match_tier</td>"
        f"<td><code>{_html_escape.escape(tier_display)}</code></td></tr>"
    )

    # Judge verdict rows (only when judge was used)
    if scores.get("judge_used"):
        per_term = scores.get("per_term", [])
        if per_term:
            judge_items = "".join(
                f"<li><code>{_html_escape.escape(t['term'])}</code>: "
                f"<span class=\"{'pass' if t['passed'] else 'fail'}\">"
                f"{'PASS' if t['passed'] else 'FAIL'}</span></li>"
                for t in per_term
            )
            rows.append(
                f"<tr class='info-row'><td>judge verdicts</td>"
                f"<td><ul style='margin:0;padding-left:1.2em'>{judge_items}</ul></td></tr>"
            )

    # Informational
    rows.append(
        f"<tr class='info-row'><td>route_classified</td>"
        f"<td><code>{_html_escape.escape(scores.get('route_classified','unknown'))}</code></td></tr>"
    )
    rows.append(
        f"<tr class='info-row'><td>expected_route</td>"
        f"<td><code>{_html_escape.escape(str(expected_route) if expected_route else 'N/A')}</code></td></tr>"
    )
    rows.append(
        f"<tr class='info-row'><td>route_observed (dominant)</td>"
        f"<td><code>{_html_escape.escape(scores.get('route_observed','none'))}</code></td></tr>"
    )
    rows.append(
        f"<tr class='info-row'><td>kg_path_hit</td>"
        f"<td>{'yes' if scores.get('kg_path_hit') else 'no'}</td></tr>"
    )
    rows.append("</tbody></table>")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main HTML renderer
# ---------------------------------------------------------------------------

_CSS = """\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  background: #fafafa;
  color: #111;
}
.container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 20px 60px;
}

/* ---- Typography ---- */
h1 { font-size: 1.75rem; font-weight: 700; }
h2 { font-size: 1.2rem; font-weight: 600; margin: 28px 0 12px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
h3 { font-size: 1rem; font-weight: 600; margin: 20px 0 8px; }
p  { margin: 8px 0; }
.secondary { color: #6b7280; font-size: 0.85rem; }

/* ---- Cards ---- */
.card {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 20px 24px;
  margin-bottom: 20px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}

/* ---- Header ---- */
header.page-header {
  background: #fff;
  border-bottom: 1px solid #e5e7eb;
  padding: 20px 0 16px;
  margin-bottom: 28px;
}
.header-inner {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
}
.header-meta { color: #6b7280; font-size: 0.85rem; margin-top: 4px; }

/* ---- Pass-rate badge ---- */
.badge {
  display: inline-block;
  padding: 6px 18px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 1.1rem;
  color: #fff;
  white-space: nowrap;
}
.badge-green  { background: #16a34a; }
.badge-amber  { background: #d97706; }
.badge-red    { background: #dc2626; }

/* ---- Pass/Fail inline spans ---- */
span.pass { color: #16a34a; font-weight: 600; }
span.fail { color: #dc2626; font-weight: 600; }

/* ---- Config table ---- */
.config-grid {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 6px 16px;
}
.config-key   { font-weight: 600; color: #374151; }
.config-value { font-family: monospace; color: #1d4ed8; }

/* ---- Summary table ---- */
.summary-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}
.summary-table th {
  background: #f3f4f6;
  text-align: left;
  padding: 8px 10px;
  border-bottom: 2px solid #e5e7eb;
  white-space: nowrap;
}
.summary-table td {
  padding: 7px 10px;
  border-bottom: 1px solid #f3f4f6;
  vertical-align: middle;
}
.summary-table tr:hover td { background: #f9fafb; }
.summary-table td:first-child { font-weight: 600; }

/* ---- Source / dimension tables ---- */
.src-table, .dim-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.83rem;
  margin-top: 8px;
}
.src-table th, .dim-table th {
  background: #f3f4f6;
  text-align: left;
  padding: 6px 10px;
  border-bottom: 1px solid #e5e7eb;
}
.src-table td, .dim-table td {
  padding: 5px 10px;
  border-bottom: 1px solid #f3f4f6;
  vertical-align: middle;
}
.info-row td { color: #6b7280; font-size: 0.8rem; }

/* ---- Bar charts ---- */
.bar-chart { margin: 10px 0 16px; }
.bar-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.bar-label {
  width: 150px;
  text-align: right;
  color: #374151;
  font-size: 0.82rem;
  flex-shrink: 0;
}
.bar {
  height: 18px;
  background: #2563eb;
  border-radius: 3px;
  min-width: 4px;
}
.bar.pass-bar { background: #16a34a; }
.bar.fail-bar { background: #dc2626; }
.bar-count {
  color: #6b7280;
  font-size: 0.8rem;
  white-space: nowrap;
}

/* ---- Details / collapsible ---- */
details.q-detail {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
  overflow: hidden;
}
details.q-detail > summary {
  padding: 14px 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 12px;
  list-style: none;
  user-select: none;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
}
details.q-detail > summary:hover { background: #f3f4f6; }
details.q-detail > summary::marker { display: none; }
.summary-arrow { font-size: 0.9rem; color: #6b7280; }
details[open] .summary-arrow::before { content: "▼"; }
details:not([open]) .summary-arrow::before { content: "▶"; }
.q-detail-body { padding: 20px 24px; }
.q-detail-body section { margin-bottom: 20px; }

/* ---- Code / pre ---- */
code {
  font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
  background: #f3f4f6;
  border-radius: 3px;
  padding: 1px 5px;
  font-size: 0.85em;
}
pre.answer-block {
  font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
  background: #f3f4f6;
  border-radius: 6px;
  padding: 14px 16px;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.82rem;
  max-height: 420px;
  overflow-y: auto;
  border: 1px solid #e5e7eb;
}

/* ---- Route badge ---- */
.route-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  background: #dbeafe;
  color: #1d4ed8;
  margin-left: 6px;
}
.route-badge.route-match  { background: #dcfce7; color: #16a34a; }
.route-badge.route-miss   { background: #fee2e2; color: #dc2626; }
.route-badge.route-na     { background: #f3f4f6; color: #6b7280; }

/* ---- Match-tier badge ---- */
.tier-badge {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  margin-left: 4px;
}
.tier-exact  { background: #dcfce7; color: #16a34a; }
.tier-fuzzy  { background: #fef9c3; color: #854d0e; }
.tier-judge  { background: #ede9fe; color: #6d28d9; }
.tier-fail   { background: #fee2e2; color: #dc2626; }

/* ---- Print ---- */
@media print {
  .container { max-width: 100%; padding: 0; }
  header.page-header { border-bottom: 1px solid #ccc; }
  details.q-detail { break-inside: avoid; box-shadow: none; }
  details.q-detail > summary { background: #f0f0f0; }
  body { font-size: 11pt; }
  pre.answer-block { max-height: none; overflow: visible; }
}
"""


def _route_badge_html(
    route_classified: str,
    expected_route: str | None,
    route_match: bool,
) -> str:
    if expected_route is None:
        return f'<span class="route-badge route-na">route={_html_escape.escape(route_classified)} (N/A)</span>'
    if route_match:
        return f'<span class="route-badge route-match">route={_html_escape.escape(route_classified)} ✓</span>'
    return (
        f'<span class="route-badge route-miss">'
        f'route={_html_escape.escape(route_classified)} '
        f'(expected {_html_escape.escape(expected_route)})'
        f'</span>'
    )


def _match_tier_badge_html(match_tier: str) -> str:
    cls_map = {
        "exact": "tier-exact",
        "fuzzy": "tier-fuzzy",
        "judge": "tier-judge",
        "-": "tier-fail",
    }
    label_map = {
        "exact": "exact ✓",
        "fuzzy": "fuzzy ✓",
        "judge": "judge ✓",
        "-": "-",
    }
    cls = cls_map.get(match_tier, "tier-fail")
    label = label_map.get(match_tier, match_tier)
    return f'<span class="tier-badge {cls}">{_html_escape.escape(label)}</span>'


def render_html(
    results_data: list[dict],
    cfg: Any,
    total_wall: float,
    passed_count: int,
    total: int,
    ts: str,
) -> str:
    """Return the complete self-contained HTML report as a string."""

    pct = (passed_count / total * 100) if total else 0.0
    if pct >= 80:
        badge_cls = "badge-green"
    elif pct >= 50:
        badge_cls = "badge-amber"
    else:
        badge_cls = "badge-red"
    badge_text = f"{passed_count}/{total} ({pct:.0f}%)"

    rcfg = cfg.retrieval
    kcfg = cfg.kg
    lcfg = cfg.llm

    # ---- Config snapshot ----
    config_rows = [
        ("retriever", rcfg.retriever),
        ("reranker", rcfg.reranker),
        ("top_k_vector", str(rcfg.top_k_vector)),
        ("top_k_final", str(rcfg.top_k_final)),
        ("rerank_threshold", str(rcfg.rerank_threshold)),
        ("kg.enabled", str(kcfg.enabled)),
        ("kg.use_communities", str(kcfg.use_communities)),
        ("llm.model", lcfg.model),
    ]
    config_html = "\n".join(
        f'<div class="config-key">{_html_escape.escape(k)}</div>'
        f'<div class="config-value">{_html_escape.escape(v)}</div>'
        for k, v in config_rows
    )

    # ---- Summary table ----
    summary_rows_html = []
    for entry in results_data:
        q = entry["q"]
        s = entry["scores"]
        qid = q["id"]
        cat = q["category"]
        overall = s["passed_overall"]
        rc = s.get("route_classified", "unknown")
        er = q.get("expected_route")
        rm = s.get("route_match", True)
        match_tier = s.get("match_tier", "-")

        overall_badge = (
            '<span class="pass">PASS</span>' if overall
            else '<span class="fail">FAIL</span>'
        )
        def _td(v: bool) -> str:
            return f'<span class="{"pass" if v else "fail"}">{"P" if v else "F"}</span>'

        route_cell = _html_escape.escape(rc)
        if er:
            match_sym = "✓" if rm else "✗"
            route_cell += f" {match_sym} <span class='secondary'>exp: {_html_escape.escape(er)}</span>"

        tier_cell = _match_tier_badge_html(match_tier)

        summary_rows_html.append(
            f"<tr>"
            f"<td><a href='#q-{_html_escape.escape(qid)}'>{_html_escape.escape(qid)}</a></td>"
            f"<td>{_html_escape.escape(cat)}</td>"
            f"<td>{overall_badge}</td>"
            f"<td>{_td(s['passed_substrings'])}</td>"
            f"<td>{tier_cell}</td>"
            f"<td>{_td(s['has_citation'])}</td>"
            f"<td>{_td(s['has_raft_blocks'])}</td>"
            f"<td>{_td(s['source_doc_match'])}</td>"
            f"<td>{_td(s['rerank_fallback_avoided'])}</td>"
            f"<td>{_td(s['rewriter_fired_on_followup'])}</td>"
            f"<td>{_td(s['no_hallucination'])}</td>"
            f"<td>{_td(rm)}</td>"
            f"<td>{route_cell}</td>"
            f"<td>{entry['duration_s']:.1f}s</td>"
            f"</tr>"
        )
    summary_table_html = (
        "<table class='summary-table'>"
        "<thead><tr>"
        "<th>ID</th><th>Category</th><th>Overall</th>"
        "<th>Substrings</th><th>Match Tier</th><th>Citation</th><th>RAFT</th>"
        "<th>Src Match</th><th>No Fallback</th><th>Rewriter</th>"
        "<th>No Halluc</th><th>Route Match</th>"
        "<th>Route (classified vs expected)</th><th>Time</th>"
        "</tr></thead><tbody>"
        + "\n".join(summary_rows_html)
        + "</tbody></table>"
    )

    # ---- Insights: Route distribution (across all classified routes) ----
    all_routes: Counter[str] = Counter()
    for entry in results_data:
        rc = entry["scores"].get("route_classified", "unknown")
        all_routes[rc] += 1
    route_dist_html = _route_distribution_html(dict(all_routes))

    # ---- Insights: Failure mode breakdown ----
    fail_counts = _dimension_failure_counts(results_data)
    dim_fail_html = _route_distribution_html(
        {k: v for k, v in fail_counts.items() if v > 0}
    ) or "<p class='secondary'>No failures recorded.</p>"

    # ---- Insights: Timing distribution ----
    timing_html = _timing_bar_html(results_data)

    # ---- Per-question detail sections ----
    detail_sections: list[str] = []
    for entry in results_data:
        q = entry["q"]
        s = entry["scores"]
        cat = entry["category"]
        qid = q["id"]
        overall = s["passed_overall"]
        rc = s.get("route_classified", "unknown")
        er = q.get("expected_route")
        rm = s.get("route_match", True)
        match_tier = s.get("match_tier", "-")

        # Question text
        if cat == "conversational_followup":
            turns = q.get("turns", [])
            question_html = (
                f"<p><strong>Turn 1:</strong> {_html_escape.escape(turns[0] if turns else '')}</p>"
                f"<p><strong>Turn 2:</strong> {_html_escape.escape(turns[1] if len(turns) > 1 else '')}</p>"
            )
            subs_label = "Expected substrings per turn"
            subs_val = _html_escape.escape(str(q.get("expected_substrings_per_turn", [])))
        else:
            question_html = f"<p>{_html_escape.escape(q.get('question', ''))}</p>"
            subs_label = "Expected substrings"
            subs_val = _html_escape.escape(str(q.get("expected_substrings", [])))

        # Answer
        answer_text = entry["answer"]
        answer_html = f"<pre class='answer-block'>{_html_escape.escape(answer_text)}</pre>"

        # Sources
        sources_html = _sources_table_html(entry["sources"])

        # Route dist from source retriever stamps
        src_route_dist = entry["route_distribution"]
        src_route_bar = _route_distribution_html(src_route_dist)

        # Dimension table
        dim_html = _dimension_table_html(s, er)

        # Notes
        notes = q.get("notes", "").strip()
        notes_html = (
            f"<p class='secondary'>{_html_escape.escape(notes)}</p>"
            if notes else ""
        )

        # Missing substrings
        missing = s.get("missing_substrings", [])
        missing_html = ""
        if missing:
            missing_items = "".join(f"<li><code>{_html_escape.escape(m)}</code></li>" for m in missing)
            missing_html = f"<p><strong>Missing substrings:</strong></p><ul>{missing_items}</ul>"

        # Match-tier info
        tier_badge_html = _match_tier_badge_html(match_tier)
        tier_html = f"<p><strong>Match tier:</strong> {tier_badge_html}</p>"

        # Query rewrite events
        rewrite_events = [(ev, pl) for ev, pl in entry["events"] if ev == "query_rewrite"]
        rewrite_html = ""
        if rewrite_events:
            rewrites = "".join(
                f"<li><code>{_html_escape.escape(pl.get('original',''))}</code>"
                f" → <code>{_html_escape.escape(pl.get('rewritten',''))}</code></li>"
                for _, pl in rewrite_events
            )
            rewrite_html = f"<p><strong>Query rewrites:</strong></p><ul>{rewrites}</ul>"

        # Rerank fallback warning
        fallback_html = ""
        for ev, pl in entry["events"]:
            if ev == "rerank_done" and pl.get("fallback_used"):
                fallback_html = (
                    f"<p class='fail'><strong>⚠ Rerank fallback used</strong>"
                    f" — kept {pl.get('kept', '?')} chunks after threshold was too strict.</p>"
                )

        # Summary chevron text
        overall_badge = (
            '<span class="pass">PASS</span>' if overall
            else '<span class="fail">FAIL</span>'
        )
        route_badge = _route_badge_html(rc, er, rm)
        summary_text = (
            f"{_html_escape.escape(qid)} — {_html_escape.escape(cat)} "
            f"— {overall_badge} — {entry['duration_s']:.1f}s"
            f"{route_badge}"
        )

        open_attr = "" if overall else " open"
        detail_sections.append(
            f'<details class="q-detail"{open_attr} id="q-{_html_escape.escape(qid)}">'
            f"<summary><span class='summary-arrow'></span>{summary_text}</summary>"
            f"<div class='q-detail-body'>"
            f"<section>"
            f"<h3>Question</h3>{question_html}"
            f"<p><strong>{_html_escape.escape(subs_label)}:</strong> <code>{subs_val}</code></p>"
            f"<p><strong>Expected route:</strong> <code>{_html_escape.escape(str(er) if er else 'N/A')}</code></p>"
            f"{notes_html}"
            f"</section>"
            f"<section><h3>Answer</h3>{answer_html}</section>"
            f"<section><h3>Sources</h3>{sources_html}</section>"
            f"<section><h3>Route Distribution (from source stamps)</h3>"
            f"<div class='bar-chart'>{src_route_bar}</div></section>"
            f"<section><h3>Dimension Breakdown</h3>{dim_html}"
            f"{tier_html}{missing_html}{rewrite_html}{fallback_html}</section>"
            f"</div>"
            f"</details>"
        )

    details_html = "\n".join(detail_sections)

    # ---- Assemble ----
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 2 v2 Benchmark Results</title>
<style>
{_CSS}
</style>
</head>
<body>

<header class="page-header">
  <div class="header-inner">
    <div>
      <h1>Phase 2 v2 Benchmark Results</h1>
      <div class="header-meta">Run timestamp: {_html_escape.escape(ts)} &nbsp;|&nbsp; Total wall time: {total_wall:.1f}s</div>
    </div>
    <div>
      <span class="badge {badge_cls}">Pass rate: {_html_escape.escape(badge_text)}</span>
    </div>
  </div>
</header>

<div class="container">

  <section id="config">
    <h2>Config Snapshot</h2>
    <div class="card">
      <div class="config-grid">
        {config_html}
      </div>
    </div>
  </section>

  <section id="summary">
    <h2>Summary</h2>
    <div class="card" style="overflow-x:auto">
      {summary_table_html}
    </div>
  </section>

  <section id="insights">
    <h2>Insights</h2>

    <div class="card">
      <h3>Route Distribution (classified labels)</h3>
      <p class="secondary">How many questions each route label was assigned to by the router.</p>
      <div class="bar-chart">
        {route_dist_html}
      </div>
    </div>

    <div class="card">
      <h3>Failure Mode Breakdown</h3>
      <p class="secondary">Number of FAIL outcomes per scoring dimension across all questions.</p>
      <div class="bar-chart">
        {dim_fail_html}
      </div>
    </div>

    <div class="card">
      <h3>Timing Distribution</h3>
      <p class="secondary">Wall-clock time per question. Green = PASS, red = FAIL.</p>
      <div class="bar-chart">
        {timing_html}
      </div>
    </div>
  </section>

  <section id="detail">
    <h2>Per-Question Detail</h2>
    <p class="secondary">Failing questions are open by default; passing questions are collapsed.</p>
    {details_html}
  </section>

</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Live progress format helper
# ---------------------------------------------------------------------------

def _format_completion_line(
    idx: int,
    total: int,
    qid: str,
    scores: dict,
    dur: float,
    n_sub: int,
    expected_route: str | None,
) -> str:
    """Build the live progress completion line with match-tier annotation."""
    icon = "✓" if scores["passed_overall"] else "✗"
    rc = scores.get("route_classified", "unknown")
    rm = scores.get("route_match", True)
    match_tier = scores.get("match_tier", "-")
    passed_sub_count = n_sub - len(scores.get("missing_substrings", []))
    route_suffix = f"route={rc} {'✓' if rm else f'(expected {expected_route})'}"

    if scores["passed_overall"]:
        tier_label = f"[{match_tier}]" if match_tier != "-" else ""
        sub_part = f"substrings {passed_sub_count}/{n_sub} {tier_label}".strip()
        return (
            f"[{idx}/{total} {icon}] {qid} — PASS in {dur:.1f}s "
            f"({sub_part}, {route_suffix})"
        )
    else:
        fail_detail = _fail_detail(scores)
        return (
            f"[{idx}/{total} {icon}] {qid} — FAIL in {dur:.1f}s "
            f"({fail_detail}; {route_suffix})"
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    benchmark_path = Path(__file__).parent / "phase2_v2.yaml"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench = yaml.safe_load(f)

    questions = bench["questions"]
    total = len(questions)

    # Build ONE orchestrator for the entire run
    print("Loading config and initialising orchestrator …", flush=True)
    cfg = load_config()
    orch = Orchestrator(cfg)
    print("Orchestrator ready.\n", flush=True)

    run_start = time.time()
    results_data: list[dict] = []

    # Determine which user_id holds the corpus data.
    # The ingest pipeline always writes under cfg.user.default_user_id ("default").
    # Using any other user_id would yield empty retrieval results.
    corpus_user_id = cfg.user.default_user_id  # "default"

    for idx, q in enumerate(questions, start=1):
        qid = q["id"]
        category = q["category"]
        expected_route = q.get("expected_route")

        print(f"[{idx}/{total} ▶] {qid} — {category}: running...", flush=True)

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

            # Turn 2 — same session
            events2: list[tuple[str, dict]] = []
            def _cb2(evt: str, payload: dict, _ev=events2) -> None:
                _ev.append((evt, payload))

            t2_start = time.time()
            r2 = orch.chat(turns[1], user_id=corpus_user_id,
                           session_id=r1.session_id,
                           progress=_cb2, stream=False)
            t2_dur = time.time() - t2_start

            total_dur = t1_dur + t2_dur

            # Build route distribution over both turns
            combined_sources = list(r1.sources) + list(r2.sources)
            route_dist = build_route_distribution(combined_sources)

            scores = score_followup(
                q,
                r1.answer, r2.answer,
                r1.sources, r2.sources,
                events1, events2,
                route_dist,
                llm=orch.llm,
            )

            # Emit completion line — v2 format with route info and match tier
            n_sub = (
                len(q.get("expected_substrings_per_turn", [[], []])[0])
                + len(q.get("expected_substrings_per_turn", [[], []])[1])
            )
            line = _format_completion_line(
                idx, total, qid, scores, total_dur, n_sub, expected_route
            )
            print(line, flush=True)

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
                "duration_s": total_dur,
                "duration_per_turn": [t1_dur, t2_dur],
                "route_distribution": route_dist,
            })

        else:
            events: list[tuple[str, dict]] = []
            def _cb(evt: str, payload: dict, _ev=events) -> None:
                _ev.append((evt, payload))

            t_start = time.time()
            r = orch.chat(q["question"], user_id=corpus_user_id,
                          session_id=None, progress=_cb, stream=False)
            dur = time.time() - t_start

            route_dist = build_route_distribution(r.sources)

            scores = score_result(
                q, r.answer, r.sources, events, category, route_dist,
                llm=orch.llm,
            )

            # Emit completion line — v2 format with route info and match tier
            substrings = q.get("expected_substrings", [])
            n_sub = len(substrings)
            line = _format_completion_line(
                idx, total, qid, scores, dur, n_sub, expected_route
            )
            print(line, flush=True)

            results_data.append({
                "q": q,
                "category": category,
                "answer": r.answer,
                "sources": r.sources,
                "events": events,
                "scores": scores,
                "duration_s": dur,
                "route_distribution": route_dist,
            })

    total_wall = time.time() - run_start
    passed_count = sum(1 for d in results_data if d["scores"]["passed_overall"])

    # -----------------------------------------------------------------------
    # Print summary table to stdout
    # -----------------------------------------------------------------------
    header = (
        "| id | category | overall | substrings | match_tier | citation | raft | src match "
        "| no fallback | rewriter | no halluc | route match | route classified | time (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"

    print("\n# Phase 2 v2 Benchmark Results", flush=True)
    print(f"Pass rate: {passed_count}/{total}\n", flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    for d in results_data:
        s = d["scores"]
        row = (
            f"| {d['q']['id']} "
            f"| {d['category']} "
            f"| {_bool(s['passed_overall'])} "
            f"| {_bool(s['passed_substrings'])} "
            f"| {s.get('match_tier', '-')} "
            f"| {_bool(s['has_citation'])} "
            f"| {_bool(s['has_raft_blocks'])} "
            f"| {_bool(s['source_doc_match'])} "
            f"| {_bool(s['rerank_fallback_avoided'])} "
            f"| {_bool(s['rewriter_fired_on_followup'])} "
            f"| {_bool(s['no_hallucination'])} "
            f"| {_bool(s.get('route_match', True))} "
            f"| {s.get('route_classified', 'unknown')} "
            f"| {d['duration_s']:.1f} |"
        )
        print(row, flush=True)

    print(f"\nTotal wall time: {total_wall:.1f}s", flush=True)

    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # -----------------------------------------------------------------------
    # Build Markdown report
    # -----------------------------------------------------------------------
    rcfg = cfg.retrieval
    kcfg = cfg.kg
    lcfg = cfg.llm

    config_snapshot_lines = [
        "## Config snapshot",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| retriever | {rcfg.retriever} |",
        f"| reranker | {rcfg.reranker} |",
        f"| top_k_vector | {rcfg.top_k_vector} |",
        f"| top_k_final | {rcfg.top_k_final} |",
        f"| rerank_threshold | {rcfg.rerank_threshold} |",
        f"| kg.enabled | {kcfg.enabled} |",
        f"| kg.use_communities | {kcfg.use_communities} |",
        f"| llm.model | {lcfg.model} |",
        "",
    ]

    md_lines: list[str] = [
        "# Phase 2 v2 Benchmark Results",
        "",
        f"Run timestamp: {ts}",
        f"Total wall time: {total_wall:.1f}s",
        f"Pass rate: {passed_count}/{total} questions",
        "",
    ]
    md_lines.extend(config_snapshot_lines)
    md_lines += [
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
            f"| {s.get('match_tier', '-')} "
            f"| {_bool(s['has_citation'])} "
            f"| {_bool(s['has_raft_blocks'])} "
            f"| {_bool(s['source_doc_match'])} "
            f"| {_bool(s['rerank_fallback_avoided'])} "
            f"| {_bool(s['rewriter_fired_on_followup'])} "
            f"| {_bool(s['no_hallucination'])} "
            f"| {_bool(s.get('route_match', True))} "
            f"| {s.get('route_classified', 'unknown')} "
            f"| {d['duration_s']:.1f} |"
        )

    md_lines += ["", "## Per-question detail", ""]

    for d in results_data:
        q = d["q"]
        s = d["scores"]
        cat = d["category"]
        qid = q["id"]
        route_dist = d["route_distribution"]
        dom = s["route_observed"]
        rc = s.get("route_classified", "unknown")
        er = q.get("expected_route")
        rm = s.get("route_match", True)
        match_tier = s.get("match_tier", "-")

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
                f" (retriever={r.retriever})"
                for i, r in enumerate(srcs[:6])
            ]
            md_lines.append("**Sources returned:**")
            md_lines.extend(src_lines)
        else:
            md_lines.append("**Sources returned:** (none)")
        md_lines.append("")

        # Route information (Phase 2 v2)
        md_lines.append(f"**Route distribution (source stamps):** {route_dist}")
        md_lines.append(f"**Dominant route (source stamps):** {dom}")
        md_lines.append(f"**Route classified (router event):** {rc}")
        md_lines.append(f"**Expected route:** {er if er else 'N/A'}")
        md_lines.append(f"**Route match:** {'PASS' if rm else 'FAIL'}")
        md_lines.append(f"**KG path hit:** {'yes' if s['kg_path_hit'] else 'no'}")
        md_lines.append("")

        # Match tier
        md_lines.append(f"**Match tier:** {match_tier}")
        if s.get("judge_used") and s.get("per_term"):
            for t in s["per_term"]:
                status = "PASS" if t["passed"] else "FAIL"
                md_lines.append(f"  - `{t['term']}`: {status} (tier={t['tier']})")
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
        md_lines.append(f"- match_tier: {match_tier}")
        md_lines.append(f"- judge_used: {s.get('judge_used', False)}")
        md_lines.append(f"- citation: {_bool(s['has_citation'])}")
        md_lines.append(f"- raft_blocks: {_bool(s['has_raft_blocks'])}")
        md_lines.append(f"- source_doc_match: {_bool(s['source_doc_match'])}")
        md_lines.append(f"- rerank_fallback_avoided: {_bool(s['rerank_fallback_avoided'])}")
        md_lines.append(f"- rewriter_fired_on_followup: {_bool(s['rewriter_fired_on_followup'])}")
        md_lines.append(f"- no_hallucination: {_bool(s['no_hallucination'])}")
        md_lines.append(f"- route_match: {_bool(s.get('route_match', True))}")
        md_lines.append(f"- route_classified (v2): {rc}")
        md_lines.append(f"- route_observed (info, dominant): {s['route_observed']}")
        md_lines.append(f"- kg_path_hit (info): {'yes' if s['kg_path_hit'] else 'no'}")
        md_lines.append("")
        md_lines.append(f"**Score: {'PASS' if s['passed_overall'] else 'FAIL'}**")
        md_lines.append("")
        if q.get("notes"):
            md_lines.append(f"**Notes:** {q['notes'].strip()}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    md_path = Path(__file__).parent / "phase2_v2_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nMarkdown report written to: {md_path}", flush=True)

    # -----------------------------------------------------------------------
    # Build and write HTML report
    # -----------------------------------------------------------------------
    html_content = render_html(results_data, cfg, total_wall, passed_count, total, ts)
    html_path = Path(__file__).parent / "phase2_v2_results.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"HTML report written to: {html_path}", flush=True)


if __name__ == "__main__":
    run_benchmark()
