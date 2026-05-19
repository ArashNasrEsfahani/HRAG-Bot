"""Phase 4 speed deltas — measures the cost of each compaction.* flag.

For each pair (flag OFF vs flag ON), runs the same question through a fresh
orchestrator session and reports wall-clock seconds. The acceptance benchmark
(`run_phase4.py`) measures correctness only; this script answers "how much
does each feature cost?" so we can pick safe defaults.

Usage (from project root):
    python tests/benchmark/run_phase4_speed.py

Outputs per-feature pairs: gate (cost on factual / saving on small-talk),
clue (cost on vague query), mask_uncertain (pure regex; should be ~0).
Dialog MST is omitted here — it amortises over many turns and the main
benchmark already measures its 20-turn total.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

from hrag.config import load_config
from hrag.orchestrator import Orchestrator


def _time_turn(orch: Orchestrator, user_id: str, q: str, sid: str | None = None) -> tuple[float, str]:
    t0 = time.time()
    r = orch.chat(q, user_id=user_id, session_id=sid)
    return time.time() - t0, r.session_id


def main() -> int:
    cfg = load_config()
    uid = cfg.user.default_user_id

    factual_q = "What is overfitting in machine learning?"
    vague_q = "tell me about that paper on managing retrieval for dialogue systems"
    smalltalk_q = "thanks!"

    print("Phase 4 speed deltas")
    print("=" * 50, flush=True)

    # --- Gate on factual: should add one cheap LLM call ---
    cfg_off = cfg.model_copy(deep=True)
    cfg_gate = cfg.model_copy(deep=True)
    cfg_gate.compaction.gate_enabled = True

    print("\n[Gate] Factual query (cost of one extra gate LLM call)", flush=True)
    t_off_fact, _ = _time_turn(Orchestrator(cfg_off), uid, factual_q)
    print(f"  flags OFF: {t_off_fact:.2f}s", flush=True)
    t_on_fact, _ = _time_turn(Orchestrator(cfg_gate), uid, factual_q)
    print(f"  gate ON  : {t_on_fact:.2f}s  (delta {t_on_fact - t_off_fact:+.2f}s)", flush=True)

    # --- Gate on small-talk: should SAVE retrieval cost ---
    print("\n[Gate] Small-talk after factual turn (saves retrieve+gen)", flush=True)
    orch_off = Orchestrator(cfg_off)
    _, sid_off = _time_turn(orch_off, uid, factual_q)
    t_off_st, _ = _time_turn(orch_off, uid, smalltalk_q, sid=sid_off)
    print(f"  flags OFF: {t_off_st:.2f}s  (full retrieve+gen pipeline)", flush=True)

    orch_g = Orchestrator(cfg_gate)
    _, sid_g = _time_turn(orch_g, uid, factual_q)
    t_on_st, _ = _time_turn(orch_g, uid, smalltalk_q, sid=sid_g)
    saving = t_off_st - t_on_st
    print(
        f"  gate ON  : {t_on_st:.2f}s  (delta {t_on_st - t_off_st:+.2f}s — "
        f"{'saved' if saving > 0 else 'cost'} {abs(saving):.2f}s)",
        flush=True,
    )

    # --- Clue on vague query: should add one LLM call ---
    cfg_clue = cfg.model_copy(deep=True)
    cfg_clue.compaction.clue_enabled = True

    print("\n[Clue] Vague query (cost of one extra clue LLM call)", flush=True)
    t_off_vague, _ = _time_turn(Orchestrator(cfg_off), uid, vague_q)
    print(f"  flags OFF: {t_off_vague:.2f}s", flush=True)
    t_on_vague, _ = _time_turn(Orchestrator(cfg_clue), uid, vague_q)
    print(
        f"  clue ON  : {t_on_vague:.2f}s  (delta {t_on_vague - t_off_vague:+.2f}s)",
        flush=True,
    )

    # --- Mask uncertain: pure regex; should be ~0 ---
    cfg_mask = cfg.model_copy(deep=True)
    cfg_mask.compaction.mask_uncertain = True

    print("\n[Mask] Factual query (pure regex post-process; should be ~0)", flush=True)
    t_off_mask, _ = _time_turn(Orchestrator(cfg_off), uid, factual_q)
    print(f"  flags OFF: {t_off_mask:.2f}s", flush=True)
    t_on_mask, _ = _time_turn(Orchestrator(cfg_mask), uid, factual_q)
    print(
        f"  mask ON  : {t_on_mask:.2f}s  (delta {t_on_mask - t_off_mask:+.2f}s)",
        flush=True,
    )

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
