"""Rerun only Phase 4 Q3 (dialog MST) — used to confirm the orchestrator
history-limit fix without sitting through 17 minutes of full benchmark."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
    sys.path.insert(0, str(_repo_root / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

import importlib.util

from hrag.config import load_config
from hrag.orchestrator import Orchestrator

_spec = importlib.util.spec_from_file_location(
    "_phase4_bench", str(Path(__file__).parent / "run_phase4.py")
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
q3_dialog_mst_preserves_old_fact = _mod.q3_dialog_mst_preserves_old_fact


def main() -> int:
    cfg = load_config()
    uid = cfg.user.default_user_id

    cfg_q3 = cfg.model_copy(deep=True)
    cfg_q3.compaction.dialog_mst_enabled = True
    cfg_q3.compaction.compact_after_turns = 12
    cfg_q3.compaction.keep_recent_turns = 6

    print("Q3 (dialog MST) standalone rerun", flush=True)
    t0 = time.time()
    ok, msg = q3_dialog_mst_preserves_old_fact(Orchestrator(cfg_q3), uid)
    dur = time.time() - t0
    print(f"\nResult: {'PASS' if ok else 'FAIL'} ({dur:.1f}s)", flush=True)
    print(f"Message: {msg}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
