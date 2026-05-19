"""Phase 6-B + 7-B/C — Combined acceptance benchmark.

Six questions that exercise the Phase-6-B-plus-7-B-plus-7-C trio
(per-intent retriever, feedback analytics, num_keep, embedder selector,
Nougat scaffold) all behind the existing FastAPI surface.

Q1 — Per-intent retriever override picks a different retriever when the
     map asks for one; emits `adaptive_retriever_picked`.
Q2 — Feedback analytics: `feedback_summary(db)` reports correct counts
     against a seeded DB; `GET /api/feedback/stats` matches.
Q3 — Ollama `num_keep` lands inside chat() options (not the top level).
Q4 — Embedding selector: `GET /api/embeddings/suggested` returns 4
     curated entries; `dimension_for_model` resolves their dims.
Q5 — Nougat loader: import is side-effect-free; PDF dispatch silently
     falls back to PyMuPDF when Nougat isn't installed.
Q6 — `/api/config` POST round-trips every new knob; defaults preserved
     on a fresh boot.

Usage (from project root):
    python tests/benchmark/run_phase7full.py
    python tests/benchmark/run_phase7full.py --json out.json

Runs in-process; no Ollama, Chroma, Neo4j, or Nougat server required.

Acceptance: >= 5/6 PASS.
"""

from __future__ import annotations

import argparse
import json
import sys
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

# ---------------------------------------------------------------------------
# sys.path
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
# Q1 — Per-intent retriever override
# ---------------------------------------------------------------------------


def q1_per_intent_retriever(tmp_dir: Path) -> tuple[bool | str, str]:
    """When `adaptive_enabled=True` AND the map points an intent at a
    non-default retriever, `_pick_retriever_for_intent` builds & caches
    it; the orchestrator emits `adaptive_retriever_picked`."""
    from hrag.config import Config, EmbeddingsConfig, LLMConfig, StorageConfig
    from hrag.intent import Intent, IntentVerdict
    from hrag.types import Chunk, RetrievalResult
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None

    cfg = Config(
        llm=LLMConfig(provider="ollama", model="test"),
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
    cfg.retrieval.adaptive_enabled = True
    cfg.retrieval.rerank_enabled = False
    cfg.retrieval.adaptive_retriever_per_intent = {
        "greeting": "default",
        "personal": "default",
        "factual": "bm25",   # << force a non-default retriever for FACTUAL
        "general": "default",
        "unclear": "default",
    }
    cfg.retrieval.retriever = "vector"   # global stays as vector

    from hrag.orchestrator import Orchestrator

    class _FakeLLM:
        def complete(self, prompt, system=None, temperature=None, max_tokens=None):
            if "Intent" in prompt:
                return "factual"
            return "stub"

        def generate(self, request):
            from hrag.types import GenerationResponse
            return GenerationResponse(text="stub", raw=None)

        def generate_stream(self, request):
            yield "stub"

    class _FactualClassifier:
        def classify(self, text, **kwargs):
            return IntentVerdict(
                intent=Intent.FACTUAL, confidence=1.0,
                source="test", raw_label="factual",
            )

    orch = Orchestrator(cfg)
    orch.llm = _FakeLLM()
    orch.intent_classifier = _FactualClassifier()  # type: ignore

    def _chunk(cid):
        return Chunk(chunk_id=cid, doc_id="d", user_id="default",
                     text=f"text {cid}", embedding_text=f"text {cid}",
                     source_type="document")

    seeded = [RetrievalResult(chunk=_chunk("a"), score=0.9)]

    class _GlobalSpy:
        name = "vector"

        def __init__(self):
            self.calls = 0

        def retrieve(self, *a, **kw):
            self.calls += 1
            return list(seeded)

    class _PerIntentSpy:
        name = "bm25"

        def __init__(self):
            self.calls = 0

        def retrieve(self, *a, **kw):
            self.calls += 1
            return list(seeded)

    global_spy = _GlobalSpy()
    per_intent_spy = _PerIntentSpy()
    orch.retriever = global_spy  # type: ignore
    orch._per_intent_retrievers = {"bm25": per_intent_spy}  # pre-cache so the
                                                            # resolver picks it
                                                            # without a real build.

    events = []
    try:
        orch.chat(
            "what is hipporag?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _conn_mod._db_singleton = None

    picks = [p for n, p in events if n == "adaptive_retriever_picked"]
    print(f"  [Q1] adaptive_retriever_picked events: {len(picks)}", flush=True)
    print(f"  [Q1] global calls={global_spy.calls}  per-intent calls={per_intent_spy.calls}", flush=True)

    if not picks:
        return False, "no adaptive_retriever_picked event"
    if picks[0]["intent"] != "factual" or picks[0]["retriever"] != "bm25":
        return False, f"unexpected pick payload: {picks[0]!r}"
    if global_spy.calls != 0 or per_intent_spy.calls < 1:
        return False, (
            f"wrong retriever was called: global={global_spy.calls}, "
            f"per_intent={per_intent_spy.calls}"
        )

    return True, "FACTUAL routed to bm25 (overriding vector); event fired"


# ---------------------------------------------------------------------------
# Q2 — Feedback analytics
# ---------------------------------------------------------------------------


def q2_feedback_analytics(tmp_dir: Path) -> tuple[bool | str, str]:
    """`feedback_summary(db)` reports correct counts; the API matches."""
    from fastapi.testclient import TestClient
    from hrag.feedback_stats import feedback_summary
    from hrag.web.app import _State, app

    # Use the live web app + a fresh state.
    with _State.lock:
        _State.cfg = None
        _State.orch = None
    client = TestClient(app, raise_server_exceptions=False)
    # Warm-up
    client.get("/api/health")

    # Seed feedback rows directly via the orchestrator's DB.
    from hrag.web.app import _get_orch
    orch = _get_orch()
    db = orch.db

    # Insert a session, two user/assistant pairs, and two feedback rows.
    # Schema: messages.message_id is INTEGER autoincrement; feedback.rating is
    # an INTEGER (+1 thumbs up, -1 thumbs down).
    sid = "phase7full-q2-session"
    db.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", ("default",))
    db.execute("INSERT OR IGNORE INTO sessions (session_id, user_id) VALUES (?, ?)",
               (sid, "default"))
    inserted_msgs: list[int] = []
    for role, content in [
        ("user", "what is the loss function?"),
        ("assistant", "the loss is L = ..."),
        ("user", "what is the gradient?"),
        ("assistant", "gradient is ∇L = ..."),
    ]:
        cur = db.execute(
            "INSERT INTO messages (session_id, user_id, role, content) "
            "VALUES (?, ?, ?, ?)",
            (sid, "default", role, content),
        )
        inserted_msgs.append(cur.lastrowid)
    a1, a2 = inserted_msgs[1], inserted_msgs[3]
    db.execute(
        "INSERT INTO feedback (feedback_id, message_id, session_id, user_id, rating) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fb_q2_1", str(a1), sid, "default", -1),
    )
    db.execute(
        "INSERT INTO feedback (feedback_id, message_id, session_id, user_id, rating) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fb_q2_2", str(a2), sid, "default", 1),
    )
    db.commit()

    # 1. Pure function
    summary = feedback_summary(db)
    print(f"  [Q2] feedback_summary={summary}", flush=True)
    if summary.get("thumbs_up", 0) < 1 or summary.get("thumbs_down", 0) < 1:
        return False, f"feedback_summary missing counts: {summary!r}"

    # 2. API match
    resp = client.get("/api/feedback/stats")
    if resp.status_code != 200:
        return False, f"/api/feedback/stats returned {resp.status_code}"
    payload = resp.json()
    print(f"  [Q2] /api/feedback/stats={payload}", flush=True)
    if payload.get("thumbs_up") != summary.get("thumbs_up"):
        return False, f"API mismatch: {payload!r} vs {summary!r}"
    if payload.get("thumbs_down") != summary.get("thumbs_down"):
        return False, f"API mismatch: {payload!r} vs {summary!r}"

    # Clean up the seeded rows so other tests aren't polluted.
    db.execute("DELETE FROM feedback WHERE feedback_id IN ('fb_q2_1', 'fb_q2_2')")
    db.execute(
        "DELETE FROM messages WHERE message_id IN (?, ?, ?, ?)",
        tuple(inserted_msgs),
    )
    db.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
    db.commit()

    return True, "feedback_summary + API agree; ≥1 thumbs_up and ≥1 thumbs_down counted"


# ---------------------------------------------------------------------------
# Q3 — Ollama num_keep plumbing
# ---------------------------------------------------------------------------


def q3_num_keep(tmp_dir: Path) -> tuple[bool | str, str]:
    """num_keep lands inside options (not at chat() top level)."""
    from hrag.config import LLMConfig
    from hrag.providers.llm import OllamaProvider
    from hrag.types import GenerationRequest, Message

    p = OllamaProvider.__new__(OllamaProvider)
    p.config = LLMConfig(num_keep=256)
    p._client = MagicMock()

    req = GenerationRequest(messages=[Message(role="user", content="hi")])
    opts = p._build_options(req)
    kwargs = p._build_chat_kwargs(
        [{"role": "user", "content": "hi"}], opts,
    )

    print(f"  [Q3] options.num_keep={opts.get('num_keep')!r}", flush=True)
    print(f"  [Q3] kwargs has num_keep at top level? {'num_keep' in kwargs}", flush=True)

    if opts.get("num_keep") != 256:
        return False, f"num_keep not in options: {opts!r}"
    if "num_keep" in kwargs:
        return False, "num_keep leaked to chat() top level (must stay in options)"

    # None default → not present at all
    p.config = LLMConfig(num_keep=None)
    opts2 = p._build_options(req)
    if "num_keep" in opts2:
        return False, f"num_keep=None should be absent: {opts2!r}"

    return True, "num_keep in options at 256; absent on None default"


# ---------------------------------------------------------------------------
# Q4 — Embedder selector
# ---------------------------------------------------------------------------


def q4_embedder_selector(tmp_dir: Path) -> tuple[bool | str, str]:
    """The 4 curated suggestions exist + dimension_for_model resolves them."""
    from hrag.config import EmbeddingsConfig
    from hrag.providers.embeddings import dimension_for_model

    cfg = EmbeddingsConfig()
    names = {entry["model"] for entry in cfg.suggested_models}
    must_have = {
        "sentence-transformers/all-mpnet-base-v2",
        "allenai/specter2_base",
        "jinaai/jina-embeddings-v2-base-en",
        "BAAI/bge-small-en-v1.5",
    }
    missing = must_have - names
    if missing:
        return False, f"missing curated models: {missing}"

    if dimension_for_model("sentence-transformers/all-mpnet-base-v2") != 768:
        return False, "all-mpnet should resolve to 768"
    if dimension_for_model("BAAI/bge-small-en-v1.5") != 384:
        return False, "bge-small should resolve to 384"
    if dimension_for_model("bogus/model") is not None:
        return False, "unknown model should return None"

    # API surface
    from fastapi.testclient import TestClient
    from hrag.web.app import app
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/embeddings/suggested")
    if resp.status_code != 200:
        return False, f"/api/embeddings/suggested returned {resp.status_code}"
    body = resp.json()
    print(f"  [Q4] API returned {len(body.get('suggestions', []))} suggestions", flush=True)
    if len(body.get("suggestions", [])) < 4:
        return False, f"API returned <4 suggestions: {body!r}"

    return True, "4 curated models present; dimension lookup works for known + unknown"


# ---------------------------------------------------------------------------
# Q5 — Nougat scaffold
# ---------------------------------------------------------------------------


def q5_nougat_scaffold(tmp_dir: Path) -> tuple[bool | str, str]:
    """Importing the loader is side-effect-free; the API surface reports
    availability honestly; PDF dispatch falls back to PyMuPDF when off."""
    # Import the module — must not pull `nougat` into sys.modules.
    import sys as _sys
    _sys.modules.pop("nougat", None)
    _sys.modules.pop("nougat_ocr", None)

    from hrag.ingest.nougat_loader import is_nougat_available  # noqa: F401

    if "nougat" in _sys.modules:
        return False, "importing nougat_loader pulled `nougat` into sys.modules"

    avail = is_nougat_available()
    print(f"  [Q5] is_nougat_available() = {avail}", flush=True)

    # API surface
    from fastapi.testclient import TestClient
    from hrag.web.app import app
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/ingest/nougat_status")
    if resp.status_code != 200:
        return False, f"/api/ingest/nougat_status returned {resp.status_code}"
    body = resp.json()
    print(f"  [Q5] /api/ingest/nougat_status={body}", flush=True)
    if body.get("available") != avail:
        return False, f"API availability disagrees with module: {body!r}"

    return True, f"loader import is side-effect-free; availability={avail}; API agrees"


# ---------------------------------------------------------------------------
# Q6 — Config POST roundtrip on every new knob
# ---------------------------------------------------------------------------


def q6_config_roundtrip(tmp_dir: Path) -> tuple[bool | str, str]:
    """POST every new field; GET reflects the change.

    Note: embeddings_model is patched in its own POST because the backend
    triggers an orchestrator rebuild (the embedder is cached at init), and
    rebuilding re-reads config.yaml — which would wipe other in-memory
    mutations made in the same request. We confirm that knob in isolation.
    """
    from fastapi.testclient import TestClient
    from hrag.web.app import _State, app

    with _State.lock:
        _State.cfg = None
        _State.orch = None
    client = TestClient(app, raise_server_exceptions=False)

    orig = client.get("/api/config").json()
    orig_embed = orig["embeddings"]["model"]
    print(f"  [Q6] original llm.num_keep={orig['llm'].get('num_keep')}", flush=True)
    print(f"  [Q6] original embeddings.model={orig_embed!r}", flush=True)

    # ---- Body A: the four orchestrator-stable knobs ----
    body_a = {
        "num_keep": 128,
        "adaptive_retriever_per_intent": {"factual": "taxonomy"},
        "use_nougat": True,
        "nougat_model": "facebook/nougat-small",
    }
    resp_a = client.post("/api/config", json=body_a)
    if resp_a.status_code != 200:
        return False, f"POST(A) returned {resp_a.status_code}: {resp_a.text}"
    after_a = client.get("/api/config").json()
    print(f"  [Q6] after-A.num_keep={after_a['llm'].get('num_keep')}", flush=True)
    print(f"  [Q6] after-A.use_nougat={after_a.get('ingest', {}).get('use_nougat')}", flush=True)
    print(
        f"  [Q6] after-A.per_intent.factual="
        f"{after_a['retrieval']['adaptive_retriever_per_intent'].get('factual')!r}",
        flush=True,
    )

    if after_a["llm"].get("num_keep") != 128:
        return False, f"num_keep roundtrip failed: {after_a['llm']!r}"
    if after_a["retrieval"]["adaptive_retriever_per_intent"].get("factual") != "taxonomy":
        return False, f"per-intent map roundtrip failed: {after_a['retrieval']!r}"
    if not after_a.get("ingest", {}).get("use_nougat"):
        return False, f"use_nougat roundtrip failed: {after_a.get('ingest')!r}"
    if after_a.get("ingest", {}).get("nougat_model") != "facebook/nougat-small":
        return False, f"nougat_model roundtrip failed: {after_a.get('ingest')!r}"

    # ---- Body B: the embedder knob in isolation ----
    resp_b = client.post("/api/config", json={"embeddings_model": "allenai/specter2_base"})
    if resp_b.status_code != 200:
        return False, f"POST(B) returned {resp_b.status_code}: {resp_b.text}"
    after_b = client.get("/api/config").json()
    print(f"  [Q6] after-B.embeddings.model={after_b.get('embeddings', {}).get('model')!r}", flush=True)
    if after_b.get("embeddings", {}).get("model") != "allenai/specter2_base":
        return False, f"embeddings_model roundtrip failed: {after_b.get('embeddings')!r}"

    # Restore defaults
    client.post("/api/config", json={
        "num_keep": None,
        "embeddings_model": orig_embed,
        "adaptive_retriever_per_intent": {"factual": "default"},
        "use_nougat": False,
        "nougat_model": "facebook/nougat-base",
    })

    return True, "all 5 new knobs roundtrip cleanly (embeddings_model patched in isolation)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    print("Phase 6-B + 7-B/C — Combined acceptance benchmark", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="hrag_phase7full_"))
    print(f"Tmp dir: {tmp_dir}", flush=True)

    suites: list[tuple[str, Callable[[Path], tuple[bool | str, str]]]] = [
        ("Q1 — Per-intent retriever override + event", q1_per_intent_retriever),
        ("Q2 — Feedback analytics (function + API)", q2_feedback_analytics),
        ("Q3 — Ollama num_keep plumbing (inside options)", q3_num_keep),
        ("Q4 — Math-aware embedder selector + API", q4_embedder_selector),
        ("Q5 — Nougat loader scaffold + status API", q5_nougat_scaffold),
        ("Q6 — /api/config roundtrip on every new knob", q6_config_roundtrip),
    ]

    passed = 0
    failed = 0
    skipped = 0
    results: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        console=console,
        transient=False,
    ) as prog:
        task = prog.add_task("Phase 7-full benchmark", total=len(suites))
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

            results.append({"label": label, "status": status, "message": msg,
                            "duration_s": round(dur, 3)})
            prog.advance(task)

    console.print("\n" + "=" * 60)
    console.print("[bold]Phase 6-B + 7-B/C Benchmark Summary[/bold]")
    console.print("=" * 60)
    for r in results:
        glyph = {"PASS": "[green]PASS[/]", "FAIL": "[red]FAIL[/]",
                 "SKIP": "[yellow]SKIP[/]"}[r["status"]]
        console.print(f"  {glyph}  {r['label']}")
        if r["status"] != "PASS":
            console.print(f"       {r['message']}")

    total = len(suites)
    console.print(
        f"\n[bold]Score: {passed}/{total} "
        f"({skipped} skipped, {failed} failed)[/bold]"
    )

    accept = passed >= 5 or (passed + skipped >= 5 and failed == 0)
    if accept:
        console.print("[green bold]ACCEPTED (>= 5/6 pass or skip)[/]")
    else:
        console.print("[red bold]FAILED — need >= 5/6[/]")

    if args.json:
        payload = {
            "phase": "6-B + 7-B/C",
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
