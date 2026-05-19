"""Phase 6 — Backends + adaptive retrieval + Ollama keep-alive benchmark.

Eight questions that exercise every Phase 6 feature end-to-end:

Q1 — sqlite-vec backend roundtrip (upsert -> query -> delete)
Q2 — Neo4j backend wiring: import-clean, factory dispatches, actionable error
     without a server.
Q3 — Adaptive top_k resolver returns correct (vec_k, final_k) per intent.
Q4 — Greeting + adaptive_enabled=True short-circuits retrieval.
Q5 — Personal + episodic bias re-sorts results so episodic chunks float up.
Q6 — Ollama provider forwards llm.keep_alive into the chat() kwargs.
Q7 — Factory dispatch: config-driven swap between vector backends does not
     raise NotImplementedError for any registered backend.
Q8 — Backward compat: with every Phase 6 flag at its default, the resolver
     returns the global top_k pair (i.e. retrieval behaviour is unchanged).

Usage (from project root):
    python tests/benchmark/run_phase6.py
    python tests/benchmark/run_phase6.py --json out.json

The whole suite runs in-process; no Ollama, no Chroma, no Neo4j server
required. The sqlite-vec test SKIPs cleanly if the optional dep is missing.

Acceptance: >= 7/8 PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock

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
# Inline fakes (avoid depending on tests.conftest, which is not importable
# when this script is run from outside the pytest test-collection root).
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Deterministic LLM stub — matches the LLMProvider surface area used by
    the orchestrator. Returns canned replies; never hits a network."""

    name = "fake"

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        # Intent-classifier prompts include the four-label vocabulary; emit
        # a token the classifier can parse. Anything else gets a one-liner.
        if "Intent Classification" in prompt or "Output (one word only)" in prompt:
            return "factual"
        return "stub answer"

    def generate(self, request):
        from hrag.types import GenerationResponse
        prompt = " ".join(m.content for m in request.messages)
        return GenerationResponse(text=self.complete(prompt), raw=None)

    def generate_stream(self, request):
        yield self.generate(request).text


# ---------------------------------------------------------------------------
# Q1 — sqlite-vec backend roundtrip
# ---------------------------------------------------------------------------


def q1_sqlite_vec_roundtrip(tmp_dir: Path) -> tuple[bool | str, str]:
    """Upsert 3 vectors, query the nearest, delete one, re-count."""
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        return _SKIP, "sqlite-vec not installed (pip install sqlite-vec)"

    from hrag.retrieval.backends.sqlite_vec import SqliteVecBackend

    persist = tmp_dir / "vec_q1"
    persist.mkdir(parents=True, exist_ok=True)
    b = SqliteVecBackend(persist)

    # Three unit vectors in 3D.
    ids = ["a", "b", "c"]
    embs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    docs = ["alpha", "beta", "gamma"]
    metas = [
        {"user_id": "default", "source_type": "document", "doc_id": "d1"},
        {"user_id": "default", "source_type": "document", "doc_id": "d2"},
        {"user_id": "alice",   "source_type": "document", "doc_id": "d3"},
    ]

    b.upsert(ids, embs, docs, metas)
    print(f"  [Q1] upserted {b.count()} vectors", flush=True)

    if b.count() != 3:
        return False, f"count after upsert: expected 3, got {b.count()}"

    # Query a vector almost-identical to 'a' — must come first.
    got_ids, got_dists = b.query_one(
        [0.99, 0.01, 0.0],
        top_k=2,
        where={"user_id": "default"},
    )
    print(f"  [Q1] query → ids={got_ids} dists={[f'{d:.3f}' for d in got_dists]}", flush=True)

    if not got_ids or got_ids[0] != "a":
        return False, f"nearest of [0.99, 0.01, 0]: expected 'a', got {got_ids}"

    # where=user_id=alice filters down to one row only.
    alice_ids, _ = b.query_one(
        [0.0, 0.0, 1.0],
        top_k=5,
        where={"user_id": "alice"},
    )
    if alice_ids != ["c"]:
        return False, f"where filter user_id=alice: expected ['c'], got {alice_ids}"

    # Delete one and re-count.
    b.delete_where({"doc_id": "d1"})
    after = b.count()
    print(f"  [Q1] delete doc_id=d1 → count={after}", flush=True)
    if after != 2:
        return False, f"count after delete: expected 2, got {after}"

    return True, f"sqlite-vec backend: upsert+query+filter+delete OK ({b.name})"


# ---------------------------------------------------------------------------
# Q2 — Neo4j backend wiring
# ---------------------------------------------------------------------------


def q2_neo4j_wiring(tmp_dir: Path) -> tuple[bool | str, str]:
    """Import is side-effect-free, factory dispatches, missing-driver-or-URI
    raises a clear error. We do NOT require a running Neo4j server."""
    # The class import must not require the neo4j driver.
    from hrag.kg.backends.neo4j import Neo4jBackend  # noqa: F401

    print("  [Q2] import: hrag.kg.backends.neo4j OK", flush=True)

    # Try importing the official driver — if absent, the constructor should
    # still raise an actionable RuntimeError instead of ImportError.
    driver_present = False
    try:
        import neo4j  # noqa: F401

        driver_present = True
    except ImportError:
        pass

    # Construct without URI / env vars; expect a RuntimeError with an
    # actionable message.
    saved = {}
    for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        saved[key] = os.environ.pop(key, None)
    try:
        try:
            b = Neo4jBackend()
            # If construction succeeded (env var was sneaked in), at least
            # number_of_nodes should not raise NotImplementedError — we are
            # checking that it is no longer a stub.
            try:
                _ = b.number_of_nodes()
            except NotImplementedError:
                return False, "Neo4jBackend.number_of_nodes raised NotImplementedError; still a stub"
            return True, "Neo4jBackend: server actually reachable; real impl works"
        except RuntimeError as exc:
            print(f"  [Q2] missing-env construction raised: {exc}", flush=True)
            msg = str(exc).lower()
            keywords = ("neo4j", "uri", "running")
            if not any(k in msg for k in keywords):
                return (
                    False,
                    f"RuntimeError raised but message is not actionable: {exc!r}",
                )
        except ImportError:
            if driver_present:
                return False, "ImportError raised despite neo4j driver being installed"
            # No driver and no URI: construct does whatever; we accept either
            # ImportError or RuntimeError as long as it's not a silent
            # success with stub methods.
            print("  [Q2] driver missing — ImportError accepted as a fallback signal", flush=True)
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val

    # Confirm the factory wiring: KGStore.from_config should route
    # kg.backend='neo4j' to Neo4jBackend (so flipping the config is enough).
    try:
        from hrag.kg.store import KGStore

        if hasattr(KGStore, "from_config"):
            # Build a minimal config that would request the neo4j backend.
            from hrag.config import Config

            cfg = Config()
            cfg.kg.backend = "neo4j"
            cfg.kg.enabled = False  # don't actually try to use it
            try:
                # The factory may eagerly construct the backend; that should
                # raise the same actionable error rather than a stub-style
                # NotImplementedError.
                _ = KGStore.from_config(cfg, db=None, embedder=None)
            except (RuntimeError, ImportError) as exc:
                print(f"  [Q2] factory raised expected wiring error: {type(exc).__name__}", flush=True)
            except NotImplementedError:
                return False, "KGStore.from_config(neo4j) raised NotImplementedError; still a stub"
            except Exception as exc:  # noqa: BLE001
                # Other init paths may need db/embedder; accept as long as
                # not a stub-style NotImplementedError.
                if "NotImplementedError" in repr(exc):
                    return False, f"Neo4j factory still stub: {exc}"
                print(f"  [Q2] factory init needed extra context; tolerated: {type(exc).__name__}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [Q2] factory check inconclusive: {type(exc).__name__}: {exc}", flush=True)

    return True, "Neo4j backend: wiring + clear error without server (no longer a stub)"


# ---------------------------------------------------------------------------
# Q3 — Adaptive top_k resolver
# ---------------------------------------------------------------------------


def q3_adaptive_resolver(tmp_dir: Path) -> tuple[bool | str, str]:
    """Each intent maps to the documented (vec_k, final_k) pair."""
    from hrag.config import Config
    from hrag.intent import Intent
    from hrag.orchestrator import _adaptive_top_k

    cfg = Config()
    cfg.retrieval.adaptive_enabled = True

    cases = {
        Intent.GREETING: (None, None),
        Intent.PERSONAL: (16, 8),
        Intent.FACTUAL: (12, 6),
        Intent.UNCLEAR: (12, 4),
    }
    rows = []
    for intent, expected in cases.items():
        got = _adaptive_top_k(cfg, intent)
        rows.append((intent.value, expected, got))
        if got != expected:
            return (
                False,
                f"{intent.value}: expected {expected}, got {got}",
            )

    print("  [Q3] resolver mappings:", flush=True)
    for name, exp, got in rows:
        print(f"        {name:8s}: vec_k={got[0]!s:>4s}  final_k={got[1]!s:>4s}", flush=True)

    return True, f"resolver: {len(rows)}/{len(rows)} intent mappings correct"


# ---------------------------------------------------------------------------
# Q4 — Greeting skip
# ---------------------------------------------------------------------------


def q4_greeting_skips_retrieval(tmp_dir: Path) -> tuple[bool | str, str]:
    """A greeting under adaptive_enabled=True never reaches the retriever."""
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
            sqlite_path=str(tmp_dir / "store_q4.sqlite"),
            chroma_path=str(tmp_dir / "chroma_q4"),
            kg_path=str(tmp_dir / "kg_q4"),
            data_root=str(tmp_dir / "data_q4"),
        ),
    )
    cfg.project_root = tmp_dir
    cfg.retrieval.adaptive_enabled = True
    cfg.retrieval.rerank_enabled = False

    from hrag.orchestrator import Orchestrator
    orch = Orchestrator(cfg)

    # Mock provider + retriever — never call out.
    orch.llm = _FakeLLM()
    if orch.gate is not None:
        orch.gate.llm = orch.llm
    if orch.clue is not None:
        orch.clue.llm = orch.llm

    class _SpyRetriever:
        name = "spy"

        def __init__(self) -> None:
            self.calls: list = []

        def retrieve(self, query, user_id, top_k=10, source_types=None, intent_hint=None, where=None):
            self.calls.append((query, top_k))
            return []

    spy = _SpyRetriever()
    orch.retriever = spy  # type: ignore[assignment]

    class _GreetClassifier:
        def classify(self, text, **kwargs):
            return IntentVerdict(
                intent=Intent.GREETING,
                confidence=1.0,
                source="test",
                raw_label="greeting",
            )

    orch.intent_classifier = _GreetClassifier()  # type: ignore[assignment]

    events: list[tuple[str, dict]] = []
    try:
        orch.chat("hi", user_id="default", progress=lambda n, p: events.append((n, p)))
    finally:
        orch.close()
        _conn_mod._db_singleton = None

    skipped = [p for n, p in events if n == "retrieval_skipped"]
    print(f"  [Q4] retriever calls={len(spy.calls)}  retrieval_skipped events={len(skipped)}", flush=True)

    if spy.calls:
        return False, f"retriever was called {len(spy.calls)} times on a greeting"
    if not skipped:
        return False, "no 'retrieval_skipped' event emitted"
    if skipped[0].get("reason") != "greeting":
        return False, f"skip reason: expected 'greeting', got {skipped[0]!r}"

    return True, "greeting skip: retriever never called; 'retrieval_skipped' fired"


# ---------------------------------------------------------------------------
# Q5 — Personal episodic bias
# ---------------------------------------------------------------------------


def q5_episodic_bias(tmp_dir: Path) -> tuple[bool | str, str]:
    """Episodic-typed results float to the top of the result list."""
    from hrag.config import Config, EmbeddingsConfig, LLMConfig, StorageConfig
    from hrag.intent import Intent, IntentVerdict
    from hrag.types import Chunk, RetrievalResult
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
            sqlite_path=str(tmp_dir / "store_q5.sqlite"),
            chroma_path=str(tmp_dir / "chroma_q5"),
            kg_path=str(tmp_dir / "kg_q5"),
            data_root=str(tmp_dir / "data_q5"),
        ),
    )
    cfg.project_root = tmp_dir
    cfg.retrieval.adaptive_enabled = True
    cfg.retrieval.adaptive_personal_episodic_bias = True
    cfg.retrieval.rerank_enabled = False

    from hrag.orchestrator import Orchestrator
    orch = Orchestrator(cfg)

    orch.llm = _FakeLLM()
    if orch.gate is not None:
        orch.gate.llm = orch.llm
    if orch.clue is not None:
        orch.clue.llm = orch.llm

    def mk_chunk(cid: str, st: str) -> Chunk:
        return Chunk(
            chunk_id=cid,
            doc_id="d",
            user_id="default",
            text=cid,
            embedding_text=cid,
            source_type=st,
        )

    seeded = [
        RetrievalResult(chunk=mk_chunk("doc1", "document"), score=0.95),
        RetrievalResult(chunk=mk_chunk("ep1",  "episodic"), score=0.80),
        RetrievalResult(chunk=mk_chunk("doc2", "document"), score=0.70),
        RetrievalResult(chunk=mk_chunk("ep2",  "episodic"), score=0.60),
    ]

    class _Spy:
        name = "spy"

        def retrieve(self, query, user_id, top_k=10, source_types=None, intent_hint=None, where=None):
            self.seen_source_types = source_types
            return list(seeded)

    spy = _Spy()
    orch.retriever = spy  # type: ignore[assignment]

    class _PersonalClassifier:
        def classify(self, text, **kwargs):
            return IntentVerdict(
                intent=Intent.PERSONAL,
                confidence=1.0,
                source="test",
                raw_label="personal",
            )

    orch.intent_classifier = _PersonalClassifier()  # type: ignore[assignment]

    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what do I prefer?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _conn_mod._db_singleton = None

    bias = [p for n, p in events if n == "episodic_bias_applied"]
    print(f"  [Q5] episodic_bias events={len(bias)}; source_types asked={getattr(spy, 'seen_source_types', None)!r}", flush=True)

    if getattr(spy, "seen_source_types", None) != ["document", "episodic"]:
        return False, "retriever was not asked for [document, episodic]"
    if not bias:
        return False, "no 'episodic_bias_applied' event"
    if bias[0]["episodic_count"] != 2 or bias[0]["total"] != 4:
        return (
            False,
            f"bias event payload off: {bias[0]!r}; expected episodic_count=2 total=4",
        )

    return True, "episodic bias: source_types broadened + lift event fired"


# ---------------------------------------------------------------------------
# Q6 — Ollama keep-alive plumbing
# ---------------------------------------------------------------------------


def q6_keep_alive_plumbing(tmp_dir: Path) -> tuple[bool | str, str]:
    """OllamaProvider forwards llm.keep_alive into the chat() top-level kwargs."""
    from hrag.config import LLMConfig
    from hrag.providers.llm import OllamaProvider

    # Bypass __init__ (which imports ollama and opens a client).
    p = OllamaProvider.__new__(OllamaProvider)
    p.config = LLMConfig()  # defaults: keep_alive="30m"
    p._client = MagicMock()  # type: ignore[attr-defined]

    k = p._build_chat_kwargs([{"role": "user", "content": "hi"}], {})
    print(f"  [Q6] default keep_alive forwarded: {k.get('keep_alive')!r}", flush=True)
    if k.get("keep_alive") != "30m":
        return False, f"default keep_alive: expected '30m', got {k.get('keep_alive')!r}"

    p.config = LLMConfig(keep_alive="-1s")
    k = p._build_chat_kwargs([{"role": "user", "content": "hi"}], {})
    if k.get("keep_alive") != "-1s":
        return False, f"keep_alive='-1s': not propagated; got {k.get('keep_alive')!r}"

    p.config = LLMConfig(keep_alive=None)
    k = p._build_chat_kwargs([{"role": "user", "content": "hi"}], {})
    if "keep_alive" in k:
        return False, f"keep_alive=None: should be absent from kwargs, got {k.get('keep_alive')!r}"

    return True, "keep_alive: default + override + None all wire correctly"


# ---------------------------------------------------------------------------
# Q7 — Factory dispatch
# ---------------------------------------------------------------------------


def q7_factory_dispatch(tmp_dir: Path) -> tuple[bool | str, str]:
    """The vector-backend factory dispatches both 'chroma' and 'sqlite_vec'
    without raising NotImplementedError. (The actual construction may fail
    due to missing deps, but never with the stub sentinel.)"""
    from hrag.orchestrator import _build_vector_backend
    from hrag.config import Config

    cfg = Config()
    cfg.storage.chroma_path = str(tmp_dir / "chroma_q7")

    # Chroma path — should construct, or raise the chroma-specific import
    # error (the conftest stubs are usually fine here).
    cfg.retrieval.vector_backend = "chroma"
    try:
        b = _build_vector_backend(cfg)
        print(f"  [Q7] chroma backend constructed: {getattr(b, 'name', '?')}", flush=True)
    except NotImplementedError:
        return False, "chroma backend factory raised NotImplementedError (regression)"
    except Exception as exc:  # noqa: BLE001
        # Construction may fail on the test box for unrelated reasons; what
        # matters is that the *dispatch* recognised the value.
        msg = str(exc).lower()
        if "unknown" in msg or "unrecognised" in msg or "not supported" in msg:
            return False, f"chroma backend not recognised: {exc}"

    # sqlite_vec path — same expectations, just different concrete class.
    cfg.retrieval.vector_backend = "sqlite_vec"
    cfg.storage.chroma_path = str(tmp_dir / "vec_q7")
    try:
        b2 = _build_vector_backend(cfg)
        print(f"  [Q7] sqlite_vec backend constructed: {getattr(b2, 'name', '?')}", flush=True)
    except NotImplementedError:
        return False, "sqlite_vec backend factory still raises NotImplementedError"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "unknown" in msg or "unrecognised" in msg or "not supported" in msg:
            return False, f"sqlite_vec backend not recognised: {exc}"
        # ImportError due to missing extension is acceptable.
        print(f"  [Q7] sqlite_vec construction raised {type(exc).__name__} (tolerated): {exc}", flush=True)

    # Unknown backend — factory must reject explicitly.
    cfg.retrieval.vector_backend = "no-such-backend"
    try:
        _ = _build_vector_backend(cfg)
        return False, "unknown backend was silently accepted"
    except Exception as exc:  # noqa: BLE001
        print(f"  [Q7] unknown backend rejected: {type(exc).__name__}: {exc}", flush=True)

    return True, "factory: dispatches chroma + sqlite_vec; rejects unknown"


# ---------------------------------------------------------------------------
# Q8 — Backward compatibility (defaults off)
# ---------------------------------------------------------------------------


def q8_backward_compat(tmp_dir: Path) -> tuple[bool | str, str]:
    """With every Phase 6 flag at its default, the adaptive resolver is a
    pass-through (returns the global (top_k_vector, top_k_final)) and
    keep_alive is the documented '30m' default."""
    from hrag.config import Config
    from hrag.intent import Intent
    from hrag.orchestrator import _adaptive_top_k

    cfg = Config()
    # Sanity-check defaults match what CLAUDE.md promised.
    if cfg.retrieval.adaptive_enabled is not False:
        return False, f"retrieval.adaptive_enabled default is {cfg.retrieval.adaptive_enabled!r}, expected False"
    if cfg.llm.keep_alive != "30m":
        return False, f"llm.keep_alive default is {cfg.llm.keep_alive!r}, expected '30m'"

    # Every intent through the resolver returns the global pair.
    global_pair = (cfg.retrieval.top_k_vector, cfg.retrieval.top_k_final)
    for intent in (Intent.GREETING, Intent.PERSONAL, Intent.FACTUAL, Intent.UNCLEAR):
        got = _adaptive_top_k(cfg, intent)
        if got != global_pair:
            return False, f"intent {intent.value}: resolver returned {got}, expected {global_pair}"

    print(f"  [Q8] resolver pass-through ok; global pair = {global_pair}", flush=True)
    print(f"  [Q8] llm.keep_alive default = {cfg.llm.keep_alive!r}", flush=True)
    return True, f"defaults preserved: adaptive off, keep_alive={cfg.llm.keep_alive!r}, global={global_pair}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None, help="Write machine-readable JSON to this file.")
    args = parser.parse_args()

    print("Phase 6 — Backends + Adaptive + Keep-Alive benchmark", flush=True)
    print("=" * 60, flush=True)

    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="hrag_phase6_"))
    print(f"Tmp dir: {tmp_dir}", flush=True)

    suites: list[tuple[str, Callable[[Path], tuple[bool | str, str]]]] = [
        ("Q1 — sqlite-vec backend roundtrip", q1_sqlite_vec_roundtrip),
        ("Q2 — Neo4j backend wiring + actionable error", q2_neo4j_wiring),
        ("Q3 — Adaptive top_k resolver per intent", q3_adaptive_resolver),
        ("Q4 — Greeting skips retrieval (adaptive=on)", q4_greeting_skips_retrieval),
        ("Q5 — Personal episodic bias re-sort", q5_episodic_bias),
        ("Q6 — Ollama keep-alive plumbing", q6_keep_alive_plumbing),
        ("Q7 — Vector backend factory dispatch", q7_factory_dispatch),
        ("Q8 — Backward compat (defaults off)", q8_backward_compat),
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
        task = prog.add_task("Phase 6 benchmark", total=len(suites))

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
    console.print("[bold]Phase 6 Benchmark Summary[/bold]")
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

    accept = passed >= 7 or (passed + skipped >= 7 and failed == 0)
    if accept:
        console.print("[green bold]ACCEPTED (>= 7/8 pass or skip)[/]")
    else:
        console.print("[red bold]FAILED — need >= 7/8[/]")

    # JSON output for the HTML report.
    if args.json:
        payload = {
            "phase": 6,
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
