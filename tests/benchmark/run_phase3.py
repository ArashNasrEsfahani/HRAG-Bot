"""Phase 3 acceptance benchmark runner.

Usage (from project root):
    python tests/benchmark/run_phase3.py

Produces:
    tests/benchmark/phase3_results.md   — Markdown summary table + per-test detail
    tests/benchmark/phase3_results.html — self-contained HTML report
    stdout                              — live per-test progress + final summary

All tests are self-contained: real ChromaDB in tmp dirs, FakeLLM and
FakeEmbedder patched into the providers. No Ollama, no pre-populated index.
"""

from __future__ import annotations

import hashlib
import html as _html_escape
import shutil
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

# ---------------------------------------------------------------------------
# sys.path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

# Line-buffered UTF-8 stdout so every print() lands on screen immediately
# and the ▶ / ✓ / ✗ progress glyphs survive Windows cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Fake providers (inlined so the runner has no dep on tests/conftest.py)
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Hash-based deterministic embedder; no torch needed."""

    name = "fake"
    _DIM = 384

    def embed(self, texts):
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            floats = []
            i = 0
            while len(floats) < self._DIM:
                floats.append((digest[i % len(digest)] / 127.5) - 1.0)
                i += 1
            out.append(floats[: self._DIM])
        return out

    def embed_one(self, t):
        return self.embed([t])[0]

    @property
    def dim(self):
        return self._DIM


class FakeLLM:
    """Returns canned text; one mode for preference-extract prompts."""

    name = "fake"
    DEFAULT_ANSWER = "This is a canned answer from FakeLLM."

    def __init__(self, canned: str | None = None) -> None:
        self._canned = canned

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        if self._canned is not None:
            return self._canned
        return self.DEFAULT_ANSWER

    def generate(self, request):
        from hrag.types import GenerationResponse  # noqa: PLC0415

        prompt = " ".join(m.content for m in request.messages)
        return GenerationResponse(text=self.complete(prompt), raw=None)

    def verify_ready(self) -> None:  # called by the chat REPL only
        return None


# ---------------------------------------------------------------------------
# Test harness helpers
# ---------------------------------------------------------------------------


def _reset_db_singleton() -> None:
    import hrag.db.connection as _conn_mod  # noqa: PLC0415

    _conn_mod._db_singleton = None


def _make_config(tmp_path: Path, **overrides) -> Any:
    """Build a Config rooted at *tmp_path* with sensible defaults for the bench."""
    from hrag.config import (  # noqa: PLC0415
        ChunkingConfig, Config, EmbeddingsConfig, KGConfig, LLMConfig,
        MemoryConfig, QualityConfig, RetrievalConfig, StorageConfig,
    )

    chunking = ChunkingConfig(quality=QualityConfig(enabled=False))
    if "chunking" in overrides:
        chunking = overrides.pop("chunking")
    if "kg" in overrides:
        kg = overrides.pop("kg")
    else:
        kg = KGConfig(enabled=False)
    if "memory" in overrides:
        memory = overrides.pop("memory")
    else:
        memory = MemoryConfig()
    if "retrieval" in overrides:
        retrieval = overrides.pop("retrieval")
    else:
        retrieval = RetrievalConfig(
            rerank_enabled=False, doc_scope_enabled=False, retriever="vector"
        )

    cfg = Config(
        llm=LLMConfig(provider="ollama", model="bench-fake"),
        embeddings=EmbeddingsConfig(
            provider="sentence-transformers",
            model="fake/all-mpnet-base-v2",
            dim=384,
        ),
        storage=StorageConfig(
            sqlite_path=str(tmp_path / "store.sqlite"),
            chroma_path=str(tmp_path / "chroma"),
            kg_path=str(tmp_path / "kg"),
            data_root=str(tmp_path / "data"),
        ),
        chunking=chunking,
        retrieval=retrieval,
        kg=kg,
        memory=memory,
        **overrides,
    )
    cfg.project_root = tmp_path
    return cfg


def _patch_providers(monkeypatch_target_mod_names: list[str], llm: FakeLLM | None = None) -> None:
    """Monkey-patch get_embedding_provider and get_llm_provider in the namespaces
    that the orchestrator imports them from."""
    import hrag.providers.embeddings as _emb  # noqa: PLC0415
    import hrag.providers.llm as _llm  # noqa: PLC0415

    fake_emb = FakeEmbedder()
    fake_llm = llm or FakeLLM()
    _emb.get_embedding_provider = lambda *a, **kw: fake_emb  # type: ignore[assignment]
    _llm.get_llm_provider = lambda *a, **kw: fake_llm  # type: ignore[assignment]
    # Orchestrator imports them like `from hrag.providers.embeddings import get_embedding_provider`,
    # so we need to patch the bound name inside orchestrator too.
    import hrag.orchestrator as _orch  # noqa: PLC0415

    _orch.get_embedding_provider = _emb.get_embedding_provider
    _orch.get_llm_provider = _llm.get_llm_provider


def _make_orch(tmp_path: Path, *, llm: FakeLLM | None = None, **cfg_overrides):
    """Build a fully-initialised Orchestrator with fake providers."""
    _reset_db_singleton()
    _patch_providers([], llm=llm)
    cfg = _make_config(tmp_path, **cfg_overrides)
    from hrag.orchestrator import Orchestrator  # noqa: PLC0415

    return Orchestrator(cfg)


# ---------------------------------------------------------------------------
# Test functions — each returns (passed: bool, message: str)
# ---------------------------------------------------------------------------


def t01_remember_writes(tmp_path: Path) -> tuple[bool, str]:
    orch = _make_orch(tmp_path)
    try:
        mid = orch.memory_store.add("default", "Postgres preferred over MySQL.")
        if not mid.startswith("episodic:default:"):
            return False, f"memory_id has wrong prefix: {mid!r}"
        row = orch.db.execute(
            "SELECT source_type, excluded FROM chunks WHERE doc_id = ?", (mid,)
        ).fetchone()
        if row is None:
            return False, "no chunk row written for the new memory"
        if row["source_type"] != "episodic":
            return False, f"source_type={row['source_type']!r}, expected 'episodic'"
        if row["excluded"] != 0:
            return False, f"chunk arrived already tombstoned (excluded={row['excluded']})"
        return True, f"memory_id={mid[:24]}…, 1 chunk persisted"
    finally:
        orch.close()


def t15_remember_latency_under_100ms(tmp_path: Path) -> tuple[bool, str]:
    orch = _make_orch(tmp_path)
    try:
        # Warm up — first call hits the chunker/Chroma cold paths.
        orch.memory_store.add("default", "warmup memory note one")
        start = time.perf_counter()
        orch.memory_store.add("default", "Hot-path measurement memory.")
        dt_ms = (time.perf_counter() - start) * 1000.0
        ok = dt_ms < 100.0
        return ok, f"hot-path /remember took {dt_ms:.1f} ms (budget 100 ms)"
    finally:
        orch.close()


def t16_bulk_200_under_10s(tmp_path: Path) -> tuple[bool, str]:
    orch = _make_orch(tmp_path)
    try:
        items = [{"text": f"bulk note number {i} about topic {i % 7}"} for i in range(200)]
        start = time.perf_counter()
        ids = orch.memory_store.add_batch("default", items)
        dt = time.perf_counter() - start
        ok = (len(ids) == 200) and (dt < 10.0)
        rate = len(ids) / max(dt, 1e-6)
        return ok, f"{len(ids)}/200 saved in {dt:.2f}s ({rate:.0f} notes/sec, budget 10s)"
    finally:
        orch.close()


def t02_recall_returns_relevant(tmp_path: Path) -> tuple[bool, str]:
    """Hash-based FakeEmbedder gives no semantic signal — fall back to exact-substring.

    The recall path still exercises Chroma + SQLite hydrate + the
    source_types='episodic' filter — what we really want to check.
    """
    orch = _make_orch(tmp_path)
    try:
        orch.memory_store.add("default", "Cats sleep most of the day.")
        target_id = orch.memory_store.add(
            "default", "Postgres is preferred over MySQL for new projects."
        )
        orch.memory_store.add("default", "Trains in Switzerland are punctual.")

        hits = orch.retriever.retrieve(
            "Postgres MySQL", "default", top_k=3, source_types=["episodic"]
        )
        if not hits:
            return False, "retrieve returned no hits"
        ids = [h.chunk.doc_id for h in hits]
        if target_id not in ids:
            return False, f"target {target_id[:24]}… missing from hits {[i[:16] for i in ids]}"
        # Document filter ironclad: every hit must be episodic.
        bad = [h.chunk.source_type for h in hits if h.chunk.source_type != "episodic"]
        if bad:
            return False, f"non-episodic source_type leaked into /recall: {bad}"
        return True, f"target in top-{len(hits)} hits; all source_type=episodic"
    finally:
        orch.close()


def t03_memory_competes_with_docs(tmp_path: Path) -> tuple[bool, str]:
    """With source_types=None, both episodic and document chunks should compete.

    We don't assert ordering (FakeEmbedder makes that flaky). We assert the
    correct behaviour: the union retrieval returns BOTH types when both
    exist and the query embedding is generic.
    """
    from hrag.types import Document  # noqa: PLC0415

    orch = _make_orch(tmp_path)
    try:
        # 1 doc, 1 memory.
        doc = Document(
            doc_id="docA",
            user_id="default",
            source_path="memory://docA",
            title="Doc A",
            text="The Earth orbits the Sun once per year.",
            source_type="document",
        )
        orch.ingest.ingest_document(doc)
        mem_id = orch.memory_store.add("default", "I prefer Postgres for new projects.")

        hits = orch.retriever.retrieve("anything", "default", top_k=10, source_types=None)
        types = {h.chunk.source_type for h in hits}
        if "document" not in types or "episodic" not in types:
            return False, f"top-k did not include both source_types — got {types}"
        return True, f"top-k contains both source_types ({sorted(types)}); memory {mem_id[:16]}… surfaced"
    finally:
        orch.close()


def t04_profile_upsert_idempotent(tmp_path: Path) -> tuple[bool, str]:
    orch = _make_orch(tmp_path)
    try:
        ps = orch.profile_store
        pid1 = ps.upsert("default", "fact", "occupation", "engineer", confidence=0.9)
        pid2 = ps.upsert("default", "fact", "occupation", "scientist", confidence=0.95)
        if pid1 != pid2:
            return False, f"upsert created two rows (pid1={pid1}, pid2={pid2})"
        prefs = ps.list_all("default")
        if len(prefs) != 1:
            return False, f"expected 1 row after upsert; got {len(prefs)}"
        if prefs[0].value != "scientist":
            return False, f"value did not update; got {prefs[0].value!r}"
        return True, "pref_id stable across upsert; value updated to 'scientist'"
    finally:
        orch.close()


def t05_profile_render_grouped(tmp_path: Path) -> tuple[bool, str]:
    orch = _make_orch(tmp_path)
    try:
        ps = orch.profile_store
        ps.upsert("default", "fact", "occupation", "engineer", confidence=0.9)
        ps.upsert("default", "fact", "low_conf", "x", confidence=0.3)
        ps.upsert("default", "style", "length", "short", confidence=0.9)
        ps.upsert("default", "like", "lang", "Python", confidence=0.9)
        ps.upsert("default", "dislike", "verbose", "yes", confidence=0.9)

        rendered = ps.render("default", min_confidence=0.5, max_items=12)
        missing = [
            label for label in ("Facts:", "Style preferences:", "Likes:", "Dislikes:")
            if label not in rendered
        ]
        if missing:
            return False, f"missing group labels: {missing}"
        if "low_conf" in rendered:
            return False, "render leaked a below-min_confidence pref"
        # max_items cap: with all 5 prefs eligible, max_items=2 must keep
        # exactly the top 2 entries (ordered by confidence DESC, then by
        # ingest order — see ProfileStore.list_all).
        capped = ps.render("default", min_confidence=0.0, max_items=2)
        rendered_topics = {"occupation", "length", "lang", "verbose", "low_conf"}
        present = {t for t in rendered_topics if t in capped}
        if len(present) != 2:
            return False, (
                f"max_items=2 should leave exactly 2 topics in the render; "
                f"got {sorted(present)} → render=\n{capped}"
            )
        return True, "groups present; min_confidence + max_items both honoured"
    finally:
        orch.close()


def t06_context_builder_into_prompt(tmp_path: Path) -> tuple[bool, str]:
    orch = _make_orch(tmp_path)
    try:
        orch.profile_store.upsert(
            "default", "fact", "occupation", "data engineer", confidence=0.95
        )
        orch.profile_store.upsert(
            "default", "style", "length", "shorter answers", confidence=0.9
        )

        # Stub the retriever — we only care about the prompt-render path.
        class _NoOp:
            def retrieve(self, *a, **kw):
                return []

        orch.retriever = _NoOp()
        orch.reranker = None
        orch.mst_organizer = None

        result = orch.chat("What's a good Python library?", "default")
        prompt = result.prompt
        for needle in (
            "Facts: occupation: data engineer",
            "Style preferences: length: shorter answers",
        ):
            if needle not in prompt:
                return False, f"prompt missing fragment: {needle!r}"
        if "(no profile yet)" in prompt:
            return False, "empty placeholder leaked despite seeded profile"
        return True, "rendered profile lines present in answer prompt"
    finally:
        orch.close()


def t07_forget_tombstones(tmp_path: Path) -> tuple[bool, str]:
    orch = _make_orch(tmp_path)
    try:
        mid = orch.memory_store.add("default", "fact to forget about postgres")
        chunks = orch.memory_store.list_recent("default")
        if len(chunks) != 1:
            return False, f"baseline list_recent should be 1; got {len(chunks)}"
        cid = chunks[0].chunk_id

        if not orch.memory_store.forget("default", cid):
            return False, "forget(chunk_id) returned False"

        row = orch.db.execute(
            "SELECT excluded FROM chunks WHERE chunk_id = ?", (cid,)
        ).fetchone()
        if row["excluded"] != 1:
            return False, f"chunks.excluded={row['excluded']} after forget"
        if orch.memory_store.count("default") != 0:
            return False, "forget did not hide the chunk from count()"
        if orch.memory_store.list_recent("default"):
            return False, "forget did not hide the chunk from list_recent()"
        return True, "excluded=1 set; list/count both hide it"
    finally:
        orch.close()


def t08_forget_by_query(tmp_path: Path) -> tuple[bool, str]:
    orch = _make_orch(tmp_path)
    try:
        orch.memory_store.add("default", "cats sleep most of the day")
        target = orch.memory_store.add("default", "Postgres preferred over MySQL")
        orch.memory_store.add("default", "trains in Switzerland are punctual")

        ids = orch.memory_store.forget_by_query(
            "default", "Postgres MySQL", top_k=3, retriever=orch.retriever
        )
        if not ids:
            return False, "forget_by_query returned no candidates"
        # The target's chunk_id must be among the returned chunk_ids.
        target_chunk = orch.db.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ?", (target,)
        ).fetchone()["chunk_id"]
        if target_chunk not in ids:
            return False, f"target chunk_id {target_chunk[:16]}… missing from {ids}"
        return True, f"target returned; {len(ids)} candidate(s)"
    finally:
        orch.close()


def t09_user_scoping(tmp_path: Path) -> tuple[bool, str]:
    orch = _make_orch(tmp_path)
    try:
        orch.db.ensure_user("alice")
        orch.db.ensure_user("bob")
        orch.db.commit()
        orch.memory_store.add("alice", "alice's secret note")
        orch.memory_store.add("bob", "bob's secret note")

        if orch.memory_store.count("alice") != 1:
            return False, "alice should see exactly 1 memory"
        if orch.memory_store.count("bob") != 1:
            return False, "bob should see exactly 1 memory"
        if orch.memory_store.count("default") != 0:
            return False, "default user should see 0 memories"

        # Cross-user recall must not leak.
        bob_hits = orch.retriever.retrieve(
            "alice secret", "bob", top_k=5, source_types=["episodic"]
        )
        if any(h.chunk.user_id != "bob" for h in bob_hits):
            return False, "bob's retrieve returned non-bob chunks"
        return True, "per-user counts + retrieve scoping clean"
    finally:
        orch.close()


def t10_bulk_import_iter(tmp_path: Path) -> tuple[bool, str]:
    """Walk a tmp dir of .md/.txt; verify per-paragraph extraction and add_batch."""
    # Build a tiny tmp tree.
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "a.md").write_text(
        "# Top\n\nIntro paragraph.\n\n## Section One\n\nBody one.\n\n## Section Two\n\nBody two.\n",
        encoding="utf-8",
    )
    (notes_dir / "b.txt").write_text("line one\nline two\nline three\n", encoding="utf-8")

    from hrag.cli import _iter_memory_texts_from_path  # noqa: PLC0415

    items = _iter_memory_texts_from_path(notes_dir, split_paragraphs=True)
    if len(items) < 5:
        return False, f"expected ≥5 items from bulk extractor; got {len(items)}"

    orch = _make_orch(tmp_path / "orch")
    try:
        ids = orch.memory_store.add_batch("default", items, source="bulk")
        if len(ids) != len(items):
            return False, f"add_batch persisted {len(ids)}/{len(items)} items"
        if orch.memory_store.count("default") != len(items):
            return False, "count() does not match items persisted"
        return True, f"extracted {len(items)} items; all persisted"
    finally:
        orch.close()


def t11_extractor_robust_json(tmp_path: Path) -> tuple[bool, str]:
    from hrag.memory.extractor import PreferenceExtractor  # noqa: PLC0415

    fenced = '```json\n[{"polarity": "fact", "topic": "city", "value": "Berlin", "confidence": 0.9}]\n```'
    prose_prefix = (
        'Sure, here are the preferences I extracted:\n'
        '[{"polarity": "like", "topic": "lang", "value": "Python", "confidence": 0.8}]'
    )

    out1 = PreferenceExtractor(FakeLLM(canned=fenced)).extract(
        [("user", "I live in Berlin.")]
    )
    if len(out1) != 1 or out1[0].topic != "city":
        return False, f"markdown-fence parse failed: {out1!r}"

    out2 = PreferenceExtractor(FakeLLM(canned=prose_prefix)).extract(
        [("user", "I love Python.")]
    )
    if len(out2) != 1 or out2[0].topic != "lang":
        return False, f"prose-prefix parse failed: {out2!r}"

    # Negative: completely malformed → empty list, no raise.
    out3 = PreferenceExtractor(FakeLLM(canned="not json at all")).extract([("user", "hi")])
    if out3 != []:
        return False, f"malformed JSON did not yield []: {out3!r}"

    return True, "fenced ✓, prose-prefix ✓, malformed → [] ✓"


def t12_auto_extractor_min_conf(tmp_path: Path) -> tuple[bool, str]:
    from hrag.memory.auto_extract import SessionAutoExtractor  # noqa: PLC0415
    from hrag.memory.extractor import PreferenceCandidate  # noqa: PLC0415

    orch = _make_orch(tmp_path)
    try:
        # Seed a session + a couple of messages.
        orch.db.execute(
            "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
            ("sess-bench", "default"),
        )
        orch.db.execute(
            "INSERT INTO messages (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            ("sess-bench", "default", "user", "I love Python and live in Berlin."),
        )
        orch.db.commit()

        class _CannedExtractor:
            def extract(self, conv):
                return [
                    PreferenceCandidate("fact", "city", "Berlin", 0.6),  # below 0.7
                    PreferenceCandidate("like", "lang", "Python", 0.95),  # above
                ]

        auto = SessionAutoExtractor(
            orch.db, _CannedExtractor(), orch.profile_store, min_confidence=0.7
        )
        auto.on_session_close("default", "sess-bench", block=True)

        prefs = orch.profile_store.list_all("default")
        topics = {p.topic for p in prefs}
        if "city" in topics:
            return False, f"low-confidence candidate slipped through: {topics}"
        if "lang" not in topics:
            return False, f"high-confidence candidate missed: {topics}"
        return True, f"only above-threshold candidate upserted; topics={sorted(topics)}"
    finally:
        orch.close()


def t13_kg_skip_episodic(tmp_path: Path) -> tuple[bool, str]:
    """Build with kg.enabled=true, monkey-patch TripleExtractor to record any call,
    ingest an episodic doc, assert it was NEVER instantiated."""
    from hrag.config import KGConfig  # noqa: PLC0415
    from hrag.types import Document  # noqa: PLC0415

    orch = _make_orch(tmp_path, kg=KGConfig(enabled=True))
    try:
        calls: list[Any] = []

        # Replace the lazy import target with a sentinel that records use.
        import hrag.kg.builder as _kgbuilder  # noqa: PLC0415

        original = _kgbuilder.TripleExtractor

        class _Tripwire:
            def __init__(self, *a, **kw):
                calls.append((a, kw))

            def extract_batch(self, *a, **kw):
                calls.append(("extract_batch", a, kw))
                return []

        _kgbuilder.TripleExtractor = _Tripwire  # type: ignore[assignment]
        try:
            doc = Document(
                doc_id="episodic:default:tripwire",
                user_id="default",
                source_path="memory://tripwire",
                title="m",
                text="Postgres is preferred over MySQL.",
                source_type="episodic",
            )
            orch.ingest.ingest_document(doc)
        finally:
            _kgbuilder.TripleExtractor = original  # type: ignore[assignment]

        if calls:
            return False, f"TripleExtractor was called {len(calls)} time(s) for episodic ingest"
        return True, "KG extraction skipped for source_type='episodic'"
    finally:
        orch.close()


def t14_quality_skip_episodic(tmp_path: Path) -> tuple[bool, str]:
    """3-token episodic must survive the chunker even with quality.enabled=True."""
    from hrag.config import ChunkingConfig, QualityConfig  # noqa: PLC0415
    from hrag.types import Document  # noqa: PLC0415

    chunking = ChunkingConfig(quality=QualityConfig(enabled=True, min_tokens=30, min_chars=80))
    orch = _make_orch(tmp_path, chunking=chunking)
    try:
        doc = Document(
            doc_id="episodic:default:tiny",
            user_id="default",
            source_path="memory://tiny",
            title="m",
            text="Postgres beats MySQL.",  # ~3 tokens
            source_type="episodic",
        )
        orch.ingest.ingest_document(doc)
        n = orch.db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?",
            (doc.doc_id,),
        ).fetchone()["n"]
        if n != 1:
            return False, f"expected 1 chunk for tiny episodic; got {n}"

        # Sanity contrast: same text as a document is dropped.
        doc2 = Document(
            doc_id="docA",
            user_id="default",
            source_path="memory://docA",
            title="m",
            text="Postgres beats MySQL.",  # too short for the filter
            source_type="document",
        )
        orch.ingest.ingest_document(doc2)
        n2 = orch.db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?", (doc2.doc_id,)
        ).fetchone()["n"]
        if n2 != 0:
            return False, f"control: short DOCUMENT should be filtered out, got {n2} chunks"
        return True, "episodic survived; document control filtered (expected)"
    finally:
        orch.close()


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


TESTS: dict[str, Callable[[Path], tuple[bool, str]]] = OrderedDict(
    [
        ("t01_remember_writes", t01_remember_writes),
        ("t15_remember_latency_under_100ms", t15_remember_latency_under_100ms),
        ("t16_bulk_200_under_10s", t16_bulk_200_under_10s),
        ("t02_recall_returns_relevant", t02_recall_returns_relevant),
        ("t03_memory_competes_with_docs", t03_memory_competes_with_docs),
        ("t04_profile_upsert_idempotent", t04_profile_upsert_idempotent),
        ("t05_profile_render_grouped", t05_profile_render_grouped),
        ("t06_context_builder_into_prompt", t06_context_builder_into_prompt),
        ("t07_forget_tombstones", t07_forget_tombstones),
        ("t08_forget_by_query", t08_forget_by_query),
        ("t09_user_scoping", t09_user_scoping),
        ("t10_bulk_import_iter", t10_bulk_import_iter),
        ("t11_extractor_robust_json", t11_extractor_robust_json),
        ("t12_auto_extractor_min_conf", t12_auto_extractor_min_conf),
        ("t13_kg_skip_episodic", t13_kg_skip_episodic),
        ("t14_quality_skip_episodic", t14_quality_skip_episodic),
    ]
)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_markdown(path: Path, spec: dict, results: list[dict], total_s: float) -> None:
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    lines: list[str] = []
    lines.append("# Phase 3 Acceptance Benchmark Results")
    lines.append("")
    lines.append(f"Run timestamp: {datetime.now(tz=timezone.utc).isoformat()}")
    lines.append(f"Total wall time: {total_s:.2f}s")
    lines.append(f"Pass rate: **{passed}/{total}**")
    lines.append("")
    lines.append("## Config snapshot")
    lines.append("")
    snap = spec.get("config_snapshot", {})
    lines.append("| key | value |")
    lines.append("|---|---|")
    for k, v in snap.items():
        v_str = str(v).strip().replace("|", "\\|")
        if "\n" in v_str:
            v_str = v_str.split("\n", 1)[0] + " …"
        lines.append(f"| {k} | {v_str} |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| id | category | result | time (s) | message |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        glyph = "PASS" if r["passed"] else "FAIL"
        msg = r["message"].replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {r['category']} | {glyph} | {r['duration_s']:.3f} | {msg} |"
        )
    lines.append("")
    lines.append("## Per-test detail")
    lines.append("")
    spec_by_id = {t["id"]: t for t in spec["tests"]}
    for r in results:
        s = spec_by_id.get(r["id"], {})
        lines.append(f"### {r['id']} — {r['category']}")
        lines.append("")
        lines.append(f"**Summary:** {s.get('summary', '')}")
        lines.append("")
        lines.append(f"**Result:** {'PASS' if r['passed'] else 'FAIL'} in {r['duration_s']:.3f}s")
        lines.append("")
        lines.append(f"**Message:** {r['message']}")
        if not r["passed"] and r.get("traceback"):
            lines.append("")
            lines.append("**Traceback:**")
            lines.append("```")
            lines.append(r["traceback"].rstrip())
            lines.append("```")
        lines.append("")
        notes = s.get("notes")
        if notes:
            lines.append(f"*Notes: {notes.strip()}*")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Run with: `python tests/benchmark/run_phase3.py`")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(path: Path, spec: dict, results: list[dict], total_s: float) -> None:
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pct = (passed / total * 100.0) if total else 0.0

    def esc(s: str) -> str:
        return _html_escape.escape(s, quote=True)

    rows = []
    for r in results:
        cls = "pass" if r["passed"] else "fail"
        glyph = "✓ PASS" if r["passed"] else "✗ FAIL"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td><code>{esc(r['id'])}</code></td>"
            f"<td>{esc(r['category'])}</td>"
            f"<td><b>{glyph}</b></td>"
            f"<td style='text-align:right'>{r['duration_s']:.3f}s</td>"
            f"<td>{esc(r['message'])}</td>"
            f"</tr>"
        )

    detail_blocks = []
    spec_by_id = {t["id"]: t for t in spec["tests"]}
    for r in results:
        s = spec_by_id.get(r["id"], {})
        cls = "pass" if r["passed"] else "fail"
        tb_block = (
            f"<details><summary>traceback</summary><pre>{esc(r.get('traceback') or '')}</pre></details>"
            if (not r["passed"] and r.get("traceback"))
            else ""
        )
        detail_blocks.append(
            f"<section class='{cls}'>"
            f"<h3><code>{esc(r['id'])}</code> · {esc(r['category'])}</h3>"
            f"<p><b>Result:</b> {'PASS' if r['passed'] else 'FAIL'} in {r['duration_s']:.3f}s</p>"
            f"<p><b>Summary:</b> {esc(s.get('summary',''))}</p>"
            f"<p><b>Message:</b> {esc(r['message'])}</p>"
            f"<p class='notes'>{esc(s.get('notes','') or '')}</p>"
            f"{tb_block}"
            f"</section>"
        )

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Phase 3 Benchmark Results</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        max-width: 1100px; margin: 1.5em auto; padding: 0 1em; color: #1f2937; }}
 h1 {{ margin-bottom: 0.2em; }}
 .summary-bar {{ background: linear-gradient(to right, #10b981 {pct:.1f}%, #ef4444 {pct:.1f}%);
                 height: 14px; border-radius: 7px; margin: 12px 0 24px; }}
 table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5em; }}
 th, td {{ padding: 6px 10px; border-bottom: 1px solid #e5e7eb; font-size: 0.92rem; }}
 th {{ background: #f3f4f6; text-align: left; }}
 tr.pass td b {{ color: #047857; }}
 tr.fail td b {{ color: #b91c1c; }}
 section {{ border-left: 4px solid #d1d5db; padding: 4px 14px; margin: 12px 0; }}
 section.pass {{ border-left-color: #10b981; background: #f0fdf4; }}
 section.fail {{ border-left-color: #ef4444; background: #fef2f2; }}
 code {{ background: #1f2937; color: #e5e7eb; padding: 1px 6px; border-radius: 4px; font-size: 0.85em; }}
 p.notes {{ color: #6b7280; font-style: italic; }}
 pre {{ background: #f3f4f6; padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 0.8rem; }}
</style></head><body>
<h1>Phase 3 Acceptance Benchmark</h1>
<p>{esc(spec.get('description','').strip())}</p>
<p><b>Pass rate: {passed}/{total}</b> ({pct:.0f}%) · total wall {total_s:.2f}s · run {datetime.now(tz=timezone.utc).isoformat()}</p>
<div class='summary-bar'></div>
<table><thead><tr>
<th>id</th><th>category</th><th>result</th><th>time</th><th>message</th>
</tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<h2>Per-test detail</h2>
{''.join(detail_blocks)}
<hr><p style='color:#6b7280'>Run: <code>python tests/benchmark/run_phase3.py</code></p>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_benchmark() -> int:
    spec_path = Path(__file__).resolve().parent / "phase3.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    tests = spec.get("tests", [])
    total = len(tests)

    print(f"\nPhase 3 acceptance benchmark — {total} tests", flush=True)
    print(f"Description: {spec.get('description','').strip()}\n", flush=True)

    results: list[dict] = []
    bench_start = time.perf_counter()
    workspace = Path(tempfile.mkdtemp(prefix="hrag-phase3-bench-"))

    for idx, t in enumerate(tests, start=1):
        tid = t["id"]
        category = t["category"]
        runner_name = t["runner"]
        runner_fn = TESTS.get(runner_name)
        if runner_fn is None:
            print(
                f"[{idx}/{total} ✗] {tid} — FAIL in 0.000s "
                f"(no runner '{runner_name}' found)",
                flush=True,
            )
            results.append(
                {
                    "id": tid, "category": category, "passed": False,
                    "duration_s": 0.0,
                    "message": f"runner '{runner_name}' not in dispatch table",
                    "traceback": "",
                }
            )
            continue

        print(f"[{idx}/{total} ▶] {tid} — {category}: running...", flush=True)

        tmp = workspace / tid
        tmp.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        try:
            passed, msg = runner_fn(tmp)
            tb = ""
        except Exception as exc:  # noqa: BLE001
            passed = False
            msg = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
        dt = time.perf_counter() - start

        glyph = "✓" if passed else "✗"
        verdict = "PASS" if passed else "FAIL"
        print(
            f"[{idx}/{total} {glyph}] {tid} — {verdict} in {dt:.3f}s ({msg})",
            flush=True,
        )

        results.append(
            {
                "id": tid, "category": category, "passed": passed,
                "duration_s": dt, "message": msg, "traceback": tb,
            }
        )

    total_s = time.perf_counter() - bench_start
    passed_n = sum(1 for r in results if r["passed"])

    print("", flush=True)
    print(f"# Phase 3 results — {passed_n}/{total} passed in {total_s:.2f}s\n", flush=True)
    header = f"{'id':36s} {'category':18s} {'result':6s} {'time(s)':>8s}  message"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        verdict = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['id']:36s} {r['category']:18s} {verdict:6s} {r['duration_s']:>8.3f}  {r['message']}",
            flush=True,
        )
    print("", flush=True)

    out_md = Path(__file__).resolve().parent / "phase3_results.md"
    out_html = Path(__file__).resolve().parent / "phase3_results.html"
    _write_markdown(out_md, spec, results, total_s)
    _write_html(out_html, spec, results, total_s)
    print(f"Markdown report: {out_md}", flush=True)
    print(f"HTML report:     {out_html}", flush=True)

    # Cleanup tmp workspace.
    try:
        shutil.rmtree(workspace, ignore_errors=True)
    except Exception:
        pass

    return 0 if passed_n == total else 1


if __name__ == "__main__":
    sys.exit(run_benchmark())
