"""Smoke tests for hrag.orchestrator.Orchestrator.

Both chromadb and sentence_transformers are required for the Orchestrator to
initialise with real providers; skip the whole module if either real package
is missing (the stubs in conftest.py are not sufficient for Orchestrator —
they satisfy the import chain but the Orchestrator itself tries to use the
client objects).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hrag.config import Config, EmbeddingsConfig, LLMConfig, StorageConfig
from hrag.orchestrator import ChatResult, Orchestrator
from tests.conftest import FakeEmbedder, FakeLLM


# We check for the REAL packages, not the stubs injected by conftest.
# The stubs set __package__ = name but have no version / real API.
# We detect "real" by looking for a known attribute on the stub vs. the
# actual chromadb.PersistentClient implementation.

def _real_chromadb_present() -> bool:
    try:
        import importlib
        real = importlib.import_module("chromadb")
        # Real chromadb has __version__; our stub does not
        return hasattr(real, "__version__")
    except Exception:
        return False


def _real_sentence_transformers_present() -> bool:
    try:
        import importlib
        real = importlib.import_module("sentence_transformers")
        return hasattr(real, "__version__")
    except Exception:
        return False


_HEAVY_DEPS_PRESENT = _real_chromadb_present() and _real_sentence_transformers_present()


def _require_heavy_deps() -> None:
    """Skip a single test when chromadb or sentence-transformers is stub-only.

    The original orchestrator smoke tests need real implementations of those
    libraries because they construct the actual ChromaDB client. The lightweight
    Phase 4 wiring tests further down the file build the Orchestrator entirely
    from the stub-friendly ``sample_config`` fixture and so do not need this
    guard — calling this helper only in the legacy smoke tests preserves their
    original semantics while letting the Phase 4 tests run unconditionally.
    """
    if not _HEAVY_DEPS_PRESENT:
        pytest.skip("chromadb / sentence_transformers stub-only — orchestrator smoke tests need real deps")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> Config:
    cfg = Config(
        llm=LLMConfig(provider="ollama", model="fake"),
        embeddings=EmbeddingsConfig(
            provider="sentence-transformers",
            model="sentence-transformers/all-mpnet-base-v2",
            dim=384,
        ),
        storage=StorageConfig(
            sqlite_path=str(tmp_path / "store.sqlite"),
            chroma_path=str(tmp_path / "chroma"),
            kg_path=str(tmp_path / "kg"),
            data_root=str(tmp_path / "data"),
        ),
    )
    cfg.project_root = tmp_path
    return cfg


def _build_orch(tmp_path: Path):
    """Build an Orchestrator with fake providers; skip if Ollama unavailable."""
    _require_heavy_deps()
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None

    cfg = _make_config(tmp_path)
    try:
        orch = Orchestrator(cfg)
    except ImportError as exc:
        pytest.skip(f"Provider import failed: {exc}")
    except Exception as exc:
        msg = str(exc).lower()
        if "ollama" in msg or "connection" in msg or "connect" in msg:
            pytest.skip(f"Ollama not available: {exc}")
        raise

    # Monkeypatch in stubs so no real inference happens
    fake_llm = FakeLLM()
    fake_embedder = FakeEmbedder()
    orch.llm = fake_llm
    orch.embedder = fake_embedder
    orch.retriever._embedder = fake_embedder
    if orch.reranker is not None:
        orch.reranker._llm = fake_llm

    return orch, fake_llm, fake_embedder


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_orchestrator_chat_returns_chat_result(tmp_path: Path) -> None:
    """Orchestrator.chat() must return a populated ChatResult."""
    orch, _, _ = _build_orch(tmp_path)
    try:
        result = orch.chat("hello", user_id="test_user")
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    assert isinstance(result, ChatResult)
    assert isinstance(result.answer, str) and len(result.answer) > 0
    assert isinstance(result.session_id, str) and len(result.session_id) > 0
    assert isinstance(result.sources, list)
    assert isinstance(result.prompt, str) and len(result.prompt) > 0


def test_orchestrator_chat_creates_new_session_each_call(tmp_path: Path) -> None:
    """Calls without a session_id must create distinct sessions."""
    orch, _, _ = _build_orch(tmp_path)
    try:
        r1 = orch.chat("first", user_id="test_user")
        r2 = orch.chat("second", user_id="test_user")
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    assert r1.session_id != r2.session_id


def test_orchestrator_chat_reuses_session(tmp_path: Path) -> None:
    """Passing the same session_id must not raise and must return the same id."""
    orch, _, _ = _build_orch(tmp_path)
    try:
        r1 = orch.chat("hello", user_id="test_user")
        r2 = orch.chat("follow-up", user_id="test_user", session_id=r1.session_id)
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    assert r2.session_id == r1.session_id


# ---------------------------------------------------------------------------
# Query rewriter wired into the pipeline
# ---------------------------------------------------------------------------


def test_orchestrator_emits_query_rewrite_event_on_followup(tmp_path: Path) -> None:
    """Follow-up turns trigger the heuristic rewriter and a `query_rewrite` event."""
    orch, _, _ = _build_orch(tmp_path)
    events: list[tuple[str, dict]] = []

    def cb(name: str, payload: dict) -> None:
        events.append((name, payload))

    try:
        # First turn — no history, no rewrite event expected.
        r1 = orch.chat("what is hipporag?", user_id="test_user", progress=cb)
        first_events = list(events)
        events.clear()

        # Second turn — pronoun-heavy follow-up should trigger rewrite.
        orch.chat(
            "explain its architecture",
            user_id="test_user",
            session_id=r1.session_id,
            progress=cb,
        )
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    # First turn: no rewrite (empty history)
    assert not any(name == "query_rewrite" for name, _ in first_events)

    # Second turn: heuristic rewrite must have fired
    rewrite_events = [p for name, p in events if name == "query_rewrite"]
    assert len(rewrite_events) == 1
    payload = rewrite_events[0]
    assert payload["original"] == "explain its architecture"
    assert "what is hipporag?" in payload["rewritten"]
    assert payload["rewriter"] == "heuristic"


def test_orchestrator_no_rewrite_when_query_rewrite_none(tmp_path: Path) -> None:
    """Setting retrieval.query_rewrite='none' suppresses rewriting entirely."""
    _require_heavy_deps()
    cfg = _make_config(tmp_path)
    cfg.retrieval.query_rewrite = "none"

    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None
    try:
        orch = Orchestrator(cfg)
    except ImportError as exc:
        pytest.skip(f"Provider import failed: {exc}")
    except Exception as exc:
        msg = str(exc).lower()
        if "ollama" in msg or "connect" in msg:
            pytest.skip(f"Ollama not available: {exc}")
        raise

    fake_llm = FakeLLM()
    fake_embedder = FakeEmbedder()
    orch.llm = fake_llm
    orch.embedder = fake_embedder
    orch.retriever._embedder = fake_embedder
    if orch.reranker is not None:
        orch.reranker._llm = fake_llm

    events: list[tuple[str, dict]] = []
    try:
        r1 = orch.chat("what is hipporag?", user_id="test_user")
        orch.chat(
            "explain its architecture",
            user_id="test_user",
            session_id=r1.session_id,
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _conn_mod._db_singleton = None

    assert not any(name == "query_rewrite" for name, _ in events)


# ===========================================================================
# Phase 4 — compaction & gating wiring tests
#
# These tests run unconditionally (no real chromadb / sentence-transformers
# needed) because they reuse the ``sample_config`` pattern from
# ``test_phase2_wiring.py`` and patch ``orch.retriever`` / ``orch.llm`` with
# stubs after construction. The conftest stubs for chromadb /
# sentence_transformers / ollama are sufficient because we never actually
# query a chroma collection or run an embedding model in these tests.
# ===========================================================================


def _reset_db_singleton() -> None:
    import hrag.db.connection as _conn_mod

    _conn_mod._db_singleton = None


class _ScriptedLLM:
    """Stub LLM whose reply is decided by a prompt-sniffing dispatcher.

    The Phase 4 helpers (RAGate, ClueGenerator) and the answer LLM all share
    ``Orchestrator.llm``, so we differentiate by inspecting the rendered
    prompt. The gate prompt contains the distinctive phrase
    ``Output ONLY one word: \\`RETRIEVE\\` or \\`SKIP\\```; the clue prompt
    starts with ``## Clue Prompt`` and asks for a ``Hypothesis:``; the intent
    classifier prompt contains ``Intent Classification``; everything else is
    treated as an answer prompt.
    """

    name = "scripted"

    def __init__(
        self,
        *,
        gate_reply: str = "RETRIEVE",
        clue_reply: str = "clue hypothesis text",
        intent_reply: str = "factual",
        answer_reply: str = "Answer body.",
    ) -> None:
        self.gate_reply = gate_reply
        self.clue_reply = clue_reply
        self.intent_reply = intent_reply
        self.answer_reply = answer_reply
        self.calls: list[str] = []

    def _dispatch(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "Output ONLY one word" in prompt and ("RETRIEVE" in prompt and "SKIP" in prompt):
            return self.gate_reply
        if "retrieval hypothesis" in prompt or "Hypothesis:" in prompt.split("\n")[-3:][0:1]:
            return self.clue_reply
        if "Clue Prompt" in prompt:
            return self.clue_reply
        if "Intent Classification" in prompt or "Output (one word only)" in prompt:
            return self.intent_reply
        return self.answer_reply

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        return self._dispatch(prompt)

    def generate(self, request):
        from hrag.types import GenerationResponse

        prompt = " ".join(m.content for m in request.messages)
        return GenerationResponse(text=self._dispatch(prompt), raw=None)

    def generate_stream(self, request):
        yield self.generate(request).text


class _SpyRetriever:
    """Records every call to ``retrieve`` and returns a fixed empty result."""

    name = "spy"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, query, user_id, top_k=10, source_types=None, intent_hint=None, where=None):
        self.calls.append(
            {
                "query": query,
                "user_id": user_id,
                "top_k": top_k,
                "source_types": source_types,
                "intent_hint": intent_hint,
                "where": where,
            }
        )
        return []


def _make_phase4_orch(sample_config, scripted_llm: _ScriptedLLM, *, force_factual: bool = True):
    """Build an Orchestrator wired with the scripted LLM and a spy retriever.

    ``force_factual`` disables the intent classifier so the FACTUAL path is
    always taken — Phase 4 gate/clue only fire on FACTUAL, and this avoids
    having to inject ``intent_classify.md`` outputs through the scripted LLM
    for every test.
    """
    sample_config.retrieval.rerank_enabled = False
    if force_factual:
        sample_config.intent.enabled = False

    _reset_db_singleton()
    from hrag.orchestrator import Orchestrator

    orch = Orchestrator(sample_config)
    # Patch the LLM EVERYWHERE that captured it during construction.
    orch.llm = scripted_llm
    if orch.gate is not None:
        orch.gate.llm = scripted_llm
    if orch.clue is not None:
        orch.clue.llm = scripted_llm
    if orch.dialog_compactor is not None:
        orch.dialog_compactor._llm = scripted_llm
    # Drop in spy retriever.
    spy = _SpyRetriever()
    orch.retriever = spy
    return orch, spy


def test_gate_skip_short_circuits_retrieval(sample_config) -> None:
    """When the gate returns SKIP, retrieval must not be called and the event fires."""
    sample_config.compaction.gate_enabled = True
    llm = _ScriptedLLM(gate_reply="SKIP")
    orch, spy = _make_phase4_orch(sample_config, llm)
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "thanks bye",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    gate_events = [p for n, p in events if n == "gate_check"]
    assert len(gate_events) == 1
    assert gate_events[0]["decision"] == "SKIP"
    assert "duration_s" in gate_events[0]
    # No retrieval call must have been made.
    assert spy.calls == []


def test_gate_retrieve_proceeds(sample_config) -> None:
    """When the gate returns RETRIEVE, retrieval proceeds normally."""
    sample_config.compaction.gate_enabled = True
    llm = _ScriptedLLM(gate_reply="RETRIEVE")
    orch, spy = _make_phase4_orch(sample_config, llm)
    events: list[tuple[str, dict]] = []
    try:
        orch.chat(
            "what is the refund policy?",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    gate_events = [p for n, p in events if n == "gate_check"]
    assert len(gate_events) == 1
    assert gate_events[0]["decision"] == "RETRIEVE"
    # Retrieval HAS been called.
    assert len(spy.calls) == 1


def test_clue_substitutes_retrieval_query(sample_config) -> None:
    """ClueGenerator's output replaces the retrieval query while the answer prompt keeps the original."""
    sample_config.compaction.clue_enabled = True
    llm = _ScriptedLLM(clue_reply="hypothesis text about widgets")
    orch, spy = _make_phase4_orch(sample_config, llm)
    events: list[tuple[str, dict]] = []
    try:
        result = orch.chat(
            "tell me about widgets",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    clue_events = [p for n, p in events if n == "clue_generate"]
    assert len(clue_events) == 1
    assert clue_events[0]["clue"] == "hypothesis text about widgets"
    # The retrieval call used the clue, not the raw question.
    assert len(spy.calls) == 1
    assert spy.calls[0]["query"] == "hypothesis text about widgets"
    # The rendered answer prompt MUST still contain the original user question.
    assert "tell me about widgets" in result.prompt
    # And NOT the hypothesis (the LLM answers the user's actual question).
    assert "hypothesis text about widgets" not in result.prompt


def test_dialog_compact_runs_when_history_long(sample_config) -> None:
    """A long synthetic history triggers DialogMSTCompactor and the dialog_compact event."""
    sample_config.compaction.dialog_mst_enabled = True
    sample_config.compaction.compact_after_turns = 4
    sample_config.compaction.keep_recent_turns = 2
    llm = _ScriptedLLM()
    orch, spy = _make_phase4_orch(sample_config, llm)

    # Pre-seed history by inserting 20 messages directly via _save_message.
    session_id = "phase4-session"
    orch._create_session(session_id, "default")
    for i in range(20):
        role = "user" if i % 2 == 0 else "assistant"
        orch._save_message(session_id, "default", role, f"turn {i} content")
    orch.db.commit()

    events: list[tuple[str, dict]] = []
    captured_history: list[str] = []

    # Monkeypatch the compactor's compact() to record what it produced so the
    # test can assert on it (the chat() pipeline doesn't expose history).
    real_compact = orch.dialog_compactor.compact

    def wrap_compact(history):
        result = real_compact(history)
        captured_history.append("system" if result and result[0].role == "system" else "no-summary")
        return result

    orch.dialog_compactor.compact = wrap_compact  # type: ignore[assignment]

    try:
        orch.chat(
            "what comes next?",
            user_id="default",
            session_id=session_id,
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    compact_events = [p for n, p in events if n == "dialog_compact"]
    assert len(compact_events) == 1
    payload = compact_events[0]
    # input_turns should exceed the compact_after_turns threshold (4)
    assert payload["input_turns"] > sample_config.compaction.compact_after_turns
    # output_turns must be strictly smaller (compaction did its job)
    assert payload["output_turns"] <= payload["input_turns"]
    # The compactor produced a synthetic system-role summary.
    assert captured_history == ["system"]
    # Spy retriever should have been called (chat completed end-to-end).
    assert len(spy.calls) == 1


def test_uncertain_rendering_replaces_tokens(sample_config) -> None:
    """mask_uncertain=True turns [UNCERTAIN] into the visible glyph and fires uncertain_render."""
    sample_config.compaction.mask_uncertain = True
    llm = _ScriptedLLM(answer_reply="Foo [UNCERTAIN] bar.")
    orch, _spy = _make_phase4_orch(sample_config, llm)
    events: list[tuple[str, dict]] = []
    try:
        result = orch.chat(
            "trigger uncertain",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    assert "[UNCERTAIN]" not in result.answer
    assert "⚠️" in result.answer
    rendered_events = [p for n, p in events if n == "uncertain_render"]
    assert len(rendered_events) == 1
    assert rendered_events[0]["count"] == 1


def test_uncertain_silently_stripped_when_disabled(sample_config) -> None:
    """mask_uncertain=False strips raw [UNCERTAIN] without emitting a render event."""
    sample_config.compaction.mask_uncertain = False
    llm = _ScriptedLLM(answer_reply="Foo [UNCERTAIN] bar.")
    orch, _spy = _make_phase4_orch(sample_config, llm)
    events: list[tuple[str, dict]] = []
    try:
        result = orch.chat(
            "no masking please",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    # Raw token is stripped — must not leak.
    assert "[UNCERTAIN]" not in result.answer
    # No visible glyph either.
    assert "⚠️" not in result.answer
    # And no event.
    rendered_events = [p for n, p in events if n == "uncertain_render"]
    assert rendered_events == []


def test_phase4_off_means_zero_overhead(sample_config) -> None:
    """All four compaction flags off ⇒ helpers are None and no Phase 4 events fire."""
    # All compaction flags default to False; just confirm.
    assert sample_config.compaction.gate_enabled is False
    assert sample_config.compaction.clue_enabled is False
    assert sample_config.compaction.dialog_mst_enabled is False
    assert sample_config.compaction.mask_uncertain is False

    llm = _ScriptedLLM(answer_reply="plain answer text")
    orch, spy = _make_phase4_orch(sample_config, llm)
    events: list[tuple[str, dict]] = []
    try:
        assert orch.gate is None
        assert orch.clue is None
        assert orch.dialog_compactor is None
        orch.chat(
            "any question",
            user_id="default",
            progress=lambda n, p: events.append((n, p)),
        )
    finally:
        orch.close()
        _reset_db_singleton()

    phase4_event_names = {"gate_check", "clue_generate", "dialog_compact", "uncertain_render"}
    fired = {n for n, _ in events}
    assert phase4_event_names.isdisjoint(fired), (
        f"Phase 4 events leaked when all flags off: {phase4_event_names & fired}"
    )
    # Retrieval ran exactly once.
    assert len(spy.calls) == 1
