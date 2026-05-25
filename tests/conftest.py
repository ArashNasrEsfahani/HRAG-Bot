"""Shared pytest fixtures for the HRAG test suite.

Boot-strapping note
-------------------
``hrag/__init__.py`` eagerly imports ``hrag.orchestrator`` which pulls in
``hrag.retrieval.vector``, which does a bare ``import chromadb`` at module
scope.  Because Python always executes the parent package's ``__init__.py``
before any sub-module, *every* ``from hrag.X import Y`` triggers this chain.

We cannot modify source files, so we inject lightweight stub modules into
``sys.modules`` via ``pytest_configure`` — which runs before pytest collects
or imports any test file.  The stubs satisfy the ``import chromadb`` /
``import sentence_transformers`` / ``import ollama`` calls just enough for
all non-orchestrator tests to proceed.  The ``test_orchestrator.py`` module
guards itself with ``pytest.importorskip`` and will be skipped if the real
packages are absent.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Sequence

import pytest


# ---------------------------------------------------------------------------
# Minimal stub injection — runs before any test file is imported
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    """Return an empty module registered under *name* (and parent packages)."""
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []          # mark as a package so sub-imports work
    mod.__hrag_stub__ = True   # let tests detect a stubbed module and skip
    sys.modules[name] = mod
    return mod


def _ensure_stub(name: str) -> None:
    """Register a stub for *name* only if the real package is absent."""
    # Walk up the dotted name so sub-modules also get stubs.
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        partial = ".".join(parts[:i])
        if partial not in sys.modules:
            _make_stub(partial)


def pytest_configure(config) -> None:  # noqa: ANN001
    """Inject stub modules before any test file is collected/imported."""
    _install_chromadb_stub()
    _install_sentence_transformers_stub()
    _install_ollama_stub()


# --- chromadb stub ----------------------------------------------------------

def _install_chromadb_stub() -> None:
    if "chromadb" in sys.modules:
        return   # real package present; nothing to do

    # Top-level chromadb namespace
    cb = _make_stub("chromadb")

    # chromadb.config.Settings
    cfg_mod = _make_stub("chromadb.config")

    class Settings:  # noqa: D101
        def __init__(self, **kw): pass

    cfg_mod.Settings = Settings
    cb.config = cfg_mod

    # chromadb.PersistentClient stub
    class _FakeCollection:
        def upsert(self, **kw): pass
        def delete(self, **kw): pass
        def query(self, **kw):
            return {"ids": [[]], "distances": [[]]}

    class _FakePersistentClient:
        def __init__(self, path=None, settings=None): pass
        def get_or_create_collection(self, name=None, metadata=None):
            return _FakeCollection()

    cb.PersistentClient = _FakePersistentClient


# --- sentence_transformers stub ---------------------------------------------

def _install_sentence_transformers_stub() -> None:
    if "sentence_transformers" in sys.modules:
        return

    st = _make_stub("sentence_transformers")

    class SentenceTransformer:  # noqa: D101
        def __init__(self, model_name_or_path=None, **kw):
            self._dim = 384

        def encode(self, sentences, **kw):
            import numpy as np  # numpy IS installed per pyproject.toml
            return np.zeros((len(sentences), self._dim), dtype="float32")

        def get_sentence_embedding_dimension(self) -> int:
            return 384

    st.SentenceTransformer = SentenceTransformer

    class CrossEncoder:  # noqa: D101
        def __init__(self, model_name=None, device=None, max_length=512, **kw):
            pass

        def predict(self, pairs, **kw):
            import numpy as np
            return np.zeros(len(pairs), dtype="float32")

    st.CrossEncoder = CrossEncoder


# --- ollama stub ------------------------------------------------------------

def _install_ollama_stub() -> None:
    if "ollama" in sys.modules:
        return

    ol = _make_stub("ollama")

    class Client:  # noqa: D101
        def __init__(self, host=None): pass
        # Accept arbitrary kwargs so the stub never breaks when OllamaProvider
        # passes new top-level params (e.g. `think`, `stream`, `keep_alive`).
        def chat(self, model=None, messages=None, options=None, **kwargs):
            return {"message": {"content": "stub-response"}}
        def list(self, **kwargs):
            return {"models": []}

    ol.Client = Client


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path):
    """Fresh SQLite DB in a temp dir; schema bootstrapped; default user present."""
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None

    from hrag.db.connection import Database
    db_path = tmp_path / "test_store.sqlite"
    db = Database(db_path)
    db.init_schema()
    db.ensure_user("default")
    db.commit()
    yield db
    db.close()
    _conn_mod._db_singleton = None


# ---------------------------------------------------------------------------
# Sample file fixtures
# ---------------------------------------------------------------------------

_SAMPLE_MD = """\
# Introduction

This is the introduction section.
It spans multiple lines.

## Background

Some background text here.
More background details.

# Methods

We used a hierarchical approach.

## Data Collection

Data was collected from various sources.
"""

_SAMPLE_TXT = """\
This is a plain text file.
It has no headings at all.
Just paragraphs of content.

Second paragraph here.
With more sentences.
"""


@pytest.fixture()
def sample_md_path(tmp_path: Path) -> Path:
    """A markdown file with multiple sections."""
    p = tmp_path / "sample.md"
    p.write_text(_SAMPLE_MD, encoding="utf-8")
    return p


@pytest.fixture()
def sample_txt_path(tmp_path: Path) -> Path:
    """A plain text file with no headings."""
    p = tmp_path / "sample.txt"
    p.write_text(_SAMPLE_TXT, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_config(tmp_path: Path):
    """A Config with all storage paths pointing at tmp_path subdirs."""
    from hrag.config import Config, EmbeddingsConfig, LLMConfig, StorageConfig
    cfg = Config(
        llm=LLMConfig(provider="ollama", model="test-model"),
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


# ---------------------------------------------------------------------------
# Fake LLM fixture
# ---------------------------------------------------------------------------

class FakeLLM:
    """Deterministic LLM stub — no network required.

    Implements the same interface as LLMProvider without inheriting from it
    (avoids triggering the LLMProvider.__init__ requirement for LLMConfig).

    * Rerank prompts (contain "Score:" or "0-3"): rotating integer responses.
    * Other prompts: returns a canned answer string.
    """

    name = "fake"
    CANNED_ANSWER = "This is a canned answer from FakeLLM."

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._counter = 0
        # Rotating scores: 0,1,2,3,2,1,0,1,...
        self._score_cycle = [0, 1, 2, 3, 2, 1]

    # LLMProvider interface -----------------------------------------------

    def generate(self, request):
        from hrag.types import GenerationResponse
        prompt = " ".join(m.content for m in request.messages)
        self.calls.append(prompt)
        text = self._dispatch(prompt)
        return GenerationResponse(text=text, raw=None)

    def complete(
        self,
        prompt: str,
        system=None,
        temperature=None,
        max_tokens=None,
    ) -> str:
        """Convenience wrapper matching LLMProvider.complete()."""
        from hrag.types import GenerationRequest, Message
        msgs = []
        if system:
            msgs.append(Message(role="system", content=system))
        msgs.append(Message(role="user", content=prompt))
        resp = self.generate(GenerationRequest(messages=msgs))
        return resp.text

    # Internal ---------------------------------------------------------------

    def _dispatch(self, prompt: str) -> str:
        if "Score:" in prompt or "0-3" in prompt or "0, 1, 2, or 3" in prompt:
            score = self._score_cycle[self._counter % len(self._score_cycle)]
            self._counter += 1
            return str(score)
        # Intent classifier prompts (intent_classify.md). Return "factual" so
        # tests exercising the full pipeline still run end-to-end without
        # having to disable the intent gate explicitly.
        if "Intent Classification" in prompt or "Output (one word only)" in prompt:
            return "factual"
        return self.CANNED_ANSWER


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


# ---------------------------------------------------------------------------
# Fake Embedder fixture
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic hash-based embedder — no torch/models needed.

    Each text gets a 384-dim float vector derived from its SHA-256 hash.
    The same text always produces the same vector (deterministic).
    """

    name = "fake"
    _DIM = 384

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            floats: list[float] = []
            i = 0
            while len(floats) < self._DIM:
                floats.append((digest[i % len(digest)] / 127.5) - 1.0)
                i += 1
            result.append(floats[: self._DIM])
        return result

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self._DIM


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# ---------------------------------------------------------------------------
# Live-DB guard — keep web tests off the real data/store.sqlite
# ---------------------------------------------------------------------------
#
# Every test that imports ``hrag.web.app`` and resets ``_State.cfg = None``
# would otherwise cause the singleton to re-init via ``load_config()``, which
# reads ``config.yaml`` and points at ``data/store.sqlite`` / ``data/chroma``
# / ``data/kg`` — the user's REAL corpus + taxonomy. Tests that seed and
# clean up still leave behind dropped taxonomy assignments and broken vector
# state, which has cost the user their freshly-built tree more than once.
#
# This autouse fixture monkey-patches ``hrag.web.app._State`` and
# ``hrag.config.load_config`` so any singleton init inside a test points at
# pytest's per-test ``tmp_path`` instead. Tests that never touch the web app
# pay zero cost (the patch is a cheap attribute assignment).
@pytest.fixture(autouse=True)
def _isolate_web_app_state(tmp_path: Path, monkeypatch):
    """Redirect ``hrag.web.app._get_orch()`` to a throwaway tmp-path config
    so no test can ever mutate the user's live ``data/`` directory."""
    try:
        from hrag.web import app as _webapp
    except Exception:
        # Web layer not importable in this environment (e.g. fastapi missing).
        # The non-web tests don't need the guard.
        yield
        return

    from hrag.config import (
        Config, EmbeddingsConfig, LLMConfig, StorageConfig, load_config as _real_load,
    )

    def _tmp_config(*args, **kwargs):
        # Start from the real config (so feature toggles like
        # ``taxonomy.enabled`` match what the tests expect) but override
        # every disk path so nothing touches the user's live ``data/``.
        try:
            cfg = _real_load(*args, **kwargs)
        except Exception:
            # If config.yaml can't be loaded (CI without one), fall back to
            # an all-defaults Config — taxonomy.enabled stays False there,
            # which is fine; tests that need it on flip it explicitly.
            cfg = Config(
                llm=LLMConfig(provider="ollama", model="test-model"),
                embeddings=EmbeddingsConfig(
                    provider="sentence-transformers",
                    model="sentence-transformers/all-mpnet-base-v2",
                    dim=384,
                ),
            )
        # Force every storage path under tmp_path. This is the SAFETY GUARD —
        # under no circumstance should a test ever read or write the user's
        # real data/store.sqlite, data/chroma, or data/kg.
        cfg.storage = StorageConfig(
            sqlite_path=str(tmp_path / "store.sqlite"),
            chroma_path=str(tmp_path / "chroma"),
            kg_path=str(tmp_path / "kg"),
            data_root=str(tmp_path / "data"),
        )
        cfg.project_root = tmp_path
        # Defensive: enable taxonomy so the existing web-taxonomy tests pass
        # even if the user's config.yaml has it switched off.
        cfg.taxonomy.enabled = True
        return cfg

    # Reset any cached singleton from a previous test and force the next
    # ``_get_orch()`` call to build against the tmp-path config.
    with _webapp._State.lock:
        _webapp._State.cfg = None
        _webapp._State.orch = None
    # Patch the module-level loader the web app imports. We patch on the
    # ``hrag.config`` module AND on the bound name inside ``hrag.web.app``
    # if it imported it directly, so we cover both lookup paths.
    monkeypatch.setattr("hrag.config.load_config", _tmp_config)
    try:
        # Some files do ``from hrag.config import load_config``; rebind there too.
        monkeypatch.setattr(_webapp, "load_config", _tmp_config, raising=False)
    except Exception:
        pass

    yield

    # After-test teardown: drop the singleton so the next test gets a fresh
    # tmp_path orchestrator (or so a manual ``hrag web`` run uses real config).
    try:
        with _webapp._State.lock:
            if _webapp._State.orch is not None:
                try:
                    _webapp._State.orch.close()
                except Exception:
                    pass
            _webapp._State.cfg = None
            _webapp._State.orch = None
    except Exception:
        pass
