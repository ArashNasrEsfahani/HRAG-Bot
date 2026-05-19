"""Phase 8.2 — PERSONAL turns surface all memories, not only those under
descended leaves, and the prompt does not copy the fallback example.

Two bugs were diagnosed from a real session:

1. ``answer_personal.md`` carried the literal fallback string
   ``"I don't have anything on file about you yet — ..."`` as an example
   inside the instructions block. Small Gemma-family models emit it
   verbatim on every PERSONAL turn, even when memories ARE rendered into
   ``{retrieved_passages}``. Fix: rewrite the main template without the
   example string and move the "nothing yet" path to a separate sibling
   template that fires only when there is genuinely nothing on file.

2. ``TaxonomyRetriever.retrieve()`` passes ``doc_ids=leaf_doc_ids`` to the
   vector store — only chunks belonging to leaf-picked documents are
   scanned. For episodic memories this is wrong: memories filed under a
   different leaf than the one the query descended to get silently
   excluded. Fix: when ``source_types`` is None or includes ``"episodic"``,
   run a SECOND vector query with no doc_id filter (and the
   ``source_types=["episodic"]`` scope) and merge into the result set.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import pytest

from hrag.config import TaxonomyConfig
from hrag.db.connection import Database
from hrag.intent import Intent, IntentVerdict
from hrag.retrieval.taxonomy import TaxonomyRetriever
from hrag.taxonomy.store import TaxonomyStore
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Local test doubles
# ---------------------------------------------------------------------------


class _DictEmbedder:
    """Deterministic 8-dim embedder with a name → vector override map.

    Mirrors the shape used by ``tests/test_taxonomy_retriever_caps.py``.
    """

    name = "dict"
    _DIM = 8

    def __init__(self, mapping=None) -> None:
        self._map = dict(mapping or {})

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        if text in self._map:
            return list(self._map[text])
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b / 255.0) * 2.0 - 1.0 for b in digest[: self._DIM]]

    @property
    def dim(self) -> int:
        return self._DIM


class _DualScopeVectorStore:
    """Vector store fake that distinguishes leaf-scoped vs global episodic queries.

    The retriever calls ``query()`` twice on the new path: first with the
    leaf-doc allow-list (returns the document chunk pool), then with
    ``source_types=["episodic"]`` and ``doc_ids=None`` (returns the global
    episodic pool). Both calls are captured so the test can assert the
    second call happened with the correct args.
    """

    def __init__(
        self,
        leaf_pool: dict[str, float] | None = None,
        episodic_pool: dict[str, float] | None = None,
    ) -> None:
        self.leaf_pool = leaf_pool or {}
        self.episodic_pool = episodic_pool or {}
        self.calls: list[dict] = []

    def query(
        self,
        *,
        user_id,
        query_embedding,
        top_k,
        source_types=None,
        doc_ids=None,
        where=None,
    ):
        self.calls.append({
            "user_id": user_id,
            "top_k": top_k,
            "source_types": list(source_types) if source_types else None,
            "doc_ids": list(doc_ids) if doc_ids else None,
            "where": where,
        })
        # Second call: source_types == ["episodic"] AND doc_ids is None →
        # return the global episodic pool.
        if (
            source_types
            and list(source_types) == ["episodic"]
            and not doc_ids
        ):
            return sorted(
                self.episodic_pool.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
        # First call: leaf-doc scope. Return the leaf pool.
        return sorted(
            self.leaf_pool.items(), key=lambda x: x[1], reverse=True
        )[:top_k]


class _DBHydrator(Database):
    """Database that pre-seeds chunk rows so ``_hydrate`` can resolve them.

    We can use the real :class:`Database` — the retriever's ``_hydrate``
    SELECTs from the ``chunks`` table by chunk_id. Inserting test rows
    keeps the contract identical to production hydration.
    """


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.sqlite"
    d = Database(path)
    d.init_schema()
    d.ensure_user("u1")
    yield d
    d.close()


def _insert_chunk(
    db: Database,
    chunk_id: str,
    doc_id: str,
    source_type: str = "document",
    text: str = "body",
) -> None:
    """Insert a minimal chunk row + parent doc row."""
    # Parent document row (FK target). Idempotent.
    try:
        db.execute(
            "INSERT INTO documents(doc_id, user_id, source_path, title, source_type) "
            "VALUES (?,?,?,?,?)",
            (doc_id, "u1", f"/tmp/{doc_id}.txt", doc_id, source_type),
        )
    except Exception:  # noqa: BLE001
        pass
    db.execute(
        "INSERT INTO chunks("
        "  chunk_id, doc_id, user_id, text, title, section, subsection,"
        "  chunk_index, token_count, source_type, excluded, metadata"
        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (chunk_id, doc_id, "u1", text, doc_id, "", "", 0, 8, source_type, 0, "{}"),
    )


# ---------------------------------------------------------------------------
# Bug 2 — TaxonomyRetriever must surface episodic memories outside leaf scope
# ---------------------------------------------------------------------------


def test_taxonomy_retriever_includes_global_episodic(db) -> None:
    """One episodic memory lives under a leaf the beam descended into; a
    SECOND episodic memory lives under a SIBLING leaf that the beam did
    NOT pick. Both must surface in the retriever output.
    """
    target_vec = [1.0] + [0.0] * 7
    other_vec  = [0.0, 1.0] + [0.0] * 6  # orthogonal — beam will not pick

    emb = _DictEmbedder({
        "query":    target_vec,
        "picked":   target_vec,
        "unpicked": other_vec,
    })

    store = TaxonomyStore(db, emb)
    root = store.ensure_root("u1")
    picked_leaf = store.add_node(
        "u1", root.node_id, "Picked", is_leaf=True,
    )
    unpicked_leaf = store.add_node(
        "u1", root.node_id, "Unpicked", is_leaf=True,
    )

    # Place one episodic "doc" under each leaf. (Memories are filed under
    # the taxonomy via the same assign_doc mechanism as regular docs.)
    db.execute(
        "INSERT INTO documents(doc_id, user_id, source_path, title, source_type) "
        "VALUES (?,?,?,?,?)",
        ("ep_doc_under_picked", "u1", "/tmp/p.md", "p", "episodic"),
    )
    db.execute(
        "INSERT INTO documents(doc_id, user_id, source_path, title, source_type) "
        "VALUES (?,?,?,?,?)",
        ("ep_doc_under_unpicked", "u1", "/tmp/u.md", "u", "episodic"),
    )
    store.upsert_doc_meta("u1", "ep_doc_under_picked", "picked", target_vec)
    store.upsert_doc_meta("u1", "ep_doc_under_unpicked", "unpicked", other_vec)
    store.assign_doc("u1", "ep_doc_under_picked", picked_leaf.node_id)
    store.assign_doc("u1", "ep_doc_under_unpicked", unpicked_leaf.node_id)

    # Seed the two chunks that the vector store will return.
    _insert_chunk(db, "c_in_leaf",  "ep_doc_under_picked",   source_type="episodic")
    _insert_chunk(db, "c_outside_leaf", "ep_doc_under_unpicked", source_type="episodic")
    db.commit()
    store.recompute_all_centroids("u1")

    vstore = _DualScopeVectorStore(
        leaf_pool={"c_in_leaf": 0.91},          # what the leaf-scoped scan finds
        episodic_pool={                          # what the global scan finds
            "c_in_leaf": 0.91,
            "c_outside_leaf": 0.83,
        },
    )

    cfg = TaxonomyConfig(
        enabled=True,
        beam_width=1,                  # force a single leaf descent
        max_depth=4,
        min_node_score=-1.0,
        beam_dominance_gap=0.0,
        min_top_score_floor=0.0,
        max_docs_pct=1.0,
    )

    retriever = TaxonomyRetriever(
        db=db,
        vector_store=vstore,           # type: ignore[arg-type]
        embedder=emb,                  # type: ignore[arg-type]
        taxonomy_store=store,
        cfg=cfg,
        fallback=None,                 # type: ignore[arg-type] — not used
    )

    results = retriever.retrieve(
        "query", "u1", top_k=10,
        source_types=["document", "episodic"],
    )
    chunk_ids = {r.chunk.chunk_id for r in results}

    # Both episodic memories surface — the under-leaf one via the
    # leaf-scoped scan, the outside-leaf one via the new global scan.
    assert "c_in_leaf" in chunk_ids
    assert "c_outside_leaf" in chunk_ids

    # The retriever issued the expected pair of queries.
    assert len(vstore.calls) == 2
    # First call is leaf-scoped: doc_ids is non-empty.
    assert vstore.calls[0]["doc_ids"]
    # Second call is the global episodic scan: doc_ids=None,
    # source_types=["episodic"].
    assert vstore.calls[1]["doc_ids"] is None
    assert vstore.calls[1]["source_types"] == ["episodic"]

    # The descend trace gets an explanatory note for the GUI panel.
    trace = retriever.describe_last_descend(user_id="u1")
    assert "note" in trace
    assert "episodic" in trace["note"]


def test_taxonomy_retriever_skips_global_episodic_when_excluded(db) -> None:
    """When ``source_types=["document"]`` the global episodic scan is
    skipped entirely — the second query is never issued."""
    target_vec = [1.0] + [0.0] * 7
    emb = _DictEmbedder({"query": target_vec, "k": target_vec})

    store = TaxonomyStore(db, emb)
    root = store.ensure_root("u1")
    leaf = store.add_node("u1", root.node_id, "L", is_leaf=True)

    db.execute(
        "INSERT INTO documents(doc_id, user_id, source_path, title) "
        "VALUES (?,?,?,?)",
        ("d1", "u1", "/tmp/d1.txt", "d1"),
    )
    store.upsert_doc_meta("u1", "d1", "k", target_vec)
    store.assign_doc("u1", "d1", leaf.node_id)
    _insert_chunk(db, "c1", "d1", source_type="document")
    db.commit()
    store.recompute_all_centroids("u1")

    vstore = _DualScopeVectorStore(
        leaf_pool={"c1": 0.9},
        episodic_pool={"c_ghost": 0.99},
    )
    cfg = TaxonomyConfig(
        enabled=True, beam_width=1, max_depth=4,
        min_node_score=-1.0, beam_dominance_gap=0.0,
        min_top_score_floor=0.0, max_docs_pct=1.0,
    )
    retriever = TaxonomyRetriever(
        db=db, vector_store=vstore, embedder=emb,    # type: ignore[arg-type]
        taxonomy_store=store, cfg=cfg, fallback=None,  # type: ignore[arg-type]
    )

    retriever.retrieve("query", "u1", top_k=10, source_types=["document"])

    # Exactly one vector-store call — no global episodic scan.
    assert len(vstore.calls) == 1
    assert vstore.calls[0]["source_types"] == ["document"]


# ---------------------------------------------------------------------------
# Bug 1 — PERSONAL prompt selection + no copy-paste example
# ---------------------------------------------------------------------------


def test_answer_personal_md_does_not_copy_paste_example() -> None:
    """The fallback example string must not live inside the main personal
    template — small Gemma models copy-paste it verbatim."""
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "hrag" / "prompts" / "answer_personal.md"
    )
    text = prompt_path.read_text(encoding="utf-8")
    assert "I don't have anything on file" not in text
    assert "feel free to share something" not in text


def test_answer_personal_empty_template_exists_and_renders() -> None:
    """The sibling 'nothing yet' template exists and renders without
    requiring the user_profile / retrieved_passages placeholders."""
    from hrag.prompts_registry import PromptRegistry
    prompts_dir = (
        Path(__file__).resolve().parents[1]
        / "src" / "hrag" / "prompts"
    )
    registry = PromptRegistry(prompts_dir)
    rendered = registry.render_personal_empty(
        conversation_history="User: hi\nAssistant: hey",
        question="do you know me?",
    )
    # Sentinel: the new template's distinctive phrase.
    assert "no memories yet" in rendered.lower()
    # And it incorporates the runtime args.
    assert "do you know me?" in rendered


# ---------------------------------------------------------------------------
# Orchestrator-level dispatch tests
# ---------------------------------------------------------------------------


class _ScriptedClassifier:
    def __init__(self, intent: Intent) -> None:
        self._intent = intent

    def classify(self, text: str, **kwargs) -> IntentVerdict:
        return IntentVerdict(
            intent=self._intent,
            confidence=1.0,
            source="test",
            raw_label=self._intent.value,
        )


class _RecordingRetriever:
    name = "recording"

    def __init__(self, results: Optional[list[RetrievalResult]] = None) -> None:
        self._results = results or []
        self.calls: list[dict] = []

    def retrieve(
        self, query, user_id, top_k=10,
        source_types=None, intent_hint=None, where=None,
    ):
        self.calls.append({"query": query, "source_types": source_types})
        return list(self._results)


class _PromptCapturingLLM:
    """LLM stub that records every prompt it sees and returns a canned reply."""

    name = "prompt-capture"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, request):
        from hrag.types import GenerationResponse
        text = " ".join(m.content for m in request.messages)
        self.prompts.append(text)
        if "Intent Classification" in text or "Output (one word only)" in text:
            return GenerationResponse(text="personal", raw=None)
        if "Score:" in text or "0-3" in text or "0, 1, 2, or 3" in text:
            return GenerationResponse(text="2", raw=None)
        return GenerationResponse(text="ok, got it.", raw=None)

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        from hrag.types import GenerationRequest, Message
        msgs = []
        if system:
            msgs.append(Message(role="system", content=system))
        msgs.append(Message(role="user", content=prompt))
        return self.generate(GenerationRequest(messages=msgs)).text

    def generate_stream(self, request):
        resp = self.generate(request)
        yield resp.text


def _make_orch_with_retriever(sample_config, intent, results):
    """Build an Orchestrator with the scripted classifier + recording retriever."""
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None
    sample_config.retrieval.rerank_enabled = False

    from hrag.orchestrator import Orchestrator
    orch = Orchestrator(sample_config)

    capture_llm = _PromptCapturingLLM()
    orch.llm = capture_llm  # type: ignore[assignment]
    if getattr(orch, "gate", None) is not None:
        orch.gate.llm = capture_llm
    if getattr(orch, "clue", None) is not None:
        orch.clue.llm = capture_llm

    orch.intent_classifier = _ScriptedClassifier(intent)  # type: ignore[assignment]
    orch.retriever = _RecordingRetriever(results=results)  # type: ignore[assignment]
    return orch, capture_llm


def _chunk(chunk_id: str, source_type: str = "episodic", text: str = "Arash is the user.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        user_id="default",
        text=text,
        embedding_text=text,
        source_type=source_type,
    )


def _result(chunk_id: str, score: float, source_type: str = "episodic") -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(chunk_id, source_type), score=score)


def test_personal_empty_uses_empty_template(sample_config) -> None:
    """Zero memories + empty profile → the orchestrator dispatches the new
    sibling 'nothing yet' template, not the main personal one."""
    orch, llm = _make_orch_with_retriever(
        sample_config, Intent.PERSONAL, results=[],
    )
    try:
        orch.chat("do you know me?", user_id="default")
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    # The answer-generation prompt is the last non-intent-classification
    # call. Find it by looking at what was passed through.
    gen_prompts = [
        p for p in llm.prompts
        if "Intent Classification" not in p
        and "Score:" not in p
        and "0-3" not in p
    ]
    assert gen_prompts, "no generation prompt captured"
    answer_prompt = gen_prompts[-1]
    # Sentinel from the new empty template.
    assert "no memories yet" in answer_prompt.lower()
    # AND the main template's instructions block must NOT be in there.
    assert "What you've remembered about the user" not in answer_prompt


def test_personal_full_uses_main_template(sample_config) -> None:
    """One memory hit + a strongly-reranked document → the orchestrator
    dispatches the main personal template (the memory-led path requires
    NO strong doc), and the rendered prompt does NOT contain the
    copy-paste bait string."""
    # Phase 8.3: include a doc-source result with a positive rerank score
    # so the dispatcher skips the memory-led path and lands on the main
    # PERSONAL template (which this test pins).
    ep = _result("c1", 0.9)
    doc = _result("c2", 0.8, source_type="document")
    doc.rerank_score = 1.5  # positive → "strong doc"
    orch, llm = _make_orch_with_retriever(
        sample_config, Intent.PERSONAL,
        results=[ep, doc],
    )
    try:
        orch.chat("do you know me?", user_id="default")
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod
        _conn_mod._db_singleton = None

    gen_prompts = [
        p for p in llm.prompts
        if "Intent Classification" not in p
        and "Score:" not in p
        and "0-3" not in p
    ]
    assert gen_prompts, "no generation prompt captured"
    answer_prompt = gen_prompts[-1]
    # Main template sentinel.
    assert "What you've remembered about the user" in answer_prompt
    # And the user's reported bug: the fallback example must not have been
    # included anywhere in the rendered prompt.
    assert "I don't have anything on file" not in answer_prompt
