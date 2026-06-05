"""Orchestrator: main RAG pipeline wiring retrieval, reranking, and generation.

Phase 1 notes:
- Gate is a no-op stub (always retrieves).
- No compaction.
- User profile is always an empty string.

Phase 4 notes (compaction & gating, all OFF by default):
- ``compaction.gate_enabled``       — RAGate (RETRIEVE vs SKIP) short-circuits
                                       retrieval when a FACTUAL question is
                                       actually small-talk.
- ``compaction.clue_enabled``       — ClueGenerator rewrites the retrieval
                                       query into a MemoRAG-style hypothesis.
                                       The LLM still answers the ORIGINAL
                                       question; only the retriever sees the clue.
- ``compaction.dialog_mst_enabled`` — DialogMSTCompactor collapses old turns
                                       into a synthetic summary system message
                                       once history exceeds ``compact_after_turns``.
- ``compaction.mask_uncertain``     — render_uncertain post-processes the LLM
                                       answer so ``[UNCERTAIN]`` tokens become a
                                       visible warning glyph. When disabled, the
                                       raw token is silently stripped.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Phase 7-A — math-meta query detector (module-level for testability)
# ---------------------------------------------------------------------------

_RE_MATH_META_QUERY = re.compile(
    r"\b(formula|formulae|equation|equations|math|maths|"
    r"mathematical|derivation|theorem|proof|loss\s+function|"
    r"objective\s+function)s?\b",
    re.IGNORECASE,
)


def _is_math_meta_query(query: str) -> bool:
    """True when the query asks about formulas / equations / math content.

    Used by the orchestrator to (a) push ``where={"has_math": True}`` into the
    retriever and (b) optionally trigger the formula-extraction LLM pass.
    Off by default — gated by ``retrieval.math_meta_filter_enabled`` and
    ``formula_extraction.enabled``.
    """
    if not query:
        return False
    return bool(_RE_MATH_META_QUERY.search(query))

from hrag.config import Config
from hrag.context.dialog_mst import DialogMSTCompactor
from hrag.db.connection import Database, init_db
from hrag.gating.clue import ClueGenerator
from hrag.gating.combined import CombinedPreflight, PreflightDecision
from hrag.gating.gate import RAGate
from hrag.gating.uncertain import extract_uncertain_spans, render_uncertain, strip_uncertain
from hrag.ingest.pipeline import IngestPipeline
from hrag.intent import (
    Intent,
    IntentClassifier,
    IntentVerdict,
    ReflectiveClassifier,
    has_reflective_anchor,
    is_reflective_query,
    is_reflective_strict,
)
from hrag.deepread import (
    DeepReadState,
    build_parts,
    distinct_chapter_labels,
    find_toc_chunk,
    is_broad_query,
    is_structural_query,
    parse_action,
    pick_target_doc,
)
from hrag.interaction import InteractionStore, ReviewDecision, maybe_pause
from hrag.prompts_registry import PromptRegistry
from hrag.providers.embeddings import get_embedding_provider
from hrag.providers.llm import get_llm_provider
from hrag.retrieval.base import Retriever
from hrag.retrieval.factory import (
    build_mst_organizer,
    build_query_rewriter,
    build_reranker,
    build_retriever,
)
from hrag.retrieval.backends import ChromaBackend, SqliteVecBackend, VectorBackend
from hrag.retrieval.policy import RetrievalPlan, RetrievalPolicy
from hrag.retrieval.vector import VectorStore
from hrag.topics import KnownTopicDetector
from hrag.types import Chunk, Message, RetrievalResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, dict], None]
"""Pipeline progress callback: (event_name, payload_dict) -> None.

Events emitted:
    "start"          — payload: {"question": str}
    "query_rewrite"  — payload: {"original": str, "rewritten": str, "rewriter": str}
    "intent_check"   — payload: {"enabled": bool, "intent": str, "confidence": float,
                        "source": "fast_path"|"llm"|"fallback", "raw_label": Optional[str],
                        "query": str}. Fires on every turn; lets the GUI / tests
                        verify the gate ran and what verdict it returned.
    "intent_route"   — payload: {"intent": str, "scope": "none"|"episodic"|"full"|"ask_clarify",
                        "top_k": Optional[int], "source_types": Optional[list[str]]}.
                        Emitted twice: once after the initial routing decision,
                        again only if the post-retrieval FACTUAL→GENERAL swap
                        fires (with extra "swapped_from" and "top_score" keys).
    "router_classify" — payload: {"label": str, "query": str}  # only when retriever=router
    "router_short_circuit" — payload: {"label": str, "retriever": str}.
                         Fires when retrieval.router_short_circuit=True AND the
                         classifier returned "entity" or "global". Emitted by
                         QueryRouter directly via its progress callback.
                         "retriever" is the name of the single retriever called
                         (e.g. "kg_ppr" for entity, "community" for global).
    "taxonomy_descend" — payload: {"trace": [...], "leaves": [...], "note": Optional[str]}
                         (only when retriever=taxonomy; same shape as
                          TaxonomyRetriever.describe_last_descend())
    "retrieve"       — payload: {"top_k": int, "duration_s": float, "n_results": int}
    "rerank_step"    — payload: {"i": int, "n": int, "score": float}
    "rerank_done"    — payload: {"duration_s": float, "kept": int, "fallback_used": bool}
    "organize_done"  — payload: {"input": int, "output": int, "dropped": int}
    "generate_start" — payload: {} (emitted just before LLM begins)
    "generate_token" — payload: {"token": str} (only when stream=True)
    "generate"       — payload: {"duration_s": float, "answer_chars": int}
    "done"           — payload: {"total_s": float}

Phase 4 events (only fire when the corresponding compaction.* flag is on):
    "dialog_compact"   — payload: {"input_turns": int, "output_turns": int,
                          "duration_s": float}. Fires when DialogMSTCompactor
                          collapses old turns (history length must exceed
                          ``compaction.compact_after_turns``).
    "gate_check"       — payload: {"decision": "RETRIEVE"|"SKIP",
                          "duration_s": float}. Fires once per FACTUAL turn
                          when ``compaction.gate_enabled`` is true.
    "clue_generate"    — payload: {"clue": str, "duration_s": float}. Fires
                          when ``compaction.clue_enabled`` is true and the
                          plan still has retrieval scope (post-gate).
    "uncertain_render" — payload: {"count": int}. Fires after the answer is
                          generated when ``compaction.mask_uncertain`` is true.
                          Silent strip path (mask disabled) does NOT emit.

Phase 6 events (only fire when ``retrieval.adaptive_enabled`` is true):
    "adaptive_top_k"        — payload: {"intent": str, "top_k_vector": int|None,
                               "top_k_final": int|None}. Fires once per turn
                               right after the resolver picks per-intent top_k.
                               (None, None) means retrieval will be skipped.
    "retrieval_skipped"     — payload: {"reason": str}. Fires when adaptive
                               resolver returns (None, None); the retriever
                               and reranker are bypassed entirely.
    "episodic_bias_applied" — payload: {"episodic_count": int, "total": int}.
                               Fires after PERSONAL-intent results are
                               re-sorted to put episodic chunks first; only
                               emitted when at least one episodic result was
                               lifted.

Phase 6-B events (only fire when ``retrieval.adaptive_enabled`` is true):
    "adaptive_retriever_picked" — payload: {"intent": str, "retriever": str,
                                  "global": str}. Fires when the per-intent
                                  override resolver picks a different retriever
                                  than ``retrieval.retriever`` (i.e. the
                                  ``adaptive_retriever_per_intent`` mapping for
                                  the active intent is non-"default" AND
                                  differs from the global). Silent when the
                                  mapping is "default" or matches the global.

Phase 7-A events (only fire when retrieval.math_meta_filter_enabled is true):
    "math_meta_filter"          — payload: {"query": str, "where": dict}.
                                   Fires when the math-meta detector triggers
                                   and a where filter is applied.
    "math_meta_filter_fallback" — payload: {"reason": str}. Fires when the
                                   filtered retrieval returned zero results
                                   and we fell back to unfiltered retrieval.
    "formula_extract"           — payload: {"duration_s": float, "chars": int}.
                                   Fires when the second LLM pass produces
                                   extracted formulas (or attempts to).
"""


# ---------------------------------------------------------------------------
# Vector-backend factory
# ---------------------------------------------------------------------------


def _build_vector_backend(config: Config) -> VectorBackend:
    """Pick the configured :class:`VectorBackend` and instantiate it.

    Both implementations resolve their persistence path against the same
    ``storage.chroma_path`` (the directory was named for Chroma; sqlite-vec
    will reuse it once implemented).
    """
    path = config.resolve(config.storage.chroma_path)
    name = config.retrieval.vector_backend
    if name == "chroma":
        return ChromaBackend(path)
    if name == "sqlite_vec":
        return SqliteVecBackend(path)
    raise ValueError(
        f"Unknown retrieval.vector_backend={name!r}; expected 'chroma' or 'sqlite_vec'."
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ChatResult:
    """Value object returned by Orchestrator.chat()."""

    answer: str
    session_id: str
    sources: list[RetrievalResult]
    prompt: str  # full rendered prompt, useful for debugging


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """End-to-end RAG pipeline: retrieve → (rerank) → generate."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Config) -> None:
        self.config = config

        # Core infrastructure
        self.db: Database = init_db(
            config.resolve(config.storage.sqlite_path),
            config.user.default_user_id,
        )
        self.embedder = get_embedding_provider(config.embeddings)
        self.llm = get_llm_provider(config.llm)

        # Vector store (always built; bm25-only mode just doesn't query it).
        # Backend selection lives behind a small factory so retrieval.vector_backend
        # can swap the underlying index (Chroma vs sqlite-vec stub) without
        # touching call-sites.
        self.vector_store = VectorStore(
            config.resolve(config.storage.chroma_path),
            self.embedder.dim,
            backend=_build_vector_backend(config),
        )

        # KG layer (Phase 2) — only constructed if cfg.kg.enabled
        self.kg_store = None
        self.community_store = None
        self.mst_organizer = None
        if config.kg.enabled:
            try:
                from hrag.kg.store import KGStore  # noqa: PLC0415
                from hrag.kg.communities import CommunityStore  # noqa: PLC0415

                kg_path = config.resolve(config.storage.kg_path)
                self.kg_store = KGStore.from_config(
                    self.db,
                    self.embedder,
                    kg_path,
                    config.kg,
                )
                # CommunityStore lives in chroma alongside the chunks collection
                chroma_path = config.resolve(config.storage.chroma_path)
                self.community_store = CommunityStore(self.db, self.embedder, chroma_path)
            except ImportError as exc:
                # KG deps not installed — degrade gracefully
                import warnings

                warnings.warn(
                    f"KG enabled in config but deps missing ({exc}); falling back to vector-only.",
                    stacklevel=2,
                )
                self.kg_store = None
                self.community_store = None

        # Taxonomy layer (Phase 2b) — always constructed when enabled.
        # Cheap to build (no model load); only the build/assign paths
        # call the LLM. Retriever code uses it directly.
        self.taxonomy_store = None
        if config.taxonomy.enabled:
            from hrag.taxonomy.store import TaxonomyStore  # noqa: PLC0415

            self.taxonomy_store = TaxonomyStore(self.db, self.embedder)
            # Phase 12 — disable the in-memory node cache if configured off.
            if not getattr(config.taxonomy, "cache_tree_in_memory", True):
                self.taxonomy_store.cache_ttl_s = 0.0

        # Retriever and reranker selected by config
        self.retriever: Retriever = build_retriever(
            config.retrieval,
            self.db,
            self.vector_store,
            self.embedder,
            llm=self.llm,
            kg_store=self.kg_store,
            community_store=self.community_store,
            kg_cfg=config.kg,
            taxonomy_store=self.taxonomy_store,
            taxonomy_cfg=config.taxonomy,
        )
        self.reranker = build_reranker(config.retrieval, self.llm)
        self.query_rewriter = build_query_rewriter(config.retrieval, self.llm)

        # Phase 6-B1 — lazy per-intent retriever cache. Populated on demand by
        # ``_pick_retriever_for_intent`` when ``retrieval.adaptive_enabled`` is
        # on and an intent maps to a non-"default" override. Each entry is the
        # built Retriever instance keyed by retriever name (e.g. "bm25").
        self._per_intent_retrievers: dict[str, Any] = {}

        # MST organizer (Phase 2) — None when KG is disabled or store is missing
        self.mst_organizer = build_mst_organizer(config.kg, self.kg_store)

        # Ingest pipeline (exposed so CLI / callers can use it directly)
        self.ingest = IngestPipeline(
            config=config,
            db=self.db,
            embedder=self.embedder,
            vector_store=self.vector_store,
            llm=self.llm,
            kg_store=self.kg_store,
            taxonomy_store=self.taxonomy_store,
        )

        # Phase 3: per-user memory layer.
        # - ProfileStore + ContextBuilder render the structured profile string
        #   into every answer prompt (~1ms hot-path cost).
        # - EpisodicMemoryStore reuses the ingest pipeline for /remember writes.
        # - SessionAutoExtractor is opt-in via memory.auto_extract; the CLI
        #   fires it on session close so it never blocks chat.
        from hrag.context import ContextBuilder  # noqa: PLC0415
        from hrag.memory import EpisodicMemoryStore, ProfileStore  # noqa: PLC0415

        self.profile_store = ProfileStore(self.db)
        self.context_builder = ContextBuilder(
            self.profile_store,
            max_items=config.memory.profile_max_items,
            min_confidence=config.memory.profile_min_confidence,
        )
        self.memory_store = EpisodicMemoryStore(self.db, self.ingest)
        # Phase 11.1 — lazy reflective LLM judge (hybrid mode fallback). Built
        # on first need so off/regex modes never construct it.
        self._reflective_classifier: Optional[ReflectiveClassifier] = None
        self.auto_extractor = None
        if config.memory.auto_extract:
            from hrag.memory.auto_extract import SessionAutoExtractor  # noqa: PLC0415
            from hrag.memory.extractor import PreferenceExtractor  # noqa: PLC0415

            self.auto_extractor = SessionAutoExtractor(
                self.db,
                PreferenceExtractor(self.llm),
                self.profile_store,
                min_confidence=config.memory.auto_extract_min_confidence,
            )

        # Prompt registry — one template per Intent (answer.md / answer_greeting.md
        # / answer_personal.md / answer_unclear.md / answer_general.md). The
        # IntentClassifier classifies the user's query into one of GREETING /
        # PERSONAL / FACTUAL / UNCLEAR; the RetrievalPolicy maps each intent to
        # a retrieval scope. GENERAL is a runtime rewrite of FACTUAL when the
        # corpus has nothing relevant to say about the question.
        prompts_dir = Path(__file__).parent / "prompts"
        self.prompts = PromptRegistry(prompts_dir)
        self.intent_classifier = IntentClassifier(
            self.llm,
            fast_only=config.intent.fast_path_only,
            max_tokens=config.intent.llm_max_tokens,
        )
        # Document-aware pre-classifier. When the query mentions any token
        # that appears in a doc title, force FACTUAL — bypasses the
        # regex/LLM. This is the safety net that catches phrasings the
        # intent classifier would otherwise misroute (e.g. "so what is
        # hipporag?" when the user has a HIPPORAG paper in their library).
        self.topic_detector = KnownTopicDetector(self.db)
        self.retrieval_policy = RetrievalPolicy(config.intent)

        # Phase 4 — compaction & gating helpers. Each is None when its flag
        # is off, giving zero overhead on the hot path. The orchestrator's
        # chat() method tests `is not None` before invoking them.
        cmp = config.compaction
        self.gate: RAGate | None = (
            RAGate(self.llm, max_tokens=cmp.gate_max_tokens)
            if cmp.gate_enabled
            else None
        )
        self.clue: ClueGenerator | None = (
            ClueGenerator(self.llm, max_tokens=cmp.clue_max_tokens)
            if cmp.clue_enabled
            else None
        )
        self.dialog_compactor: DialogMSTCompactor | None = (
            DialogMSTCompactor(
                self.llm,
                self.embedder,
                compact_after_turns=cmp.compact_after_turns,
                keep_recent_turns=cmp.keep_recent_turns,
                summary_target_tokens=cmp.summary_target_tokens,
            )
            if cmp.dialog_mst_enabled
            else None
        )

        # Phase 9.6 — combined gate + clue + intent preflight. Only constructed
        # when the flag is on AND every prerequisite (gate / clue / intent) is
        # also enabled — otherwise the per-stage events the GUI subscribes to
        # would have no defined source. ``decide()`` returns None on parse
        # failure and the orchestrator falls back to the three separate calls.
        self.combined_preflight: CombinedPreflight | None = (
            CombinedPreflight(self.llm, max_tokens=cmp.combined_preflight_max_tokens)
            if (
                cmp.combined_preflight_enabled
                and cmp.gate_enabled
                and cmp.clue_enabled
                and config.intent.enabled
            )
            else None
        )
        # Phase 9.13 — re-usable compactor for over-budget recovery even when
        # the long-running dialog-MST flag is off. Lazily instantiated below
        # the first time the budget is exceeded so cold paths pay nothing.
        self._budget_compactor: DialogMSTCompactor | None = None

        # Phase 8 — interactive review store. Shared with the /resume HTTP
        # endpoint so the orchestrator thread (blocked on
        # ``wait_for_decision``) and the web thread (calling
        # ``submit_decision``) talk through the same in-memory pending-turn
        # registry. Always created — gating happens inside
        # :func:`maybe_pause` via ``cfg.interaction.review_enabled``, so the
        # zero-overhead default-off contract is preserved by the callee.
        self.interaction_store: InteractionStore = InteractionStore(
            ttl_s=float(config.interaction.review_timeout_s) + 30.0,
            reap_interval_s=30.0,
        )

        # Phase 8.3 — per-session "what was the previous turn's intent /
        # what entities did its memories surface" tracker. Keyed by
        # session_id. Used by the intent classifier's follow-up fast path:
        # when the previous turn was PERSONAL and the new message mentions
        # any tracked entity (e.g. "Mahmoud"), we short-circuit to PERSONAL
        # without an LLM call. The dict is bounded per-session and the
        # whole tracker is process-local — fine for a single-user FastAPI
        # process. Persistence across restarts is intentionally
        # NOT provided: the entities will be re-extracted from retrieved
        # memories on the first PERSONAL turn after a restart.
        from hrag.intent import ENTITY_TRACKER_CAP as _ENTITY_CAP  # noqa: PLC0415

        self._entity_cap: int = _ENTITY_CAP
        self._session_last_intent: dict[str, Intent] = {}
        # OrderedDict-of-entities per session (oldest-first iteration).
        # Stored as list so we can pop from the head when over cap.
        self._session_memory_entities: dict[str, list[str]] = {}

        # Boot self-test: exercise the classifier's fast-path against a
        # golden set ("what is hipporag" → FACTUAL, "what's my name" →
        # PERSONAL, …). Mismatches are surfaced as one logger.warning line
        # — discoverable in the server log if a fast-path rule regresses.
        _self_test_failures = self.intent_classifier.self_test()
        if _self_test_failures:
            logger.warning(
                "Intent classifier self-test detected %d misclassifications: %s",
                len(_self_test_failures),
                "; ".join(
                    f"{q!r} expected={e.value} got={a.value}"
                    for q, e, a in _self_test_failures
                ),
            )

        # Phase 9.4 — Ollama warm-up + num_keep auto-tune.
        # Delegated to a private method so tests can call it on a MagicMock
        # instance without spinning up a full Orchestrator.
        self._maybe_warmup_llm(config)

        # Phase 10 — refuse to start when the stored embedding dim disagrees
        # with cfg.embeddings.dim. Catches model-swap-without-reingest before
        # any retrieval occurs so the user gets a clear error instead of
        # silent garbage results.
        self._check_embedding_dim_match()

    # ------------------------------------------------------------------
    # Phase 6-B1 — per-intent retriever resolver
    # ------------------------------------------------------------------

    def _pick_retriever_for_intent(self, intent: Intent):
        """Return the Retriever to use for *intent*.

        Lookup order:

        1. ``retrieval.adaptive_enabled`` is False → ``self.retriever`` (no-op).
        2. The intent's mapping is ``"default"`` → ``self.retriever``.
        3. The mapping equals ``retrieval.retriever`` (the global) → reuse
           ``self.retriever`` (the global is already this flavour).
        4. Otherwise build the alternative retriever once via
           :func:`build_retriever` (cloning ``cfg.retrieval`` with a tweaked
           ``retriever`` field) and cache it in ``self._per_intent_retrievers``.

        If the alternative cannot be built (missing pre-requisite store, e.g.
        "taxonomy" with no ``taxonomy_store``, or an unknown name), the
        resolver silently falls back to ``self.retriever`` with a
        ``logger.warning`` — the chat path never crashes here.
        """
        cfg = self.config
        if not cfg.retrieval.adaptive_enabled:
            return self.retriever

        mapping = cfg.retrieval.adaptive_retriever_per_intent or {}
        target = mapping.get(intent.value, "default")
        if target == "default":
            return self.retriever
        if target == cfg.retrieval.retriever:
            return self.retriever

        cached = self._per_intent_retrievers.get(target)
        if cached is not None:
            return cached

        try:
            tweaked_cfg = cfg.retrieval.model_copy(update={"retriever": target})
            built = build_retriever(
                tweaked_cfg,
                self.db,
                self.vector_store,
                self.embedder,
                llm=self.llm,
                kg_store=self.kg_store,
                community_store=self.community_store,
                kg_cfg=cfg.kg,
                taxonomy_store=self.taxonomy_store,
                taxonomy_cfg=cfg.taxonomy,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "adaptive_retriever_per_intent: failed to build %r for intent=%s "
                "(%s); falling back to global retriever %r",
                target, intent.value, exc, getattr(self.retriever, "name", "?"),
            )
            return self.retriever

        self._per_intent_retrievers[target] = built
        return built

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        question: str,
        user_id: str,
        session_id: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
        stream: bool = False,
    ) -> ChatResult:
        """Run one conversational turn and return a ChatResult.

        Parameters
        ----------
        question:   The user's latest message.
        user_id:    Owning user; determines retrieval scope.
        session_id: Existing session UUID hex string, or None to start a new session.
        """
        cfg = self.config
        t_start = time.time()

        def _emit(event: str, payload: dict) -> None:
            if progress is not None:
                try:
                    progress(event, payload)
                except Exception:
                    pass  # never let UI errors break the pipeline

        # Phase 8 — allocate the turn_id up front. Surfaced on the ``start``
        # event so the frontend can wire ``/api/chat/turns/{id}/resume``
        # before any review_required event might fire.
        turn_id = uuid.uuid4().hex

        _emit("start", {"question": question, "turn_id": turn_id})

        # 1. Ensure the user exists; create session row if needed.
        self.db.ensure_user(user_id)
        if session_id is None:
            session_id = uuid.uuid4().hex
            self._create_session(session_id, user_id)

        # Phase 9.3 — activate the per-session embedding cache for the whole
        # turn. Retrievers nested below see this via the contextvar in
        # ``hrag.providers.embeddings``; no signature changes required.
        # Capture the token so we can reset on exit (test isolation: leaving
        # the var set would leak the ambient session into other test cases
        # that share the pytest process/thread).
        _emb_token = None
        try:
            from hrag.providers.embeddings import _session_var as _emb_session_var  # noqa: PLC0415
            _emb_token = _emb_session_var.set(session_id)
        except Exception:  # noqa: BLE001
            pass

        # 2. Persist the user's message.
        self._save_message(session_id, user_id, "user", question)

        # 3. Load conversation history BEFORE the current turn.
        # Default cap is 10 turns; when the dialog compactor is enabled we must
        # load enough history to actually exceed ``compact_after_turns``, otherwise
        # the threshold is unreachable and the compactor never fires.
        history_limit = 10
        if self.dialog_compactor is not None:
            history_limit = max(
                history_limit,
                cfg.compaction.compact_after_turns + cfg.compaction.keep_recent_turns,
            )
        history_rows = self._load_history(session_id, limit=history_limit)

        # 3a. Phase 4 — DialogMSTCompactor.
        # Once the conversation grows past ``compaction.compact_after_turns``,
        # collapse the older portion into a single synthetic system message so
        # the answer prompt isn't crowded out by ancient turns. The compactor
        # speaks ``list[Message]`` while the rest of the pipeline still uses
        # ``list[tuple[role, content]]``; we convert in and out around the call.
        if (
            self.dialog_compactor is not None
            and len(history_rows) > cfg.compaction.compact_after_turns
        ):
            before = len(history_rows)
            t0 = time.time()
            try:
                msgs_in = [Message(role=r, content=c) for r, c in history_rows]
                msgs_out = self.dialog_compactor.compact(msgs_in)
                history_rows = [(m.role, m.content) for m in msgs_out]
            except Exception:  # noqa: BLE001 — never let compaction break chat
                logger.exception("DialogMSTCompactor failed; using raw history")
            _emit(
                "dialog_compact",
                {
                    "input_turns": before,
                    "output_turns": len(history_rows),
                    "duration_s": time.time() - t0,
                },
            )

        # 3b. Rewrite the retrieval query so follow-up turns ground their pronouns.
        # The original `question` is still what the answer LLM sees; only the
        # retriever and reranker get the rewritten form.
        retrieval_query = self.query_rewriter.rewrite(question, history_rows)
        if retrieval_query != question:
            _emit(
                "query_rewrite",
                {
                    "original": question,
                    "rewritten": retrieval_query,
                    "rewriter": self.query_rewriter.name,
                },
            )

        # 3b-i. Phase 9.6 — combined gate+clue+intent.
        # When the flag is on AND no document-aware topic match short-circuits
        # below, one LLM call returns all three decisions and stashes them for
        # the existing branches to pick up. ``preflight_decision`` stays None
        # when the combined call wasn't run, parsed badly, or was overridden by
        # the topic detector — the three serial calls then fire as before.
        preflight_decision: Optional[PreflightDecision] = None

        # 3b-ii. Phase 9.2 — async pre-retrieval future bag.
        # When ``retrieval.async_preflight_enabled`` is True AND the combined
        # preflight is off, the three pre-retrieval LLM calls (gate, clue,
        # classifier) are dispatched concurrently below; the per-stage blocks
        # later in this function then ``.result()`` the future instead of
        # making a fresh blocking call. ``async_preflight_results`` carries
        # the resolved values so each block stays consumer-shaped.
        async_preflight_results: dict[str, Any] = {}

        # 3c. Intent classification — decides retrieval scope and prompt template.
        # The classifier itself only ever returns GREETING / PERSONAL / FACTUAL /
        # UNCLEAR. GENERAL is a downstream rewrite of FACTUAL when retrieval
        # returns no corpus-relevant content (see "post-retrieval re-route" below).
        #
        # The classifier sees the REWRITTEN retrieval query — not the bare
        # original — so follow-ups like "search the documents for it" inherit
        # the prior turn's topic ("it" → "HippoRAG") and get routed to FACTUAL
        # instead of being interpreted as ambiguous chitchat. The heuristic
        # rewriter only fires on genuine follow-up signals (pronouns, opener
        # phrases) so plain greetings like "hey" still arrive at the classifier
        # unmodified.
        classifier_input = retrieval_query

        # Document-aware pre-classifier — if the query mentions any token
        # that exists in the user's document titles (e.g. "hipporag",
        # "naacl", "remindrag"), we ALREADY know it's about the corpus and
        # short-circuit straight to FACTUAL. This bypasses both the regex
        # fast-path and the LLM, which have both mis-flagged corpus queries
        # in the past. Source = "named_topic" so the UI surfaces *why*.
        verdict: IntentVerdict
        if cfg.intent.enabled:
            try:
                matched_topics = self.topic_detector.detect(classifier_input, user_id)
            except Exception:  # noqa: BLE001
                matched_topics = set()
            if matched_topics:
                topic_sample = ", ".join(sorted(matched_topics)[:3])
                verdict = IntentVerdict(
                    intent=Intent.FACTUAL,
                    confidence=0.98,
                    source="named_topic",
                    raw_label=topic_sample,
                )
                logger.info(
                    "intent verdict (named_topic): query=%r matched=%s",
                    question[:80], sorted(matched_topics),
                )
            else:
                # Phase 9.6 — combined preflight (one call returning intent +
                # gate + clue). When it succeeds, we build the IntentVerdict
                # from its result and stash gate/clue for the per-stage blocks
                # below. The downstream code's `is not None` checks on
                # ``self.gate`` / ``self.clue`` still gate the per-stage logic.
                if self.combined_preflight is not None:
                    preflight_history = [Message(role=r, content=c) for r, c in history_rows]
                    preflight_decision = self.combined_preflight.decide(
                        classifier_input, preflight_history
                    )
                if preflight_decision is not None:
                    intent_str = preflight_decision.intent
                    intent_map = {
                        "factual": Intent.FACTUAL,
                        "personal": Intent.PERSONAL,
                        "greeting": Intent.GREETING,
                        "unclear": Intent.UNCLEAR,
                    }
                    verdict = IntentVerdict(
                        intent=intent_map.get(intent_str, Intent.UNCLEAR),
                        confidence=0.85,
                        source="combined_preflight",
                        raw_label=intent_str,
                    )
                elif (
                    cfg.retrieval.async_preflight_enabled
                    and self.combined_preflight is None
                ):
                    # Phase 9.2 — dispatch the three pre-retrieval calls in
                    # parallel. ``classify`` is what we MUST have to build the
                    # verdict; the gate / clue futures are resolved later in
                    # their own blocks. We let the executor stay alive for the
                    # life of the chat() call so the per-stage blocks can read
                    # the futures without re-instantiating threads.
                    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

                    prev_intent_for_session = self._session_last_intent.get(session_id)
                    prev_entities_for_session = set(
                        self._session_memory_entities.get(session_id, [])
                    )
                    exec_msgs = [Message(role=r, content=c) for r, c in history_rows]
                    pool = ThreadPoolExecutor(max_workers=3)
                    async_preflight_results["_executor"] = pool
                    if self.gate is not None:
                        async_preflight_results["gate"] = pool.submit(
                            self.gate.decide, question, exec_msgs
                        )
                    if self.clue is not None:
                        async_preflight_results["clue"] = pool.submit(
                            self.clue.generate, classifier_input, exec_msgs
                        )
                    fut = pool.submit(
                        self.intent_classifier.classify,
                        classifier_input,
                        history=history_rows,
                        prev_intent=prev_intent_for_session,
                        prev_memory_entities=prev_entities_for_session,
                    )
                    verdict = fut.result()
                    _emit("async_preflight", {
                        "stages_dispatched": [
                            k for k in ("gate", "clue") if k in async_preflight_results
                        ] + ["intent"],
                    })
                else:
                    # Phase 8.3 — feed the last 2 exchanges into the LLM prompt
                    # so it can recognise a personal follow-up, AND pass the
                    # previous turn's intent + memory entities so the
                    # follow-up entity fast path can short-circuit without
                    # an LLM call.
                    prev_intent_for_session = self._session_last_intent.get(session_id)
                    prev_entities_for_session: set[str] = set(
                        self._session_memory_entities.get(session_id, [])
                    )
                    verdict = self.intent_classifier.classify(
                        classifier_input,
                        history=history_rows,
                        prev_intent=prev_intent_for_session,
                        prev_memory_entities=prev_entities_for_session,
                    )
        else:
            # Disabled-by-config emergency bypass: treat everything as factual.
            verdict = IntentVerdict(
                intent=Intent.FACTUAL,
                confidence=1.0,
                source="fallback",
                raw_label=None,
            )
        intent: Intent = verdict.intent

        logger.info(
            "intent verdict: query=%r intent=%s confidence=%.2f source=%s",
            question[:80], intent.value, verdict.confidence, verdict.source,
        )
        _emit("intent_check", {
            "enabled": cfg.intent.enabled,
            "intent": intent.value,
            "confidence": float(verdict.confidence),
            "source": verdict.source,
            "raw_label": verdict.raw_label,
            "query": question,
        })

        # 3c-bis. Phase 9.6 — late combined preflight.
        # When intent was determined by the regex fast-path / named_topic path
        # (so the LLM intent classifier was skipped), we still benefit from
        # merging gate + clue into one LLM round-trip. Fire the combined call
        # here unconditionally when the flag is on AND intent is FACTUAL AND
        # we haven't already populated ``preflight_decision`` from the intent
        # branch above.
        if (
            self.combined_preflight is not None
            and preflight_decision is None
            and intent == Intent.FACTUAL
        ):
            late_history = [Message(role=r, content=c) for r, c in history_rows]
            preflight_decision = self.combined_preflight.decide(
                question, late_history
            )

        # 3d-pre. Phase 11.1 — reflective / opinion personal questions.
        # "What do you think about me?" wants an *impression* synthesised from
        # everything we know, not a recital of one saved fact. Detection is
        # switchable (cfg.retrieval.reflection_mode: off|regex|hybrid) and the
        # precision is asymmetric:
        #   * COERCION (override a non-PERSONAL verdict) requires a HIGH-precision
        #     signal — strict regex or the LLM judge — so "describe me a function"
        #     (FACTUAL) is never hijacked.
        #   * WITHIN an already-PERSONAL turn we accept the looser recall-tier
        #     match, where a false positive is cheap (synthesis vs recital).
        mode = getattr(cfg.retrieval, "reflection_mode", "hybrid")
        strict_hit = mode != "off" and is_reflective_strict(question)
        loose_hit = mode != "off" and is_reflective_query(question)
        # The hybrid LLM judge is a small, error-prone model: a lone false "yes"
        # on a neutral message ("Ok i want to test you") must not be enough to
        # turn the reply into a reflective self-portrait. Require a textual
        # reflective anchor (opinion cue / me-my-myself / "who I am") before we
        # consult the judge OR let it coerce — for BOTH non-PERSONAL coercion
        # and within-PERSONAL turns. Genuine reflective questions always carry
        # an anchor; bare subject "i" does not.
        llm_hit = (
            mode == "hybrid"
            and not strict_hit
            and has_reflective_anchor(question)
            and self._reflective_llm_signal(question, history_rows, preflight_decision)
        )
        if mode != "off" and intent != Intent.PERSONAL and (strict_hit or llm_hit):
            logger.info(
                "reflective query coerced intent %s -> personal: %r",
                intent.value, question[:80],
            )
            intent = Intent.PERSONAL
        reflective_turn = (
            mode != "off"
            and intent == Intent.PERSONAL
            and (loose_hit or strict_hit or llm_hit)
        )
        _emit("reflective_check", {
            "mode": mode,
            "strict": strict_hit,
            "loose": loose_hit,
            "llm": llm_hit,
            "reflective": reflective_turn,
        })

        # 3d-deep. Phase 13 / 13.1 — agentic multi-stage deep read. Reads the
        # single best-matching document by choosing a navigation action each pass
        # (read_part / search / answer), visualised live. Triggers on a broad
        # *or* a structural/meta question ("how many chapters", "table of
        # contents", "structure of X") — the latter so pointed factual questions
        # whose answer lives in the document's structure get the iterative
        # treatment too (Phase 13.1 smart escalation, part a). Stream-only (the
        # document-map panel needs the SSE channel); skipped under the Phase-8
        # review loop. A weak-one-pass FACTUAL turn ALSO escalates here later
        # (part b, just before the FACTUAL→GENERAL swap).
        _dr_structural = cfg.deep_read.structural_trigger and is_structural_query(question)
        if (
            stream
            and cfg.deep_read.enabled
            and cfg.deep_read.auto_trigger
            and intent in (Intent.FACTUAL, Intent.GENERAL)
            and not reflective_turn
            and not cfg.interaction.review_enabled
            and (is_broad_query(question) or _dr_structural)
        ):
            dr = self._run_deep_read(
                question=question,
                user_id=user_id,
                session_id=session_id,
                history_rows=history_rows,
                emit=_emit,
                t_start=t_start,
                structural=_dr_structural,
            )
            if dr is not None:
                if _emb_token is not None:
                    try:
                        from hrag.providers.embeddings import (  # noqa: PLC0415
                            _session_var as _evs,
                        )
                        _evs.reset(_emb_token)
                    except Exception:  # noqa: BLE001
                        pass
                return dr
            # No suitable document → fall through to the normal one-pass path.

        # 3d. Retrieval-policy dispatch.
        plan: RetrievalPlan = self.retrieval_policy.plan(intent)
        _emit("intent_route", {
            "intent": intent.value,
            "scope": plan.scope,
            "top_k": plan.top_k_override,
            "source_types": plan.source_types,
        })

        # 3d-bis. Phase 11 — reflective-turn query augmentation.
        # When the turn is reflective (detected above), augment the retrieval
        # query with the user's saved-profile terms so document chunks about
        # them clear reranking — an opinion-shaped query alone matches them
        # poorly. The render section further down chooses the synthesis prompt.
        if reflective_turn:
            profile_terms = ""
            try:
                profile_terms = self.profile_store.render(user_id)
            except Exception:  # noqa: BLE001
                profile_terms = ""
            if profile_terms and profile_terms != "(no profile yet)":
                # Flatten the multi-line profile render into query terms so the
                # retriever pulls document chunks that mention the user's known
                # facts (employer, research topics, …) alongside the question.
                flat_profile = " ".join(profile_terms.split())
                retrieval_query = f"{question} {flat_profile}"
            _emit("reflective_personal", {
                "augmented": retrieval_query != question,
                "query": retrieval_query[:120],
            })

        # 3d-0. Phase 6 — adaptive top_k resolver.
        # No-op when ``retrieval.adaptive_enabled`` is False (returns the global
        # defaults). When enabled, this gives every intent its own retrieve
        # depth — including ``(None, None)`` for greetings, which short-circuits
        # retrieval entirely below.
        adaptive_vec_k, adaptive_final_k = _adaptive_top_k(cfg, intent)
        _emit("adaptive_top_k", {
            "intent": intent.value,
            "top_k_vector": adaptive_vec_k,
            "top_k_final": adaptive_final_k,
        })
        # Hard skip when the resolver returns (None, None). We bypass the
        # gate, clue, retriever, reranker, and organizer — the answer
        # template still runs with no retrieved passages.
        adaptive_skip = (
            cfg.retrieval.adaptive_enabled
            and adaptive_vec_k is None
            and adaptive_final_k is None
        )
        if adaptive_skip:
            _emit("retrieval_skipped", {"reason": "greeting"})
            plan = dataclasses.replace(plan, scope="none")

        # 3d-i. Phase 4 — RAGate. A single cheap LLM call decides RETRIEVE vs
        # SKIP for FACTUAL questions; small-talk that slipped past the intent
        # classifier is short-circuited to "no retrieval" by rewriting the plan
        # to scope="none". RetrievalPlan is a frozen dataclass — use
        # dataclasses.replace to produce a new instance.
        if self.gate is not None and intent == Intent.FACTUAL:
            t0 = time.time()
            gate_source = "gate"
            if preflight_decision is not None:
                # Reuse the combined-preflight gate verdict (no extra LLM call).
                gate_decision = preflight_decision.gate
                gate_source = "combined"
            elif "gate" in async_preflight_results:
                try:
                    gate_decision = async_preflight_results["gate"].result()
                except Exception:  # noqa: BLE001
                    logger.exception("Async RAGate failed; defaulting to RETRIEVE")
                    gate_decision = "RETRIEVE"
                gate_source = "async"
            else:
                try:
                    gate_history = [Message(role=r, content=c) for r, c in history_rows]
                    gate_decision = self.gate.decide(question, gate_history)
                except Exception:  # noqa: BLE001 — fail open so chat never breaks
                    logger.exception("RAGate failed; defaulting to RETRIEVE")
                    gate_decision = "RETRIEVE"
            _emit(
                "gate_check",
                {
                    "decision": gate_decision,
                    "duration_s": time.time() - t0,
                    "source": gate_source,
                },
            )
            if gate_decision == "SKIP":
                plan = dataclasses.replace(plan, scope="none")
                # Rewrite the intent so the answer prompt picks a no-retrieval
                # template (GENERAL → answer_general.md, which expects no
                # {retrieved_passages} slot). This mirrors the post-retrieval
                # FACTUAL→GENERAL swap below.
                intent = Intent.GENERAL

        # 3d-ii. Phase 4 — ClueGenerator. When the question is FACTUAL and we
        # are still going to retrieve, swap the retrieval query for a
        # MemoRAG-style "hypothesis" — a draft of what the answer probably
        # looks like, in source-document vocabulary. The original `question`
        # is preserved for the answer prompt (the LLM answers what the user
        # actually asked, not the hypothesis).
        # Phase 8 — captured so the interactive review pause can surface them
        # in the modal payload. ``router_label_so_far`` is the most recent
        # value emitted on ``router_classify`` (None when no router was in
        # the chain). ``last_descend_payload`` mirrors the taxonomy_descend
        # event payload (None when no taxonomy retriever ran).
        router_label_so_far: Optional[str] = None
        last_descend_payload: Optional[dict] = None
        clue_text: Optional[str] = None  # surfaced in the review payload

        if (
            self.clue is not None
            and intent == Intent.FACTUAL
            and plan.scope != "none"
        ):
            t0 = time.time()
            clue_source = "clue"
            if preflight_decision is not None and preflight_decision.clue:
                clue_text = preflight_decision.clue
                clue_source = "combined"
            elif "clue" in async_preflight_results:
                try:
                    clue_text = async_preflight_results["clue"].result()
                except Exception:  # noqa: BLE001
                    logger.exception("Async ClueGenerator failed; using raw query")
                    clue_text = retrieval_query
                clue_source = "async"
            else:
                try:
                    clue_history = [Message(role=r, content=c) for r, c in history_rows]
                    clue_text = self.clue.generate(retrieval_query, clue_history)
                except Exception:  # noqa: BLE001 — fail soft to the raw query
                    logger.exception("ClueGenerator failed; using raw query")
                    clue_text = retrieval_query
            _emit(
                "clue_generate",
                {
                    "clue": clue_text,
                    "duration_s": time.time() - t0,
                    "source": clue_source,
                },
            )
            if clue_text and clue_text.strip():
                retrieval_query = clue_text

        # Phase 9.2 — release the async preflight executor once both futures
        # have been consumed (or skipped). Safe to call multiple times.
        _pool = async_preflight_results.pop("_executor", None)
        if _pool is not None:
            try:
                _pool.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass

        results: list[RetrievalResult] = []
        fallback_used = False
        # Phase 7-A: tracked across the retrieve/rerank/extract path so the
        # formula-extraction block at the end of chat() can see it.
        math_meta = False

        if plan.scope == "full":
            # ---- Full retrieve path (factual, on-corpus) ----

            # Phase 6-B1 — pick the per-intent retriever (or fall back to the
            # global one). All subsequent retrieve() calls + diagnostic walkers
            # for THIS turn use ``active_retriever``.
            active_retriever = self._pick_retriever_for_intent(intent)
            if cfg.retrieval.adaptive_enabled:
                picked_name = getattr(active_retriever, "name", "?")
                global_name = getattr(self.retriever, "name", "?")
                if picked_name != global_name:
                    _emit("adaptive_retriever_picked", {
                        "intent": intent.value,
                        "retriever": picked_name,
                        "global": global_name,
                    })

            # Surface router classification as a diagnostic event so users can
            # see which retrieval path the query took. Classification is cached
            # on the router so the actual retrieve() call below won't re-run it.
            # Walk through wrappers (e.g. DocScopedRetriever) to find the router.
            # Phase 8: also capture the label on ``router_label_so_far`` so the
            # interactive review modal can surface ROUTER_AMBIGUOUS.
            inner = active_retriever
            for _ in range(4):  # bounded walk; avoid infinite loop on cycles
                if getattr(inner, "name", "") == "router":
                    break
                inner = getattr(inner, "wrapped", None) or getattr(inner, "inner", None)
                if inner is None:
                    break
            if inner is not None and getattr(inner, "name", "") == "router":
                try:
                    label = inner.classify(retrieval_query)
                except Exception:  # noqa: BLE001
                    label = "unknown"
                router_label_so_far = label
                if progress:
                    _emit("router_classify", {"label": label, "query": retrieval_query})
                # Phase 9.11 — wire the per-turn progress callback into the router
                # so it can emit ``router_short_circuit`` events from inside
                # retrieve(). The callback is cleared after retrieval to avoid
                # stale references when progress is per-turn.
                if hasattr(inner, "_progress"):
                    inner._progress = _emit if progress else None

            t0 = time.time()
            # Adaptive top_k: when retrieval.adaptive_enabled is true, vec_k is
            # the per-intent value (widened by 2x for reranker slack); else the
            # global default. Falls back defensively if the resolver somehow
            # returned None on a non-skip path.
            # Precedence: adaptive per-intent value (when enabled) > the
            # policy's per-intent override (e.g. PERSONAL → cfg.personal_top_k)
            # > the global default. PERSONAL is the only intent that sets an
            # override today, so FACTUAL is unchanged (override is None there).
            if adaptive_vec_k is not None:
                full_vec_k = adaptive_vec_k
            elif plan.top_k_override is not None:
                full_vec_k = plan.top_k_override
            else:
                full_vec_k = cfg.retrieval.top_k_vector
            full_final_k = adaptive_final_k if adaptive_final_k is not None else cfg.retrieval.top_k_final

            # Phase 7-A — math-meta filter. When the query asks about
            # formulas/equations/math AND the flag is on, push
            # where={"has_math": True} into the retriever so equation-bearing
            # chunks float to the top. Falls back to unfiltered retrieval if
            # the filtered call returns nothing (e.g. has_math metadata not
            # yet backfilled).
            where_filter: Optional[dict] = None
            math_meta = (
                cfg.retrieval.math_meta_filter_enabled
                and _is_math_meta_query(question)
            )
            if math_meta:
                where_filter = {"has_math": True}
                _emit("math_meta_filter", {
                    "query": question,
                    "where": where_filter,
                })

            # Phase 8.1 — memories must always be findable on FACTUAL turns too.
            # For the "full" scope, plan.source_types is None (no filter), but
            # some retrievers (e.g. TaxonomyRetriever) scope to documents picked
            # by the tree and miss episodic memories. Pass an explicit
            # ["document", "episodic"] list so retrievers that DO honour the
            # filter see both pools. Retrievers that ignore source_types (or
            # only filter chunks they would have returned anyway) keep their
            # previous behaviour.
            full_source_types = plan.source_types
            if getattr(cfg.retrieval, "always_include_episodic", True):
                if isinstance(full_source_types, (list, tuple)) and "episodic" not in full_source_types:
                    full_source_types = list(full_source_types) + ["episodic"]
                elif full_source_types is None:
                    full_source_types = ["document", "episodic"]

            results = active_retriever.retrieve(
                retrieval_query,
                user_id,
                top_k=full_vec_k,
                source_types=full_source_types,
                intent_hint=intent,
                where=where_filter,
            )

            # Math-meta fallback: if the filter eliminated everything, retry
            # without it so the user always gets some retrieval signal even if
            # backfill hasn't run.
            if math_meta and not results:
                _emit("math_meta_filter_fallback", {"reason": "no_matches"})
                results = active_retriever.retrieve(
                    retrieval_query,
                    user_id,
                    top_k=full_vec_k,
                    source_types=full_source_types,
                    intent_hint=intent,
                )

            # If a TaxonomyRetriever is in the chain, surface its descend trace
            # so the GUI can render the tree-navigation visual.
            # Phase 8: also stash the payload on ``last_descend_payload`` so
            # the review modal can fire BRANCH_THRESHOLD.
            tx = active_retriever
            for _ in range(4):  # walk through wrappers
                if getattr(tx, "name", "") == "taxonomy":
                    break
                tx = getattr(tx, "wrapped", None) or getattr(tx, "inner", None)
                if tx is None:
                    break
            if tx is not None and getattr(tx, "name", "") == "taxonomy":
                try:
                    last_descend_payload = tx.describe_last_descend(user_id)
                except Exception:  # noqa: BLE001
                    last_descend_payload = None
                if progress is not None and last_descend_payload is not None:
                    _emit("taxonomy_descend", last_descend_payload)

            _emit(
                "retrieve",
                {
                    "top_k": full_vec_k,
                    "duration_s": time.time() - t0,
                    "n_results": len(results),
                },
            )

            # Rerank or plain truncate.
            if self.reranker is not None:
                t0 = time.time()
                unreranked = list(results)  # keep a copy for fallback

                def _rerank_progress(i: int, n: int, score: float) -> None:
                    _emit("rerank_step", {"i": i, "n": n, "score": score})

                # Phase 7-A: lower the rerank threshold on math-meta queries
                # so the MS-MARCO cross-encoder doesn't drop math passages
                # whose embeddings (heavy on glyphs / TeX) score weakly against
                # natural-language queries.
                effective_threshold = (
                    cfg.retrieval.math_meta_rerank_threshold
                    if math_meta
                    else cfg.retrieval.rerank_threshold
                )
                reranked = self.reranker.rerank(
                    retrieval_query,
                    results,
                    threshold=effective_threshold,
                    top_k=full_final_k,
                    progress=_rerank_progress,
                )

                # Fallback: if the reranker dropped everything, fall back to the
                # top-k unreranked vector results so the LLM has *something* to
                # work with rather than answering "I couldn't find that".
                if not reranked and unreranked:
                    reranked = unreranked[: full_final_k]
                    fallback_used = True
                    # Phase 9.9 — telemetry. Capture the chunk_ids the reranker
                    # threw out so the user can mine which queries the
                    # cross-encoder mis-scores.
                    if cfg.retrieval.rerank_fallback_telemetry_enabled:
                        try:
                            self._log_rerank_fallback(
                                turn_id=turn_id,
                                session_id=session_id,
                                user_id=user_id,
                                query=retrieval_query,
                                dropped_chunk_ids=[
                                    r.chunk.chunk_id for r in unreranked
                                ],
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception("rerank-fallback telemetry write failed")

                results = reranked

                # Phase 9.15 — feedback-weighted re-ranking.
                # After the cross-encoder scores are final, nudge each result's
                # rerank_score by EMA(thumbs_up - thumbs_down) for its chunk.
                # Gated by cfg.retrieval.feedback_reranking_enabled (default OFF).
                if cfg.retrieval.feedback_reranking_enabled and results:
                    from hrag.feedback_scoring import (  # noqa: PLC0415
                        FeedbackScorer,
                        apply_feedback_to_rerank_score,
                    )
                    _fb_scorer = FeedbackScorer(
                        self.db,
                        alpha=cfg.retrieval.feedback_reranking_alpha,
                    )
                    _chunk_ids = [r.chunk.chunk_id for r in results]
                    _fb_scores = _fb_scorer.score_many(_chunk_ids)
                    _n_with_feedback = 0
                    _score_shifts: list[float] = []
                    for _r in results:
                        _fs = _fb_scores.get(_r.chunk.chunk_id, 0.0)
                        _r.feedback_score = _fs
                        if _fs != 0.0:
                            _n_with_feedback += 1
                            _old = _r.rerank_score if _r.rerank_score is not None else 0.0
                            _new = apply_feedback_to_rerank_score(
                                _old, _fs,
                                weight=cfg.retrieval.feedback_reranking_weight,
                            )
                            _score_shifts.append(_new - _old)
                            _r.rerank_score = _new
                    # Re-sort after nudge so ordering reflects adjusted scores.
                    results.sort(
                        key=lambda _r2: (
                            _r2.rerank_score if _r2.rerank_score is not None else float("-inf"),
                            _r2.score,
                        ),
                        reverse=True,
                    )
                    _mean_shift = (
                        sum(_score_shifts) / len(_score_shifts)
                        if _score_shifts else 0.0
                    )
                    _emit("feedback_rerank_applied", {
                        "n_results": len(results),
                        "n_with_feedback": _n_with_feedback,
                        "mean_shift": round(_mean_shift, 4),
                    })

                _emit(
                    "rerank_done",
                    {
                        "duration_s": time.time() - t0,
                        "kept": len(results),
                        "fallback_used": fallback_used,
                    },
                )
            else:
                results = results[: full_final_k]

            # Phase 2: MST organizer (KG2RAG redundancy filter + tree-ordering).
            # No-op when KG is empty / disabled.
            if self.mst_organizer is not None:
                organized_count_before = len(results)
                results = self.mst_organizer.organize(results)
                if progress:
                    _emit("organize_done", {
                        "input": organized_count_before,
                        "output": len(results),
                        "dropped": organized_count_before - len(results),
                    })

            # Phase 6 — episodic bias for PERSONAL turns. PERSONAL now routes
            # through this "full" scope (so uploaded documents are searched too),
            # so the stable-sort that floats saved memories above documents must
            # run here, AFTER rerank/organize, rather than in the (now-unused)
            # episodic branch below. No-op unless adaptive + the bias flag are on
            # and the turn is PERSONAL.
            episodic_bias_on = (
                cfg.retrieval.adaptive_enabled
                and cfg.retrieval.adaptive_personal_episodic_bias
                and intent == Intent.PERSONAL
            )
            if episodic_bias_on and results:
                episodic_count = sum(
                    1 for r in results if r.chunk.source_type == "episodic"
                )
                results.sort(
                    key=lambda r: 0 if r.chunk.source_type == "episodic" else 1
                )
                if episodic_count > 0:
                    _emit("episodic_bias_applied", {
                        "episodic_count": episodic_count,
                        "total": len(results),
                    })

        elif plan.scope == "episodic":
            # ---- Memory-only retrieval (personal questions) ----
            # No rerank, no organize. Defensively swallow retriever failures —
            # an empty memory store is an expected state, not an error.
            t0 = time.time()

            # Phase 6-B1 — pick the per-intent retriever (or fall back to the
            # global one). Emit the override event when it's not a no-op.
            active_retriever = self._pick_retriever_for_intent(intent)
            if cfg.retrieval.adaptive_enabled:
                picked_name = getattr(active_retriever, "name", "?")
                global_name = getattr(self.retriever, "name", "?")
                if picked_name != global_name:
                    _emit("adaptive_retriever_picked", {
                        "intent": intent.value,
                        "retriever": picked_name,
                        "global": global_name,
                    })

            # Phase 6: when adaptive is enabled, prefer the adaptive top_k.
            # Otherwise fall back to plan.top_k_override → config default.
            if cfg.retrieval.adaptive_enabled and adaptive_final_k is not None:
                top_k = adaptive_final_k
            else:
                top_k = plan.top_k_override or cfg.retrieval.top_k_vector

            # Phase 6 — episodic bias. When enabled, broaden source_types
            # so the retriever sees both documents and episodic memories;
            # after the call we re-sort to lift episodic chunks to the top.
            source_types = plan.source_types
            episodic_bias_on = (
                cfg.retrieval.adaptive_enabled
                and cfg.retrieval.adaptive_personal_episodic_bias
                and intent == Intent.PERSONAL
            )
            if episodic_bias_on:
                # Pre-existing Phase 6 behaviour: hard-override to both pools.
                source_types = ["document", "episodic"]

            # Phase 8.1 — memories must always be findable, not only on PERSONAL turns.
            # Always include "episodic" in source_types unless explicitly disabled.
            # Episodic chunks will compete with documents on relevance; for PERSONAL
            # turns the stable sort below still lifts them to the top.
            include_episodic_always = getattr(cfg.retrieval, "always_include_episodic", True)
            if include_episodic_always:
                if isinstance(source_types, (list, tuple)) and "episodic" not in source_types:
                    source_types = list(source_types) + ["episodic"]
                elif source_types is None:
                    source_types = ["document", "episodic"]

            try:
                results = active_retriever.retrieve(
                    retrieval_query,
                    user_id,
                    top_k=top_k,
                    source_types=source_types,
                    intent_hint=intent,
                )
            except Exception:  # noqa: BLE001
                results = []
            _emit("retrieve", {
                "top_k": top_k,
                "duration_s": time.time() - t0,
                "n_results": len(results),
            })
            # Stable sort: episodic chunks float to the top, within-group
            # order preserved. Emit a progress event only if at least one
            # episodic chunk was found (no-op otherwise).
            if episodic_bias_on and results:
                episodic_count = sum(
                    1 for r in results if r.chunk.source_type == "episodic"
                )
                results.sort(
                    key=lambda r: 0 if r.chunk.source_type == "episodic" else 1
                )
                if episodic_count > 0:
                    _emit("episodic_bias_applied", {
                        "episodic_count": episodic_count,
                        "total": len(results),
                    })
            # Surface the actual memory snippets so the GUI's reasoning trace
            # can render them — without this event, the user sees that 2
            # memories were retrieved but not which ones.
            if results:
                memory_previews: list[dict] = []
                for r in results[:10]:
                    chunk = r.chunk
                    raw_text = (chunk.text or "").strip().replace("\n", " ")
                    memory_previews.append({
                        "title": (chunk.title or "memory").strip()[:80],
                        "text": raw_text[:240],
                        "score": float(r.score),
                        "chunk_id": chunk.chunk_id,
                    })
                _emit("personal_memories", {"memories": memory_previews})

        # plan.scope == "none" or "ask_clarify": no retrieval. The chosen
        # prompt template does not consume {retrieved_passages}, so we leave
        # `results` empty and proceed straight to generation.

        # --------------------------------------------------------------
        # 3d-iii. Phase 8 — interactive review pause.
        # --------------------------------------------------------------
        # Sits between retrieval/organize and the FACTUAL→GENERAL swap.
        # When ``cfg.interaction.review_enabled`` is False, ``maybe_pause``
        # short-circuits to ``ReviewDecision(action="continue")`` with zero
        # side effects — preserves the default-off contract.
        #
        # The pause is BEFORE the swap so the user gets a chance to act on
        # the same weak signal that would normally drive the swap.
        review_decision: ReviewDecision = ReviewDecision(action="continue")
        review_persistence: Optional[dict[str, Any]] = None

        if cfg.interaction.review_enabled and intent != Intent.GREETING:
            # Compute the "swap imminent" flag without actually performing
            # the swap — the user decides first.
            factual_general_swap_imminent = False
            if intent == Intent.FACTUAL:
                floor_imminent = float(cfg.intent.corpus_relevance_floor)
                top_score_imminent = max((r.score for r in results), default=0.0)
                if not results or (floor_imminent > 0.0 and top_score_imminent < floor_imminent):
                    factual_general_swap_imminent = True

            review_decision = maybe_pause(
                cfg=cfg.interaction,
                results=results,
                descend=last_descend_payload,
                intent_verdict=verdict,
                router_label=router_label_so_far,
                factual_general_swap_imminent=factual_general_swap_imminent,
                clue=clue_text,
                question=question,
                retrieval_query=retrieval_query,
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                progress=progress,
                store=self.interaction_store,
                llm=self.llm,
            )

            # If the decision carries non-empty reasons, a real pause happened
            # — stash a structured payload for the messages.metadata column.
            if review_decision.reasons:
                review_persistence = {
                    "phase8": {
                        "turn_id": turn_id,
                        "action": review_decision.action,
                        "reasons": list(review_decision.reasons),
                        "timed_out": bool(review_decision.timed_out),
                        "selected_chunk_ids": list(review_decision.selected_chunk_ids),
                        "rewritten_query": review_decision.rewritten_query,
                        "expand_from_doc_id": review_decision.expand_from_doc_id,
                        "redirect_taxonomy_node_id": review_decision.redirect_taxonomy_node_id,
                        "include_episodic": bool(review_decision.include_episodic),
                    }
                }

            # Apply the decision before prompt rendering.
            if review_decision.action == "abort":
                _emit("done", {
                    "total_s": time.time() - t_start,
                    "aborted_by_user": True,
                })
                # Persist a marker assistant message so session replay
                # surfaces the abort point.
                aborted_text = "_(Turn aborted by user during review.)_"
                meta_json: Optional[str] = None
                if cfg.interaction.persistence_enabled and review_persistence is not None:
                    try:
                        meta_json = json.dumps(review_persistence)
                    except Exception:  # noqa: BLE001
                        meta_json = None
                self._save_message(
                    session_id, user_id, "assistant", aborted_text,
                    metadata=meta_json,
                )
                self.db.commit()
                return ChatResult(
                    answer=aborted_text,
                    session_id=session_id,
                    sources=[],
                    prompt="",
                )

            if review_decision.action == "general":
                # User opted out of corpus retrieval; answer from general
                # knowledge instead.
                intent = Intent.GENERAL
                results = []
                plan = dataclasses.replace(plan, scope="none")

            elif review_decision.action == "filter" and review_decision.selected_chunk_ids:
                selected = set(review_decision.selected_chunk_ids)
                results = [r for r in results if r.chunk.chunk_id in selected]

            elif review_decision.action == "rephrase" and review_decision.rewritten_query:
                retrieval_query = review_decision.rewritten_query
                try:
                    active_retriever = self._pick_retriever_for_intent(intent)
                    results = active_retriever.retrieve(
                        retrieval_query,
                        user_id,
                        top_k=cfg.retrieval.top_k_final,
                        intent_hint=intent,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Phase 8 rephrase: re-retrieval failed")
                    results = []

            elif review_decision.action == "clarify":
                # Generate a clarifying question and return it as the assistant
                # answer. Subsequent user turns will pick up the clarification.
                try:
                    clarify_template = (
                        Path(__file__).parent / "prompts" / "clarify.md"
                    ).read_text(encoding="utf-8")
                    clarify_prompt = clarify_template.format(
                        question=question,
                        clue=clue_text or "",
                    )
                    clarify_answer = self.llm.complete(
                        clarify_prompt,
                        temperature=0.3,
                        max_tokens=120,
                    ).strip()
                except Exception:  # noqa: BLE001
                    logger.exception("Phase 8 clarify: LLM call failed")
                    clarify_answer = "Could you tell me a bit more about what you mean?"
                if not clarify_answer:
                    clarify_answer = "Could you tell me a bit more about what you mean?"

                meta_json = None
                if cfg.interaction.persistence_enabled and review_persistence is not None:
                    try:
                        meta_json = json.dumps(review_persistence)
                    except Exception:  # noqa: BLE001
                        meta_json = None
                self._save_message(
                    session_id, user_id, "assistant", clarify_answer,
                    metadata=meta_json,
                )
                self.db.commit()
                _emit("done", {
                    "total_s": time.time() - t_start,
                    "action": "clarify",
                })
                return ChatResult(
                    answer=clarify_answer,
                    session_id=session_id,
                    sources=results,
                    prompt="",
                )

            elif review_decision.action == "expand_doc" and review_decision.expand_from_doc_id:
                try:
                    from hrag.retrieval.doc_scope import DocScopedRetriever
                    scoped = DocScopedRetriever(
                        wrapped=self.retriever,
                        db=self.db,
                        embedder=self.embedder,
                    )
                    # DocScopedRetriever doesn't accept a hard doc_id —
                    # use a transient alias map.
                    scoped._aliases = {review_decision.expand_from_doc_id: [
                        review_decision.expand_from_doc_id
                    ]}
                    results = scoped.retrieve(
                        retrieval_query,
                        user_id=user_id,
                        top_k=cfg.retrieval.top_k_final,
                        intent_hint=intent,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Phase 8 expand_doc: re-retrieval failed")

            elif review_decision.action == "redescend" and review_decision.redirect_taxonomy_node_id:
                target_retriever = self._pick_retriever_for_intent(intent)
                tx_inner = target_retriever
                for _ in range(4):
                    if getattr(tx_inner, "name", "") == "taxonomy":
                        break
                    tx_inner = (
                        getattr(tx_inner, "wrapped", None)
                        or getattr(tx_inner, "inner", None)
                    )
                    if tx_inner is None:
                        break
                if tx_inner is not None and hasattr(tx_inner, "retrieve_from_node"):
                    try:
                        results = tx_inner.retrieve_from_node(
                            retrieval_query,
                            node_id=review_decision.redirect_taxonomy_node_id,
                            user_id=user_id,
                            top_k=cfg.retrieval.top_k_final,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("Phase 8 redescend failed")
                else:
                    _emit("review_warning", {
                        "reason": "redescend_unsupported",
                        "retriever": getattr(target_retriever, "name", "?"),
                    })

            # Optional episodic toggle (orthogonal to the action choice).
            if review_decision.include_episodic and plan.scope == "full":
                try:
                    active_retriever = self._pick_retriever_for_intent(intent)
                    extra = active_retriever.retrieve(
                        retrieval_query,
                        user_id=user_id,
                        top_k=cfg.retrieval.top_k_vector,
                        source_types=["episodic"],
                        intent_hint=intent,
                    )
                except Exception:  # noqa: BLE001
                    extra = []
                merged = list(results) + list(extra)
                seen: set[str] = set()
                dedup: list[RetrievalResult] = []
                for r in merged:
                    cid = r.chunk.chunk_id
                    if cid in seen:
                        continue
                    seen.add(cid)
                    dedup.append(r)
                results = dedup

        # 3d-iv. Phase 9.16 — CRAG-style automatic re-routing.
        # Light, no-user-prompt cousin of Phase 8's pause: when the post-rerank
        # top score is below the configured floor AND the user did NOT already
        # steer the turn via the interactive review, do ONE rewrite + retry
        # retrieval pass before falling through to the GENERAL swap. Capped at
        # one retry by design.
        if (
            cfg.retrieval.crag_enabled
            and review_decision.action == "continue"
            and intent == Intent.FACTUAL
            and plan.scope == "full"
        ):
            floor_crag = float(cfg.retrieval.crag_score_floor)
            top_score_crag = max((r.score for r in results), default=0.0)
            if not results or (floor_crag > 0.0 and top_score_crag < floor_crag):
                t0 = time.time()
                # Cheap heuristic broadening: ask the LLM (or the heuristic
                # rewriter) for an alternative phrasing one more time, then
                # retry retrieval with a wider top_k_vector.
                try:
                    crag_history = [Message(role=r, content=c) for r, c in history_rows]
                    crag_query = self.query_rewriter.rewrite(
                        question + " (alternative phrasing)",
                        crag_history,
                    )
                except Exception:  # noqa: BLE001
                    crag_query = retrieval_query
                widened_k = max(
                    int((adaptive_vec_k or cfg.retrieval.top_k_vector)
                        * float(cfg.retrieval.crag_retry_top_k_multiplier)),
                    cfg.retrieval.top_k_vector,
                )
                try:
                    crag_results = self._pick_retriever_for_intent(intent).retrieve(
                        crag_query,
                        user_id,
                        top_k=widened_k,
                        intent_hint=intent,
                    )
                except Exception:  # noqa: BLE001
                    crag_results = []
                if crag_results:
                    # Re-rank if we have a reranker; otherwise truncate.
                    if self.reranker is not None:
                        try:
                            crag_results = self.reranker.rerank(
                                crag_query,
                                crag_results,
                                threshold=cfg.retrieval.rerank_threshold,
                                top_k=adaptive_final_k or cfg.retrieval.top_k_final,
                            ) or crag_results[: cfg.retrieval.top_k_final]
                        except Exception:  # noqa: BLE001
                            crag_results = crag_results[: cfg.retrieval.top_k_final]
                    results = crag_results
                    retrieval_query = crag_query
                _emit("crag_reroute", {
                    "duration_s": time.time() - t0,
                    "n_results_before": 0 if not results else len(results),
                    "rewritten_query": crag_query,
                    "widened_top_k": widened_k,
                })

        # 3e. Post-retrieval re-route: a FACTUAL query that produced no
        # meaningfully-relevant results gets rewritten to GENERAL — the LLM
        # will then answer from world knowledge with a brief caveat instead
        # of forcing a thin RAFT answer about unrelated chunks.
        # Phase 8: only fire when the user did NOT explicitly choose a
        # different action above. ``continue`` covers both the no-pause case
        # (the default) and the explicit accept-defaults case.
        if review_decision.action == "continue" and intent == Intent.FACTUAL:
            floor = float(cfg.intent.corpus_relevance_floor)
            top_score = max((r.score for r in results), default=0.0)
            rerank_top = max(
                (r.rerank_score for r in results if r.rerank_score is not None),
                default=None,
            )
            weak = (
                (not results)
                or (floor > 0.0 and top_score < floor)
                or (rerank_top is not None and rerank_top < cfg.deep_read.weak_answer_floor)
            )

            # 3e-pre. Phase 13.1 (smart escalation, part b) — a weak FACTUAL turn
            # whose one-pass retrieval is thin gets ESCALATED into the agentic
            # deep read BEFORE generation (no wasted/duplicate streaming), rather
            # than degrading to a thin RAFT answer or the GENERAL swap. Requires
            # at least one result so the reader has a document to anchor on; a
            # zero-result off-corpus question stays GENERAL (handled by the swap).
            if (
                weak
                and results
                and stream
                and cfg.deep_read.enabled
                and cfg.deep_read.auto_trigger
                and cfg.deep_read.escalate_on_weak_answer
                and not reflective_turn
                and not cfg.interaction.review_enabled
            ):
                logger.info(
                    "deep_read: escalating weak one-pass FACTUAL "
                    "(top_score=%.3f floor=%.3f rerank_top=%s)",
                    top_score, floor, rerank_top,
                )
                esc = self._run_deep_read(
                    question=question,
                    user_id=user_id,
                    session_id=session_id,
                    history_rows=history_rows,
                    emit=_emit,
                    t_start=t_start,
                    escalated=True,
                )
                if esc is not None:
                    if _emb_token is not None:
                        try:
                            from hrag.providers.embeddings import (  # noqa: PLC0415
                                _session_var as _evs,
                            )
                            _evs.reset(_emb_token)
                        except Exception:  # noqa: BLE001
                            pass
                    return esc
                # No document matched → fall through to the swap below.

            if not results or (floor > 0.0 and top_score < floor):
                logger.info(
                    "FACTUAL → GENERAL swap: top_score=%.3f floor=%.3f n_results=%d",
                    top_score, floor, len(results),
                )
                intent = Intent.GENERAL
                results = []  # discard weak passages — GENERAL prompt has no slot
                _emit("intent_route", {
                    "intent": intent.value,
                    "scope": "none",
                    "swapped_from": "factual",
                    "top_score": top_score,
                })

        # 4. Render the prompt via the registry. kwargs must match each
        # template's placeholders exactly.
        ctx = self.context_builder.build(user_id)
        user_profile = ctx["user_profile"] or "(no profile yet)"
        conversation_history = _format_history(history_rows)

        if intent == Intent.FACTUAL:
            retrieved_passages = _format_passages(results)
            prompt = self.prompts.render(
                Intent.FACTUAL,
                user_profile=user_profile,
                conversation_history=conversation_history,
                retrieved_passages=retrieved_passages,
                question=question,
                detail_hint=_detail_hint(question),
            )
        elif intent == Intent.PERSONAL and reflective_turn:
            # Phase 11 — reflective impression. Synthesise from the user's saved
            # profile + episodic memories. Document chunks are included ONLY when
            # they actually mention the user: a reflective "impression of you"
            # must never recite a document's CONTENT (a novel's plot, a paper's
            # narrative) as the user's biography. A weak local model will ignore
            # a prompt instruction to that effect, so the robust guard is to keep
            # unrelated document text out of the prompt entirely.
            episodic_results = [
                r for r in results
                if getattr(r.chunk, "source_type", "") == "episodic"
            ]
            _uterms = _user_identifying_terms(user_profile)
            doc_results = [
                r for r in results
                if getattr(r.chunk, "source_type", "") != "episodic"
                and _chunk_is_about_user(getattr(r.chunk, "text", "") or "", _uterms)
            ]
            memories_block = (
                _format_passages(episodic_results)
                if episodic_results else "(nothing saved yet)"
            )
            docs_block = (
                _format_passages(doc_results)
                if doc_results else "(no personal documents on file)"
            )
            prompt = self.prompts.render_personal_reflect(
                user_profile=user_profile,
                retrieved_memories=memories_block,
                retrieved_docs=docs_block,
                conversation_history=conversation_history,
                question=question,
            )
        elif intent == Intent.PERSONAL:
            # Phase 8.2: dispatch to a sibling "empty" template when we
            # genuinely have nothing on file — no memories AND no profile.
            # The main personal template no longer carries the literal
            # fallback string as an example, because small Gemma-family
            # models were copy-pasting it verbatim on every PERSONAL turn.
            has_profile = bool(user_profile) and user_profile != "(no profile yet)"
            has_memory_context = bool(results) or has_profile

            # Phase 8.3 — friendly memory-led dispatch. When we have at
            # least one episodic memory AND no document chunk earned a
            # meaningfully-positive rerank score, the bot leads with the
            # memory, admits the limit, and offers to dig further. This
            # is the "I want a friend and assistant" path.
            has_episodic = any(
                getattr(r.chunk, "source_type", "") == "episodic"
                for r in results
            )
            has_strong_doc = any(
                getattr(r.chunk, "source_type", "") != "episodic"
                and (r.rerank_score is not None and r.rerank_score > 0.0)
                for r in results
            )
            if has_episodic and not has_strong_doc:
                episodic_results = [
                    r for r in results
                    if getattr(r.chunk, "source_type", "") == "episodic"
                ]
                memories_block = _format_passages(episodic_results)
                doc_results = [
                    r for r in results
                    if getattr(r.chunk, "source_type", "") != "episodic"
                ]
                if not doc_results:
                    docs_summary = "(no relevant documents found)"
                else:
                    titles = ", ".join(
                        f'"{(r.chunk.title or "untitled")[:60]}"'
                        for r in doc_results[:3]
                    )
                    docs_summary = (
                        "I looked but nothing matched well: " + titles
                    )
                prompt = self.prompts.render_personal_known(
                    retrieved_memories=memories_block,
                    retrieved_docs_summary=docs_summary,
                    conversation_history=conversation_history,
                    question=question,
                )
            elif has_memory_context:
                retrieved_passages = (
                    _format_passages(results) if results else "(none on file)"
                )
                prompt = self.prompts.render(
                    Intent.PERSONAL,
                    user_profile=user_profile,
                    conversation_history=conversation_history,
                    retrieved_passages=retrieved_passages,
                    question=question,
                )
            else:
                prompt = self.prompts.render_personal_empty(
                    conversation_history=conversation_history,
                    question=question,
                )
        elif intent == Intent.GREETING:
            prompt = self.prompts.render(
                Intent.GREETING,
                user_profile=user_profile,
                conversation_history=conversation_history,
                question=question,
            )
        elif intent == Intent.GENERAL:
            prompt = self.prompts.render(
                Intent.GENERAL,
                user_profile=user_profile,
                conversation_history=conversation_history,
                question=question,
            )
        else:  # Intent.UNCLEAR or any unmapped future intent
            prompt = self.prompts.render(
                Intent.UNCLEAR,
                user_profile=user_profile,
                conversation_history=conversation_history,
                question=question,
            )

        # 8b. Phase 9.13 — context compression for over-budget prompts.
        # When the rendered prompt blows past the configured budget, we collapse
        # the oldest half of history via DialogMSTCompactor (lazily instantiated
        # so the cold path pays nothing) AND drop the bottom-quartile of
        # retrieved passages by rerank_score, then re-render the prompt. Capped
        # at one pass — if it is still over budget, the model truncates and the
        # user gets a partial answer rather than an OOM.
        if (
            cfg.compaction.context_compression_enabled
            and len(prompt) > cfg.compaction.context_budget_chars
            and intent == Intent.FACTUAL
        ):
            t0 = time.time()
            chars_before = len(prompt)
            results_before = len(results)
            # Drop the lowest-scoring quartile of results.
            if results:
                k_keep = max(1, (len(results) * 3 + 3) // 4)
                results = sorted(
                    results,
                    key=lambda r: (
                        r.rerank_score if r.rerank_score is not None else r.score
                    ),
                    reverse=True,
                )[:k_keep]
            # Compress the oldest half of history.
            compressed_history_rows = history_rows
            if len(history_rows) >= 4:
                if self._budget_compactor is None:
                    self._budget_compactor = DialogMSTCompactor(
                        self.llm,
                        self.embedder,
                        compact_after_turns=max(2, len(history_rows) // 2),
                        keep_recent_turns=max(2, len(history_rows) // 2),
                        summary_target_tokens=cfg.compaction.summary_target_tokens,
                    )
                try:
                    msgs_in = [Message(role=r, content=c) for r, c in history_rows]
                    msgs_out = self._budget_compactor.compact(msgs_in)
                    compressed_history_rows = [(m.role, m.content) for m in msgs_out]
                except Exception:  # noqa: BLE001
                    logger.exception("Context compression: dialog compactor failed")
            conversation_history = _format_history(compressed_history_rows)
            # Re-render the prompt with the trimmed inputs.
            retrieved_passages = _format_passages(results)
            prompt = self.prompts.render(
                Intent.FACTUAL,
                user_profile=user_profile,
                conversation_history=conversation_history,
                retrieved_passages=retrieved_passages,
                question=question,
                detail_hint=_detail_hint(question),
            )
            _emit("context_compress", {
                "chars_before": chars_before,
                "chars_after": len(prompt),
                "passages_dropped": results_before - len(results),
                "duration_s": time.time() - t0,
            })

        # 9. Generate (streaming or one-shot).
        # Per-intent token caps: greetings should be one or two sentences,
        # not a 2048-token essay. Keeps "hey" turns under a second on a
        # small model fully on GPU.
        _intent_max_tokens: dict = {
            Intent.GREETING: 80,
            Intent.PERSONAL: 400,
            Intent.FACTUAL:  700,   # was uncapped → typical answer was ~1250 tok;
                                    # 700 covers a full RAFT response with a few quotes.
            Intent.GENERAL:  600,
            Intent.UNCLEAR:  200,
        }
        gen_max_tokens = _intent_max_tokens.get(intent)  # None → use cfg default
        t0 = time.time()
        _emit("generate_start", {"max_tokens": gen_max_tokens})
        first_token_ms: Optional[float] = None  # Phase 9.10
        if stream:
            from hrag.types import GenerationRequest  # noqa: PLC0415

            req = GenerationRequest(
                messages=[Message(role="user", content=prompt)],
                max_tokens=gen_max_tokens,
            )
            parts: list[str] = []
            for piece in self.llm.generate_stream(req):
                if first_token_ms is None and cfg.retrieval.first_token_latency_enabled:
                    first_token_ms = (time.time() - t0) * 1000.0
                parts.append(piece)
                _emit("generate_token", {"token": piece})
            answer = "".join(parts)
        else:
            answer = self.llm.complete(prompt, max_tokens=gen_max_tokens)
        gen_payload: dict[str, Any] = {
            "duration_s": time.time() - t0,
            "answer_chars": len(answer),
        }
        if first_token_ms is not None:
            gen_payload["first_token_ms"] = round(first_token_ms, 2)
        _emit("generate", gen_payload)

        # 9b-pre. Phase 9.17 — Self-RAG span re-retrieval.
        # Run BEFORE render/strip_uncertain so we still have the raw
        # ``[UNCERTAIN]`` tokens to anchor on. For each uncertain span we
        # extract, run one extra retrieval pass and append the snippets as a
        # "Sources for uncertain claims" block. Capped at
        # ``self_rag_max_spans`` queries to bound latency.
        if (
            cfg.compaction.self_rag_enabled
            and answer
            and "[UNCERTAIN]" in answer
            and intent in (Intent.FACTUAL, Intent.GENERAL)
        ):
            spans = extract_uncertain_spans(answer)[: cfg.compaction.self_rag_max_spans]
            extra_blocks: list[str] = []
            if spans:
                t0 = time.time()
                active_retriever_sr = self._pick_retriever_for_intent(intent)
                for span in spans:
                    try:
                        span_results = active_retriever_sr.retrieve(
                            span,
                            user_id,
                            top_k=3,
                            intent_hint=intent,
                        )
                    except Exception:  # noqa: BLE001
                        span_results = []
                    if not span_results:
                        continue
                    extra_blocks.append(
                        f"_For: \"{span[:120]}\"_\n" + _format_passages(span_results[:3])
                    )
                _emit("self_rag", {
                    "n_spans": len(spans),
                    "n_blocks_added": len(extra_blocks),
                    "duration_s": time.time() - t0,
                })
            if extra_blocks:
                answer = (
                    answer
                    + "\n\n---\n\n**Sources for uncertain claims:**\n\n"
                    + "\n\n".join(extra_blocks)
                )

        # 9b. Phase 4 — [UNCERTAIN] post-processing.
        # The answer prompt tells the LLM to write `[UNCERTAIN]` after any
        # sub-claim it cannot back with a quote. When ``mask_uncertain`` is on
        # we render those tokens as a visible warning glyph so the user sees
        # the model's self-doubt; otherwise we silently strip the raw token
        # so it never leaks. Stripping is unconditional (off-path stays quiet).
        if cfg.compaction.mask_uncertain:
            answer, n_uncertain = render_uncertain(answer)
            _emit("uncertain_render", {"count": n_uncertain})
        else:
            answer = strip_uncertain(answer)

        # 9c. Phase 7-A — formula extraction.
        # When the user asked a math-meta question and we have non-empty
        # results, run one extra LLM call that pulls every formula verbatim
        # from the retrieved passages and appends a clearly-marked block to
        # the answer. Failures are non-fatal — we keep the RAFT answer.
        if cfg.formula_extraction.enabled and math_meta and results:
            t0 = time.time()
            extraction_prompt = self.prompts.render_extract_formulas(
                retrieved_passages=_format_passages(results),
            )
            try:
                formulas_text = self.llm.complete(
                    extraction_prompt,
                    max_tokens=cfg.formula_extraction.max_tokens,
                    temperature=0.0,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Formula extraction failed; skipping")
                formulas_text = ""
            _emit("formula_extract", {
                "duration_s": time.time() - t0,
                "chars": len(formulas_text),
            })
            if formulas_text.strip():
                answer = (
                    answer
                    + "\n\n---\n\n**Extracted formulas:**\n\n"
                    + formulas_text.strip()
                )

        # 9d. Phase 8 — follow-up chip generation.
        # One extra LLM call producing up to 3 short follow-up questions.
        # Gated by both ``interaction.review_enabled`` and
        # ``interaction.followups_enabled``; default-off.
        if (
            cfg.interaction.review_enabled
            and cfg.interaction.followups_enabled
            and answer
        ):
            try:
                fu_template = (
                    Path(__file__).parent / "prompts" / "followups.md"
                ).read_text(encoding="utf-8")
                fu_prompt = fu_template.format(
                    question=question,
                    answer=answer[:1500],
                )
                raw_fu = self.llm.complete(
                    fu_prompt, temperature=0.5, max_tokens=120
                )
            except Exception:  # noqa: BLE001
                logger.exception("Phase 8 followups: LLM call failed")
                raw_fu = ""
            followups: list[str] = []
            for line in (raw_fu or "").splitlines():
                cleaned = _strip_bullet(line)
                if cleaned:
                    followups.append(cleaned)
                if len(followups) >= 3:
                    break
            if followups:
                _emit("followups", {"chips": followups})

        # 9c. Phase 8.3 — update the per-session "last intent + memory
        # entities" tracker. The intent classifier reads these on the NEXT
        # turn so a follow-up about a person the user already mentioned
        # ("what does Mahmoud do?") short-circuits to PERSONAL without
        # paying for an LLM call. Bounded at ``self._entity_cap`` per
        # session, evicting oldest entries first.
        self._session_last_intent[session_id] = intent
        if intent == Intent.PERSONAL and results:
            from hrag.intent import extract_entities  # noqa: PLC0415

            harvested: set[str] = set()
            for r in results:
                if getattr(r.chunk, "source_type", "") != "episodic":
                    continue
                harvested |= extract_entities(getattr(r.chunk, "text", "") or "")
                title = getattr(r.chunk, "title", "") or ""
                if title:
                    harvested |= extract_entities(title)
            if harvested:
                bucket = self._session_memory_entities.setdefault(session_id, [])
                seen = set(bucket)
                for ent in harvested:
                    if ent in seen:
                        continue
                    bucket.append(ent)
                    seen.add(ent)
                # Cap eviction: drop oldest entries until <= cap.
                if len(bucket) > self._entity_cap:
                    drop = len(bucket) - self._entity_cap
                    del bucket[:drop]

        # 10. Persist the assistant message.
        # Phase 8: when a real review pause happened AND persistence is on,
        # write the decision payload to ``messages.metadata``. The column
        # stays NULL on the no-pause / no-persistence happy path so existing
        # downstream readers see no change.
        assistant_metadata: Optional[str] = None
        metadata_payload: dict[str, Any] = {}
        if (
            cfg.interaction.persistence_enabled
            and review_persistence is not None
        ):
            metadata_payload.update(review_persistence)
        if cfg.retrieval.first_token_latency_enabled and first_token_ms is not None:
            metadata_payload["latency"] = {
                "first_token_ms": round(first_token_ms, 2),
            }
        if metadata_payload:
            try:
                assistant_metadata = json.dumps(metadata_payload)
            except Exception:  # noqa: BLE001
                assistant_metadata = None
        self._save_message(
            session_id, user_id, "assistant", answer,
            metadata=assistant_metadata,
        )
        self.db.commit()

        _emit("done", {"total_s": time.time() - t_start})

        # Phase 9.3 — release the ambient session before returning so the
        # contextvar does not leak across turns (or into other test cases).
        if _emb_token is not None:
            try:
                from hrag.providers.embeddings import _session_var as _emb_session_var  # noqa: PLC0415
                _emb_session_var.reset(_emb_token)
            except Exception:  # noqa: BLE001
                pass

        return ChatResult(
            answer=answer,
            session_id=session_id,
            sources=results,
            prompt=prompt,
        )

    def close(self) -> None:
        """Release database resources and stop background threads."""
        # Phase 8 — shut down the interaction-store reaper. Safe to call
        # multiple times; subsequent calls are no-ops.
        try:
            self.interaction_store.shutdown()
        except Exception:  # noqa: BLE001
            pass
        self.db.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reflective_llm_signal(
        self,
        question: str,
        history_rows: list[tuple[str, str]],
        preflight_decision: Optional[PreflightDecision],
    ) -> bool:
        """Hybrid-mode LLM reflective signal.

        Prefers the free signal from the combined preflight when it supplied a
        ``reflective`` field this turn; otherwise lazily builds and consults a
        cached :class:`ReflectiveClassifier` (one tiny yes/no call). The caller
        only invokes this when the regex tiers missed and the turn is plausibly
        personal, so the LLM cost stays on the rare ambiguous tail.
        """
        if preflight_decision is not None and preflight_decision.reflective is not None:
            return bool(preflight_decision.reflective)
        if self._reflective_classifier is None:
            try:
                self._reflective_classifier = ReflectiveClassifier(self.llm)
            except Exception:  # noqa: BLE001
                logger.exception("ReflectiveClassifier construction failed")
                return False
        return self._reflective_classifier.classify(question, history=history_rows)

    # ------------------------------------------------------------------
    # Phase 13 — agentic deep read
    # ------------------------------------------------------------------
    def _run_deep_read(
        self,
        *,
        question: str,
        user_id: str,
        session_id: str,
        history_rows: list[tuple[str, str]],
        emit: Callable[[str, dict], None],
        t_start: float,
        structural: bool = False,
        escalated: bool = False,
    ) -> Optional[ChatResult]:
        """Agentic multi-stage read of the single best-matching document.

        Each pass the planner picks ONE action from a closed menu — open a
        specific chapter deterministically (``read_part``), search within the
        doc (``search``), or stop and answer (``answer``) — with the full
        document map (parts + read/unread + page labels) always in front of it.
        ``read_part``/``read_page`` fetch a chunk_index range straight from
        SQLite (no vector search), implementing "now go read chapter 7".

        Returns a ChatResult, or None to fall back to the normal one-pass path
        (no document matched). Emits ``deep_read_start`` / ``section_opened`` /
        ``deep_read_pass`` (carrying the chosen ``action`` + ``arg``) /
        ``generate_token`` / ``followups``."""
        cfg = self.config
        dcfg = cfg.deep_read
        prompts_dir = Path(__file__).parent / "prompts"
        has_page = self._chunks_has_page()

        # 1. Seed retrieval → choose the document to read.
        try:
            seed = self.retriever.retrieve(question, user_id, top_k=dcfg.seed_top_k)
        except Exception:  # noqa: BLE001
            logger.exception("deep_read: seed retrieval failed")
            return None
        target = pick_target_doc(seed)
        if target is None:
            return None
        doc_id, doc_title = target

        # 2. Document map from the doc's chunks.
        cols = (
            "chunk_id, chunk_index, section, chapter, page"
            if has_page else "chunk_id, chunk_index, section"
        )
        rows = self.db.execute(
            f"SELECT {cols} FROM chunks "  # noqa: S608 — cols is a literal allow-list
            "WHERE doc_id = ? AND user_id = ? AND source_type = 'document' "
            "ORDER BY chunk_index",
            (doc_id, user_id),
        ).fetchall()
        if not rows:
            return None
        cid2idx = {r["chunk_id"]: (r["chunk_index"] or 0) for r in rows}

        def _row_page(r) -> Optional[int]:
            if not has_page:
                return None
            try:
                p = r["page"]
            except Exception:  # noqa: BLE001
                return None
            return p if (p is not None and p >= 0) else None

        def _row_label(r) -> str:
            if has_page:
                try:
                    ch = r["chapter"]
                except Exception:  # noqa: BLE001
                    ch = None
                if ch:
                    return ch
            return r["section"] or ""

        parts = build_parts(
            [((r["chunk_index"] or 0), _row_label(r)) for r in rows], n_parts=10
        )
        idx2page = {(r["chunk_index"] or 0): _row_page(r) for r in rows}
        pages_available = any(v is not None for v in idx2page.values())

        def _part_pages(p) -> tuple[Optional[int], Optional[int]]:
            pgs = [idx2page.get(i) for i in range(p.lo, p.hi + 1)]
            pgs = [x for x in pgs if x is not None]
            return (min(pgs), max(pgs)) if pgs else (None, None)

        part_pages = {p.idx: _part_pages(p) for p in parts}

        def _part_public(p) -> dict:
            d = p.public()
            lo_pg, hi_pg = part_pages.get(p.idx, (None, None))
            if lo_pg is not None:
                d["page_lo"], d["page_hi"] = lo_pg, hi_pg
            return d

        state = DeepReadState(doc_id=doc_id, doc_title=doc_title, parts=parts)
        emit("deep_read_start", {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "question": question,
            "parts": [_part_public(p) for p in parts],
            "pages_available": pages_available,
            "structural": structural,
            "escalated": escalated,
        })
        logger.info(
            "deep_read: reading %r across %d parts (structural=%s escalated=%s)",
            doc_title, len(parts), structural, escalated,
        )

        all_results: list[RetrievalResult] = []

        def _page_range_to_idx(lo_pg, hi_pg) -> tuple[Optional[int], Optional[int]]:
            idxs = [
                i for i, pg in idx2page.items()
                if pg is not None and lo_pg <= pg <= hi_pg
            ]
            return (min(idxs), max(idxs)) if idxs else (None, None)

        def _record(fresh: list[RetrievalResult]) -> None:
            touched: dict = {}
            for r in fresh:
                cid = getattr(r.chunk, "chunk_id", None)
                if cid:
                    state.seen_chunk_ids.add(cid)
                idx = cid2idx.get(cid, getattr(r.chunk, "chunk_index", 0) or 0)
                p, _ = state.open_for_index(idx)
                if p is not None:
                    touched[p.idx] = p
            all_results.extend(fresh)
            for p in sorted(touched.values(), key=lambda x: x.idx):
                payload = {"idx": p.idx, "label": p.label, "quotes": p.quotes}
                lo_pg, hi_pg = part_pages.get(p.idx, (None, None))
                if lo_pg is not None:
                    payload["page_lo"], payload["page_hi"] = lo_pg, hi_pg
                emit("section_opened", payload)

        def _search_within_doc(q: str) -> list[RetrievalResult]:
            try:
                cand = self.retriever.retrieve(
                    q or question, user_id,
                    top_k=max(40, dcfg.chunks_per_pass * 6),
                    where={"doc_id": {"$eq": doc_id}},
                )
            except Exception:  # noqa: BLE001
                logger.exception("deep_read: search retrieval failed")
                cand = []
            fresh: list[RetrievalResult] = []
            for r in cand:
                ch = r.chunk
                if getattr(ch, "doc_id", None) != doc_id:
                    continue
                cid = getattr(ch, "chunk_id", None)
                if not cid or cid in state.seen_chunk_ids:
                    continue
                fresh.append(r)
                if len(fresh) >= dcfg.chunks_per_pass:
                    break
            return fresh

        def _force_next_unread() -> list[RetrievalResult]:
            for p in state.parts:
                if p.status != "read":
                    got = self._read_part_chunks(
                        doc_id, user_id, p.lo, p.hi,
                        exclude=state.seen_chunk_ids,
                        limit=dcfg.chunks_per_pass, has_page=has_page,
                    )
                    if got:
                        return got
            return []

        # 3. Pass 0 — seed gives the planner something to plan from.
        last_results: list[RetrievalResult] = []
        for r in seed:
            ch = r.chunk
            if getattr(ch, "doc_id", None) != doc_id:
                continue
            cid = getattr(ch, "chunk_id", None)
            if not cid or cid in state.seen_chunk_ids:
                continue
            last_results.append(r)
            if len(last_results) >= dcfg.chunks_per_pass:
                break
        if last_results:
            _record(last_results)

        # 4. Read→plan passes — the planner picks one navigation action each pass.
        for pass_no in range(1, dcfg.max_passes + 1):
            state.passes = pass_no
            action = self._deep_read_plan_action(
                prompts_dir,
                state=state, question=question, doc_title=doc_title,
                last_results=last_results, pages_available=pages_available,
                part_pages=part_pages,
            )
            if action.note:
                state.notes.append(action.note)

            # Narration arg (read by the live document-map panel) — built BEFORE
            # executing so the UI can say "Opening Chapter 7" immediately.
            arg: dict = {}
            if action.kind == "read_part" and action.part_idx is not None:
                tp = state.parts[action.part_idx]
                arg = {"idx": tp.idx, "label": tp.label}
                lo_pg, hi_pg = part_pages.get(tp.idx, (None, None))
                if lo_pg is not None:
                    arg["page_lo"], arg["page_hi"] = lo_pg, hi_pg
            elif action.kind == "read_page":
                arg = {"page_lo": action.lo, "page_hi": action.hi}
            elif action.kind == "search":
                arg = {"query": action.query}
            emit("deep_read_pass", {
                "pass": pass_no,
                "action": action.kind,
                "arg": arg,
                "note": action.note,
                # Back-compat keys for older frontends:
                "next_query": action.query if action.kind == "search" else "",
                "sections_read": [],
            })

            # Decide & execute the action.
            if action.kind == "answer":
                if pass_no >= dcfg.min_passes:
                    break
                fresh = _force_next_unread()  # too early to stop — keep reading
            elif action.kind == "read_part":
                fresh = self._read_part_chunks(
                    doc_id, user_id, action.lo, action.hi,
                    exclude=state.seen_chunk_ids,
                    limit=dcfg.chunks_per_pass, has_page=has_page,
                )
            elif action.kind == "read_page":
                lo_i, hi_i = _page_range_to_idx(action.lo, action.hi)
                fresh = (
                    _search_within_doc(question) if lo_i is None
                    else self._read_part_chunks(
                        doc_id, user_id, lo_i, hi_i,
                        exclude=state.seen_chunk_ids,
                        limit=dcfg.chunks_per_pass, has_page=has_page,
                    )
                )
            else:  # search
                fresh = _search_within_doc(action.query)

            if not fresh:
                if pass_no >= dcfg.min_passes:
                    break
                fresh = _force_next_unread()
                if not fresh:
                    break

            _record(fresh)
            last_results = fresh
            if state.remaining() == 0 and pass_no >= dcfg.min_passes:
                break

        # 4b. Structural scan-all (optional) — open every remaining part for full
        # coverage before a counting/structure synthesis. Cheap (all SQL).
        if structural and dcfg.structural_scan_all:
            for p in state.parts:
                if p.status != "read":
                    got = self._read_part_chunks(
                        doc_id, user_id, p.lo, p.hi,
                        exclude=state.seen_chunk_ids,
                        limit=dcfg.chunks_per_pass, has_page=has_page,
                    )
                    if got:
                        _record(got)

        # 5. Structural block — for "how many chapters / table of contents" the
        # honest answer is derived from the document's visible structure, not
        # from any single chunk. Enumerate in code; the model presents it with a
        # caveat and never invents chapters.
        structural_block = ""
        if structural:
            labels = distinct_chapter_labels(rows)
            toc = find_toc_chunk(self._deep_read_early_rows(doc_id, user_id))
            toc_block = (
                toc[1].strip() if toc
                else "(no explicit table of contents found in the text)"
            )
            listing = (
                "\n".join(f"{i}. {lab}" for i, lab in enumerate(labels, 1))
                or "(no clean headings detected)"
            )
            structural_block = (
                "### Document structure (derived from headings/TOC — may be approximate)\n"
                f"Distinct chapter/section headings detected: {len(labels)}\n"
                f"Coarse parts the reader split the document into: {len(parts)}\n\n"
                f"Headings:\n{listing}\n\n"
                f"Table of contents found in the document:\n{toc_block}\n\n"
                "When the user asks how many chapters/sections/pages or for the table "
                "of contents, base your answer on this structure. State plainly that "
                "the count is derived from the document's visible headings and may be "
                "approximate. Never invent chapters not listed here.\n"
            )

        # 6. Final synthesis (streamed into the answer bubble).
        synth_prompt = (prompts_dir / "deep_read_synthesize.md").read_text(
            encoding="utf-8"
        ).format(
            doc_title=doc_title,
            question=question,
            notes="\n".join(f"- {n}" for n in state.notes) or "(no notes gathered)",
            passages=_format_passages(all_results[:8]),
            structural_block=structural_block,
        )
        emit("generate_start", {"deep_read": True})
        from hrag.types import GenerationRequest  # noqa: PLC0415

        req = GenerationRequest(
            messages=[Message(role="user", content=synth_prompt)], max_tokens=900
        )
        out: list[str] = []
        for piece in self.llm.generate_stream(req):
            out.append(piece)
            emit("generate_token", {"token": piece})
        answer = strip_uncertain("".join(out).strip())
        emit("generate", {"answer_chars": len(answer), "deep_read": True})

        # 7. Follow-up suggestions grounded in what was read.
        chips = self._deep_read_followups(prompts_dir, question, answer, dcfg.followups)
        if chips:
            emit("followups", {"chips": chips})  # reuse the existing chip UI

        # 8. Persist + return.
        self._save_message(session_id, user_id, "assistant", answer)
        self.db.commit()
        self._session_last_intent[session_id] = Intent.FACTUAL
        emit("done", {
            "total_s": time.time() - t_start, "deep_read": True,
            "passes": state.passes, "parts_read": len(parts) - state.remaining(),
            "structural": structural, "escalated": escalated,
        })
        return ChatResult(
            answer=answer, session_id=session_id, sources=all_results, prompt=synth_prompt
        )

    def _chunks_has_page(self) -> bool:
        """True when the ``chunks`` table carries the Phase-13.1 ``page`` +
        ``chapter`` columns. Cached after the first PRAGMA so the deep read
        degrades gracefully (page reads → part reads) on a pre-13.1 DB."""
        cached = getattr(self, "_chunks_has_page_cache", None)
        if cached is not None:
            return cached
        has = False
        try:
            info = self.db.execute("PRAGMA table_info(chunks)").fetchall()
            names = {row["name"] for row in info}
            has = "page" in names and "chapter" in names
        except Exception:  # noqa: BLE001
            has = False
        self._chunks_has_page_cache = has
        return has

    def _read_part_chunks(
        self, doc_id: str, user_id: str, lo: Optional[int], hi: Optional[int],
        *, exclude: set, limit: int, has_page: bool = False,
    ) -> list[RetrievalResult]:
        """Deterministically read the document's chunks in the ``[lo, hi]``
        chunk_index range straight from SQLite — NO vector search. This is the
        engine of the ``read_part`` / ``read_page`` actions ("go read chapter
        7"). Skips already-seen chunks; caps at ``limit``."""
        if lo is None or hi is None:
            return []
        cols = (
            "chunk_id, doc_id, user_id, text, title, section, subsection, "
            "chunk_index, source_type"
        )
        if has_page:
            cols += ", page, chapter"
        rows = self.db.execute(
            f"SELECT {cols} FROM chunks "  # noqa: S608 — cols is a literal allow-list
            "WHERE doc_id = ? AND user_id = ? AND source_type = 'document' "
            "AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index",
            (doc_id, user_id, lo, hi),
        ).fetchall()
        out: list[RetrievalResult] = []
        for r in rows:
            cid = r["chunk_id"]
            if cid in exclude:
                continue
            page: Optional[int] = None
            meta: dict = {}
            if has_page:
                try:
                    pg = r["page"]
                    page = pg if (pg is not None and pg >= 0) else None
                except Exception:  # noqa: BLE001
                    page = None
                try:
                    if r["chapter"]:
                        meta["chapter"] = r["chapter"]
                except Exception:  # noqa: BLE001
                    pass
            text = r["text"] or ""
            chunk = Chunk(
                chunk_id=cid,
                doc_id=r["doc_id"],
                user_id=r["user_id"],
                text=text,
                embedding_text=text,
                title=r["title"] or "",
                section=r["section"] or "",
                subsection=r["subsection"] or "",
                chunk_index=r["chunk_index"] or 0,
                source_type=r["source_type"] or "document",
                page=page,
                metadata=meta,
            )
            out.append(RetrievalResult(chunk=chunk, score=0.0, retriever="deep_read"))
            if len(out) >= limit:
                break
        return out

    def _deep_read_early_rows(self, doc_id: str, user_id: str, limit: int = 20) -> list:
        """First ``limit`` chunks (chunk_index, text) of a document — fed to
        ``find_toc_chunk`` to surface a real table-of-contents chunk."""
        return self.db.execute(
            "SELECT chunk_index, text FROM chunks "
            "WHERE doc_id = ? AND user_id = ? AND source_type = 'document' "
            "ORDER BY chunk_index LIMIT ?",
            (doc_id, user_id, limit),
        ).fetchall()

    def _deep_read_plan_action(
        self, prompts_dir: Path, *, state: DeepReadState, question: str,
        doc_title: str, last_results: list[RetrievalResult],
        pages_available: bool, part_pages: dict,
    ):
        """One planning call: render the document map + the passages just read,
        ask the weak model to pick ONE action, then parse + harden it via the
        pure ``parse_action`` (clamps / redirects / downgrades — never raises)."""
        dcfg = self.config.deep_read
        structure = _format_doc_map(state, part_pages)
        prompt = (prompts_dir / "deep_read_pass.md").read_text(encoding="utf-8").format(
            doc_title=doc_title,
            question=question,
            structure=structure,
            notes_so_far="\n".join(f"- {n}" for n in state.notes) or "(nothing yet)",
            passages=_format_passages(last_results),
        )
        try:
            raw = self.llm.complete(
                prompt, temperature=0.1, max_tokens=dcfg.plan_max_tokens
            )
        except Exception:  # noqa: BLE001
            logger.exception("deep_read: plan LLM call failed")
            # Keep the read alive: answer past min_passes, else search the
            # original question for the next-best unseen chunks.
            fallback = (
                {"action": "answer"} if state.passes >= dcfg.min_passes
                else {"action": "search", "query": ""}
            )
            return parse_action(fallback, state, pages_available=pages_available)
        return parse_action(_loose_json(raw), state, pages_available=pages_available)

    def _deep_read_followups(
        self, prompts_dir: Path, question: str, answer: str, n: int
    ) -> list[str]:
        try:
            tmpl = (prompts_dir / "followups.md").read_text(encoding="utf-8")
            raw = self.llm.complete(
                tmpl.format(question=question, answer=answer[:1500]),
                temperature=0.5, max_tokens=120,
            )
        except Exception:  # noqa: BLE001
            return []
        chips: list[str] = []
        for line in (raw or "").splitlines():
            c = _strip_bullet(line)
            if c:
                chips.append(c)
            if len(chips) >= n:
                break
        return chips

    def _create_session(self, session_id: str, user_id: str) -> None:
        with self.db.conn:
            self.db.execute(
                "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
                (session_id, user_id),
            )

    def _save_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[str] = None,
    ) -> None:
        """Insert a message row. ``metadata`` is an optional JSON string
        persisted to the Phase-8 ``messages.metadata`` column (nullable)."""
        with self.db.conn:
            self.db.execute(
                """
                INSERT INTO messages (session_id, user_id, role, content, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, role, content, metadata),
            )

    def _log_rerank_fallback(
        self,
        *,
        turn_id: str,
        session_id: str,
        user_id: str,
        query: str,
        dropped_chunk_ids: list[str],
    ) -> None:
        """Insert a Phase-9.9 rerank-fallback telemetry row."""
        with self.db.conn:
            self.db.execute(
                """
                INSERT INTO rerank_fallback_events
                    (event_id, turn_id, session_id, user_id, query, dropped_chunk_ids)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    turn_id,
                    session_id,
                    user_id,
                    query,
                    json.dumps(dropped_chunk_ids),
                ),
            )

    def _load_history(
        self, session_id: str, limit: int = 10
    ) -> list[tuple[str, str]]:
        """Return up to *limit* (role, content) pairs, oldest-first.

        Excludes the message just inserted (we pick the last N before the newest
        user turn by fetching N+1 and dropping the last row, but since we already
        inserted the user message above we use a sub-select with LIMIT to get the
        rows that existed *before* this turn).
        """
        # We want the 10 messages before the most recently inserted user message.
        # Because we just inserted it, the most recent row is ours; skip it.
        cursor = self.db.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY message_id ASC
            LIMIT ?
            OFFSET MAX(0,
                (SELECT COUNT(*) FROM messages WHERE session_id = ?) - ?
            )
            """,
            (session_id, limit, session_id, limit + 1),
        )
        rows = cursor.fetchall()
        # The last row is the current user turn — exclude it.
        pairs = [(r["role"], r["content"]) for r in rows]
        if pairs and pairs[-1][0] == "user":
            pairs = pairs[:-1]
        return pairs

    # ------------------------------------------------------------------
    # Phase 9.4 — Ollama warm-up + num_keep auto-tune
    # ------------------------------------------------------------------

    def _maybe_warmup_llm(self, cfg: Config) -> None:
        """Pre-load the Ollama model into VRAM and optionally auto-set num_keep.

        Split into its own method so tests can exercise the logic directly on a
        MagicMock-based "orchestrator" without instantiating a full pipeline.

        Warm-up: fires a 1-token chat() with the configured ``keep_alive`` so
        the model is resident before the user's first turn.  Guarded by both the
        config flag (``cfg.llm.warmup_on_init``) and a provider sniff
        (``self.llm.name == "ollama"``).  Errors are logged at DEBUG and
        silently swallowed — a missing/sleeping Ollama server must not crash
        startup.

        Auto-tune: when ``cfg.llm.num_keep_auto`` is True **and**
        ``cfg.llm.num_keep`` is not already set, estimates the prefix length
        from ``prompts/answer.md`` and stores it back into ``cfg.llm`` for the
        session.
        """
        # --- auto-tune num_keep (runs before warmup so warmup picks it up) ---
        if getattr(cfg.llm, "num_keep_auto", False) and not getattr(cfg.llm, "num_keep", None):
            try:
                from hrag.providers.llm import estimate_num_keep  # noqa: PLC0415

                prompt_path = Path(__file__).parent / "prompts" / "answer.md"
                prompt_text = prompt_path.read_text(encoding="utf-8")
                prefix = prompt_text.split("{user_profile}")[0]
                estimated = estimate_num_keep(prefix)
                cfg.llm.num_keep = estimated
                logger.debug(
                    "[orchestrator] auto-set num_keep=%d from answer.md prefix", estimated
                )
            except Exception as exc:
                logger.debug("[orchestrator] num_keep auto-tune skipped: %s", exc)

        # --- warm-up ping ---
        if (
            getattr(cfg.llm, "warmup_on_init", True)
            and getattr(self.llm, "name", "") == "ollama"
        ):
            try:
                logger.debug("[orchestrator] firing Ollama warm-up ping for model=%s", cfg.llm.model)
                self.llm.warmup()
                logger.debug("[orchestrator] Ollama warm-up complete")
            except Exception as exc:
                logger.debug("[orchestrator] LLM warmup skipped: %s", exc)
        else:
            logger.debug(
                "[orchestrator] LLM warmup skipped (warmup_on_init=%s, provider=%s)",
                getattr(cfg.llm, "warmup_on_init", True),
                getattr(self.llm, "name", "unknown"),
            )

    # ------------------------------------------------------------------
    # Phase 10 — embedding dim mismatch guard
    # ------------------------------------------------------------------

    def _check_embedding_dim_match(self) -> None:
        """Refuse to start when the stored index dim disagrees with cfg.embeddings.dim.

        Reads one vector from the populated chunks collection via the backend's
        ``dim()`` helper. Skips silently when the collection is empty (fresh
        install or pre-ingest state) so the check is a true no-op on first run.

        Raises :class:`RuntimeError` with a clear re-ingest command when a
        mismatch is detected. Any other unexpected error (e.g. the backend is a
        test stub without a real ``dim()`` implementation) is caught and logged
        at WARNING so tests and non-Chroma paths are never broken by this guard.
        """
        try:
            backend = self.vector_store._backend
            stored_dim = backend.dim()
            if stored_dim is None:
                # Collection is empty or backend cannot introspect dim — skip.
                return
            configured_dim: int = self.config.embeddings.dim
            if stored_dim != configured_dim:
                raise RuntimeError(
                    f"Embedding dim mismatch: the index has dim={stored_dim} but "
                    f"config.embeddings.dim={configured_dim}. The embedding model "
                    f"was changed without re-ingesting the corpus. Run:\n"
                    f"  hrag init --wipe && hrag ingest <your-corpus>\n"
                    f"to rebuild the index under the new model."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("[orchestrator] dim-check skipped: %s", exc)


# ---------------------------------------------------------------------------
# Adaptive top_k resolver (Phase 6)
# ---------------------------------------------------------------------------


def _adaptive_top_k(cfg, intent) -> tuple[Optional[int], Optional[int]]:
    """Return ``(top_k_vector, top_k_final)`` for *intent*.

    When ``retrieval.adaptive_enabled`` is False, returns the global defaults
    so the rest of the pipeline is byte-for-byte unchanged.

    When adaptive is enabled, looks up ``retrieval.adaptive_top_k[intent]``;
    a value of 0 signals "skip retrieval entirely" and is reported as
    ``(None, None)``. Otherwise ``top_k_vector`` is widened to
    ``max(per_intent * 2, 12)`` so the reranker has slack to filter from.

    Unknown intent keys fall back to ``cfg.retrieval.top_k_final``.
    """
    if not cfg.retrieval.adaptive_enabled:
        return cfg.retrieval.top_k_vector, cfg.retrieval.top_k_final
    # Intent is a str-Enum, so .value is already lowercase ("greeting", etc.)
    key = intent.value.lower()
    per_intent = cfg.retrieval.adaptive_top_k.get(key, cfg.retrieval.top_k_final)
    if per_intent == 0:
        return None, None  # skip retrieval
    vec_k = max(per_intent * 2, 12)
    return vec_k, per_intent


# ---------------------------------------------------------------------------
# Detail-hint helper
# ---------------------------------------------------------------------------

# Question-shape signals that bias the answer toward a more thorough response.
_DETAIL_PATTERNS = (
    "in detail", "in depth", "in-depth",
    "elaborate", "walk me through", "thoroughly",
    "explain fully", "comprehensive", "step by step",
    "step-by-step", "deep dive", "full explanation",
)


def _detail_hint(question: str) -> str:
    q = question.lower()
    if any(p in q for p in _DETAIL_PATTERNS):
        return (
            "The user asked for depth — write a thorough, multi-paragraph "
            "answer. Cover the topic from multiple angles using every relevant "
            "source. Aim for several hundred words when the material supports it."
        )
    return (
        "Match length to question complexity. Short questions deserve short "
        "answers; substantive questions deserve substantive ones."
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


_RE_LEADING_BULLET = re.compile(r"^[\s\-*•]*\d*[.)]?\s*")


def _strip_bullet(line: str) -> str:
    """Strip a leading bullet / numbering from a follow-up line.

    Used by the Phase 8 follow-up chip generator to clean up bullet lists
    the LLM occasionally returns despite the prompt asking for plain lines.
    """
    return _RE_LEADING_BULLET.sub("", line).strip()


def _format_passages(results: list[RetrievalResult]) -> str:
    if not results:
        return "(no passages retrieved)"
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        chunk = r.chunk
        header = f"[Source {i} | {chunk.title or 'Untitled'} | {chunk.section or 'N/A'}]"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)


def _format_doc_map(state: "DeepReadState", part_pages: dict) -> str:
    """Render the document map as a numbered pick-list for the deep-read planner.
    Reflects each part's live read/unread status so the weak model only has to
    choose an index from a visible list. Phase 13.1."""
    lines: list[str] = []
    for p in state.parts:
        status = "READ" if p.status == "read" else "unread"
        lo_pg, hi_pg = part_pages.get(p.idx, (None, None))
        pg = f" (pages {lo_pg}-{hi_pg})" if lo_pg is not None else ""
        lines.append(f" [{p.idx}] {p.label}{pg}    {status}")
    return "\n".join(lines) or "(document has no parts)"


# Words that show up in profile renders but don't identify the user — ignored
# when deciding whether a document chunk is actually ABOUT the user (Phase 11
# reflective-synthesis guard).
_PROFILE_STOPWORDS = frozenset({
    "facts", "fact", "know", "knows", "about", "your", "yours", "user",
    "users", "profile", "none", "name", "named", "call", "called", "prefers",
    "prefer", "likes", "like", "interested", "interest", "interests", "works",
    "work", "working", "based", "from", "with", "that", "this", "they", "them",
    "their", "here", "what", "have", "has", "the", "and", "for", "you",
})


def _user_identifying_terms(profile: str) -> set[str]:
    """Distinctive tokens (≥4 chars, non-stopword) from a profile render, used
    to test whether a document chunk is actually about the user. Empty set for
    an empty / "(no profile yet)" profile — callers then exclude ALL documents
    from the reflective synthesis."""
    if not profile or profile.strip().lower() in {"", "(no profile yet)"}:
        return set()
    return {
        t for t in re.findall(r"[a-z0-9]{4,}", profile.lower())
        if t not in _PROFILE_STOPWORDS
    }


def _chunk_is_about_user(text: str, terms: set[str]) -> bool:
    """True only when a document chunk references a distinctive user term — so a
    reflective impression can never recast unrelated document content (e.g. a
    novel's plot) as the user's biography. Empty terms ⇒ never about the user."""
    if not terms or not text:
        return False
    low = text.lower()
    return any(t in low for t in terms)


def _loose_json(raw: str) -> dict:
    """Best-effort parse of the first ``{...}`` object in an LLM reply. Returns
    {} on failure — small models wrap JSON in prose or fences."""
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _format_history(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return "(no prior conversation)"
    lines: list[str] = []
    for role, content in pairs:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)
