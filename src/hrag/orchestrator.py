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
from hrag.gating.gate import RAGate
from hrag.gating.uncertain import render_uncertain, strip_uncertain
from hrag.ingest.pipeline import IngestPipeline
from hrag.intent import Intent, IntentClassifier, IntentVerdict
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
from hrag.types import Message, RetrievalResult

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
        # whole tracker is process-local — fine for a single-user Streamlit
        # / FastAPI process. Persistence across restarts is intentionally
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
        # — discoverable in the streamlit log if a fast-path rule regresses.
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

        # 3d. Retrieval-policy dispatch.
        plan: RetrievalPlan = self.retrieval_policy.plan(intent)
        _emit("intent_route", {
            "intent": intent.value,
            "scope": plan.scope,
            "top_k": plan.top_k_override,
            "source_types": plan.source_types,
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
            try:
                gate_history = [Message(role=r, content=c) for r, c in history_rows]
                gate_decision = self.gate.decide(question, gate_history)
            except Exception:  # noqa: BLE001 — fail open so chat never breaks
                logger.exception("RAGate failed; defaulting to RETRIEVE")
                gate_decision = "RETRIEVE"
            _emit(
                "gate_check",
                {"decision": gate_decision, "duration_s": time.time() - t0},
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
            try:
                clue_history = [Message(role=r, content=c) for r, c in history_rows]
                clue_text = self.clue.generate(retrieval_query, clue_history)
            except Exception:  # noqa: BLE001 — fail soft to the raw query
                logger.exception("ClueGenerator failed; using raw query")
                clue_text = retrieval_query
            _emit(
                "clue_generate",
                {"clue": clue_text, "duration_s": time.time() - t0},
            )
            if clue_text and clue_text.strip():
                retrieval_query = clue_text

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

            t0 = time.time()
            # Adaptive top_k: when retrieval.adaptive_enabled is true, vec_k is
            # the per-intent value (widened by 2x for reranker slack); else the
            # global default. Falls back defensively if the resolver somehow
            # returned None on a non-skip path.
            full_vec_k = adaptive_vec_k if adaptive_vec_k is not None else cfg.retrieval.top_k_vector
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

                results = reranked
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
        if stream:
            from hrag.types import GenerationRequest  # noqa: PLC0415

            req = GenerationRequest(
                messages=[Message(role="user", content=prompt)],
                max_tokens=gen_max_tokens,
            )
            parts: list[str] = []
            for piece in self.llm.generate_stream(req):
                parts.append(piece)
                _emit("generate_token", {"token": piece})
            answer = "".join(parts)
        else:
            answer = self.llm.complete(prompt, max_tokens=gen_max_tokens)
        _emit(
            "generate",
            {"duration_s": time.time() - t0, "answer_chars": len(answer)},
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
        if (
            cfg.interaction.persistence_enabled
            and review_persistence is not None
        ):
            try:
                assistant_metadata = json.dumps(review_persistence)
            except Exception:  # noqa: BLE001
                assistant_metadata = None
        self._save_message(
            session_id, user_id, "assistant", answer,
            metadata=assistant_metadata,
        )
        self.db.commit()

        _emit("done", {"total_s": time.time() - t_start})

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


def _format_history(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return "(no prior conversation)"
    lines: list[str] = []
    for role, content in pairs:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)
