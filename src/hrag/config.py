"""Pydantic-backed configuration loader.

Loads `config.yaml` from the project root by default. Environment variables prefixed
with HRAG_ override fields using `__` as the section separator, e.g.
HRAG_LLM__MODEL=gemma:7b sets `llm.model`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class QualityConfig(BaseModel):
    """Settings for the post-chunking quality filter (hrag.ingest.quality)."""

    enabled: bool = True
    min_tokens: int = 30
    min_chars: int = 80
    min_alpha_ratio: float = 0.4
    drop_references_sections: bool = True
    drop_bibliography_chunks: bool = True
    drop_page_artifacts: bool = True
    dedupe: bool = True


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "gemma3:latest"
    temperature: float = 0.2
    max_tokens: int = 1024
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Per-provider keys — stored separately so all providers can be
    # pre-configured at once. Each provider falls back to api_key, then its
    # env var (OPENROUTER_API_KEY / GEMINI_API_KEY).
    openrouter_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    # Ollama-only: context window size sent as `num_ctx`. Ollama otherwise defaults
    # to the model's max (e.g. 131k for gemma4), allocating a huge KV cache that
    # forces CPU spill on small GPUs. 8192 covers hrag's realistic prompts.
    num_ctx: Optional[int] = 8192
    # Ollama-only: when the model declares the `thinking` capability, set this to
    # False to suppress reasoning tokens. hrag's gate / clue / classifier prompts
    # are short and the few-token caps cut off mid-thought, so reasoning here is
    # pure latency tax. None = leave the model's default behaviour.
    think: Optional[bool] = False
    # Phase 6 — Ollama only. Forwarded to ollama.Client.chat(keep_alive=...).
    # The Ollama server unloads a model from VRAM after `keep_alive` of
    # inactivity (default 5m). hrag chats are bursty (one user turn, then
    # silence, then another turn 30s later), and reloading the model is a
    # several-second hit. "30m" keeps it resident through normal use; "-1s"
    # never unloads. None = let the Ollama server use its own default.
    keep_alive: Optional[str] = "30m"
    # Phase 6-B — Ollama options.num_keep. The Ollama server uses the first
    # N tokens of the previous prompt to seed the KV cache on the next chat,
    # so a stable system + history prefix can be reused across turns. Effect
    # is best-effort: some models / quantizations ignore it. None or 0 leaves
    # Ollama to its own default (usually 4 BOS tokens). Set to roughly the
    # token count of your system prompt + a small slack to reuse it across
    # turns in the same session.
    num_keep: Optional[int] = None
    # Phase 9.5 — Anthropic-only. When True, the AnthropicProvider wraps the
    # system prompt and the last user message in cache_control={"type":
    # "ephemeral"} blocks so Anthropic's prompt cache can reuse them across
    # turns. Cached writes cost 1.25x, cached reads cost 0.1x — a multi-turn
    # FACTUAL session with a stable system + retrieval prefix saves 30–80% on
    # cost and latency. Default OFF: existing tests + callers see a string
    # `system=` and string `content=` (byte-identical to today). No effect on
    # Ollama or OpenAI paths.
    anthropic_prompt_caching: bool = False
    # Phase 9.4 — Ollama warm-up ping. When True (the default), the
    # Orchestrator fires a 1-token chat() during __init__ to pre-load the
    # model into VRAM. This eliminates the 2–4 s cold-load tax on the user's
    # first turn. The flag is a no-op for OpenAI/Anthropic (no concept of a
    # resident model). Default ON because it is a pure speed win on Ollama and
    # costs nothing when the model is already loaded.
    warmup_on_init: bool = True
    # Phase 9.4 — auto-compute num_keep from the answer-prompt prefix. When
    # True AND num_keep has NOT been explicitly set, the Orchestrator reads
    # prompts/answer.md, takes the prefix before {user_profile}, estimates its
    # token count at ~4 chars/token, and stores the result in cfg.llm.num_keep
    # so it persists through the session. Default OFF — opt in when you want
    # the KV-cache prefix to survive across turns without guessing the right
    # num_keep by hand.
    num_keep_auto: bool = False


class EmbeddingsConfig(BaseModel):
    provider: str = "sentence-transformers"
    model: str = "sentence-transformers/all-mpnet-base-v2"
    dim: int = 768
    # Phase 7-B — curated alternative model presets for the GUI selector.
    # The provider already supports any Hugging Face model; this list just
    # surfaces the ones we recommend in the docs. Each entry is
    # ``(label, model_id, dim)``. Flipping ``model`` to one of these requires
    # a full re-embed: existing chunks were vectorized under the old model
    # and will not be compatible. The GUI surfaces this warning prominently.
    suggested_models: list[dict] = Field(default_factory=lambda: [
        {"label": "all-mpnet-base-v2 (general, default)",
         "model": "sentence-transformers/all-mpnet-base-v2", "dim": 768},
        {"label": "specter2 (academic papers)",
         "model": "allenai/specter2_base", "dim": 768},
        {"label": "jina-embeddings-v2-base-en (long-context, technical)",
         "model": "jinaai/jina-embeddings-v2-base-en", "dim": 768},
        {"label": "bge-small-en-v1.5 (fast, 384-d)",
         "model": "BAAI/bge-small-en-v1.5", "dim": 384},
        {"label": "potion-retrieval-32M (model2vec, ~500x faster, static)",
         "model": "minishlab/potion-retrieval-32M", "dim": 512, "provider": "model2vec"},
    ])

    # Phase 9.3 — per-session query embedding LRU cache. Eliminates redundant
    # embed_one() calls when the same query repeats across Phase-8 review
    # actions (rephrase, redescend, filter). Cache key is (session_id, query)
    # — calls that omit session_id always bypass the cache.
    query_cache_enabled: bool = True
    query_cache_size: int = 64

    # Phase 9.7 — precision / backend selector for SentenceTransformersProvider.
    # Kept as a back-compat alias; new code should set `precision` instead.
    # Supported modes (valid for both `embed_precision` and `precision`):
    #   "fp32"      (default): full float32, same as today, no behavioural change.
    #   "fp16":                half-precision on GPU. ~50% VRAM cut, ~1.5–2x
    #                          throughput. Auto-falls back to fp32 on CPU.
    #   "bf16":                bfloat16 on GPU; better numerical stability than
    #                          fp16. Auto-falls back to fp32 on CPU.
    #   "onnx":                native sentence-transformers ONNX backend. CPU-
    #                          friendly; no quantization. Requires optimum.
    #   "onnx_int8":           ONNX + int8 dynamic quantization. Best CPU
    #                          throughput (>2x). Requires the `quantize` optional
    #                          dep group (`optimum[onnxruntime]` + `onnxruntime`).
    #   "openvino":            Intel-optimized backend (mainly Intel CPUs).
    #                          Requires the `openvino` optional dep group.
    # When `precision` is set (non-None) it supersedes `embed_precision`.
    embed_precision: str = "fp32"
    # Device hint for SentenceTransformersProvider. None = let the library pick
    # (CUDA if available, else CPU). Set to "cuda", "cuda:0", "cpu", or "mps".
    embed_device: Optional[str] = None

    # Phase 10 — modular embedding backends.
    # When set, supersedes `embed_precision`. See the mode table in the comment
    # above for valid values.
    precision: Optional[str] = None

    # Phase 10 — batch size used at ingest time by IngestPipeline._embed_chunks.
    # Was hardcoded as _EMBED_BATCH_SIZE=8 in pipeline.py; raised to 32 by
    # default for better throughput, configurable per-corpus. Progress events
    # still emit per-batch, just with larger batches.
    embed_batch_size: int = 32

    # Phase 10 — where sentence-transformers caches the exported ONNX model.
    # None lets the library pick (typically ~/.cache/huggingface/...). Set this
    # when you want the export to live alongside your project data.
    embed_onnx_cache_dir: Optional[str] = None

    # Phase 10 — ONNX graph optimization level passed to sentence-transformers'
    # export_optimized_onnx_model. Valid: "O1" | "O2" | "O3" | "O4". O3 is the
    # recommended trade-off between optimization and load time.
    embed_onnx_optimization: str = "O3"


class StorageConfig(BaseModel):
    sqlite_path: str = "data/store.sqlite"
    chroma_path: str = "data/chroma"
    kg_path: str = "data/kg"
    data_root: str = "data"


class KGConfig(BaseModel):
    """Phase 2 knowledge-graph layer settings."""

    enabled: bool = False                  # master switch; opt-in for Phase 1 users
    backend: str = "networkx"              # KG backend: "networkx" (default) | "neo4j"
                                           # (both real; neo4j needs the `neo4j` extra +
                                           # NEO4J_URI, else errors on first op).
    use_communities: bool = False          # GraphRAG community detection + summarization
                                           # (slow at ingest, marginal accuracy gain;
                                           # default off — opt in for global/sensemaking
                                           # corpora)
    parallel_workers: int = 8              # ThreadPoolExecutor size for triple/summary calls
    ner: str = "spacy"                     # "spacy" (default) | "llm"
    damping: float = 0.5                   # PPR damping factor (HippoRAG default)
    synonym_threshold: float = 0.8         # cosine threshold for entity merging
    leiden_seed: int = 42                  # deterministic communities
    community_levels: list[int] = Field(default_factory=lambda: [0, 1, 2])

    # HippoRAG-faithful retrieval knobs ----------------------------------
    seed_top_k: int = 3                    # broaden PPR seed: per query entity,
                                           # match the top-K phrase nodes by cosine
                                           # similarity and assign reset probability
                                           # proportional to similarity. K=1 reproduces
                                           # the prior single-match behavior.
    passage_node_alpha: float = 0.5        # mix passage-node PPR scores (alpha) with
                                           # the phrase-aggregate (1 - alpha) score.
                                           # alpha = 0 -> phrase-only (legacy);
                                           # alpha = 1 -> passage-node-only.
                                           # Set to 0 if KG has no passage nodes.
    section_depth_beta: float = 0.0        # coefficient on section-depth penalty;
                                           # 0.0 = no-op (default). Multiply each
                                           # passage score by (1 - beta * depth)
                                           # when chunk.section_depth is available.

    # Phase 9.12 — cross-chunk triple deduplication.
    # When True (default), the extractor consults kg_canonical_triples after
    # each LLM call. Each unique (subject, relation, object) triple is
    # inserted once and gets a freq counter incremented on every subsequent
    # sighting. The per-chunk passage edge is still created in the graph
    # (mirror correctness). The per-chunk hash short-circuit in
    # kg_triple_cache is independent and always on when ``db`` is provided
    # to the extractor. Disable for debugging or to force re-extraction of
    # every triple.
    dedup_enabled: bool = True


class RetrievalConfig(BaseModel):
    top_k_vector: int = 10
    top_k_final: int = 6
    rerank_enabled: bool = True

    # Vector-store backend selector: "chroma" (default) | "sqlite_vec".
    # See hrag.retrieval.backends for the protocol. Both are real;
    # sqlite_vec needs the `sqlite-vec` extra installed.
    vector_backend: str = "chroma"

    # Retriever choice: "vector" | "bm25" | "hybrid"
    # Phase 2 adds: "kg_ppr" | "community" | "router"
    # Phase 2b adds: "taxonomy" (hierarchical category tree)
    retriever: str = "vector"
    rrf_k: int = 60                       # RRF smoothing constant
    rrf_weights: Optional[list[float]] = None  # only used for hybrid; default = equal

    # Reranker choice: "cross_encoder" | "llm" | "batched_llm"
    reranker: str = "cross_encoder"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Phase 9.8 — opt-in INT8 ONNX export of the cross-encoder. Requires the
    # `quantize` optional dep group (`pip install -e .[quantize]`). Silent
    # fallback to FP32 when optimum/onnxruntime is missing.
    rerank_quantize: bool = False

    # Threshold semantics depend on the reranker:
    #   cross_encoder -> float logit (default -5 = permissive; only filter clearly bad)
    #   llm / batched_llm -> int 0-3 (set to 2 manually if switching reranker)
    rerank_threshold: float = -5.0

    # Conversational query rewriting before retrieval.
    # "heuristic" -> cheap rule-based, no LLM call (default)
    # "llm"       -> one extra LLM call to rewrite the query
    # "none"      -> passthrough; the bare question is sent to retrieval
    query_rewrite: str = "heuristic"

    # Phase 2 features
    use_kg: bool = False                # build & query the KG (HippoRAG-style)
    use_communities: bool = False       # build & query GraphRAG community summaries
    use_router: bool = False            # use the LLM-based QueryRouter

    # Document-scoped retrieval wrapper (regime A: explicit-title hard-filter,
    # regime B: coarse-doc -> fine-chunk two-stage). When true, whichever
    # retriever is built is wrapped with DocScopedRetriever before being
    # returned by the factory. Default true: the wrapper degrades gracefully
    # (falls through unchanged when nothing matches), so it's safe-by-default.
    doc_scope_enabled: bool = True

    # Phase 6 — adaptive retrieval per intent.
    # When True, the orchestrator overrides top_k_vector / top_k_final on a
    # per-intent basis, and may skip retrieval entirely for GREETING. Defaults
    # below are conservative: greetings retrieve zero (the LLM answers
    # directly), personal queries broaden episodic recall, factual stays at
    # the global default. Off by default — flip to true once a corpus is
    # ingested.
    adaptive_enabled: bool = False
    # Per-intent top_k overrides. Keys: greeting | personal | factual |
    # general | unclear. Value is the post-rerank cap. top_k_vector is set
    # to max(value, 12) before retrieval so the reranker has slack.
    # Greeting=0 skips retrieval entirely.
    adaptive_top_k: dict = Field(
        default_factory=lambda: {
            "greeting": 0,
            "personal": 8,
            "factual": 6,
            "general": 4,
            "unclear": 4,
        }
    )
    # On PERSONAL intent, bias the retrieved set toward episodic memories.
    # Concretely: the retriever is asked for both document and episodic
    # chunks, and the result list is re-sorted so episodic comes first.
    # No-op when adaptive_enabled is False or the intent is not PERSONAL.
    adaptive_personal_episodic_bias: bool = True

    # Phase 8.1 — memory recall fix.
    # When True (default), episodic memories are included in retrieval for ALL
    # intents, not just PERSONAL. Cosine similarity decides whether they
    # appear in results; for PERSONAL turns the existing stable-sort still
    # lifts them to the top. Set to False to revert to strict per-intent
    # source_types (the pre-Phase-8.1 behaviour) — useful for users with
    # massive corpora where memories should never compete with documents.
    always_include_episodic: bool = True

    # Phase 11.1 — reflective / opinion personal-question detection mode.
    # On a reflective turn the orchestrator (a) broadens retrieval to pull
    # document chunks about the user in addition to episodic memories, and
    # (b) renders a synthesis prompt that forms a genuine warm impression
    # instead of reciting one saved fact and offering to search.
    #   "hybrid" — hardened regex fast-path, then an LLM reflective signal
    #              when the regex misses (best recall + multilingual).
    #   "regex"  — pure regex only, deterministic, no LLM (contract-19 style).
    #   "off"    — feature disabled (true no-op; equals the old
    #              reflection_enabled=False).
    # Switchable live from the web side panel. See orchestrator 3d-pre/3d-bis
    # and `is_reflective_strict` / `is_reflective_query` in intent.py.
    reflection_mode: str = "hybrid"

    # Phase 7-A — math-meta query handling. Detect "formula"/"equation" meta-queries
    # and pass where={"has_math": True} into the retriever. Off by default — run
    # scripts/backfill_has_math.py first to populate metadata on existing chunks.
    math_meta_filter_enabled: bool = False
    # Lower rerank threshold for math-meta queries so MS-MARCO doesn't drop them.
    math_meta_rerank_threshold: float = -10.0

    # Phase 6-B1 — per-intent retriever override.
    # When ``retrieval.adaptive_enabled`` is True, the orchestrator may also
    # pick a different *retriever* for each intent (not just a different
    # top_k). Each value must match the global ``retriever`` enum values
    # ("vector" | "bm25" | "hybrid" | "kg_ppr" | "community" | "router" |
    # "taxonomy") or be the sentinel "default" meaning "use the global
    # ``retrieval.retriever`` setting". Off by default (everything maps to
    # "default") so behaviour is byte-identical until the user opts in.
    adaptive_retriever_per_intent: dict = Field(
        default_factory=lambda: {
            "greeting": "default",
            "personal": "default",
            "factual": "default",
            "general": "default",
            "unclear": "default",
        }
    )

    # Phase 9.15 — Feedback-weighted re-ranking.
    # When True, historical 👍/👎 feedback is used to nudge reranker scores.
    # Each chunk's EMA(thumbs_up - thumbs_down) score is computed from the
    # feedback table (joined to messages.metadata.sources) and added to the
    # cross-encoder logit as: final_score = rerank_score + weight * feedback_score.
    # Default OFF — users with no feedback history should see no change.
    feedback_reranking_enabled: bool = False
    # Weight of the feedback nudge (default 0.3 — keeps feedback as a hint,
    # not a steamroller). Tune upward once a substantial feedback corpus exists.
    feedback_reranking_weight: float = 0.3
    # EMA decay rate for the feedback scorer (alpha in [0,1]).
    # Higher alpha = more weight on recent feedback; lower = smoother history.
    feedback_reranking_alpha: float = 0.3

    # Phase 9.2 — Async pre-retrieval stages.
    # When True AND ``compaction.combined_preflight_enabled`` is False, gate /
    # clue / intent-classifier calls are dispatched concurrently via a small
    # ThreadPoolExecutor. Mutually exclusive with the combined-preflight path
    # (combined wins when both are on). Default OFF — safest behaviour matches
    # pre-Phase-9 serial dispatch. Set to True once you have observed that the
    # 3 calls are independent on your LLM provider (Anthropic / OpenAI / a
    # second Ollama instance behind a load-balancer).
    async_preflight_enabled: bool = False

    # Phase 9.9 — Rerank-fallback telemetry.
    # When True, the orchestrator logs a row to the ``feedback`` table every
    # time the cross-encoder zero-filter trips and the unreranked top-k is
    # used as a fallback. Surfaces in ``hrag feedback-stats`` so users can see
    # which queries the reranker mis-scores. Default OFF — the fallback itself
    # is unchanged either way; this only flips the logging.
    rerank_fallback_telemetry_enabled: bool = False

    # Phase 9.10 — First-token latency tracker.
    # When True, the orchestrator records the wall-clock time between
    # ``generate_start`` and the first streamed token, emits it on the
    # ``generate`` event as ``first_token_ms``, and persists it into
    # ``messages.metadata`` under ``{"latency": {"first_token_ms": ...}}``.
    # Default OFF — the ``generate`` event payload otherwise stays
    # byte-identical to pre-Phase-9.10.
    first_token_latency_enabled: bool = False

    # Phase 9.16 — CRAG-style automatic re-routing.
    # When True AND the post-rerank top score is below ``crag_score_floor``
    # AND no Phase-8 review pause already redirected the turn, the
    # orchestrator runs one query rewrite + retry retrieval pass before
    # falling through to the FACTUAL→GENERAL swap. Lighter cousin of the
    # Phase-8 user-mediated review. Cap of 1 retry to bound latency.
    crag_enabled: bool = False
    crag_score_floor: float = 0.0
    crag_retry_top_k_multiplier: float = 2.0

    # Phase 9.11 — QueryRouter speculative short-circuit.
    # When True (default ON — this is a pure speed win with no semantic change),
    # the QueryRouter skips the multi-retriever RRF fusion for clearly-routed
    # ``entity`` or ``global`` queries. Instead of fanning out to kg_ppr +
    # bm25 + vector (entity) or community + vector (global), it calls ONLY the
    # primary retriever for that route:
    #   entity  -> kg_ppr (falls back to bm25, then vector if kg_ppr is absent)
    #   global  -> community (falls back to vector if community is absent)
    # For ``cross_document`` and ``ambiguous``, the full RRF fusion is always
    # kept — those routes exist precisely to hedge across multiple sources.
    # Set to False to restore the pre-Phase-9.11 broad-fanout behaviour.
    router_short_circuit: bool = True


class ChunkingConfig(BaseModel):
    max_tokens: int = 400
    overlap_tokens: int = 60
    metadata_fusion: bool = True
    drop_after_references: bool = True
    quality: QualityConfig = Field(default_factory=QualityConfig)


class IngestConfig(BaseModel):
    """Phase 7-C — loader-level toggles for the ingest pipeline.

    Defaults preserve byte-identical behaviour to pre-Phase-7 (PyMuPDF for
    every PDF). The Nougat loader is opt-in and silently falls back to
    PyMuPDF if the ``nougat-ocr`` package can't be imported.
    """

    use_nougat: bool = False                   # academic-PDF → LaTeX OCR via
                                                # Meta's Nougat. Requires
                                                # ``pip install nougat-ocr`` and
                                                # ~800 MB model download on first
                                                # use. If False (default) or the
                                                # import fails, falls back to
                                                # PyMuPDF as today.
    nougat_model: str = "facebook/nougat-base"  # checkpoint id; "facebook/nougat-small"
                                                # is a faster lower-fidelity option.

    # Phase 9.14 — Contextual Retrieval (Anthropic technique).
    # When True, run one LLM call per chunk at ingest to generate a
    # 50-100-token "context line" describing the chunk's role in its parent
    # document. The line is PREPENDED to ``chunk.embedding_text`` (HeteRAG
    # fusion target - what the embedder sees). ``chunk.text`` (what the LLM
    # sees at answer time) is left untouched. Per-chunk results cached in
    # ``ingest_context_cache`` keyed by (chunk_text, doc_hash, model_name)
    # so re-ingesting an unchanged doc is free.
    #
    # Off by default because it adds 1 LLM call per chunk x every doc at
    # ingest, and turning it on requires a re-ingest of existing documents
    # to take effect.
    #
    # Anthropic reports ~35% reduction in retrieval failures (~67% when
    # combined with BM25 + reranking).
    contextual_retrieval_enabled: bool = False
    contextual_retrieval_max_workers: int = 4          # ThreadPoolExecutor size
                                                        # (Ollama serialises;
                                                        # OpenAI/Anthropic get
                                                        # true parallelism).
    contextual_retrieval_max_doc_chars: int = 60_000   # cap on full document
                                                        # text fed into the
                                                        # prompt. ~15k tokens;
                                                        # fits gemma3's 8k
                                                        # num_ctx after the
                                                        # chunk itself. If the
                                                        # doc is much larger we
                                                        # truncate with a [...]
                                                        # marker. Less context
                                                        # in -> less informative
                                                        # augmentation.
    contextual_retrieval_max_context_tokens: int = 100  # max_tokens cap for
                                                         # the LLM call.
                                                         # Anthropic's technique
                                                         # targets 50-100 tokens;
                                                         # budget some slack.


class ContextConfig(BaseModel):
    history_token_budget: int = 4000
    gate_enabled: bool = False          # deprecated, see compaction.*
    compact_after_turns: int = 12       # deprecated, see compaction.*


class UserConfig(BaseModel):
    default_user_id: str = "default"


class MemoryConfig(BaseModel):
    """Phase 3 per-user memory layer settings."""

    auto_extract: bool = False
    auto_extract_min_confidence: float = 0.7
    profile_max_items: int = 12
    profile_min_confidence: float = 0.5
    bulk_chunk_per_paragraph: bool = True
    forget_confirm: bool = True


class IntentConfig(BaseModel):
    """Intent-routing layer (replaces the older ChitchatConfig).

    A hybrid IntentClassifier classifies every user turn into one of
    {GREETING, PERSONAL, FACTUAL, UNCLEAR}. The orchestrator dispatches
    retrieval scope and prompt template by intent, eliminating the brittle
    regex-only chitchat gate and the canned "I couldn't find that in your
    documents" failure mode.

    GENERAL is a fifth intent that the classifier itself never emits — the
    orchestrator rewrites FACTUAL → GENERAL when the top retrieval score is
    below ``corpus_relevance_floor``, meaning the question is substantive but
    the local corpus has nothing on it. The LLM then answers from its general
    knowledge with a brief disclaimer.
    """

    enabled: bool = True                       # master switch for the intent layer
    fast_path_only: bool = False               # skip the LLM fallback (regex-only)
    llm_max_tokens: int = 30                   # cap on the LLM classifier's output
    personal_top_k: int = 8                    # chunks pulled on PERSONAL (documents + episodic)
    corpus_relevance_floor: float = 0.15       # max retrieval score below which a
                                                # FACTUAL query is rewritten to GENERAL.
                                                # Cosine-space (0..1) since r.score
                                                # is the vector similarity. 0.15 swaps
                                                # only on clearly off-corpus queries.
                                                # Set to 0.0 to disable the swap.


class TaxonomyConfig(BaseModel):
    """Hierarchical document-taxonomy retriever (Phase 2b).

    A user-editable tree where leaves hold documents. Retrieval beams down
    the tree by query/centroid cosine, then runs chunk retrieval scoped to
    the few docs at the picked leaves.
    """

    enabled: bool = False                              # master switch
    beam_width: int = 3                                # nodes kept per level
    max_depth: int = 6                                 # safety bound
    fallback_retriever: str = "vector"                 # used when tree is empty
    # Build / refresh knobs ---------------------------------------------
    auto_assign_on_ingest: bool = True                 # file new docs as they arrive
    include_episodic: bool = True                      # also file episodic memories
                                                        # (notes saved via /remember).
                                                        # The whole point of a personal
                                                        # tree is that memories live in
                                                        # it too — default true.
    propose_sample_size: int = 80                      # docs per LLM proposal call
    max_children_per_node: int = 12                    # cap LLM-proposed branches
    min_docs_per_leaf: int = 1                         # below = merge into parent
    summary_max_chars: int = 280                       # one-line per-doc summary cap
    use_llm_doc_summaries: bool = True                 # if false, use title only (cheap)
    # Retrieval knobs ----------------------------------------------------
    chunk_top_k_per_leaf: int = 8                      # chunks pulled from each leaf
    min_node_score: float = 0.05                       # prune branches below this cos sim
    beam_dominance_gap: float = 0.10                   # adaptive beam: if the gap between
                                                        # the i-th and (i-1)-th score
                                                        # exceeds this, drop everything
                                                        # from rank i onward. A clear
                                                        # winner (e.g. 0.37 vs 0.13) thus
                                                        # narrows the beam to 1 even when
                                                        # beam_width=3.
    min_top_score_floor: float = 0.30                  # confidence floor on the root
                                                        # level. If the best level-0 node
                                                        # scores below this, the corpus
                                                        # has no real match for the query
                                                        # (e.g. "hey!") — narrow the beam
                                                        # to 1 so we don't open 23 of 24
                                                        # docs out of desperation. Set to
                                                        # 0.0 to disable.
    max_docs_pct: float = 0.40                         # cap on the share of the user's
                                                        # corpus the picked leaves can
                                                        # collectively open. If exceeded,
                                                        # the lowest-scoring leaves are
                                                        # dropped until the cap holds.
                                                        # Set to 1.0 to disable.
    # Phase 12 — hybrid (dense + keyword) routing --------------------------
    keyword_routing_enabled: bool = True               # blend per-node keyword overlap
                                                        # into beam-descend scoring. Default
                                                        # ON for accuracy; a true no-op
                                                        # until a tree has keywords (built
                                                        # with the keyword-aware propose
                                                        # prompt or backfilled via
                                                        # `hrag taxonomy keywords`).
    keyword_weight: float = 0.20                       # combined = cosine + weight*overlap
    keywords_per_node: int = 8                         # target keywords per node
    keyword_tiebreak_skip: bool = True                 # DocAssigner: skip the LLM tiebreak
                                                        # when the keyword signal already
                                                        # separates the top-2 candidates.
    cache_tree_in_memory: bool = True                  # cache the per-user node set for
                                                        # the beam-descend hot path; a write
                                                        # through the store invalidates it,
                                                        # and it self-expires after a short
                                                        # TTL so external rebuilds are seen.


class CompactionConfig(BaseModel):
    """Phase 4 — compaction & gating toggles.

    Each feature is OFF by default. Phases 1–3 behaviour is unchanged when
    no flag is set.
    """

    # RAGate: cheap LLM call that decides RETRIEVE vs SKIP.
    gate_enabled: bool = False
    gate_max_tokens: int = 8

    # Clue generation: MemoRAG-style retrieval hypothesis used as the search query.
    clue_enabled: bool = False
    clue_max_tokens: int = 200

    # Dialog MST: turn-level history compaction when conversation grows beyond a budget.
    dialog_mst_enabled: bool = False
    compact_after_turns: int = 12
    keep_recent_turns: int = 6
    summary_target_tokens: int = 400

    # [UNCERTAIN] masking: render `[UNCERTAIN]` tokens visibly in answers.
    mask_uncertain: bool = False

    # Phase 9.6 — combined gate+clue+intent in one LLM call.
    # When True AND all three of gate_enabled / clue_enabled / intent.enabled are
    # also True, the orchestrator renders prompts/combined_preflight.md and
    # parses {intent, gate_decision, clue} from one JSON response. Saves 2 round
    # trips on each FACTUAL turn. The per-stage progress events still fire (with
    # source="combined") so the GUI / benchmark stay compatible.
    # Default OFF — opt in once you have observed that the 3 calls are the
    # critical path on your hardware.
    combined_preflight_enabled: bool = False
    combined_preflight_max_tokens: int = 280

    # Phase 9.13 — context compression for over-budget prompts.
    # When the rendered prompt exceeds ``context_budget_chars`` characters,
    # summarize the oldest half of conversation history via the existing
    # DialogMSTCompactor and drop the lowest-scoring quartile of retrieved
    # passages. Default OFF — long prompts work as-is; this is a recovery path
    # for genuinely over-budget turns on small num_ctx Ollama models.
    context_compression_enabled: bool = False
    context_budget_chars: int = 12_000

    # Phase 9.17 — Self-RAG span re-retrieval.
    # When True AND the answer contains [UNCERTAIN] markers, run one extra
    # retrieval pass per uncertain span (cap at ``self_rag_max_spans``) and
    # append the retrieved snippets as a "Sources for uncertain claims"
    # block below the answer. Off by default; the [UNCERTAIN] tokens are
    # already rendered / stripped by mask_uncertain in either case.
    self_rag_enabled: bool = False
    self_rag_max_spans: int = 2


class FormulaExtractionConfig(BaseModel):
    """Phase 7-A — second LLM pass that extracts equations verbatim from
    retrieved passages and appends them to the RAFT answer. Off by default;
    only fires when math_meta_filter_enabled also triggers."""

    enabled: bool = False
    max_tokens: int = 400


class InteractionConfig(BaseModel):
    """Phase 8 — interactive retrieval review loop.

    Default-off. When `review_enabled` is False, no new events fire, no new
    LLM calls are made, no DB writes happen. Existing 829 tests stay byte-
    identical pass.
    """

    review_enabled: bool = False
    review_mode: Literal["smart_auto", "always", "off"] = "smart_auto"
    review_score_floor: float = -3.0
    review_ambiguity_delta: float = 0.4
    review_branch_threshold: int = 2
    review_timeout_s: float = 300.0
    rephrasings_enabled: bool = True
    followups_enabled: bool = True
    persistence_enabled: bool = True
    why_source_enabled: bool = True
    clarify_enabled: bool = True
    expand_doc_enabled: bool = True


class DeepReadConfig(BaseModel):
    """Phase 13 — agentic iterative "deep read" over a single document.

    On a broad/exploratory question the orchestrator reads a document
    section-by-section across several passes (each pass plans the next from what
    it just learned), visualised live in the GUI, then proposes follow-ups.
    """

    enabled: bool = True            # master switch
    auto_trigger: bool = True       # fire automatically on broad questions
    max_passes: int = 4             # hard cap on read→plan iterations
    min_passes: int = 2             # always iterate at least this many (while doc has more)
    seed_top_k: int = 12            # initial corpus retrieval used to pick the doc
    chunks_per_pass: int = 6        # chunks pulled per pass (scoped to the doc)
    min_rerank_score: float = -7.0  # stop when a pass surfaces nothing above this
    followups: int = 3              # follow-up suggestions offered at the end


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    user: UserConfig = Field(default_factory=UserConfig)
    kg: KGConfig = Field(default_factory=KGConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    formula_extraction: FormulaExtractionConfig = Field(default_factory=FormulaExtractionConfig)
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
    deep_read: DeepReadConfig = Field(default_factory=DeepReadConfig)

    project_root: Path = Field(default_factory=lambda: Path.cwd())

    @model_validator(mode="after")
    def _mirror_legacy_compaction(self) -> "Config":
        # If a legacy config sets context.gate_enabled / context.compact_after_turns,
        # propagate to compaction.* unless compaction.* was already explicitly set.
        # (This is best-effort: pydantic v2 doesn't easily distinguish default vs explicit,
        # so we only mirror when the legacy value differs from its default AND compaction
        # still has the default.)
        if self.context.gate_enabled and not self.compaction.gate_enabled:
            self.compaction.gate_enabled = True
        if self.context.compact_after_turns != 12 and self.compaction.compact_after_turns == 12:
            self.compaction.compact_after_turns = self.context.compact_after_turns
        return self

    def resolve(self, relative: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(relative)
        return p if p.is_absolute() else (self.project_root / p)


def _apply_env_overrides(data: dict) -> dict:
    """HRAG_LLM__MODEL=foo overrides data['llm']['model']."""
    for env_key, value in os.environ.items():
        if not env_key.startswith("HRAG_"):
            continue
        path = env_key[len("HRAG_"):].lower().split("__")
        cursor = data
        for piece in path[:-1]:
            cursor = cursor.setdefault(piece, {})
        cursor[path[-1]] = value
    return data


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML; if path is None, look for config.yaml in cwd."""
    project_root = Path.cwd()
    if path is None:
        path = project_root / "config.yaml"
    path = Path(path)
    raw: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    raw = _apply_env_overrides(raw)
    cfg = Config(**raw)
    cfg.project_root = project_root
    return cfg
