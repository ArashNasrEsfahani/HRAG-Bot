"""Pydantic-backed configuration loader.

Loads `config.yaml` from the project root by default. Environment variables prefixed
with HRAG_ override fields using `__` as the section separator, e.g.
HRAG_LLM__MODEL=gemma:7b sets `llm.model`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

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
    ])


class StorageConfig(BaseModel):
    sqlite_path: str = "data/store.sqlite"
    chroma_path: str = "data/chroma"
    kg_path: str = "data/kg"
    data_root: str = "data"


class KGConfig(BaseModel):
    """Phase 2 knowledge-graph layer settings."""

    enabled: bool = False                  # master switch; opt-in for Phase 1 users
    backend: str = "networkx"              # KG backend: "networkx" (default) | "neo4j"
                                           # (Phase 5 stub — raises NotImplementedError
                                           # on every op until implemented).
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


class RetrievalConfig(BaseModel):
    top_k_vector: int = 10
    top_k_final: int = 6
    rerank_enabled: bool = True

    # Vector-store backend selector: "chroma" (default) | "sqlite_vec" (stub).
    # See hrag.retrieval.backends for the protocol. Switching is a refactor
    # seam — sqlite_vec is not implemented yet and raises NotImplementedError
    # on any operation.
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
    personal_top_k: int = 3                    # episodic memories pulled on PERSONAL
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


class FormulaExtractionConfig(BaseModel):
    """Phase 7-A — second LLM pass that extracts equations verbatim from
    retrieved passages and appends them to the RAFT answer. Off by default;
    only fires when math_meta_filter_enabled also triggers."""

    enabled: bool = False
    max_tokens: int = 400


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
