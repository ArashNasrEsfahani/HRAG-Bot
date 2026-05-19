# HRAG-Bot — Hierarchical RAG Chatbot

A personal RAG chatbot for 10–1000 documents with **hierarchical reasoning**, **growing per-user memory**, and a **modular** architecture you can incrementally extend. Built around a synthesis of recent RAG papers (HippoRAG2, GraphRAG, HeteRAG, RAFT, RAGate, MemoRAG, KG2RAG, ACP-RAG, and others).

> **Status**: Phases 1–3 complete (vector RAG · KG hierarchy · personalization with memory + taxonomy). 537 unit tests passing. Phases 4–5 (compaction & gating · polish) on the roadmap below.

## What's working today (Phases 1–3)

- Ingest **PDFs, DOCX, Markdown, plain text** documents with structure-aware chunking (HeteRAG metadata fusion).
- Pluggable retriever: **vector / BM25 / hybrid (RRF) / kg_ppr / community / router / taxonomy** (default).
- Pluggable reranker: **cross-encoder (default) / LLM 0-3 / batched LLM** — or none.
- RAFT-style answer generation with `##begin_quote##` evidence citations.
- **Knowledge graph layer** (Phase 2): OpenIE triple extraction, Personalized PageRank retrieval, GraphRAG-style Leiden community summaries, LLM-routed dispatch, KG2RAG MST organizer.
- **Personalization layer** (Phase 3):
  - **Per-user memory** — `/remember`, `/recall`, `/forget`; structured preferences rendered into every answer prompt.
  - **Hierarchical document taxonomy** — LLM proposes a category tree over your docs and memories, you edit it in the GUI, retrieval beam-searches the tree and only opens chunks from a few picked leaves. See [Hierarchical taxonomy](#hierarchical-taxonomy-phase-3) below.
- Pluggable **Ollama / OpenAI / Anthropic** LLM backends behind one interface.
- Pluggable **sentence-transformers / OpenAI** embedding backends.
- Multi-user-ready SQLite schema (single user is the default).
- Interactive CLI with **live progress display** and a `/status` command to see session state.
- **Streamlit GUI dashboard** (`hrag gui`) with Chat · Memories · Documents · 🌳 Taxonomy · Profile · Users · Sessions · Stats pages.
- 537 unit tests passing.

## Quick start

### 1. Install Python 3.10+ and dependencies

```bash
cd "D:\Hierachical RAG based ChatBot"
python -m pip install -e .
```

This pulls: `pydantic`, `pyyaml`, `rich`, `click`, `tiktoken`, `chromadb`, `networkx`, `sentence-transformers`, `ollama`, `openai`, `anthropic`, `pymupdf`, `python-docx`, `markdown-it-py`, `scipy`, `numpy`.

> **Note (Python 3.14):** Some heavy deps (`chromadb`, `sentence-transformers`, `scipy`) may not have prebuilt wheels yet. If `pip install -e .` fails on those, fall back to Python 3.11 or 3.12.

### 2. Make sure Ollama is running with Gemma

```bash
ollama pull gemma3
ollama serve   # runs at http://localhost:11434 by default
```

If your tag is different (e.g. `gemma:7b`), edit `config.yaml`:

```yaml
llm:
  provider: ollama
  model: gemma:7b
```

### 3. Initialize the database

```bash
hrag init
```

This creates `data/store.sqlite` with the v1 schema and a `default` user.

### 4. Ingest documents

```bash
# A single file
hrag ingest "F:/Dynamic RAG Papers/HIPPORAG.pdf"

# A whole directory (recursive)
hrag ingest "F:/Dynamic RAG Papers" --recursive
```

You'll see one line per document: `[ingest] HIPPORAG: 47 chunks`.

### 5. Chat

```bash
hrag chat
```

```
you> What's the core technique behind HippoRAG?
[answer panel with cited evidence]
[Sources: 1. HIPPORAG — Section 3 (0.83)]
        2. HIPPORAG — Method (0.79)
        3. HIPPORAG — Abstract (0.77)
```

Slash commands inside the REPL:
- `/help` — list commands
- `/sources` — show full text of the last response's top 5 sources
- `/remember <text>` — save an episodic memory inline
- `/recall <query>` — semantic search across episodic memories only
- `/forget <query|chunk_id>` — tombstone matching memories
- `/profile [add|forget …]` — inspect or edit the user profile rendered into every prompt
- `/stats` — corpus + KG counts; `/status` — session info
- `/exit` or `/quit` — leave (Ctrl-D works too)

Other CLI commands:
- `hrag list-docs` — table of ingested documents and chunk counts.
- `hrag ingest --user alice <path>` — ingest under a different user (multi-user-ready schema).
- `hrag remember "<text>"` · `hrag remember <path>` — save an episodic memory inline or bulk-import notes.
- `hrag memory list` · `hrag memory extract --session <id>` — inspect / mine episodic memories.
- `hrag taxonomy build` · `hrag taxonomy show` · `hrag taxonomy clear` — build/inspect/drop the hierarchical category tree.
- `hrag gui` — launch the Streamlit dashboard (Chat · Memories · 🌳 Taxonomy · Documents · Profile · Users · Sessions · Stats).

## Configuration

`config.yaml` at the project root controls everything. Key knobs:

```yaml
llm:
  provider: ollama          # ollama | openai | anthropic
  model: gemma3:latest

embeddings:
  provider: sentence-transformers
  model: sentence-transformers/all-mpnet-base-v2

retrieval:
  top_k_vector: 20          # candidates from retriever
  top_k_final: 15           # final cap fed to the answer LLM

  retriever: taxonomy       # vector | bm25 | hybrid | kg_ppr | community | router | taxonomy
  reranker: cross_encoder   # cross_encoder | llm | batched_llm
  rerank_enabled: true
  rerank_threshold: -8.0    # logit for cross_encoder; 2 for llm/batched_llm

taxonomy:
  enabled: true             # build & query the LLM-proposed category tree
  beam_width: 3             # branches kept per descent level
  include_episodic: true    # file /remember'd memories under the same tree

memory:
  auto_extract: false       # opt-in: mine prefs from conversation on session close

chunking:
  max_tokens: 400
  overlap_tokens: 60
  metadata_fusion: true     # HeteRAG: prepend title/section to embedded text
```

**Environment variable overrides** with double-underscore section separator:
```bash
HRAG_LLM__MODEL=gemma:7b
HRAG_RETRIEVAL__TOP_K_FINAL=8
```

## Choosing a retriever and reranker

You can pick any combination via `config.yaml` or per-session CLI flags:

```bash
hrag chat --retriever hybrid --reranker cross_encoder    # best quality
hrag chat --retriever vector --no-rerank                 # fastest (1 LLM call total)
hrag chat --reranker batched_llm                         # 1 LLM rerank call instead of 10
```

### Retrievers

| Mode | What it does | Pros | Cons |
|---|---|---|---|
| `vector` | Dense embedding search via ChromaDB | Semantic; catches paraphrase | Misses exact-keyword queries |
| `bm25` | Sparse keyword search (Okapi BM25) | Instant; no embedding cost at query time | No semantic match |
| `hybrid` | Vector + BM25 fused via Reciprocal Rank Fusion | Best of both worlds | 2× retrieval cost (still cheap) |
| `kg_ppr` | NER → seeds → Personalized PageRank over the KG → passage hydration (HippoRAG-style) | Multi-hop reasoning across entities | Requires `kg.enabled: true` + ingest pass |
| `community` | Search GraphRAG community summaries | Best for global / sensemaking queries | Requires `kg.use_communities: true` |
| `router` | LLM classifies query → routes to kg_ppr / community / hybrid, RRF-fuses cross-doc paths | Auto-picks the right backend per query | One extra LLM call per query (cached) |
| `taxonomy` *(default)* | Beam-search a user-editable category tree, then chunk-fetch from the few docs at the picked leaves | Scales to many docs; emits a visual descent trace; tree is editable in the GUI | Needs an initial `hrag taxonomy build`; falls back to `taxonomy.fallback_retriever` (default `vector`) when the tree is empty |

### Rerankers

| Mode | What it does | Per-query cost on Gemma e4b | Pros | Cons |
|---|---|---|---|---|
| `cross_encoder` *(default)* | Local cross-encoder (`ms-marco-MiniLM-L-6-v2`, 22M params) scores all 10 pairs in one batched forward pass | ~100–300 ms | Fast; no LLM call; trained for relevance ranking | One-time ~80MB model download |
| `llm` | Per-chunk LLM call returning 0–3 score (ACP-RAG) | ~20–60 s | Highest quality on hard queries; works with the LLM you already trust | 10 LLM calls per query; runs hot |
| `batched_llm` | Single LLM call scoring all chunks at once via JSON output | ~5–15 s | 1 LLM call total; quality close to per-chunk | Smaller models can produce malformed JSON |
| `--no-rerank` | Skip reranking; use top-k vector results directly | 0 ms | Fastest possible; only the answer LLM runs | No filter for off-topic neighbours |

Pick by use case:

- **Default — scale to many docs without paying retrieval cost per doc:** `taxonomy` + `cross_encoder` (the defaults; build the tree once via `hrag taxonomy build`).
- **Want it fast on a laptop with no tree yet:** `vector` + `cross_encoder`.
- **Want better recall on keyword-heavy queries:** `hybrid` + `cross_encoder`.
- **Want zero new ML models:** `vector` + `batched_llm` (or `llm` if you're patient).
- **Just exploring:** `vector` + `--no-rerank` is the cheapest possible setup.

## Hierarchical taxonomy (Phase 3)

Once you have more than a handful of documents, brute-force vector search becomes wasteful: every query embeds against every chunk in your corpus. The taxonomy retriever fixes this by organising your documents (and your `/remember`'d memories) under a *category tree* — and then **navigating that tree** rather than scanning the whole corpus.

How it works:

1. **One-time build** (`hrag taxonomy build`) — every doc + memory gets a one-line summary; their (title + summary) embeddings are clustered by an LLM into a labelled tree (root → branches → leaves). Each leaf holds a small set of docs.
2. **Edit in the GUI** (`hrag gui` → 🌳 Taxonomy) — rename labels, move nodes, split a leaf, delete a category, unfile a doc. The tree is yours; the LLM only proposes.
3. **At query time** — embed the question, beam-descend from root by cosine similarity against each node's centroid, stop when you've reached a leaf. Then run normal chunk retrieval **scoped to just the docs at the picked leaves** (typically 1–4 docs). The beam keeps the top-B branches at each level so multi-topic queries still work.
4. **Visual trace** — every chat reply has a collapsible **🌳 Tree navigation** block under the answer that shows which branches were considered vs kept at each level, with cosine scores.

New documents auto-file themselves on `hrag ingest` via `DocAssigner` (greedy cosine descent + LLM tiebreak when two branches are close). Memories saved via `/remember` are filed into the same tree by default (controlled by `taxonomy.include_episodic`).

CLI:
```bash
hrag taxonomy build              # build / rebuild from the current corpus
hrag taxonomy show               # print the tree as a text outline
hrag taxonomy clear              # drop the tree (preserves doc-meta cache)
```

## Tuning retrieval quality

If retrieval feels off (single bad source, "I couldn't find that", reranker drops everything), check these:

### Chunk quality filter (on by default)
PDF→chunk extraction can produce a lot of garbage: 1-line "paragraphs", figure captions, `Page 12 of 50`, bibliography entries. The filter at `chunking.quality.*` drops these at ingest time. Typical academic papers see **70–85% of chunks dropped** as low-value, leaving the substantive content. To inspect what's being dropped, watch the line printed during `hrag ingest`:

```
[ingest] HIPPORAG: 106 chunks kept (528 dropped: {'too_short': 523, 'bibliography_chunk': 3, 'duplicate': 2})
```

If you'd rather keep everything, set `chunking.quality.enabled: false` in `config.yaml`.

**After tweaking the filter, you must re-ingest** (the index doesn't update retroactively).

### Cross-encoder threshold
Default is `-5.0` (permissive — only filter clearly off-topic chunks). If the reranker is dropping useful content, lower further (e.g. `-10.0`). If too much garbage is getting through, raise toward `0.0`. Threshold is a logit; ms-marco scores typically range -12 to +5.

### Rerank-empty fallback
If the reranker drops everything, the orchestrator now falls back to the unreranked top-k vector results (you'll see `(rerank dropped all; fell back to top-k vector)` in yellow). This prevents "I couldn't find that" answers when retrieval did succeed but the reranker was too strict.

### Inspect what was actually retrieved
Inside the chat REPL, type `/sources` after any answer to see the full text + scores of every chunk passed to the LLM.

## Speed tips for laptops

The dominant cost on local hardware is the **answer generation LLM**, not retrieval. Streaming is on by default so you see tokens as they generate (no more staring at a spinner). Other knobs:

| Tip | Effect |
|---|---|
| `hrag chat --fast` | top_k=4, no rerank. Lowest latency, lowest heat. |
| Smaller LLM | `ollama pull llama3.2:1b` or `gemma2:2b`, then edit `config.yaml` (`llm.model`). **Often 5–10× faster** than gemma4:e4b. |
| Lower `max_tokens` | `config.yaml` → `llm.max_tokens: 512`. Shorter answers, less wait. |
| Drop `top_k_final` to 3 | Smaller prompt → faster generation. |
| Use OpenAI/Anthropic API | Cloud is much faster than local 9GB models. Set `llm.provider: openai` and `OPENAI_API_KEY`. |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  CLI (chat REPL · /remember · hrag taxonomy build · /sources)    │
├──────────────────────────────────────────────────────────────────┤
│  GUI (Streamlit · Chat · Memories · 🌳 Taxonomy · Documents · …) │
├──────────────────────────────────────────────────────────────────┤
│  Orchestrator (Gate → Router → Retriever → Rerank → LLM)         │
├──────────────────────────────────────────────────────────────────┤
│  Retrievers: VectorRetriever · BM25 · HybridRetriever (RRF)      │
│              KGPPRRetriever · CommunityRetriever · QueryRouter   │
│              TaxonomyRetriever (beam descent → leaf-scoped chunk)│
├──────────────────────────────────────────────────────────────────┤
│  Personalization: ProfileStore · EpisodicMemoryStore             │
│                   TaxonomyStore (tree + centroids + assignments) │
├──────────────────────────────────────────────────────────────────┤
│  Storage: ChromaDB (vectors) │ NetworkX (KG) │ SQLite            │
├──────────────────────────────────────────────────────────────────┤
│  Providers: LLMProvider {Ollama, OpenAI, Anthropic}              │
│             EmbeddingProvider {sentence-transformers, OpenAI}    │
└──────────────────────────────────────────────────────────────────┘
```

Every layer has a single small Python module. To swap a layer (e.g. trade ChromaDB for sqlite-vec, or NetworkX for Neo4j), implement the interface and update the factory.

### Source layout

```
src/hrag/
├── cli.py                       # click-based CLI (incl. hrag taxonomy / hrag remember / hrag gui)
├── orchestrator.py              # main pipeline
├── config.py                    # pydantic config + env overrides
├── types.py                     # Document, Chunk, RetrievalResult, ...
├── providers/
│   ├── llm.py                   # LLMProvider + Ollama/OpenAI/Anthropic
│   └── embeddings.py            # EmbeddingProvider + ST/OpenAI
├── ingest/
│   ├── loaders.py               # pdf/docx/md/txt loaders
│   ├── metadata.py              # section detection
│   ├── chunker.py               # token-budgeted chunking + HeteRAG fusion
│   ├── quality.py               # post-chunk filter (drops noise)
│   └── pipeline.py              # ingest orchestrator (calls KG + taxonomy hooks)
├── retrieval/
│   ├── base.py                  # Retriever ABC
│   ├── factory.py               # build_retriever / build_reranker
│   ├── vector.py                # ChromaDB store
│   ├── vector_retriever.py
│   ├── bm25.py · hybrid.py
│   ├── kg_ppr.py                # phase 2 — HippoRAG-style PPR
│   ├── community.py             # phase 2 — GraphRAG community summaries
│   ├── router.py                # phase 2 — LLM-classified dispatch
│   ├── mst.py                   # phase 2 — KG2RAG MST organizer
│   ├── taxonomy.py              # phase 3 — TaxonomyRetriever
│   ├── doc_scope.py             # title-aware hard-filter wrapper
│   └── reranker.py · cross_encoder_reranker.py · batched_llm_reranker.py
├── kg/                          # phase 2 — knowledge graph
│   ├── builder.py · store.py · ner.py · ppr.py · communities.py
├── memory/                      # phase 3 — per-user memory
│   ├── profile.py · store.py · extractor.py · auto_extract.py
├── context/
│   └── builder.py               # renders {user_profile} into the answer prompt
├── taxonomy/                    # phase 3 — hierarchical taxonomy
│   ├── types.py                 # TaxonomyNode, NodeScore, LevelTrace, DescendResult
│   ├── store.py                 # TaxonomyStore (CRUD + centroids + beam descend)
│   ├── builder.py               # TaxonomyBuilder (LLM-proposed tree)
│   └── assigner.py              # DocAssigner (cosine + LLM tiebreak)
├── gui/                         # Streamlit dashboard
│   ├── app.py · state.py
│   └── pages/                   # Chat · Memories · Documents · Profile · Users · Sessions · Stats · 🌳 Taxonomy
├── prompts/                     # all prompt templates as .md
│   ├── answer.md                # RAFT-style CoT with ##begin_quote## ... ##end_quote##
│   ├── rerank.md
│   ├── triple_extraction.md     # phase 2
│   ├── community_summary.md     # phase 2
│   ├── router.md                # phase 2
│   ├── preference_extract.md    # phase 3 (memory)
│   ├── taxonomy_doc_summary.md  # phase 3 (taxonomy)
│   ├── taxonomy_propose.md      # phase 3 (taxonomy)
│   ├── taxonomy_relabel.md      # phase 3 (taxonomy)
│   ├── taxonomy_route_tiebreak.md  # phase 3 (taxonomy)
│   ├── gate.md · clue.md        # phase 4 (placeholders)
└── db/
    ├── schema.sql               # incl. kg_taxonomy_nodes / kg_taxonomy_assignments / kg_taxonomy_doc_meta
    ├── migrations.py
    └── connection.py
```

## Roadmap

| Phase | Feature | Status | Key paper(s) |
|------:|---------|--------|--------------|
| 1 | Walking skeleton: ingest + vector retrieve + rerank + answer | ✅ Done | HeteRAG, RAFT, ACP-RAG, Vladika & Matthes |
| 2 | Hierarchical retrieval: KG + Personalized PageRank + Leiden community summaries + query router + KG2RAG MST | ✅ Done | HippoRAG2, GraphRAG, KG2RAG, GFM-RAG |
| 3 | **Personalization**: per-user memory (`/remember` + profile + episodic store) **and** hierarchical document taxonomy (LLM-proposed + user-editable tree, beam-search retrieval over docs + memories) | ✅ Done | SimRAG, ReMindRAG; tree retrieval is novel here |
| 4 | Compaction & gating: RAGate + clue-generation + dialog MST filter + `[UNCERTAIN]` masking | Next | RAGate, MemoRAG, KG2RAG, CoopRAG |
| 5 | Polish: streaming refinements, fine-tuning hooks, optional Neo4j / sqlite-vec backends, locked equation-aware ingest | | DynamicRAG, RAFT (FT), DRAE (DPMM) |


## Running the tests

```bash
pytest -q
```

Heavy-dependency tests skip cleanly if their deps aren't installed (ChromaDB/sentence-transformers/Ollama). The conftest stubs let lightweight tests run anywhere.

## Paper-to-component mapping

| Component (Phase 1) | Paper |
|---|---|
| Metadata-fused chunk embeddings | HeteRAG (2504.10529) |
| Token cap at top-12 final | Vladika & Matthes (NAACL 2025) |
| LLM 0–3 reranker | ACP-RAG (2025.naacl-long.575) |
| `##begin_quote##` CoT answer | RAFT (2403.10131) |

| Component (Phase 2+) | Paper |
|---|---|
| KG triples + passage nodes | HippoRAG2 (2502.14802) |
| Personalized PageRank retrieval | HippoRAG / HippoRAG2 |
| Leiden community summaries | GraphRAG (2404.16130) |
| MST context organization | KG2RAG (2025.naacl-long.449) |
| RAGate-Prompt | RAGate (2025.findings-naacl.30) |
| Clue-generation compaction | MemoRAG / HawkRAG |
| `[UNCERTAIN]` query masking | CoopRAG (2512.10422) |
| Sub-query decomposition | CoRAG (2501.14342) |
| Round-trip consistency filter | SimRAG (2025.naacl-long.575) |
| Edge-embedding memory replay | ReMindRAG |

## License

Personal project. Not for redistribution.
