-- HRAG-Bot SQLite schema (v1).
-- Multi-user-ready: every table has a user_id column.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    display_name TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    title        TEXT,
    source_type  TEXT NOT NULL DEFAULT 'document',  -- 'document' | 'episodic'
    metadata     TEXT,                              -- JSON blob
    ingested_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    text         TEXT NOT NULL,
    title        TEXT,
    section      TEXT,
    subsection   TEXT,
    chunk_index  INTEGER NOT NULL DEFAULT 0,
    token_count  INTEGER NOT NULL DEFAULT 0,
    source_type  TEXT NOT NULL DEFAULT 'document',
    excluded     INTEGER NOT NULL DEFAULT 0,        -- /forget tombstone
    page         INTEGER,                           -- Phase 13.1: 1-based source page (PDF only; NULL otherwise)
    chapter      TEXT,                              -- Phase 13.1: normalised, forward-filled heading label
    metadata     TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_excluded ON chunks(excluded);
-- idx_chunks_doc_page is created in migrations.py AFTER the page column is
-- guaranteed to exist. It must NOT live here: init_schema() runs this file whole
-- via executescript(), and an existing pre-13.1 chunks table has no `page`
-- column, so indexing it here would abort the entire schema bootstrap.

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at    TEXT,
    summary     TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS messages (
    message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,                     -- 'system' | 'user' | 'assistant'
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    metadata    TEXT,                              -- Phase 8: JSON-encoded review decision (nullable)
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS preferences (
    pref_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           TEXT NOT NULL,
    polarity          TEXT NOT NULL,                -- 'like' | 'dislike' | 'fact' | 'style'
    topic             TEXT NOT NULL,
    value             TEXT,
    confidence        REAL NOT NULL DEFAULT 0.5,
    last_updated      TEXT NOT NULL DEFAULT (datetime('now')),
    source_session_id TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (source_session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_preferences_user_topic ON preferences(user_id, topic);

-- Phase 2 placeholder tables (empty in Phase 1, declared for forward compat).
CREATE TABLE IF NOT EXISTS kg_nodes (
    node_id   TEXT PRIMARY KEY,
    user_id   TEXT NOT NULL,
    doc_id    TEXT,                                 -- nullable; set on ingest (Phase 2)
    label     TEXT NOT NULL,
    node_type TEXT NOT NULL,                        -- 'phrase' | 'passage'
    metadata  TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_user ON kg_nodes(user_id);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_user_doc ON kg_nodes(user_id, doc_id);

CREATE TABLE IF NOT EXISTS kg_edges (
    edge_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT NOT NULL,
    doc_id    TEXT,                                 -- nullable; set on ingest (Phase 2)
    src       TEXT NOT NULL,
    relation  TEXT NOT NULL,
    dst       TEXT NOT NULL,
    weight    REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_kg_edges_user ON kg_edges(user_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_src ON kg_edges(src);
CREATE INDEX IF NOT EXISTS idx_kg_edges_user_doc ON kg_edges(user_id, doc_id);

-- Phase 2: GraphRAG community summaries (one row per community, per level).
CREATE TABLE IF NOT EXISTS kg_communities (
    community_id    TEXT PRIMARY KEY,                  -- "level0_c12" etc.
    user_id         TEXT NOT NULL,
    level           INTEGER NOT NULL,                  -- Leiden resolution level
    summary         TEXT NOT NULL,
    member_chunks   TEXT NOT NULL,                     -- JSON list of chunk_ids
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_kg_communities_user_level
    ON kg_communities(user_id, level);

-- Phase 2: triple-extraction cache. Keyed by SHA-256(chunk_text + extractor_model)
-- so re-ingesting an unchanged chunk with the same model returns cached triples
-- instantly. Invalidate by clearing rows for a model or by clearing the table.
CREATE TABLE IF NOT EXISTS kg_triple_cache (
    cache_key       TEXT PRIMARY KEY,            -- sha256(chunk_text + model_name)
    model_name      TEXT NOT NULL,
    triples_json    TEXT NOT NULL,               -- list[{head, relation, tail, source_chunk_id}]
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_kg_triple_cache_model ON kg_triple_cache(model_name);

-- Phase 3 (taxonomy sublayer): hierarchical taxonomy tree. Documents are
-- filed under leaf nodes; query navigation beam-searches the tree top-down
-- and only opens leaves' documents.
--   - depth      : 0 = synthetic root, 1+ = real categories
--   - is_leaf    : 1 = docs may be assigned here directly. Computed at write time.
--   - centroid   : packed float32 little-endian; len = embeddings.dim * 4 bytes.
--                  Average of (children centroids) for internal nodes; average of
--                  (assigned doc centroids) for leaves. NULL until first populated.
--   - doc_summary: short LLM-or-title summary of an assigned doc; cached per-doc
--                  on `kg_taxonomy_doc_meta` so re-builds don't repay the LLM cost.
CREATE TABLE IF NOT EXISTS kg_taxonomy_nodes (
    node_id        TEXT PRIMARY KEY,                       -- "root::<user>" or "tx_<hex>"
    user_id        TEXT NOT NULL,
    parent_id      TEXT,                                   -- NULL for root
    label          TEXT NOT NULL,
    description    TEXT,                                   -- short tooltip for LLM + GUI
    depth          INTEGER NOT NULL DEFAULT 0,
    is_leaf        INTEGER NOT NULL DEFAULT 0,
    centroid       BLOB,                                   -- packed float32 (embeddings.dim)
    centroid_dim   INTEGER,                                -- sanity-check dim
    doc_count      INTEGER NOT NULL DEFAULT 0,             -- cached count of descendants
    keywords       TEXT,                                   -- Phase 12: JSON list for hybrid routing
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES kg_taxonomy_nodes(node_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_kg_taxonomy_user ON kg_taxonomy_nodes(user_id);
CREATE INDEX IF NOT EXISTS idx_kg_taxonomy_parent ON kg_taxonomy_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_kg_taxonomy_user_depth ON kg_taxonomy_nodes(user_id, depth);

-- A document is filed under one or more leaf nodes. `is_primary` marks the
-- canonical home; secondary assignments let multi-topic docs surface under
-- multiple branches.
CREATE TABLE IF NOT EXISTS kg_taxonomy_assignments (
    assignment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    node_id        TEXT NOT NULL,                          -- leaf node
    score          REAL NOT NULL DEFAULT 1.0,
    is_primary     INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (node_id) REFERENCES kg_taxonomy_nodes(node_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_kg_taxonomy_assign_doc ON kg_taxonomy_assignments(doc_id);
CREATE INDEX IF NOT EXISTS idx_kg_taxonomy_assign_node ON kg_taxonomy_assignments(node_id);
CREATE INDEX IF NOT EXISTS idx_kg_taxonomy_assign_user ON kg_taxonomy_assignments(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_taxonomy_assign_unique
    ON kg_taxonomy_assignments(user_id, doc_id, node_id);

-- Per-doc summary + centroid cache so build/rebuild can reuse expensive
-- embeddings/LLM-summary work. Keyed by (user_id, doc_id).
CREATE TABLE IF NOT EXISTS kg_taxonomy_doc_meta (
    user_id        TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    summary        TEXT,                                   -- one-line LLM summary; nullable
    centroid       BLOB,                                   -- packed float32 (embeddings.dim)
    centroid_dim   INTEGER,
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, doc_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

-- Phase 5 — background ingest jobs (and any future long-running task).
-- Each row is one piece of work the orchestrator runs in a daemon thread.
-- The web layer polls GET /api/jobs/{id} for live progress.
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,                         -- uuid hex
    user_id      TEXT NOT NULL,
    kind         TEXT NOT NULL,                            -- 'ingest' | (future kinds)
    status       TEXT NOT NULL DEFAULT 'queued',           -- 'queued' | 'running' | 'done' | 'failed'
    progress     INTEGER NOT NULL DEFAULT 0,               -- items completed
    total        INTEGER NOT NULL DEFAULT 0,               -- items expected (0 = unknown)
    message      TEXT,                                     -- latest status line / error
    result       TEXT,                                     -- JSON blob of final result; nullable
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

-- Track D1 — per-message thumbs-up/down feedback.
-- message_id is the INTEGER primary key from the messages table (stored as TEXT
-- here so the FK round-trips cleanly regardless of how the caller serialises it).
-- rating: +1 (good) | -1 (bad) | 0 (neutral/cleared).
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id  TEXT PRIMARY KEY,           -- uuid hex
    message_id   TEXT NOT NULL,              -- assistant message_id from messages
    session_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    rating       INTEGER NOT NULL,           -- +1 | -1 | 0
    note         TEXT,                       -- optional free-text comment
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_message_unique ON feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_rating ON feedback(user_id, rating);

-- Phase 9.9 — Rerank-fallback telemetry. One row per turn where the
-- cross-encoder zero-filter tripped and the orchestrator fell back to the
-- unreranked top-k. Surfaces in `hrag feedback-stats` so users can see which
-- queries the reranker mis-scores.
CREATE TABLE IF NOT EXISTS rerank_fallback_events (
    event_id          TEXT PRIMARY KEY,
    turn_id           TEXT NOT NULL,
    session_id        TEXT,
    user_id           TEXT,
    query             TEXT NOT NULL,
    dropped_chunk_ids TEXT,                              -- JSON list of chunk_ids
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_rerank_fallback_session
    ON rerank_fallback_events(session_id);
