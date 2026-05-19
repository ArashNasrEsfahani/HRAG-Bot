"""FastAPI app exposing the Orchestrator over HTTP + SSE.

Endpoints:
    GET  /                          → index.html (SPA shell)
    GET  /static/*                  → CSS / JS / fonts
    GET  /api/config                → frontend bootstrap (user, model, flags)
    POST /api/config                → mutate live config (think mode, retriever, …)
                                      Phase 6 knobs also exposed: keep_alive,
                                      adaptive_enabled, adaptive_personal_episodic_bias,
                                      adaptive_top_k. Phase 7-A knobs: math_meta_filter_enabled,
                                      math_meta_rerank_threshold, formula_extraction_enabled,
                                      formula_extraction_max_tokens. Backend swaps
                                      (vector_backend, kg.backend) are display-only — require
                                      a restart.
    GET  /api/sessions              → list sessions for the active user
    GET  /api/sessions/{id}         → message history for a session
    DELETE /api/sessions/{id}       → delete a session
    POST /api/chat                  → SSE stream {event, data} of progress + tokens
    GET  /api/sources/{message_id}  → retrieval evidence for an assistant turn
    POST /api/users/switch          → change active user

Streaming uses Server-Sent Events (SSE) — simple, no WebSocket framing,
plays well with the browser fetch + ReadableStream APIs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

logger = logging.getLogger("hrag.web")
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hrag.config import Config, load_config
from hrag.orchestrator import Orchestrator

_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Singleton orchestrator (cached for the lifetime of the process)
# ---------------------------------------------------------------------------


class _State:
    cfg: Optional[Config] = None
    orch: Optional[Orchestrator] = None
    lock = threading.Lock()


def _get_orch() -> Orchestrator:
    with _State.lock:
        if _State.orch is None:
            _State.cfg = load_config()
            _State.orch = Orchestrator(_State.cfg)
        return _State.orch


def _get_cfg() -> Config:
    _get_orch()
    assert _State.cfg is not None
    return _State.cfg


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ConfigPatch(BaseModel):
    """Live-mutable config fields accepted by POST /api/config.

    Backend swaps (vector_backend, kg.backend) require a restart and are
    intentionally not patchable here — send them and they will be silently
    ignored by Pydantic's extra='ignore' default.
    """

    model: Optional[str] = None
    think: Optional[bool] = None
    num_ctx: Optional[int] = None
    temperature: Optional[float] = None
    retriever: Optional[str] = None
    reranker: Optional[str] = None
    rerank_enabled: Optional[bool] = None
    gate_enabled: Optional[bool] = None
    clue_enabled: Optional[bool] = None
    dialog_mst_enabled: Optional[bool] = None
    mask_uncertain: Optional[bool] = None
    # Phase 6 live-mutable knobs
    keep_alive: Optional[str] = None
    adaptive_enabled: Optional[bool] = None
    adaptive_personal_episodic_bias: Optional[bool] = None
    adaptive_top_k: Optional[dict] = None  # intent → int; unknown keys are dropped
    # Phase 7-A live-mutable knobs
    math_meta_filter_enabled: Optional[bool] = None
    math_meta_rerank_threshold: Optional[float] = None
    formula_extraction_enabled: Optional[bool] = None
    formula_extraction_max_tokens: Optional[int] = None
    # Phase 6-B / 7-B / 7-C live-mutable knobs
    num_keep: Optional[int] = None
    embeddings_model: Optional[str] = None
    adaptive_retriever_per_intent: Optional[dict] = None
    use_nougat: Optional[bool] = None
    nougat_model: Optional[str] = None


class UserSwitch(BaseModel):
    user_id: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


app = FastAPI(title="HRAG-Bot Web", docs_url=None, redoc_url=None)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    fav = _STATIC_DIR / "favicon.svg"
    if fav.exists():
        return FileResponse(fav, media_type="image/svg+xml")
    raise HTTPException(404)


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    cfg = _get_cfg()
    return {
        "user_id": cfg.user.default_user_id,
        "llm": {
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
            "think": cfg.llm.think,
            "num_ctx": cfg.llm.num_ctx,
            "temperature": cfg.llm.temperature,
            "keep_alive": cfg.llm.keep_alive,
            "num_keep": cfg.llm.num_keep,
        },
        "embeddings": {
            "model": cfg.embeddings.model,
            "dim": cfg.embeddings.dim,
        },
        "retrieval": {
            "retriever": cfg.retrieval.retriever,
            "reranker": cfg.retrieval.reranker,
            "rerank_enabled": cfg.retrieval.rerank_enabled,
            "top_k_vector": cfg.retrieval.top_k_vector,
            "top_k_final": cfg.retrieval.top_k_final,
            "vector_backend": cfg.retrieval.vector_backend,
            "adaptive_enabled": cfg.retrieval.adaptive_enabled,
            "adaptive_personal_episodic_bias": cfg.retrieval.adaptive_personal_episodic_bias,
            "adaptive_top_k": cfg.retrieval.adaptive_top_k,
            "adaptive_retriever_per_intent": cfg.retrieval.adaptive_retriever_per_intent,
            "math_meta_filter_enabled": cfg.retrieval.math_meta_filter_enabled,
            "math_meta_rerank_threshold": cfg.retrieval.math_meta_rerank_threshold,
        },
        "ingest": {
            "use_nougat": cfg.ingest.use_nougat,
            "nougat_model": cfg.ingest.nougat_model,
        },
        "formula_extraction": {
            "enabled": cfg.formula_extraction.enabled,
            "max_tokens": cfg.formula_extraction.max_tokens,
        },
        "compaction": {
            "gate_enabled": cfg.compaction.gate_enabled,
            "clue_enabled": cfg.compaction.clue_enabled,
            "dialog_mst_enabled": cfg.compaction.dialog_mst_enabled,
            "mask_uncertain": cfg.compaction.mask_uncertain,
        },
        "kg": {
            "enabled": cfg.kg.enabled,
            "backend": cfg.kg.backend,
        },
        "taxonomy": {
            "enabled": cfg.taxonomy.enabled,
            "beam_width": cfg.taxonomy.beam_width,
            "max_depth": cfg.taxonomy.max_depth,
            "propose_sample_size": cfg.taxonomy.propose_sample_size,
            "max_children_per_node": cfg.taxonomy.max_children_per_node,
            "min_top_score_floor": cfg.taxonomy.min_top_score_floor,
            "max_docs_pct": cfg.taxonomy.max_docs_pct,
        },
    }


@app.post("/api/config")
def patch_config(patch: ConfigPatch) -> dict[str, Any]:
    cfg = _get_cfg()
    orch = _get_orch()
    needs_retriever_rebuild = False
    needs_reranker_rebuild = False

    # LLM knobs — read fresh per call by OllamaProvider, so no rebuild needed.
    if patch.model is not None and patch.model != cfg.llm.model:
        cfg.llm.model = patch.model
    if patch.think is not None:
        cfg.llm.think = patch.think
    if patch.num_ctx is not None:
        cfg.llm.num_ctx = patch.num_ctx
    if patch.temperature is not None:
        cfg.llm.temperature = patch.temperature

    # Retrieval module swaps — rebuild the affected component.
    if patch.retriever is not None and patch.retriever != cfg.retrieval.retriever:
        cfg.retrieval.retriever = patch.retriever
        needs_retriever_rebuild = True
    if patch.reranker is not None and patch.reranker != cfg.retrieval.reranker:
        cfg.retrieval.reranker = patch.reranker
        # default sensible threshold per type so the swap "just works"
        if patch.reranker in ("llm", "batched_llm"):
            cfg.retrieval.rerank_threshold = 2.0
        else:
            cfg.retrieval.rerank_threshold = -5.0
        needs_reranker_rebuild = True
    if patch.rerank_enabled is not None:
        cfg.retrieval.rerank_enabled = patch.rerank_enabled

    # Compaction flags — read at chat() time, no rebuild needed.
    if patch.gate_enabled is not None:
        cfg.compaction.gate_enabled = patch.gate_enabled
    if patch.clue_enabled is not None:
        cfg.compaction.clue_enabled = patch.clue_enabled
    if patch.dialog_mst_enabled is not None:
        cfg.compaction.dialog_mst_enabled = patch.dialog_mst_enabled
    if patch.mask_uncertain is not None:
        cfg.compaction.mask_uncertain = patch.mask_uncertain

    # Phase 6 knobs — all live-mutable; the next chat() call picks them up.
    if patch.keep_alive is not None:
        cfg.llm.keep_alive = patch.keep_alive
    if patch.adaptive_enabled is not None:
        cfg.retrieval.adaptive_enabled = patch.adaptive_enabled
    if patch.adaptive_personal_episodic_bias is not None:
        cfg.retrieval.adaptive_personal_episodic_bias = patch.adaptive_personal_episodic_bias
    if patch.adaptive_top_k is not None:
        allowed = {"greeting", "personal", "factual", "general", "unclear"}
        cfg.retrieval.adaptive_top_k = {
            k: int(v) for k, v in patch.adaptive_top_k.items() if k in allowed
        }

    # Phase 7-A knobs — read at chat() time, no rebuild needed.
    if patch.math_meta_filter_enabled is not None:
        cfg.retrieval.math_meta_filter_enabled = patch.math_meta_filter_enabled
    if patch.math_meta_rerank_threshold is not None:
        cfg.retrieval.math_meta_rerank_threshold = patch.math_meta_rerank_threshold
    if patch.formula_extraction_enabled is not None:
        cfg.formula_extraction.enabled = patch.formula_extraction_enabled
    if patch.formula_extraction_max_tokens is not None:
        cfg.formula_extraction.max_tokens = patch.formula_extraction_max_tokens

    # Phase 6-B / 7-B / 7-C knobs.
    if patch.num_keep is not None:
        cfg.llm.num_keep = patch.num_keep

    if patch.embeddings_model is not None and patch.embeddings_model != cfg.embeddings.model:
        # Changing the embedding model requires a full re-ingest because all
        # existing chunk vectors were produced by the old model.  We apply the
        # config change so it takes effect for any NEW ingest, but we do NOT
        # hot-swap the live embedder (the orchestrator caches it).  The caller
        # MUST run `hrag ingest --recursive` to rebuild the index.
        import logging as _logging  # noqa: PLC0415
        _logging.getLogger("hrag.web").warning(
            "embeddings.model changed to %r — existing vectors are stale. "
            "Run `hrag ingest` to rebuild the index before using retrieval.",
            patch.embeddings_model,
        )
        cfg.embeddings.model = patch.embeddings_model
        result = get_config()
        result["warning"] = (
            "Embedding model changed. Existing vectors are stale — "
            "run `hrag ingest` to rebuild the index."
        )
        return result

    if patch.adaptive_retriever_per_intent is not None:
        _valid_retrievers = {
            "default", "vector", "bm25", "hybrid",
            "kg_ppr", "community", "router", "taxonomy",
        }
        _valid_intents = {
            "greeting", "personal", "factual", "general", "unclear",
        }
        bad_values = {
            k: v for k, v in patch.adaptive_retriever_per_intent.items()
            if v not in _valid_retrievers
        }
        if bad_values:
            raise HTTPException(
                400,
                f"Invalid retriever value(s) in adaptive_retriever_per_intent: {bad_values}. "
                f"Allowed values: {sorted(_valid_retrievers)}",
            )
        # Merge new keys into existing dict; only allow known intent keys.
        merged = dict(cfg.retrieval.adaptive_retriever_per_intent)
        merged.update({
            k: v for k, v in patch.adaptive_retriever_per_intent.items()
            if k in _valid_intents
        })
        cfg.retrieval.adaptive_retriever_per_intent = merged

    # Nougat knobs — read at ingest time, no rebuild needed.
    if patch.use_nougat is not None:
        cfg.ingest.use_nougat = patch.use_nougat
    if patch.nougat_model is not None:
        cfg.ingest.nougat_model = patch.nougat_model

    # Live rebuilds (imported lazily to avoid touching heavy deps on cold path).
    if needs_retriever_rebuild:
        from hrag.retrieval.factory import build_retriever  # noqa: PLC0415
        orch.retriever = build_retriever(
            cfg.retrieval,
            orch.db,
            orch.vector_store,
            orch.embedder,
            llm=orch.llm,
            kg_store=getattr(orch, "kg_store", None),
            community_store=getattr(orch, "community_store", None),
            kg_cfg=cfg.kg,
            taxonomy_store=getattr(orch, "taxonomy_store", None),
            taxonomy_cfg=cfg.taxonomy,
        )
    if needs_reranker_rebuild:
        from hrag.retrieval.factory import build_reranker  # noqa: PLC0415
        orch.reranker = build_reranker(cfg.retrieval, orch.llm)

    return get_config()


@app.get("/api/llm/models")
def list_llm_models() -> dict[str, Any]:
    """Enumerate installed Ollama models. Returns {provider, models: [...]}.

    For OpenAI/Anthropic providers, returns an empty list — those don't
    enumerate installed models the same way.
    """
    cfg = _get_cfg()
    if cfg.llm.provider != "ollama":
        return {"provider": cfg.llm.provider, "models": []}
    try:
        import ollama  # noqa: PLC0415
        client = ollama.Client(host=cfg.llm.base_url) if cfg.llm.base_url else ollama.Client()
        resp = client.list()
        # client returns either an object with .models or a dict
        raw = getattr(resp, "models", None)
        if raw is None and isinstance(resp, dict):
            raw = resp.get("models", [])
        out = []
        for m in raw or []:
            name = getattr(m, "model", None)
            size = getattr(m, "size", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
                size = m.get("size")
            if not name:
                continue
            # Normalize: strip trailing ":latest" so user-facing names match
            # what the config typically stores. Ollama treats them the same.
            if name.endswith(":latest"):
                name = name[: -len(":latest")]
            out.append({
                "name": name,
                "size_gb": (size / 1e9) if isinstance(size, (int, float)) else None,
            })
        # current model first, then alphabetical
        cur = cfg.llm.model
        out.sort(key=lambda x: (x["name"] != cur, x["name"]))
        return {"provider": "ollama", "models": out, "current": cur}
    except Exception as exc:  # noqa: BLE001
        return {"provider": "ollama", "models": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


@app.get("/api/users")
def list_users() -> list[dict[str, Any]]:
    orch = _get_orch()
    rows = orch.db.execute("SELECT user_id, created_at FROM users ORDER BY created_at DESC").fetchall()
    return [{"user_id": r["user_id"], "created_at": r["created_at"]} for r in rows]


@app.post("/api/users/switch")
def switch_user(body: UserSwitch) -> dict[str, str]:
    cfg = _get_cfg()
    orch = _get_orch()
    orch.db.ensure_user(body.user_id)
    cfg.user.default_user_id = body.user_id
    return {"user_id": body.user_id}


# ---------------------------------------------------------------------------
# Sessions + messages
# ---------------------------------------------------------------------------


@app.get("/api/sessions")
def list_sessions(user_id: Optional[str] = None) -> list[dict[str, Any]]:
    orch = _get_orch()
    uid = user_id or _get_cfg().user.default_user_id
    rows = orch.db.execute(
        "SELECT s.session_id, s.started_at, "
        "       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS n, "
        "       (SELECT content FROM messages m WHERE m.session_id = s.session_id "
        "          AND m.role = 'user' ORDER BY m.created_at ASC LIMIT 1) AS first_user "
        "FROM sessions s WHERE s.user_id = ? "
        "ORDER BY s.started_at DESC LIMIT 200",
        (uid,),
    ).fetchall()
    out = []
    for r in rows:
        first = r["first_user"] or ""
        title = first[:60] + ("…" if len(first) > 60 else "") if first else "(empty)"
        out.append({
            "session_id": r["session_id"],
            "started_at": r["started_at"],
            "n_messages": r["n"],
            "title": title,
        })
    return out


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    orch = _get_orch()
    rows = orch.db.execute(
        "SELECT message_id, role, content, created_at "
        "FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": r["message_id"],
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"],
            }
            for r in rows
        ],
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    orch = _get_orch()
    with orch.db.conn:
        orch.db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        orch.db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    orch.db.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat — SSE streaming
# ---------------------------------------------------------------------------


def _sse_pack(event: str, data: Any) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, default=str)
    # Multi-line payloads must each be prefixed with "data: ".
    lines = payload.splitlines() or [""]
    body = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event}\n{body}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    orch = _get_orch()
    cfg = _get_cfg()
    uid = req.user_id or cfg.user.default_user_id
    orch.db.ensure_user(uid)
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")

    q: queue.Queue[tuple[str, Any]] = queue.Queue()
    _END = object()

    def _runner() -> None:
        try:
            def _progress(event: str, payload: dict[str, Any]) -> None:
                q.put((event, payload))
            result = orch.chat(
                msg,
                user_id=uid,
                session_id=req.session_id,
                progress=_progress,
                stream=True,
            )
            # Fetch the message_id of the assistant turn just saved so the
            # frontend can attach feedback without a second round-trip.
            msg_row = orch.db.execute(
                "SELECT message_id FROM messages "
                "WHERE session_id = ? AND role = 'assistant' "
                "ORDER BY message_id DESC LIMIT 1",
                (result.session_id,),
            ).fetchone()
            assistant_message_id = msg_row["message_id"] if msg_row else None
            q.put(("__final__", {
                "session_id": result.session_id,
                "answer": result.answer,
                "message_id": assistant_message_id,
                "sources": [
                    {
                        "title": (s.chunk.title or "Untitled"),
                        "section": s.chunk.section or "",
                        "source_type": s.chunk.source_type or "document",
                        "score": float(s.score) if s.score is not None else None,
                        "rerank_score": (float(s.rerank_score)
                                          if s.rerank_score is not None else None),
                        "text": (s.chunk.text or "")[:1200],
                        "doc_id": getattr(s.chunk, "doc_id", None),
                        "chunk_id": getattr(s.chunk, "chunk_id", None),
                    }
                    for s in (result.sources or [])
                ],
            }))
        except Exception as exc:  # noqa: BLE001
            q.put(("__error__", {"message": str(exc), "type": type(exc).__name__}))
        finally:
            q.put((_END, None))

    threading.Thread(target=_runner, daemon=True).start()

    async def _stream():
        # Initial open event so the client knows the stream is alive.
        yield _sse_pack("open", {"ts": time.time()})
        loop = asyncio.get_event_loop()
        while True:
            event, payload = await loop.run_in_executor(None, q.get)
            if event is _END:
                yield _sse_pack("done", {"ts": time.time()})
                break
            if event == "__final__":
                yield _sse_pack("final", payload)
                continue
            if event == "__error__":
                yield _sse_pack("error", payload)
                continue
            # Per-token events are routed as their own SSE event so the
            # frontend can append efficiently.
            if event == "generate_token":
                yield _sse_pack("token", payload)
                continue
            # Everything else (retrieve_start, rerank_done, …) goes through
            # as a generic progress event.
            yield _sse_pack("progress", {"event": event, "payload": payload})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering
        },
    )


# ---------------------------------------------------------------------------
# Docs / memories — read-only listings + CRUD
# ---------------------------------------------------------------------------


@app.get("/api/docs")
def list_docs(user_id: Optional[str] = None) -> list[dict[str, Any]]:
    orch = _get_orch()
    uid = user_id or _get_cfg().user.default_user_id
    rows = orch.db.execute(
        "SELECT d.doc_id, d.title, d.source_path, d.source_type, d.ingested_at, "
        "       (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.doc_id AND c.excluded = 0) AS n_chunks, "
        "       a.node_id "
        "FROM documents d "
        "LEFT JOIN kg_taxonomy_assignments a ON a.doc_id = d.doc_id AND a.user_id = ? "
        "WHERE d.user_id = ? AND d.source_type = 'document' "
        "ORDER BY d.ingested_at DESC",
        (uid, uid),
    ).fetchall()
    return [
        {
            "doc_id": r["doc_id"],
            "title": r["title"],
            "source_path": r["source_path"],
            "source_type": r["source_type"],
            "ingested_at": r["ingested_at"],
            "n_chunks": r["n_chunks"],
            "node_id": r["node_id"],
        }
        for r in rows
    ]


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, Any]:
    """Return metadata + content preview + taxonomy assignment for a document."""
    orch = _get_orch()
    uid = _get_cfg().user.default_user_id
    row = orch.db.execute(
        "SELECT doc_id, title, source_path, source_type, ingested_at "
        "FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
        raise HTTPException(404, "document not found")
    # First few chunks for a content preview.
    preview_rows = orch.db.execute(
        "SELECT chunk_index, text FROM chunks WHERE doc_id = ? AND excluded = 0 "
        "ORDER BY chunk_index ASC LIMIT 3",
        (doc_id,),
    ).fetchall()
    preview = "\n\n".join(r["text"] for r in preview_rows if r["text"])[:800]
    n_chunks = orch.db.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ? AND excluded = 0",
        (doc_id,),
    ).fetchone()[0]
    # Current taxonomy assignment if any.
    asg = orch.db.execute(
        "SELECT node_id FROM kg_taxonomy_assignments WHERE doc_id = ? AND user_id = ?",
        (doc_id, uid),
    ).fetchone()
    return {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "source_path": row["source_path"],
        "source_type": row["source_type"],
        "ingested_at": row["ingested_at"],
        "n_chunks": int(n_chunks),
        "preview": preview,
        "node_id": asg["node_id"] if asg else None,
    }


@app.get("/api/documents/{doc_id}/chunks")
def get_document_chunks(doc_id: str) -> dict[str, Any]:
    """Return every (non-excluded) chunk of a document, ordered.

    Used by the right-side preview panel's "View all N chunks" button so
    the user can read the whole document without re-ingesting it.
    """
    orch = _get_orch()
    row = orch.db.execute(
        "SELECT doc_id, title FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
        raise HTTPException(404, "document not found")
    rows = orch.db.execute(
        "SELECT chunk_index, title AS section_title, section, text, token_count "
        "FROM chunks WHERE doc_id = ? AND excluded = 0 "
        "ORDER BY chunk_index ASC",
        (doc_id,),
    ).fetchall()
    return {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "n_chunks": len(rows),
        "chunks": [
            {
                "chunk_index": int(r["chunk_index"]),
                "section": r["section"] or "",
                "section_title": r["section_title"] or "",
                "text": r["text"] or "",
                "token_count": int(r["token_count"] or 0),
            }
            for r in rows
        ],
    }


@app.delete("/api/documents/{doc_id}")
def delete_document_endpoint(doc_id: str) -> dict[str, Any]:
    """Hard-delete a document: chunks + vectors + KG nodes/edges + taxonomy assignments.

    Episodic memories (doc_id prefixed with ``episodic:``) must be deleted via
    DELETE /api/memories/{memory_id} instead; calling this endpoint with an
    episodic id returns 400.

    Deleting a non-existent doc is idempotent — returns 200 with deleted_chunks=0.
    """
    if doc_id.startswith("episodic:"):
        raise HTTPException(400, "Use DELETE /api/memories/{memory_id} for memories.")
    orch = _get_orch()
    uid = _get_cfg().user.default_user_id
    print(f"[doc-delete] starting removal of doc {doc_id!r}", flush=True)

    # Count chunks before removal so we can report a useful number.
    n_chunks = orch.db.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0]

    # 1. Vector store — delete embeddings by (user_id, doc_id).
    try:
        orch.vector_store.delete_doc(uid, doc_id)
        print(f"[doc-delete] vectors removed for {doc_id!r}", flush=True)
    except Exception:
        logger.exception("vector delete failed for %s", doc_id)

    # 2. KG — drop passage nodes and pruned phrase edges from this doc.
    kg_store = getattr(orch, "kg_store", None)
    if kg_store is not None:
        try:
            kg_store.delete_doc(uid, doc_id)
            print(f"[doc-delete] KG nodes/edges removed for {doc_id!r}", flush=True)
        except Exception:
            logger.exception("kg delete failed for %s", doc_id)

    # 3. Taxonomy assignments.
    orch.db.execute(
        "DELETE FROM kg_taxonomy_assignments WHERE doc_id = ?", (doc_id,)
    )

    # 4. Chunks + document row.
    orch.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    orch.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    orch.db.commit()

    print(
        f"[doc-delete] done: doc={doc_id!r} chunks_removed={n_chunks}", flush=True
    )
    return {"doc_id": doc_id, "deleted_chunks": int(n_chunks)}


@app.get("/api/memories")
def list_memories(user_id: Optional[str] = None, limit: int = 200) -> list[dict[str, Any]]:
    """One row per memory (doc_id), with chunks concatenated in chunk_index order.

    Episodic memories are stored as one Document → N chunks (most are 1).
    The web UI treats a memory as a single unit; we recombine chunks here so
    the user sees one card per `/remember` call, not one per chunk.
    """
    orch = _get_orch()
    uid = user_id or _get_cfg().user.default_user_id
    rows = orch.db.execute(
        "SELECT d.doc_id AS memory_id, d.title, d.ingested_at AS created_at "
        "FROM documents d "
        "WHERE d.user_id = ? AND d.source_type = 'episodic' "
        "  AND EXISTS (SELECT 1 FROM chunks c "
        "              WHERE c.doc_id = d.doc_id AND c.excluded = 0) "
        "ORDER BY d.ingested_at DESC LIMIT ?",
        (uid, limit),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        chunk_rows = orch.db.execute(
            "SELECT text FROM chunks WHERE doc_id = ? AND excluded = 0 "
            "ORDER BY chunk_index ASC",
            (r["memory_id"],),
        ).fetchall()
        text = "\n\n".join(c["text"] for c in chunk_rows if c["text"])
        out.append({
            "memory_id": r["memory_id"],
            "title": r["title"],
            "text": text,
            "created_at": r["created_at"],
        })
    return out


class MemoryEdit(BaseModel):
    text: str
    title: Optional[str] = None


@app.put("/api/memories/{memory_id}")
def update_memory(memory_id: str, body: MemoryEdit) -> dict[str, Any]:
    """Replace a memory's text. Implemented as forget-old + add-new (since we
    embed at chunk granularity; in-place vector edit is not safe across chunk
    boundary changes). Returns the new memory_id.

    The taxonomy assignment (if any) is migrated to the new id so the GUI's
    node popover keeps pointing at a real, queryable memory. Also any orphan
    rows left behind by `forget_memory` (tombstoned chunks + document row)
    are cleaned up here so they don't leak into the doc list later.
    """
    orch = _get_orch()
    uid = _get_cfg().user.default_user_id
    if not body.text.strip():
        raise HTTPException(400, "memory text must be non-empty")
    # confirm the memory belongs to the active user before touching anything
    row = orch.db.execute(
        "SELECT doc_id FROM documents WHERE doc_id = ? AND user_id = ? "
        "AND source_type = 'episodic'",
        (memory_id, uid),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "memory not found")
    # Capture current taxonomy assignment (we'll migrate it to the new id).
    old_assignment = orch.db.execute(
        "SELECT node_id, score, is_primary FROM kg_taxonomy_assignments "
        "WHERE doc_id = ? AND user_id = ?",
        (memory_id, uid),
    ).fetchone()
    orch.memory_store.forget_memory(uid, memory_id)
    # forget_memory only tombstones (sets excluded=1); the document row + the
    # assignment + the tombstoned chunks would otherwise stay around as
    # orphans visible nowhere but cluttering the taxonomy's doc lists. Wipe
    # them so the taxonomy stays clean.
    orch.db.execute("DELETE FROM kg_taxonomy_assignments WHERE doc_id = ?", (memory_id,))
    orch.db.execute("DELETE FROM chunks WHERE doc_id = ?", (memory_id,))
    orch.db.execute("DELETE FROM documents WHERE doc_id = ?", (memory_id,))
    orch.db.commit()
    new_id = orch.memory_store.add(uid, body.text, title=body.title)
    # Migrate the taxonomy assignment to the new id.
    if old_assignment is not None:
        orch.db.execute(
            "INSERT OR REPLACE INTO kg_taxonomy_assignments "
            "(user_id, doc_id, node_id, score, is_primary) "
            "VALUES (?, ?, ?, ?, ?)",
            (uid, new_id, old_assignment["node_id"],
             float(old_assignment["score"] or 1.0),
             int(old_assignment["is_primary"] or 1)),
        )
        orch.db.commit()
    return {"old_memory_id": memory_id, "new_memory_id": new_id}


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str) -> dict[str, Any]:
    """Hard-delete a memory: tombstone its chunks, then remove the document
    row, its (now-tombstoned) chunks, and any taxonomy assignment.

    Phase 7's `forget_memory` only sets ``excluded=1`` on the chunks so they
    drop out of retrieval, but the rows + assignment lingered as orphans —
    visible in the taxonomy popover's chip list yet 404 on lookup. The
    DELETE endpoint now wipes the whole footprint so a deleted memory really
    is gone everywhere.
    """
    orch = _get_orch()
    uid = _get_cfg().user.default_user_id
    row = orch.db.execute(
        "SELECT doc_id FROM documents WHERE doc_id = ? AND user_id = ? "
        "AND source_type = 'episodic'",
        (memory_id, uid),
    ).fetchone()
    if row is None:
        # Idempotent: deleting an already-gone memory just removes any
        # leftover assignment row (handles orphan cleanup from the GUI).
        n = orch.db.execute(
            "DELETE FROM kg_taxonomy_assignments WHERE doc_id = ? AND user_id = ?",
            (memory_id, uid),
        ).rowcount
        orch.db.commit()
        return {"memory_id": memory_id, "forgotten_chunks": 0, "removed_orphan_assignment": bool(n)}
    count = orch.memory_store.forget_memory(uid, memory_id)
    orch.db.execute("DELETE FROM kg_taxonomy_assignments WHERE doc_id = ?", (memory_id,))
    orch.db.execute("DELETE FROM chunks WHERE doc_id = ?", (memory_id,))
    orch.db.execute("DELETE FROM documents WHERE doc_id = ?", (memory_id,))
    orch.db.commit()
    return {"memory_id": memory_id, "forgotten_chunks": int(count)}


class MemoryCreate(BaseModel):
    text: str
    title: Optional[str] = None


@app.post("/api/memories")
def create_memory(body: MemoryCreate) -> dict[str, Any]:
    orch = _get_orch()
    uid = _get_cfg().user.default_user_id
    if not body.text.strip():
        raise HTTPException(400, "memory text must be non-empty")
    new_id = orch.memory_store.add(uid, body.text, title=body.title)
    return {"memory_id": new_id}


# ---------------------------------------------------------------------------
# Smart Remember — propose memorable items extracted from the conversation
# ---------------------------------------------------------------------------


class ExtractMemoryRequest(BaseModel):
    session_id: Optional[str] = None
    max_items: int = 8


def _format_candidate_text(polarity: str, topic: str, value: str) -> str:
    """Render a PreferenceCandidate as a single self-contained sentence."""
    polarity = (polarity or "").strip().lower()
    topic = (topic or "").strip()
    value = (value or "").strip()
    if not topic and not value:
        return ""
    if polarity == "like":
        body = f"User likes {topic}" + (f" ({value})" if value else "")
    elif polarity == "dislike":
        body = f"User dislikes {topic}" + (f" ({value})" if value else "")
    elif polarity == "style":
        body = f"User prefers response style: {topic}" + (
            f" — {value}" if value else ""
        )
    else:  # "fact" or anything else: render as a fact
        if topic and value:
            body = f"User's {topic}: {value}"
        else:
            body = f"User: {topic or value}"
    # Keep it bounded — the spec is ≤140 chars.
    return body[:140]


@app.post("/api/memories/extract")
def extract_memories_endpoint(req: ExtractMemoryRequest) -> dict[str, Any]:
    """Analyze a conversation and propose candidate memories to save.

    Reuses ``PreferenceExtractor`` (Phase 3 auto-extract pipeline) and
    reshapes its ``PreferenceCandidate`` output into the response envelope
    the modal expects: ``{text, category, confidence}``.

    Behavior:
    * No ``session_id`` given → pick the active user's most recent session.
    * Session with no messages → ``{items: [], n_turns_considered: 0}``.
    * Session_id pointing at a non-existent session → same empty response
      (mirrors the "no messages" path; we deliberately do not 404 so the
      modal still renders an empty state instead of an error toast).
    * LLM crash or garbage output → ``{items: []}`` with a 200 (warning
      logged); PreferenceExtractor already swallows exceptions internally.
    """
    from hrag.memory.extractor import PreferenceExtractor  # noqa: PLC0415

    orch = _get_orch()
    cfg = _get_cfg()
    uid = cfg.user.default_user_id
    max_items = max(1, int(req.max_items or 8))

    sid = req.session_id
    if not sid:
        row = orch.db.execute(
            "SELECT session_id FROM sessions WHERE user_id = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (uid,),
        ).fetchone()
        if not row:
            return {"items": [], "session_id": None, "n_turns_considered": 0}
        sid = row["session_id"]

    rows = orch.db.execute(
        "SELECT role, content FROM messages WHERE session_id = ? "
        "ORDER BY created_at ASC",
        (sid,),
    ).fetchall()
    if not rows:
        return {"items": [], "session_id": sid, "n_turns_considered": 0}

    # Cap to the last ~30 messages or ~6000 chars (whichever is smaller) so
    # the extraction prompt stays bounded on long sessions.
    MAX_TURNS = 30
    MAX_CHARS = 6000
    tail = list(rows[-MAX_TURNS:])
    char_total = sum(len((r["content"] or "")) for r in tail)
    while tail and char_total > MAX_CHARS and len(tail) > 1:
        dropped = tail.pop(0)
        char_total -= len(dropped["content"] or "")
    conversation = [(r["role"], r["content"] or "") for r in tail]

    extractor = PreferenceExtractor(orch.llm)
    try:
        candidates = extractor.extract(conversation)
    except Exception as exc:  # noqa: BLE001 - defensive belt
        logger.warning(
            "memories/extract: PreferenceExtractor raised %s: %s",
            type(exc).__name__, exc,
        )
        candidates = []

    items: list[dict[str, Any]] = []
    for c in (candidates or [])[:max_items]:
        text = _format_candidate_text(c.polarity, c.topic, c.value)
        if not text:
            continue
        items.append({
            "text": text,
            "category": c.polarity,            # like / dislike / fact / style
            "confidence": float(c.confidence),
        })
    return {
        "items": items,
        "session_id": sid,
        "n_turns_considered": len(conversation),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Feedback — Track D1 (per-message thumbs up/down + training-pair export)
# ---------------------------------------------------------------------------


class FeedbackCreate(BaseModel):
    message_id: str
    rating: int           # +1, -1, or 0 (clear)
    note: Optional[str] = None


@app.post("/api/feedback")
def upsert_feedback(body: FeedbackCreate) -> dict[str, Any]:
    """Create or replace feedback for a single assistant message."""
    import uuid  # noqa: PLC0415
    orch = _get_orch()
    if body.rating not in (-1, 0, 1):
        raise HTTPException(400, "rating must be -1, 0, or 1")
    # Resolve session_id / user_id from the messages table.
    row = orch.db.execute(
        "SELECT session_id, user_id FROM messages WHERE message_id = ?",
        (str(body.message_id),),
    ).fetchone()
    if row is None:
        raise HTTPException(404, f"message {body.message_id!r} not found")
    session_id = row["session_id"]
    user_id = row["user_id"]
    feedback_id = uuid.uuid4().hex
    with orch.db.conn:
        # DELETE-then-INSERT so the unique index on message_id stays clean.
        orch.db.execute(
            "DELETE FROM feedback WHERE message_id = ?",
            (str(body.message_id),),
        )
        orch.db.execute(
            "INSERT INTO feedback (feedback_id, message_id, session_id, user_id, rating, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (feedback_id, str(body.message_id), session_id, user_id, body.rating, body.note),
        )
    orch.db.commit()
    return {
        "feedback_id": feedback_id,
        "message_id": str(body.message_id),
        "session_id": session_id,
        "user_id": user_id,
        "rating": body.rating,
    }


@app.get("/api/feedback")
def list_feedback(session_id: str) -> list[dict[str, Any]]:
    """Return all feedback rows for a session."""
    orch = _get_orch()
    rows = orch.db.execute(
        "SELECT feedback_id, message_id, rating, note, created_at "
        "FROM feedback WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@app.delete("/api/feedback/{message_id}")
def delete_feedback(message_id: str) -> dict[str, Any]:
    """Clear the feedback for a single message (sets rating back to neutral)."""
    orch = _get_orch()
    with orch.db.conn:
        cur = orch.db.execute(
            "DELETE FROM feedback WHERE message_id = ?", (message_id,)
        )
    orch.db.commit()
    return {"message_id": message_id, "deleted": cur.rowcount > 0}


# ---------------------------------------------------------------------------
# Document upload (multipart) + background ingest jobs
# ---------------------------------------------------------------------------


_INGEST_SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def _create_job(uid: str, kind: str, total: int = 0, message: str = "queued") -> str:
    """Insert a `jobs` row and return its id. Caller owns the worker thread."""
    import uuid  # noqa: PLC0415
    orch = _get_orch()
    job_id = uuid.uuid4().hex
    with orch.db.conn:
        orch.db.execute(
            "INSERT INTO jobs (job_id, user_id, kind, status, progress, total, message) "
            "VALUES (?, ?, ?, 'queued', 0, ?, ?)",
            (job_id, uid, kind, total, message),
        )
    orch.db.commit()
    return job_id


def _update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    total: Optional[int] = None,
    message: Optional[str] = None,
    result: Optional[dict] = None,
    completed: bool = False,
) -> None:
    orch = _get_orch()
    sets, vals = [], []
    if status is not None: sets.append("status = ?"); vals.append(status)
    if progress is not None: sets.append("progress = ?"); vals.append(progress)
    if total is not None: sets.append("total = ?"); vals.append(total)
    if message is not None: sets.append("message = ?"); vals.append(message)
    if result is not None: sets.append("result = ?"); vals.append(json.dumps(result, default=str))
    if completed: sets.append("completed_at = datetime('now')")
    if not sets: return
    vals.append(job_id)
    with orch.db.conn:
        orch.db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", vals)
    orch.db.commit()


def _run_ingest_job(job_id: str, uid: str, dest_path: Path, original_name: str) -> None:
    """Worker thread: runs IngestPipeline.ingest_path and updates the job row."""
    try:
        _update_job(job_id, status="running", message=f"ingesting {original_name}…")
        orch = _get_orch()
        doc = orch.ingest.ingest_path(str(dest_path), uid)
        n_chunks = orch.db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ? AND excluded = 0",
            (doc.doc_id,),
        ).fetchone()["n"]
        _update_job(
            job_id,
            status="done",
            progress=1, total=1,
            message=f"{n_chunks} chunks",
            result={
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source_path": str(dest_path),
                "n_chunks": n_chunks,
            },
            completed=True,
        )
    except Exception as exc:  # noqa: BLE001
        try: dest_path.unlink(missing_ok=True)
        except Exception: pass
        _update_job(
            job_id,
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
            completed=True,
        )


@app.post("/api/ingest")
async def ingest_upload(file: UploadFile = File(...), background: bool = True) -> dict[str, Any]:
    """Document ingest from a browser upload.

    When ``background=true`` (default) returns immediately with a ``job_id``;
    poll ``GET /api/jobs/{job_id}`` for progress. When ``background=false``
    blocks until done and returns the final summary directly.
    """
    cfg = _get_cfg()
    uid = cfg.user.default_user_id

    name = (file.filename or "upload").strip()
    if not name:
        raise HTTPException(400, "filename required")
    suffix = Path(name).suffix.lower()
    if suffix not in _INGEST_SUPPORTED:
        raise HTTPException(
            415,
            f"Unsupported file type {suffix!r}. Supported: {sorted(_INGEST_SUPPORTED)}",
        )

    uploads_dir = cfg.resolve("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    import uuid  # noqa: PLC0415
    safe_stem = Path(name).stem.replace("/", "_").replace("\\", "_")
    dest = uploads_dir / f"{safe_stem}-{uuid.uuid4().hex[:8]}{suffix}"

    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")
    dest.write_bytes(content)

    job_id = _create_job(uid, kind="ingest", total=1, message=f"queued: {name}")
    if background:
        threading.Thread(
            target=_run_ingest_job, args=(job_id, uid, dest, name), daemon=True
        ).start()
        return {"job_id": job_id, "kind": "ingest", "status": "queued", "filename": name}

    # Synchronous path — useful for the smoke benchmark.
    _run_ingest_job(job_id, uid, dest, name)
    row = _get_job(job_id)
    if row["status"] == "failed":
        raise HTTPException(500, row["message"] or "ingest failed")
    return {"job_id": job_id, **(json.loads(row["result"]) if row["result"] else {})}


def _get_job(job_id: str) -> dict:
    orch = _get_orch()
    row = orch.db.execute(
        "SELECT job_id, user_id, kind, status, progress, total, message, result, "
        "       created_at, completed_at "
        "FROM jobs WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "job not found")
    return dict(row)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    row = _get_job(job_id)
    if row.get("result"):
        try: row["result"] = json.loads(row["result"])
        except Exception: pass
    return row


@app.get("/api/jobs")
def list_jobs(user_id: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
    orch = _get_orch()
    uid = user_id or _get_cfg().user.default_user_id
    rows = orch.db.execute(
        "SELECT job_id, kind, status, progress, total, message, "
        "       created_at, completed_at "
        "FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (uid, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Phase 6-B / 7-B / 7-C — additional GET endpoints
# ---------------------------------------------------------------------------


@app.get("/api/embeddings/suggested")
def get_embeddings_suggested() -> dict[str, Any]:
    """Return the current embedding model and the curated suggestion list.

    The suggestions list is populated from ``cfg.embeddings.suggested_models``
    which is defined in config.py.  Changing the model requires a full re-ingest
    so the endpoint surfaces this cost clearly.
    """
    cfg = _get_cfg()
    return {
        "current": cfg.embeddings.model,
        "current_dim": cfg.embeddings.dim,
        "suggestions": cfg.embeddings.suggested_models,
    }


@app.get("/api/ingest/nougat_status")
def get_nougat_status() -> dict[str, Any]:
    """Return whether the optional Nougat OCR loader is available + configured."""
    cfg = _get_cfg()
    from hrag.ingest.nougat_loader import is_nougat_available  # noqa: PLC0415
    return {
        "available": is_nougat_available(),
        "model": cfg.ingest.nougat_model,
        "use_nougat": cfg.ingest.use_nougat,
    }


@app.get("/api/feedback/stats")
def get_feedback_stats() -> dict[str, Any]:
    """Return aggregate feedback statistics (thumbs-up/down totals + top negative)."""
    orch = _get_orch()
    from hrag.feedback_stats import feedback_summary  # noqa: PLC0415
    return feedback_summary(orch.db)


# ---------------------------------------------------------------------------
# Taxonomy CRUD + SSE recompute
#
# Surface for the live drag-and-drop tree editor. CRUD endpoints wrap
# ``TaxonomyStore``; the SSE recompute / assign-unfiled endpoints spawn a
# daemon thread, feed builder progress callbacks into a Queue, and drain the
# Queue from the streaming response.
# ---------------------------------------------------------------------------


class _TaxonomyState:
    """Optional dependency-injection seams used by tests.

    ``builder_factory`` / ``assigner_factory`` are normally None and the real
    ``TaxonomyBuilder`` / ``DocAssigner`` are constructed on demand. Tests can
    swap in a fake to avoid running the real LLM-driven build.
    """

    builder_factory: Optional[Any] = None
    assigner_factory: Optional[Any] = None


def _require_taxonomy_store():
    orch = _get_orch()
    store = getattr(orch, "taxonomy_store", None)
    if store is None:
        raise HTTPException(
            503,
            "taxonomy is disabled in config (taxonomy.enabled=false)",
        )
    return store


def _build_taxonomy_builder():
    if _TaxonomyState.builder_factory is not None:
        return _TaxonomyState.builder_factory()
    from hrag.taxonomy.builder import TaxonomyBuilder  # noqa: PLC0415
    cfg = _get_cfg()
    orch = _get_orch()
    return TaxonomyBuilder(
        orch.db, orch.llm, orch.embedder, _require_taxonomy_store(), cfg.taxonomy,
    )


def _build_doc_assigner():
    if _TaxonomyState.assigner_factory is not None:
        return _TaxonomyState.assigner_factory()
    from hrag.taxonomy.assigner import DocAssigner  # noqa: PLC0415
    cfg = _get_cfg()
    orch = _get_orch()
    return DocAssigner(
        orch.db, orch.llm, orch.embedder, _require_taxonomy_store(), cfg.taxonomy,
    )


def _node_to_dict(node: Any, *, with_centroid: bool = False) -> dict[str, Any]:
    out = {
        "id": node.node_id,
        "node_id": node.node_id,
        "user_id": node.user_id,
        "parent_id": node.parent_id,
        "label": node.label,
        "description": node.description or "",
        "depth": int(node.depth),
        "is_leaf": bool(node.is_leaf),
        "doc_count": int(node.doc_count),
        "created_at": node.created_at,
        "updated_at": node.updated_at,
    }
    if with_centroid:
        out["centroid_dim"] = (
            len(node.centroid) if node.centroid is not None else None
        )
    return out


def _live_doc_counts(user_id: str) -> dict[str, int]:
    """Count assignments per node *right now* from kg_taxonomy_assignments.

    ``kg_taxonomy_nodes.doc_count`` is a cached column written at build time;
    it does NOT decrement when a document is deleted, so reading it would
    leave stale numbers on every node card after a delete. Compute live.
    """
    orch = _get_orch()
    rows = orch.db.execute(
        "SELECT node_id, COUNT(*) AS n FROM kg_taxonomy_assignments "
        "WHERE user_id = ? GROUP BY node_id",
        (user_id,),
    ).fetchall()
    return {r["node_id"]: int(r["n"]) for r in rows}


def _build_tree_dict(store, user_id: str) -> dict[str, Any]:
    """Walk the user's taxonomy and return a nested dict suitable for D3."""
    nodes = list(store.list_nodes(user_id))
    if not nodes:
        return {
            "user_id": user_id,
            "node_count": 0,
            "doc_count": 0,
            "unfiled_count": _count_unfiled(user_id),
            "root": None,
        }
    children_of: dict[str, list] = {n.node_id: [] for n in nodes}
    root_node = None
    for n in nodes:
        if n.parent_id is None:
            root_node = n
        else:
            children_of.setdefault(n.parent_id, []).append(n)
    # Stable label-sort within each parent.
    for nid in children_of:
        children_of[nid].sort(key=lambda x: (not x.is_leaf, x.label or ""))

    if root_node is None:
        return {
            "user_id": user_id,
            "node_count": len(nodes),
            "doc_count": 0,
            "unfiled_count": _count_unfiled(user_id),
            "root": None,
        }

    live = _live_doc_counts(user_id)
    total_docs = 0

    def _serialize(n) -> dict[str, Any]:
        nonlocal total_docs
        children = [_serialize(c) for c in children_of.get(n.node_id, [])]
        # Leaf: count direct assignments. Internal: sum its children.
        if n.is_leaf:
            count = live.get(n.node_id, 0)
            total_docs += count
        else:
            count = sum(c["doc_count"] for c in children)
        return {
            "id": n.node_id,
            "node_id": n.node_id,
            "label": n.label,
            "description": n.description or "",
            "depth": int(n.depth),
            "is_leaf": bool(n.is_leaf),
            "doc_count": count,
            "parent_id": n.parent_id,
            "children": children,
        }

    root_dict = _serialize(root_node)
    return {
        "user_id": user_id,
        "node_count": len(nodes),
        "doc_count": total_docs,
        "unfiled_count": _count_unfiled(user_id),
        "root": root_dict,
    }


def _count_unfiled(user_id: str) -> int:
    """Count documents not present in kg_taxonomy_assignments."""
    orch = _get_orch()
    row = orch.db.execute(
        "SELECT COUNT(*) AS n FROM documents d "
        "WHERE d.user_id = ? "
        "  AND NOT EXISTS ("
        "    SELECT 1 FROM kg_taxonomy_assignments a "
        "    WHERE a.doc_id = d.doc_id AND a.user_id = d.user_id"
        "  )",
        (user_id,),
    ).fetchone()
    return int(row["n"]) if row else 0


# ---- Read endpoints --------------------------------------------------------


@app.get("/api/taxonomy/tree")
def taxonomy_tree(user_id: Optional[str] = None) -> dict[str, Any]:
    store = _require_taxonomy_store()
    uid = user_id or _get_cfg().user.default_user_id
    return _build_tree_dict(store, uid)


@app.get("/api/taxonomy/unfiled")
def taxonomy_unfiled(user_id: Optional[str] = None) -> list[dict[str, Any]]:
    """List docs that have no taxonomy assignment yet."""
    _require_taxonomy_store()
    orch = _get_orch()
    uid = user_id or _get_cfg().user.default_user_id
    rows = orch.db.execute(
        "SELECT d.doc_id, d.title, d.source_path, d.ingested_at, d.source_type "
        "FROM documents d "
        "WHERE d.user_id = ? "
        "  AND NOT EXISTS ("
        "    SELECT 1 FROM kg_taxonomy_assignments a "
        "    WHERE a.doc_id = d.doc_id AND a.user_id = d.user_id"
        "  ) "
        "ORDER BY d.ingested_at DESC",
        (uid,),
    ).fetchall()
    return [
        {
            "doc_id": r["doc_id"],
            "title": r["title"] or "",
            "source_path": r["source_path"] or "",
            "ingested_at": r["ingested_at"],
            "source_type": r["source_type"] or "document",
        }
        for r in rows
    ]


@app.get("/api/taxonomy/nodes/{node_id}/docs")
def taxonomy_node_docs(node_id: str) -> dict[str, Any]:
    """List docs currently assigned to a single node (leaf semantics)."""
    store = _require_taxonomy_store()
    node = store.get_node(node_id)
    if node is None:
        raise HTTPException(404, f"node {node_id!r} not found")
    doc_ids = store.get_docs_at(node_id)
    if not doc_ids:
        return {"node_id": node_id, "label": node.label, "docs": []}
    orch = _get_orch()
    placeholders = ",".join("?" for _ in doc_ids)
    rows = orch.db.execute(
        f"SELECT doc_id, title, source_path, ingested_at, source_type "
        f"FROM documents WHERE doc_id IN ({placeholders})",
        doc_ids,
    ).fetchall()
    by_id = {r["doc_id"]: r for r in rows}
    docs = []
    for did in doc_ids:
        r = by_id.get(did)
        if r is None:
            docs.append({"doc_id": did, "title": "", "missing": True})
            continue
        docs.append({
            "doc_id": r["doc_id"],
            "title": r["title"] or "",
            "source_path": r["source_path"] or "",
            "ingested_at": r["ingested_at"],
            "source_type": r["source_type"] or "document",
        })
    return {"node_id": node_id, "label": node.label, "docs": docs}


# ---- Mutate endpoints ------------------------------------------------------


class TaxonomyNodeCreate(BaseModel):
    label: str
    description: Optional[str] = ""
    parent_id: Optional[str] = None
    is_leaf: Optional[bool] = False


class TaxonomyNodePatch(BaseModel):
    label: Optional[str] = None
    description: Optional[str] = None
    parent_id: Optional[str] = None


class TaxonomyMoveDoc(BaseModel):
    doc_id: str
    node_id: str


class TaxonomyRecomputeBody(BaseModel):
    user_id: Optional[str] = None


@app.post("/api/taxonomy/nodes")
def taxonomy_create_node(body: TaxonomyNodeCreate) -> dict[str, Any]:
    store = _require_taxonomy_store()
    cfg = _get_cfg()
    uid = cfg.user.default_user_id
    label = (body.label or "").strip()
    if not label:
        raise HTTPException(400, "label is required")
    parent_id = body.parent_id
    if parent_id is None:
        # Default-parent: the user's root node (ensure_root creates if absent).
        parent_id = store.ensure_root(uid).node_id
    try:
        node = store.add_node(
            uid,
            parent_id,
            label,
            (body.description or "").strip(),
            is_leaf=bool(body.is_leaf),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    print(f"[taxonomy] created node {node.node_id} ({label}) under {parent_id}", flush=True)
    return _node_to_dict(node)


@app.put("/api/taxonomy/nodes/{node_id}")
def taxonomy_update_node(node_id: str, body: TaxonomyNodePatch) -> dict[str, Any]:
    store = _require_taxonomy_store()
    existing = store.get_node(node_id)
    if existing is None:
        raise HTTPException(404, f"node {node_id!r} not found")
    if existing.parent_id is None and (
        body.label is not None or body.description is not None or body.parent_id is not None
    ):
        # Allow no-op updates on root, but reject rename/reparent.
        if body.parent_id is not None:
            raise HTTPException(400, "cannot move the root node")
        if body.label is not None or body.description is not None:
            raise HTTPException(400, "cannot rename the root node")

    try:
        if body.label is not None or body.description is not None:
            store.update_node(node_id, label=body.label, description=body.description)
        if body.parent_id is not None and body.parent_id != existing.parent_id:
            store.move_node(node_id, body.parent_id)
    except ValueError as exc:
        # cycle / unknown parent / cross-user → 400
        raise HTTPException(400, str(exc)) from exc

    updated = store.get_node(node_id)
    assert updated is not None
    print(f"[taxonomy] updated node {node_id}", flush=True)
    return _node_to_dict(updated)


@app.delete("/api/taxonomy/nodes/{node_id}")
def taxonomy_delete_node(
    node_id: str,
    reparent_children_to: Optional[str] = None,
) -> dict[str, Any]:
    store = _require_taxonomy_store()
    node = store.get_node(node_id)
    if node is None:
        raise HTTPException(404, f"node {node_id!r} not found")
    if node.parent_id is None:
        raise HTTPException(400, "cannot delete the root node")

    cfg = _get_cfg()
    uid = node.user_id or cfg.user.default_user_id

    # Reparent children manually (TaxonomyStore.delete_node only reparents
    # assignments, not child nodes; SQLite FK CASCADE would wipe them).
    target_parent = reparent_children_to or node.parent_id
    target_node = store.get_node(target_parent) if target_parent else None
    if target_node is None:
        raise HTTPException(400, f"reparent target {target_parent!r} not found")
    if target_node.user_id != node.user_id:
        raise HTTPException(400, "cross-user reparent is not allowed")

    children = store.get_children(node_id)
    for child in children:
        try:
            store.move_node(child.node_id, target_parent)
        except ValueError as exc:
            raise HTTPException(400, f"reparent failed: {exc}") from exc

    # Try to reassign docs to the same target if it is a leaf; otherwise drop
    # the assignments (they will show up in /unfiled).
    try:
        if target_node.is_leaf:
            store.delete_node(node_id, reassign_docs_to=target_parent)
        else:
            store.delete_node(node_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    print(f"[taxonomy] deleted node {node_id} (user={uid})", flush=True)
    return {"node_id": node_id, "deleted": True}


@app.post("/api/taxonomy/move-doc")
def taxonomy_move_doc(body: TaxonomyMoveDoc) -> dict[str, Any]:
    store = _require_taxonomy_store()
    cfg = _get_cfg()
    uid = cfg.user.default_user_id
    target = store.get_node(body.node_id)
    if target is None:
        raise HTTPException(404, f"node {body.node_id!r} not found")
    if not target.is_leaf:
        raise HTTPException(400, "move-doc target must be a leaf")
    # Confirm the doc exists for this user.
    orch = _get_orch()
    row = orch.db.execute(
        "SELECT doc_id FROM documents WHERE doc_id = ? AND user_id = ?",
        (body.doc_id, uid),
    ).fetchone()
    if row is None:
        raise HTTPException(404, f"doc {body.doc_id!r} not found for user {uid!r}")
    # Wipe existing assignments and re-assign as primary.
    store.unassign_doc(uid, body.doc_id)
    try:
        store.assign_doc(uid, body.doc_id, body.node_id, score=1.0, is_primary=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    print(f"[taxonomy] moved doc {body.doc_id} -> {body.node_id}", flush=True)
    return {"doc_id": body.doc_id, "node_id": body.node_id, "moved": True}


@app.post("/api/taxonomy/clear")
def taxonomy_clear(user_id: Optional[str] = None) -> dict[str, Any]:
    """Drop the user's tree. Preserves the doc-meta cache (per Phase-3 contract)."""
    store = _require_taxonomy_store()
    uid = user_id or _get_cfg().user.default_user_id
    store.clear(uid)
    print(f"[taxonomy] cleared tree for user {uid}", flush=True)
    return {"user_id": uid, "cleared": True}


class TaxonomyOptionsPatch(BaseModel):
    """Six knobs from ``cfg.taxonomy`` that the editor surfaces inline."""

    beam_width: Optional[int] = None
    max_depth: Optional[int] = None
    propose_sample_size: Optional[int] = None
    max_children_per_node: Optional[int] = None
    min_top_score_floor: Optional[float] = None
    max_docs_pct: Optional[float] = None


@app.post("/api/taxonomy/options")
def taxonomy_options(patch: TaxonomyOptionsPatch) -> dict[str, Any]:
    """Live-mutable taxonomy tuning knobs surfaced by the GUI's options drawer.

    Each field is read at the next recompute / assign call — no orchestrator
    rebuild is needed. Unset fields are left untouched.
    """
    cfg = _get_cfg()
    if patch.beam_width is not None:
        cfg.taxonomy.beam_width = max(1, int(patch.beam_width))
    if patch.max_depth is not None:
        cfg.taxonomy.max_depth = max(1, int(patch.max_depth))
    if patch.propose_sample_size is not None:
        cfg.taxonomy.propose_sample_size = max(1, int(patch.propose_sample_size))
    if patch.max_children_per_node is not None:
        cfg.taxonomy.max_children_per_node = max(1, int(patch.max_children_per_node))
    if patch.min_top_score_floor is not None:
        cfg.taxonomy.min_top_score_floor = float(patch.min_top_score_floor)
    if patch.max_docs_pct is not None:
        cfg.taxonomy.max_docs_pct = float(patch.max_docs_pct)
    return {
        "taxonomy": {
            "beam_width": cfg.taxonomy.beam_width,
            "max_depth": cfg.taxonomy.max_depth,
            "propose_sample_size": cfg.taxonomy.propose_sample_size,
            "max_children_per_node": cfg.taxonomy.max_children_per_node,
            "min_top_score_floor": cfg.taxonomy.min_top_score_floor,
            "max_docs_pct": cfg.taxonomy.max_docs_pct,
        }
    }


# ---- SSE compute endpoints -------------------------------------------------


def _stream_builder_progress(
    runner: Callable[[Callable[[str, dict], None]], None],
):
    """Adapter: turn a worker that takes a ``progress(stage, payload)``
    callback into a StreamingResponse generator that emits SSE events.

    Per-stage progress events are streamed as ``event: stage``. Errors are
    emitted as ``event: error``; a terminal ``event: done`` always fires
    last so the client can close cleanly.
    """
    q: queue.Queue[tuple[str, Any]] = queue.Queue()
    _END = object()

    def _progress(stage: str, payload: dict[str, Any]) -> None:
        # Always include the stage name in the payload for clients that look
        # at the data field only (some SSE libs do).
        body = dict(payload or {})
        body.setdefault("stage", stage)
        q.put((stage, body))

    def _worker() -> None:
        try:
            runner(_progress)
        except Exception as exc:  # noqa: BLE001
            q.put(("__error__", {
                "type": type(exc).__name__,
                "message": str(exc),
            }))
        finally:
            q.put((_END, None))

    threading.Thread(target=_worker, daemon=True).start()

    async def _gen():
        loop = asyncio.get_event_loop()
        yield _sse_pack("open", {"ts": time.time()})
        while True:
            event, payload = await loop.run_in_executor(None, q.get)
            if event is _END:
                yield _sse_pack("done", {"ts": time.time()})
                break
            if event == "__error__":
                yield _sse_pack("error", payload)
                continue
            yield _sse_pack("stage", payload)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/taxonomy/recompute")
def taxonomy_recompute(body: TaxonomyRecomputeBody | None = None) -> StreamingResponse:
    _require_taxonomy_store()
    cfg = _get_cfg()
    uid = (body.user_id if body and body.user_id else cfg.user.default_user_id)
    builder = _build_taxonomy_builder()
    print(f"[taxonomy] recompute start (user={uid})", flush=True)

    def _run(progress_cb: Callable[[str, dict], None]) -> None:
        # The parallel agent's contract is build(user_id, *, progress=...).
        # The existing impl is build_for_user(user_id, *, progress=...,
        # force=...). Try both so we don't break either contract.
        kwargs = {"progress": progress_cb}
        method = getattr(builder, "build", None)
        if callable(method):
            method(uid, **kwargs)
            return
        method = getattr(builder, "build_for_user", None)
        if callable(method):
            method(uid, **kwargs)
            return
        raise RuntimeError(
            "TaxonomyBuilder has neither .build nor .build_for_user"
        )

    return _stream_builder_progress(_run)


@app.post("/api/taxonomy/assign-unfiled")
def taxonomy_assign_unfiled(body: TaxonomyRecomputeBody | None = None) -> StreamingResponse:
    _require_taxonomy_store()
    cfg = _get_cfg()
    uid = (body.user_id if body and body.user_id else cfg.user.default_user_id)
    assigner = _build_doc_assigner()
    print(f"[taxonomy] assign-unfiled start (user={uid})", flush=True)

    def _run(progress_cb: Callable[[str, dict], None]) -> None:
        # The parallel agent renames assign_all -> assign_unfiled. Support
        # both names; both must accept ``progress=``.
        for name in ("assign_unfiled", "assign_all"):
            method = getattr(assigner, name, None)
            if callable(method):
                method(uid, progress=progress_cb)
                return
        raise RuntimeError(
            "DocAssigner has neither .assign_unfiled nor .assign_all"
        )

    return _stream_builder_progress(_run)
