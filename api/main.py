"""
api/main.py — FastAPI application for DSA TubeRAG.

Endpoints (Phase 1):
  POST /api/ask    — full RAG pipeline, returns RAGResponse JSON
  GET  /api/stats  — collection statistics
  GET  /           — health-check / welcome message

Heavy models (Qdrant client, Gemini client, bge-m3) are all initialized
once at application startup and reused across requests via module-level
singletons. No per-request model loading.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running from the project root as `python -m api.main`
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config

config.validate()   # Fail fast if secrets are missing

import embed        # noqa: E402 — imports after path fix
import index        # noqa: E402
from answer import answer as rag_answer  # noqa: E402
from models import AskRequest, RAGResponse, StatsResponse  # noqa: E402
from preprocess import preprocess_query  # noqa: E402


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up dense and sparse embedding models and verify Qdrant connectivity.
    This avoids cold-start delays on the first request.
    """
    print("[startup] Warming up hybrid embedding models (dense + sparse BM25) …")
    embed.hybrid_embedder.embed_query("warmup")
    print("[startup] Verifying Qdrant connectivity …")
    client = index.get_client()
    if not index.collection_exists(client):
        print(
            f"[startup] WARNING: Collection '{config.COLLECTION_NAME}' not found. "
            "Run `tuberag reindex --force` before making queries."
        )
    else:
        stats = index.get_collection_stats(client)
        is_hyb = index.is_hybrid_collection(client)
        mode_str = "Hybrid (Dense + BM25)" if is_hyb else "Dense-only"
        print(f"[startup] Qdrant OK — {stats['total_chunks']:,} chunks in collection [{mode_str}]")
    yield
    print("[shutdown] Bye!")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DSA TubeRAG API",
    description=(
        "Timestamp-accurate RAG system for the 'Padho with Pratyush' DSA playlist. "
        "Queries are answered in Hinglish using only the lecture transcripts."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["Server-Timing"],
)


# ── Routes ─────────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", tags=["UI & Health"])
async def root():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {
        "service": "DSA TubeRAG",
        "version": "1.0.0",
        "status": "ok",
        "docs": "/docs",
    }


@app.post("/api/ask", response_model=RAGResponse, tags=["RAG"])
async def ask(request: AskRequest, response: Response) -> RAGResponse:
    """
    Full Phase 2 RAG pipeline with high-resolution latency instrumentation.

    1. Preprocess query (acronym expansion + LeetCode normalization).
    2. Encode with HybridEmbedder (dense bge-m3 + FastEmbed BM25) with LRU caching.
    3. Hybrid retrieval in Qdrant with Reciprocal Rank Fusion (RRF).
    4. Distance cutoff guard (MAX_DISTANCE = 0.5):
       - If best distance > 0.5 → refusal with 0 tokens and 0ms generation.
    5. Gemini generation with strict system prompt.
    6. Return structured RAGResponse with answer, badge, citations, and latency breakdown.
    7. Server-Timing HTTP header for downstream performance tracing.
    """
    t0 = time.perf_counter()
    timing: dict[str, float] = {}

    try:
        client = index.get_client()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}")

    # 1. Preprocess query
    query_processed = preprocess_query(request.query)

    is_hyb = index.is_hybrid_collection(client)

    # 2. Embedding generation (timed, benefits from LRU cache)
    t_embed_start = time.perf_counter()
    if is_hyb:
        query_dense, query_sparse = embed.hybrid_embedder.embed_query(query_processed)
    else:
        query_dense = embed.encode_query(query_processed)
        query_sparse = None
    timing["embed_ms"] = round((time.perf_counter() - t_embed_start) * 1000, 2)

    # 3. Vector search (timed)
    t_ret_start = time.perf_counter()
    try:
        if index.is_hybrid_collection(client) and query_sparse is not None:
            points, best_distance = index.search_hybrid(
                client=client,
                query_dense=query_dense,
                query_sparse=query_sparse,
                top_k=request.top_k,
            )
        else:
            points = index.search_dense(client, query_dense, top_k=request.top_k)
            best_distance = 1.0 - float(points[0].score) if points else 1.0
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search failed: {exc}")
    timing["retrieval_ms"] = round((time.perf_counter() - t_ret_start) * 1000, 2)

    # 4. Generate answer (includes all guardrails, records generation_ms and total_ms)
    try:
        resp = rag_answer(
            query=request.query,
            points=points,
            retrieval_distance=best_distance,
            query_processed=query_processed,
            t0=t0,
            timing=timing,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    # 5. Attach standard HTTP Server-Timing header
    # Format: Server-Timing: embed;dur=X, retrieval;dur=Y, generation;dur=Z, total;dur=W
    e_dur = resp.timing.get("embed_ms", 0.0)
    r_dur = resp.timing.get("retrieval_ms", 0.0)
    g_dur = resp.timing.get("generation_ms", 0.0)
    tot_dur = resp.timing.get("total_ms", 0.0)
    response.headers["Server-Timing"] = (
        f"embed;dur={e_dur}, retrieval;dur={r_dur}, generation;dur={g_dur}, total;dur={tot_dur}"
    )

    return resp


@app.get("/api/stats", response_model=StatsResponse, tags=["Monitoring"])
async def stats() -> StatsResponse:
    """Return collection statistics from Qdrant Cloud."""
    client = index.get_client()

    if not index.collection_exists(client):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{config.COLLECTION_NAME}' not found. Run reindex first.",
        )

    raw_stats = index.get_collection_stats(client)
    indexed_ids = index.get_indexed_video_ids(client)
    total_chunks = raw_stats["total_chunks"] or 0
    total_videos = len(indexed_ids)

    return StatsResponse(
        collection_name=config.COLLECTION_NAME,
        total_chunks=total_chunks,
        total_videos=total_videos,
        embed_model=config.EMBED_MODEL,
        embed_dim=config.EMBED_DIM,
        avg_chunks_per_video=round(total_chunks / max(total_videos, 1), 1),
    )


# ── Dev server entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
