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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config

config.validate()   # Fail fast if secrets are missing

import embed        # noqa: E402 — imports after path fix
import index        # noqa: E402
from answer import answer as rag_answer  # noqa: E402
from models import AskRequest, RAGResponse, StatsResponse  # noqa: E402


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up the embedding model and verify Qdrant connectivity.
    This avoids a cold-start delay on the very first request.
    """
    print("[startup] Warming up embedding model …")
    embed.encode_query("warmup")          # loads bge-m3 into memory
    print("[startup] Verifying Qdrant connectivity …")
    client = index.get_client()
    if not index.collection_exists(client):
        print(
            f"[startup] WARNING: Collection '{config.COLLECTION_NAME}' not found. "
            "Run `tuberag reindex` before making queries."
        )
    else:
        stats = index.get_collection_stats(client)
        print(f"[startup] Qdrant OK — {stats['total_chunks']:,} chunks in collection")
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
async def ask(request: AskRequest) -> RAGResponse:
    """
    Full RAG pipeline.

    1. Encode query with bge-m3.
    2. Dense cosine search in Qdrant (top_k).
    3. Distance cutoff guard (MAX_DISTANCE = 0.5):
       - If no good match → refusal with 0 tokens.
    4. Gemini 2.5 Flash generation with strict system prompt.
    5. Return structured RAGResponse with answer, badge, and citations.
    """
    t0 = time.perf_counter()

    try:
        client = index.get_client()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}")

    # Vector search
    query_vec = embed.encode_query(request.query)
    try:
        points = index.search_dense(client, query_vec, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search failed: {exc}")

    # Generate answer (includes all guardrails)
    try:
        resp = rag_answer(
            query=request.query,
            points=points,
            query_processed=request.query,
            t0=t0,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

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
