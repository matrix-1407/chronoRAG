"""
config.py — Single source of truth for all application configuration.

Reads from .env via python-dotenv. All constants are typed and validated
at import time so errors surface immediately at startup.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────
_project_root = Path(__file__).parent
load_dotenv(_project_root / ".env", override=False)

# ── Secrets (required) ─────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
QDRANT_URL: str = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")

# ── Optional secrets ───────────────────────────────────────────────────────
OPENROUTER_API_KEY: str | None = os.environ.get("OPENROUTER_API_KEY") or None

# ── Collection & model names ───────────────────────────────────────────────
COLLECTION_NAME: str = os.environ.get("YTRAG_COLLECTION", "dsa_lectures_1024")
EMBED_MODEL: str = os.environ.get("YTRAG_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM: int = 1024  # bge-m3 fixed output dimension
SPARSE_MODEL: str = os.environ.get("YTRAG_SPARSE_MODEL", "Qdrant/bm25")
GEMINI_MODEL: str = os.environ.get("YTRAG_GEMINI_MODEL", "gemini-2.5-flash")

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_DURATION: int = int(os.environ.get("YTRAG_CHUNK_SECONDS", 75))
CHUNK_OVERLAP: int = int(os.environ.get("YTRAG_CHUNK_OVERLAP", 15))
CHUNK_STEP: int = CHUNK_DURATION - CHUNK_OVERLAP          # 60 s advance per window
PLAYBACK_REWIND: int = int(os.environ.get("YTRAG_LINK_REWIND", 5))
MIN_CHUNK_WORDS: int = 20   # skip near-silent windows
UNIQUE_RATIO_THRESHOLD: float = 0.35   # Whisper repetition filter

# ── Retrieval (Hybrid & RRF) ────────────────────────────────────────────────
MAX_DISTANCE: float = float(os.environ.get("YTRAG_MAX_DISTANCE", 0.46))
TOP_K: int = 5
TOP_K_DENSE: int = int(os.environ.get("YTRAG_TOP_K_DENSE", 25))
TOP_K_SPARSE: int = int(os.environ.get("YTRAG_TOP_K_SPARSE", 25))
RRF_K: int = int(os.environ.get("YTRAG_RRF_K", 60))

# ── Generation ─────────────────────────────────────────────────────────────
MAX_OUTPUT_TOKENS: int = int(os.environ.get("YTRAG_MAX_OUTPUT_TOKENS", 450))
TEMPERATURE: float = 0.1
TOP_P: float = 0.8
TOP_K_GEMINI: int = 20

# ── Paths ──────────────────────────────────────────────────────────────────
TRANSCRIPTS_DIR: Path = _project_root / "transcripts"

# ── Startup validation ─────────────────────────────────────────────────────
def validate() -> None:
    """Raise ValueError if any required secret is missing."""
    missing: list[str] = []
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example → .env and fill in your credentials."
        )
