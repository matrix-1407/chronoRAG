"""
models.py — Pydantic v2 data models (immutable data containers only).

No business logic lives here. All models use model_config frozen=True
to prevent accidental mutation after construction.
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field

import config


# ── Raw transcript data (from JSON files) ─────────────────────────────────

class Segment(BaseModel):
    """A single Whisper-transcribed speech segment."""
    model_config = ConfigDict(frozen=True)

    start: float       # seconds from video start
    end: float         # seconds from video start
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class TranscriptDoc(BaseModel):
    """Parsed representation of one transcript JSON file."""
    model_config = ConfigDict(frozen=True)

    video_id: str
    title: str
    language: str = "hi"
    duration: float                 # total video length in seconds
    segments: list[Segment]

    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.segments)


# ── Processed chunk ────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """A time-windowed chunk ready for embedding and Qdrant upsert."""
    model_config = ConfigDict(frozen=True)

    id: str                        # uuid5 deterministic ID
    video_id: str
    title: str
    start_sec: float
    end_sec: float
    seek_sec: float                # max(0, start_sec - PLAYBACK_REWIND)
    text: str                      # raw body text (no title prefix)
    text_for_embed: str            # "{title}\n\n{text}" (title prefix included)
    word_count: int
    youtube_url: str               # deep-link with timestamp
    segments: list[Segment]        # constituent segments (for UI sync)

    @classmethod
    def make_id(cls, video_id: str, start_sec: float) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}:{int(start_sec)}"))

    def to_qdrant_payload(self) -> dict:
        """Serialize to Qdrant point payload (excludes embed-only fields)."""
        return {
            "video_id": self.video_id,
            "title": self.title,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "seek_sec": self.seek_sec,
            "text": self.text,
            "word_count": self.word_count,
            "youtube_url": self.youtube_url,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in self.segments],
        }


# ── API request / response models ─────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    """A single retrieved chunk surfaced to the caller."""
    model_config = ConfigDict(frozen=True)

    video_id: str
    title: str
    start_sec: float
    seek_sec: float
    youtube_url: str
    chunk_preview: str             # first 120 chars of chunk text
    distance: float                # cosine distance (lower = more relevant)
    segments: list[dict]           # [{start, end, text}] for frontend sync


class ComplexityBadge(BaseModel):
    model_config = ConfigDict(frozen=True)

    time_complexity: str
    space_complexity: str
    pattern: str

    def display(self) -> str:
        return f"[Time: {self.time_complexity} | Space: {self.space_complexity} | Pattern: {self.pattern}]"


_BADGE_RE = re.compile(
    r"\[Time:\s*([^|]+?)\s*\|\s*Space:\s*([^|]+?)\s*\|\s*Pattern:\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)


def parse_badge(text: str) -> Optional[ComplexityBadge]:
    """Extract ComplexityBadge from LLM response text. Returns None if not found."""
    m = _BADGE_RE.search(text)
    if not m:
        return None
    return ComplexityBadge(
        time_complexity=m.group(1).strip(),
        space_complexity=m.group(2).strip(),
        pattern=m.group(3).strip(),
    )


class RAGResponse(BaseModel):
    answer: str
    complexity_badge: Optional[ComplexityBadge] = None
    citations: list[Citation] = []
    retrieval_distance: float = 1.0
    tokens_used: int = 0
    is_refused: bool = False
    query_processed: str = ""
    latency_ms: int = 0


class StatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    total_videos: int
    embed_model: str
    embed_dim: int
    avg_chunks_per_video: float
