"""
chunk.py — Time-domain chunking engine for Whisper transcript JSONs.

Core algorithm:
  - Greedy time-window: slides a CHUNK_DURATION-second window forward
    by CHUNK_STEP (= CHUNK_DURATION - CHUNK_OVERLAP) each iteration.
  - Collects all Whisper segments that overlap the window.
  - Applies Whisper repetition filters before joining.
  - Prepends lecture title for semantic boosting in the embedding.
  - Emits deterministic uuid5-identified Chunk objects.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

import config
from models import Chunk, Segment, TranscriptDoc


# ── Whisper hallucination filters ──────────────────────────────────────────

_CONSECUTIVE_RE = re.compile(r"\b(\w+)(?:\s+\1){3,}\b", re.IGNORECASE)
_FILLER = frozenset({
    "haan", "hmm", "uh", "um", "okay", "ok", "theek", "theek hai",
    "acha", "accha", "right", "yes", "no", "nahi",
})


def _is_repetition_loop(text: str) -> bool:
    """Return True if any word repeats 4+ consecutive times (Whisper loop artifact)."""
    return bool(_CONSECUTIVE_RE.search(text))


def _unique_ratio(text: str) -> float:
    """
    Fraction of unique words vs total words.
    A very low ratio (< UNIQUE_RATIO_THRESHOLD) signals a Whisper repetition loop.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _is_filler_only(text: str) -> bool:
    """Return True if every word in the segment is a known filler word."""
    words = {w.strip(".,!?") for w in text.lower().split()}
    return bool(words) and words.issubset(_FILLER)


def _keep_segment(seg: Segment) -> bool:
    """
    Return True if segment passes all Whisper quality filters.
    Filters out: too-short, repetition loops, low unique-ratio, filler-only.
    """
    if seg.word_count < config.MIN_CHUNK_WORDS // 4:  # < ~5 words
        return False
    if _is_repetition_loop(seg.text):
        return False
    if _unique_ratio(seg.text) < config.UNIQUE_RATIO_THRESHOLD:
        return False
    if _is_filler_only(seg.text):
        return False
    return True


# ── Core chunker ───────────────────────────────────────────────────────────

def chunk_transcript(doc: TranscriptDoc) -> list[Chunk]:
    """
    Slice a TranscriptDoc into time-windowed Chunks.

    Window advances by CHUNK_STEP (60 s) each iteration.
    A segment is included if it *starts* within the window
    [t_start, t_start + CHUNK_DURATION).
    """
    chunks: list[Chunk] = []
    t_start = 0

    while t_start < doc.duration:
        t_end = t_start + config.CHUNK_DURATION

        # Collect segments whose start falls in [t_start, t_end)
        window_segs = [
            s for s in doc.segments
            if t_start <= s.start < t_end
        ]

        # Apply quality filters
        kept = [s for s in window_segs if _keep_segment(s)]

        if not kept:
            t_start += config.CHUNK_STEP
            continue

        body = " ".join(s.text.strip() for s in kept)

        # Skip windows that are nearly empty after filtering
        if len(body.split()) < config.MIN_CHUNK_WORDS:
            t_start += config.CHUNK_STEP
            continue

        # Title prefix for semantic boosting (used only in embedding)
        text_for_embed = f"{doc.title}\n\n{body}"

        seek_sec = max(0.0, t_start - config.PLAYBACK_REWIND)
        youtube_url = f"https://youtu.be/{doc.video_id}?t={int(seek_sec)}"

        chunk = Chunk(
            id=Chunk.make_id(doc.video_id, t_start),
            video_id=doc.video_id,
            title=doc.title,
            start_sec=float(t_start),
            end_sec=float(t_end),
            seek_sec=seek_sec,
            text=body,
            text_for_embed=text_for_embed,
            word_count=len(body.split()),
            youtube_url=youtube_url,
            segments=kept,
        )
        chunks.append(chunk)
        t_start += config.CHUNK_STEP

    return chunks


# ── File I/O ───────────────────────────────────────────────────────────────

def load_transcript(path: Path) -> TranscriptDoc | None:
    """
    Parse a single transcript JSON file into a TranscriptDoc.
    Returns None and logs a warning on schema errors (so batch jobs continue).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"])
            for s in raw.get("segments", [])
        ]
        return TranscriptDoc(
            video_id=raw["video_id"],
            title=raw.get("title", path.stem),
            language=raw.get("language", "hi"),
            duration=float(raw.get("duration", 0)),
            segments=segments,
        )
    except Exception as exc:
        print(f"[WARN] Skipping {path.name}: {exc}")
        return None


def load_all_transcripts(
    dir_path: Path = config.TRANSCRIPTS_DIR,
) -> list[TranscriptDoc]:
    """Load all *.json files from dir_path, skipping malformed ones."""
    docs: list[TranscriptDoc] = []
    paths = sorted(dir_path.glob("*.json"))
    for p in tqdm(paths, desc="Loading transcripts", unit="file"):
        doc = load_transcript(p)
        if doc is not None:
            docs.append(doc)
    return docs


def chunk_all(docs: list[TranscriptDoc]) -> list[Chunk]:
    """Chunk every TranscriptDoc, return a flat list ordered by (video_id, start_sec)."""
    all_chunks: list[Chunk] = []
    for doc in tqdm(docs, desc="Chunking", unit="video"):
        all_chunks.extend(chunk_transcript(doc))
    return all_chunks
