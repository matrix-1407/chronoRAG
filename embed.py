"""
embed.py — Dense embedding wrapper for BAAI/bge-m3.

Singleton model loading: the SentenceTransformer is loaded once at module
import (lazy, on first call) and reused for the lifetime of the process.
All embeddings are L2-normalized (normalize_embeddings=True).
"""
from __future__ import annotations

import threading
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """Lazy-load bge-m3 once; thread-safe."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                print(f"[embed] Loading {config.EMBED_MODEL} …")
                _model = SentenceTransformer(
                    config.EMBED_MODEL,
                    trust_remote_code=True,
                )
                print(f"[embed] Model loaded (dim={config.EMBED_DIM})")
    return _model


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single query string for retrieval.

    Returns float32 ndarray of shape (1024,), L2-normalized.
    Uses the query instruction prefix expected by bge-m3 for retrieval tasks.
    """
    model = _get_model()
    vec = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)


def encode_chunks(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of chunk texts (title-prefixed) for indexing.

    Args:
        texts:          List of text_for_embed strings.
        batch_size:     Encoding batch size (32 is safe for 8 GB VRAM / CPU).
        show_progress:  Show tqdm progress bar.

    Returns:
        float32 ndarray of shape (N, 1024), each row L2-normalized.
    """
    model = _get_model()
    all_vecs: list[np.ndarray] = []

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    iterator = tqdm(batches, desc="Encoding chunks", unit="batch") if show_progress else batches

    for batch in iterator:
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_vecs.append(vecs.astype(np.float32))

    return np.vstack(all_vecs) if all_vecs else np.empty((0, config.EMBED_DIM), dtype=np.float32)
