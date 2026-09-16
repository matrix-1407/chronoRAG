"""
embed.py — Hybrid embedding wrappers for dense (BAAI/bge-m3) and sparse (FastEmbed BM25).

Singleton model loading: both the SentenceTransformer and SparseTextEmbedding are loaded
once lazily and thread-safely, reused across queries and batch indexing.
"""
from __future__ import annotations

import collections
import re
import threading
from typing import Sequence

from fastembed import SparseTextEmbedding
import numpy as np
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

_dense_model: SentenceTransformer | None = None
_sparse_model: SparseTextEmbedding | None = None
_lock = threading.Lock()


def _get_dense_model() -> SentenceTransformer:
    """Lazy-load bge-m3 once; thread-safe."""
    global _dense_model
    if _dense_model is None:
        with _lock:
            if _dense_model is None:
                print(f"[embed] Loading dense model {config.EMBED_MODEL} …")
                _dense_model = SentenceTransformer(
                    config.EMBED_MODEL,
                    trust_remote_code=True,
                )
                print(f"[embed] Dense model loaded (dim={config.EMBED_DIM})")
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    """Lazy-load FastEmbed BM25 sparse embedder once; thread-safe."""
    global _sparse_model
    if _sparse_model is None:
        with _lock:
            if _sparse_model is None:
                print(f"[embed] Loading sparse model {config.SPARSE_MODEL} …")
                _sparse_model = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
                print(f"[embed] Sparse model loaded ({config.SPARSE_MODEL})")
    return _sparse_model


# ── Query Normalization & LRU Caching ──────────────────────────────────────

def normalize_query(query: str) -> str:
    """
    Normalize query for deterministic cache keying:
    - Strips leading/trailing whitespace
    - Lowercases
    - Collapses consecutive whitespace into a single space
    - Strips trailing punctuation (? ! . , ; :)
    """
    q = query.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[?!.,;:]+$", "", q).strip()
    return q


class LRUQueryCache:
    """
    Thread-safe in-memory LRU cache for query embeddings (dense ndarray + sparse SparseVector).
    Eliminates redundant neural forward passes on equivalent queries.
    """

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = maxsize
        self._cache: collections.OrderedDict[str, tuple[np.ndarray, qm.SparseVector]] = collections.OrderedDict()
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> tuple[np.ndarray, qm.SparseVector] | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                dense, sparse = self._cache[key]
                # Return deep copy of dense array and reconstructed SparseVector to prevent mutation
                return (
                    dense.copy(),
                    qm.SparseVector(indices=list(sparse.indices), values=list(sparse.values)),
                )
            self.misses += 1
            return None

    def put(self, key: str, value: tuple[np.ndarray, qm.SparseVector]) -> None:
        dense, sparse = value
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (
                dense.copy(),
                qm.SparseVector(indices=list(sparse.indices), values=list(sparse.values)),
            )
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def get_info(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "maxsize": self.maxsize,
                "currsize": len(self._cache),
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0


_query_cache = LRUQueryCache(maxsize=512)


def get_cache_info() -> dict[str, int]:
    """Expose query embedding LRU cache telemetry: hits, misses, maxsize, currsize."""
    return _query_cache.get_info()


def clear_cache() -> None:
    """Clear query embedding cache."""
    _query_cache.clear()


# ── Dense encoding functions ───────────────────────────────────────────────

def _encode_dense_raw(query: str) -> np.ndarray:
    model = _get_dense_model()
    vec = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single query string into a dense vector for retrieval.
    Returns float32 ndarray of shape (1024,), L2-normalized.
    """
    norm_key = normalize_query(query)
    cached = _query_cache.get(norm_key)
    if cached is not None:
        return cached[0]
    return _encode_dense_raw(query)


def encode_chunks(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of chunk texts into dense vectors for indexing.
    Returns float32 ndarray of shape (N, 1024), each row L2-normalized.
    """
    model = _get_dense_model()
    all_vecs: list[np.ndarray] = []

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    iterator = tqdm(batches, desc="Encoding dense chunks", unit="batch") if show_progress else batches

    for batch in iterator:
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_vecs.append(vecs.astype(np.float32))

    return np.vstack(all_vecs) if all_vecs else np.empty((0, config.EMBED_DIM), dtype=np.float32)


# ── Sparse encoding functions ──────────────────────────────────────────────

def _encode_sparse_raw(query: str) -> qm.SparseVector:
    model = _get_sparse_model()
    embeddings = list(model.embed([query]))
    if not embeddings:
        return qm.SparseVector(indices=[], values=[])
    emb = embeddings[0]
    return qm.SparseVector(
        indices=emb.indices.tolist(),
        values=emb.values.tolist(),
    )


def encode_sparse_query(query: str) -> qm.SparseVector:
    """
    Encode a single query string into a Qdrant SparseVector (BM25).
    """
    norm_key = normalize_query(query)
    cached = _query_cache.get(norm_key)
    if cached is not None:
        return cached[1]
    return _encode_sparse_raw(query)


def encode_sparse_chunks(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[qm.SparseVector]:
    """
    Encode a list of chunk texts into Qdrant SparseVectors using FastEmbed BM25.
    """
    model = _get_sparse_model()
    sparse_vectors: list[qm.SparseVector] = []

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    iterator = tqdm(batches, desc="Encoding sparse chunks", unit="batch") if show_progress else batches

    for batch in iterator:
        batch_embeddings = list(model.embed(batch))
        for emb in batch_embeddings:
            sparse_vectors.append(
                qm.SparseVector(
                    indices=emb.indices.tolist(),
                    values=emb.values.tolist(),
                )
            )

    return sparse_vectors


# ── Unified Hybrid Embedder ────────────────────────────────────────────────

class HybridEmbedder:
    """
    Unified embedder generating both dense (1024-dim Cosine) and sparse (BM25)
    vector representations for queries and chunks.
    Thread-safe query caching backed by LRUQueryCache.
    """

    def __init__(self) -> None:
        self.dense_dim = config.EMBED_DIM
        self.sparse_model_name = config.SPARSE_MODEL

    def embed_query(self, query: str) -> tuple[np.ndarray, qm.SparseVector]:
        """
        Embed a search query into dense and sparse representations.
        Checks thread-safe normalized LRU cache first to avoid redundant neural passes.
        Returns:
            (dense_vector, sparse_vector)
        """
        norm_key = normalize_query(query)
        cached = _query_cache.get(norm_key)
        if cached is not None:
            return cached

        dense_vec = _encode_dense_raw(query)
        sparse_vec = _encode_sparse_raw(query)
        _query_cache.put(norm_key, (dense_vec, sparse_vec))
        return dense_vec, sparse_vec

    def embed_chunks(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[qm.SparseVector]]:
        """
        Batch embed texts into dense ndarray and sparse vector list.
        Returns:
            (dense_vectors_array, list_of_sparse_vectors)
        """
        dense_vecs = encode_chunks(texts, batch_size=batch_size, show_progress=show_progress)
        sparse_vecs = encode_sparse_chunks(texts, batch_size=batch_size * 2, show_progress=show_progress)
        return dense_vecs, sparse_vecs


# Default singleton instance
hybrid_embedder = HybridEmbedder()

