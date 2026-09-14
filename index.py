"""
index.py — Qdrant Cloud collection management and idempotent upsert pipeline.

Collection: dsa_lectures_1024
Vectors:    dense (1024-dim Cosine) — Phase 1 only (sparse BM25 added in Phase 2)
IDs:        uuid5(NAMESPACE_DNS, f"{video_id}:{int(start_sec)}") — deterministic

Calling upsert_chunks() multiple times with the same chunks is safe:
Qdrant upsert() overwrites existing points at the same ID.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

import config
from models import Chunk


# ── Client factory ─────────────────────────────────────────────────────────

def get_client() -> QdrantClient:
    """Create and return an authenticated QdrantClient."""
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60,
    )


# ── Collection lifecycle ───────────────────────────────────────────────────

def collection_exists(client: QdrantClient) -> bool:
    """Check whether the target collection is present in Qdrant Cloud."""
    return client.collection_exists(config.COLLECTION_NAME)


def create_collection(client: QdrantClient) -> None:
    """
    Create 'dsa_lectures_1024' with a single dense vector field (Cosine).
    Phase 2 will add a sparse 'sparse' vector field for BM25.
    Raises if collection already exists — check with collection_exists() first.
    """
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=qm.VectorParams(
            size=config.EMBED_DIM,
            distance=qm.Distance.COSINE,
            on_disk=False,
        ),
    )
    print(f"[index] Created collection '{config.COLLECTION_NAME}' (dim={config.EMBED_DIM}, Cosine)")


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection if it doesn't already exist."""
    if not collection_exists(client):
        create_collection(client)
    else:
        print(f"[index] Collection '{config.COLLECTION_NAME}' already exists — skipping creation")


def delete_collection(client: QdrantClient) -> None:
    """
    ⚠️  Permanently delete the collection and ALL its data.
    Use only when force-reindexing.
    """
    client.delete_collection(config.COLLECTION_NAME)
    print(f"[index] Deleted collection '{config.COLLECTION_NAME}'")


# ── Incremental sync helpers ───────────────────────────────────────────────

def get_indexed_video_ids(client: QdrantClient) -> set[str]:
    """
    Scroll through ALL points and collect unique video_id payload values.
    Used to skip already-indexed videos during incremental ingestion.
    """
    indexed: set[str] = set()
    offset: Any = None
    while True:
        records, offset = client.scroll(
            collection_name=config.COLLECTION_NAME,
            scroll_filter=None,
            limit=1000,
            offset=offset,
            with_payload=["video_id"],
            with_vectors=False,
        )
        for rec in records:
            if rec.payload and "video_id" in rec.payload:
                indexed.add(rec.payload["video_id"])
        if offset is None:
            break
    return indexed


def get_collection_stats(client: QdrantClient) -> dict:
    """Return point count and other collection-level info."""
    info = client.get_collection(config.COLLECTION_NAME)
    return {
        "total_chunks": info.points_count,
        "status": info.status,
        "optimizer_status": str(info.optimizer_status),
    }


# ── Upsert ─────────────────────────────────────────────────────────────────

def upsert_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    dense_vectors: np.ndarray,
    batch_size: int = 100,
) -> int:
    """
    Idempotent upsert of chunks + their dense vectors into Qdrant Cloud.

    Args:
        client:         Authenticated QdrantClient.
        chunks:         List of Chunk objects (length N).
        dense_vectors:  float32 ndarray of shape (N, 1024).
        batch_size:     Number of points per upsert call (100 is safe).

    Returns:
        Total number of points successfully upserted.

    The uuid5-based point IDs make repeated upserts idempotent:
    Qdrant will overwrite an existing point at the same ID rather than
    creating a duplicate.
    """
    assert len(chunks) == len(dense_vectors), (
        f"Chunk count ({len(chunks)}) ≠ vector count ({len(dense_vectors)})"
    )

    total_upserted = 0
    batches = [
        (chunks[i : i + batch_size], dense_vectors[i : i + batch_size])
        for i in range(0, len(chunks), batch_size)
    ]

    for chunk_batch, vec_batch in tqdm(batches, desc="Upserting to Qdrant", unit="batch"):
        points = [
            qm.PointStruct(
                id=chunk.id,
                vector=vec.tolist(),
                payload=chunk.to_qdrant_payload(),
            )
            for chunk, vec in zip(chunk_batch, vec_batch)
        ]

        # Retry up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                client.upsert(
                    collection_name=config.COLLECTION_NAME,
                    points=points,
                    wait=True,
                )
                total_upserted += len(points)
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Qdrant upsert failed after 3 attempts: {exc}") from exc
                wait_s = 2 ** attempt
                print(f"[index] Upsert attempt {attempt+1} failed ({exc}), retrying in {wait_s}s …")
                time.sleep(wait_s)

    return total_upserted


# ── Dense retrieval ────────────────────────────────────────────────────────

def search_dense(
    client: QdrantClient,
    query_vector: np.ndarray,
    top_k: int = config.TOP_K_DENSE,
) -> list[qm.ScoredPoint]:
    """
    Cosine similarity search in the dense vector space.

    Uses client.query_points() (qdrant-client >= 1.7 API).
    Returns ScoredPoints where .score is cosine similarity (0–1).
    Distance = 1 - score; lower distance = more relevant.
    """
    response = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return response.points
