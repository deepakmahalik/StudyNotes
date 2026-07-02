"""
Qdrant vector store layer.

Responsibilities
----------------
- Create/ensure the collection exists with the right vector config
- Upsert embedded chunks as Qdrant points
- Dense vector search (semantic)
- Full-text keyword search (using Qdrant payload filter)
- Delete all points belonging to a user (for re-ingestion)
"""

import logging
import uuid
from typing import Any

from . import config

logger = logging.getLogger(__name__)

_client: Any = None


def _get_client():
    global _client
    if _client is None:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise RuntimeError(
                "qdrant-client is not installed. Run: uv add qdrant-client"
            )
        url = config.QDRANT_URL.strip().lower()

        if url in (":memory:", "memory"):
            # Volatile in-memory mode
            _client = QdrantClient(":memory:")
            logger.info("Qdrant running in IN-MEMORY mode (data lost on restart)")

        elif url in ("local", ""):
            # Persistent local disk store — no Docker required
            import os
            local_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "qdrant_storage")
            )
            os.makedirs(local_path, exist_ok=True)
            _client = QdrantClient(path=local_path)
            logger.info("Qdrant running in LOCAL DISK mode: %s", local_path)

        else:
            # Remote Qdrant server (Docker / cloud)
            kwargs: dict = {"url": config.QDRANT_URL}
            if config.QDRANT_API_KEY:
                kwargs["api_key"] = config.QDRANT_API_KEY
            _client = QdrantClient(**kwargs)
            logger.info("Qdrant client connected: %s", config.QDRANT_URL)
    return _client


def ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if config.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection: %s", config.QDRANT_COLLECTION)
    else:
        logger.debug("Qdrant collection already exists: %s", config.QDRANT_COLLECTION)


def upsert_chunks(chunks: list[dict]) -> int:
    """
    Upsert embedded chunks into Qdrant.
    Each chunk must have 'embedding', 'chunk_id', 'text', and 'metadata'.
    Returns the number of points upserted.
    """
    from qdrant_client.models import PointStruct

    if not chunks:
        return 0

    ensure_collection()
    client = _get_client()

    points: list[PointStruct] = []
    for chunk in chunks:
        # Derive a deterministic UUID from chunk_id so re-ingestion is idempotent
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
        payload = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "chunk_index": chunk.get("chunk_index", 0),
            **chunk.get("metadata", {}),
        }
        points.append(
            PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload=payload,
            )
        )

    # Upsert in batches of 256
    batch_size = 256
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=batch)
        total += len(batch)

    logger.info("Upserted %d points into Qdrant collection '%s'", total, config.QDRANT_COLLECTION)
    return total


def dense_search(
    query_vector: list[float],
    user_id: int,
    top_k: int | None = None,
) -> list[dict]:
    """
    Semantic (cosine) nearest-neighbour search filtered by user_id.
    Returns a list of result dicts: {chunk_id, text, score, metadata}.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    ensure_collection()
    k = top_k or config.TOP_K
    client = _get_client()

    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=k,
        with_payload=True,
    )

    return [
        {
            "chunk_id": r.payload.get("chunk_id", ""),
            "doc_id": r.payload.get("doc_id", ""),
            "text": r.payload.get("text", ""),
            "score": r.score,
            "metadata": {key: val for key, val in r.payload.items() if key not in ("text", "chunk_id", "doc_id")},
        }
        for r in response.points
    ]


def keyword_search(
    keyword: str,
    user_id: int,
    top_k: int | None = None,
) -> list[dict]:
    """
    Scroll-based keyword search: retrieves all user docs and returns those
    whose text contains the keyword (case-insensitive).
    Used as the BM25 candidate pool; scoring happens in retrieval.py.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    k = (top_k or config.TOP_K) * 4  # fetch more candidates for BM25 re-scoring
    client = _get_client()

    results, _ = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=min(k, 1000),
        with_payload=True,
        with_vectors=False,
    )

    kw_lower = keyword.lower()
    matched = [
        {
            "chunk_id": r.payload.get("chunk_id", ""),
            "doc_id": r.payload.get("doc_id", ""),
            "text": r.payload.get("text", ""),
            "score": 0.0,
            "metadata": {
                key: val
                for key, val in r.payload.items()
                if key not in ("text", "chunk_id", "doc_id")
            },
        }
        for r in results
        if kw_lower in r.payload.get("text", "").lower()
    ]
    return matched


def delete_user_data(user_id: int) -> None:
    """Remove all Qdrant points belonging to *user_id*. Safe if collection doesn't exist yet."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    ensure_collection()
    client = _get_client()
    client.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
    )
    logger.info("Deleted Qdrant points for user_id=%s", user_id)


def get_existing_hashes(user_id: int) -> dict[str, str]:
    """
    Return { doc_id: content_hash } for all chunks already stored for this user.
    Used by incremental ingest to skip unchanged nodes.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    ensure_collection()
    client = _get_client()
    results, _ = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    seen: dict[str, str] = {}
    for r in results:
        doc_id = r.payload.get("doc_id", "")
        content_hash = r.payload.get("content_hash", "")
        if doc_id and doc_id not in seen:
            seen[doc_id] = content_hash
    return seen


def delete_doc_chunks(doc_id: str) -> None:
    """Delete all Qdrant chunks belonging to a single doc_id (one node)."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = _get_client()
    client.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )
    logger.debug("Deleted chunks for doc_id=%s", doc_id)


def get_all_user_chunks(user_id: int) -> list[dict]:
    """Fetch all stored chunks for a user (used by BM25 index build)."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    ensure_collection()
    client = _get_client()
    results, _ = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {
            "chunk_id": r.payload.get("chunk_id", ""),
            "doc_id": r.payload.get("doc_id", ""),
            "text": r.payload.get("text", ""),
            "score": 0.0,
            "metadata": {
                k: v
                for k, v in r.payload.items()
                if k not in ("text", "chunk_id", "doc_id")
            },
        }
        for r in results
    ]
