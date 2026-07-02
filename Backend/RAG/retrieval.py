"""
Retrieval layer: hybrid search (dense + BM25) fused with RRF,
followed by optional cross-encoder re-ranking.

Flow
----
1. Embed query → dense search in Qdrant
2. BM25 search over all user chunks fetched from Qdrant
3. Reciprocal Rank Fusion (RRF) to merge ranked lists
4. Cross-encoder re-rank top candidates
5. Return top-N results
"""

import logging
import math
from typing import Any

from . import config
from .embeddings import embed_query
from . import vector_store

logger = logging.getLogger(__name__)

# ── BM25 ──────────────────────────────────────────────────────────────────────

def _bm25_search(
    query: str,
    corpus: list[dict],
    top_k: int,
) -> list[tuple[dict, float]]:
    """Run BM25 over *corpus* and return [(chunk, score)] sorted descending."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not installed; skipping BM25. Run: uv add rank-bm25")
        return []

    tokenize = lambda t: t.lower().split()
    corpus_tokens = [tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ── RRF ───────────────────────────────────────────────────────────────────────

def _rrf(
    dense_results: list[dict],
    bm25_results: list[tuple[dict, float]],
    k: int = 60,
    alpha: float | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion.
    alpha: weight for dense ranks (1-alpha for BM25).
    """
    w_dense = alpha if alpha is not None else config.HYBRID_ALPHA
    w_bm25 = 1.0 - w_dense

    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + w_dense * (1.0 / (k + rank))
        chunk_map[cid] = item

    for rank, (item, _) in enumerate(bm25_results, start=1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + w_bm25 * (1.0 / (k + rank))
        chunk_map[cid] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in fused:
        chunk = dict(chunk_map[cid])
        chunk["rrf_score"] = score
        results.append(chunk)
    return results


# ── Cross-encoder re-rank ─────────────────────────────────────────────────────

_reranker: Any = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading re-ranker: %s", config.RERANK_MODEL)
            _reranker = CrossEncoder(config.RERANK_MODEL)
        except ImportError:
            logger.warning(
                "sentence-transformers not available; skipping re-ranking. "
                "Run: uv add sentence-transformers"
            )
    return _reranker


def _rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    reranker = _get_reranker()
    if not reranker or not candidates:
        return candidates[:top_n]

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    results = []
    for chunk, score in ranked[:top_n]:
        c = dict(chunk)
        c["rerank_score"] = float(score)
        results.append(c)
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    user_id: int,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline for *query* scoped to *user_id*.

    Returns a list of ranked chunk dicts with fields:
        chunk_id, doc_id, text, score, rrf_score, rerank_score, metadata
    """
    k = top_k or config.TOP_K
    top_n = rerank_top_n or config.RERANK_TOP_N

    # 1. Dense search
    query_vec = embed_query(query)
    dense_hits = vector_store.dense_search(query_vec, user_id=user_id, top_k=k * 2)

    # 2. BM25 — build corpus from stored chunks
    corpus = vector_store.get_all_user_chunks(user_id)
    bm25_hits = _bm25_search(query, corpus, top_k=k * 2)

    if not dense_hits and not bm25_hits:
        logger.info("No results from Qdrant for user_id=%s", user_id)
        return []

    # 3. RRF fusion
    fused = _rrf(dense_hits, bm25_hits)[:k * 2]

    # 4. Cross-encoder re-rank
    reranked = _rerank(query, fused, top_n=top_n)

    logger.info(
        "Retrieval complete: dense=%d, bm25=%d, fused=%d, reranked=%d",
        len(dense_hits), len(bm25_hits), len(fused), len(reranked),
    )
    return reranked
