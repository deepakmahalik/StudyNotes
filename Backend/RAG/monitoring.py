"""
Monitoring and evaluation for the RAG pipeline.

Metrics tracked
---------------
- Query count (total, by mode)
- Latency (per query)
- Cache hit rate
- Precision@k, Recall@k, MRR (when ground-truth labels are supplied)

All metrics are stored in-memory and exposed via `get_stats()`.
A rotating log file is also written for persistence across restarts.
"""

import json
import logging
import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs", "rag_metrics.jsonl")
)
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)

_lock = Lock()
_stats: dict[str, Any] = {
    "total_queries": 0,
    "rag_queries": 0,
    "web_queries": 0,
    "cache_hits": 0,
    "total_latency_ms": 0.0,
    "precision_sum": 0.0,
    "recall_sum": 0.0,
    "mrr_sum": 0.0,
    "eval_count": 0,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _append_log(record: dict) -> None:
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.debug("Could not write metrics log: %s", exc)


# ── public API ────────────────────────────────────────────────────────────────

def record_query(
    query: str,
    mode: str,           # "rag" | "web" | "cache"
    latency_ms: float,
    user_id: int = 0,
) -> None:
    with _lock:
        _stats["total_queries"] += 1
        _stats["total_latency_ms"] += latency_ms
        if mode == "rag":
            _stats["rag_queries"] += 1
        elif mode == "web":
            _stats["web_queries"] += 1
        elif mode == "cache":
            _stats["cache_hits"] += 1

    _append_log(
        {
            "event": "query",
            "user_id": user_id,
            "mode": mode,
            "latency_ms": round(latency_ms, 2),
            "query_len": len(query),
            "ts": time.time(),
        }
    )


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> dict[str, float]:
    """
    Compute Precision@k, Recall@k, and MRR for a single query.
    *retrieved_ids* — ordered list of retrieved chunk/doc IDs.
    *relevant_ids*  — ground-truth relevant IDs (any order).

    Returns a dict with keys: precision_at_k, recall_at_k, mrr.
    Updates running totals for aggregate stats.
    """
    if not relevant_ids:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

    at_k = k or len(retrieved_ids)
    top = retrieved_ids[:at_k]
    relevant_set = set(relevant_ids)

    hits = [1 if rid in relevant_set else 0 for rid in top]
    precision = sum(hits) / at_k if at_k else 0.0
    recall = sum(hits) / len(relevant_set) if relevant_set else 0.0

    mrr = 0.0
    for rank, rid in enumerate(top, start=1):
        if rid in relevant_set:
            mrr = 1.0 / rank
            break

    with _lock:
        _stats["precision_sum"] += precision
        _stats["recall_sum"] += recall
        _stats["mrr_sum"] += mrr
        _stats["eval_count"] += 1

    _append_log(
        {
            "event": "eval",
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "mrr": round(mrr, 4),
            "ts": time.time(),
        }
    )

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
    }


def get_stats() -> dict[str, Any]:
    """Return a snapshot of aggregate statistics."""
    with _lock:
        s = dict(_stats)

    n = s["total_queries"] or 1
    ev = s["eval_count"] or 1
    avg_lat = s["total_latency_ms"] / n
    cache_rate = s["cache_hits"] / n

    return {
        "total_queries": s["total_queries"],
        "rag_queries": s["rag_queries"],
        "web_queries": s["web_queries"],
        "cache_hits": s["cache_hits"],
        "cache_hit_rate": round(cache_rate, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "avg_precision_at_k": round(s["precision_sum"] / ev, 4) if s["eval_count"] else None,
        "avg_recall_at_k": round(s["recall_sum"] / ev, 4) if s["eval_count"] else None,
        "avg_mrr": round(s["mrr_sum"] / ev, 4) if s["eval_count"] else None,
        "eval_count": s["eval_count"],
    }


def reset_stats() -> None:
    with _lock:
        for key in _stats:
            _stats[key] = 0 if isinstance(_stats[key], int) else 0.0
