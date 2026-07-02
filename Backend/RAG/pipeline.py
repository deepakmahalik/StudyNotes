"""
RAG Pipeline Orchestrator

Flow
----
User Query
    │
    ├─ Cache hit? ──────────────────────────────► Return cached response
    │
    ├─ Guardrail: validate query
    │
    ├─ Intent classification
    │       ├─ NOTES  → RAG path
    │       │       ├─ Embed query
    │       │       ├─ Hybrid search (dense + BM25 + RRF)
    │       │       ├─ Cross-encoder re-rank
    │       │       ├─ Build prompt with context
    │       │       └─ Generate with Gemini
    │       └─ EXTERNAL → Web search path
    │               ├─ DuckDuckGo search
    │               └─ Generate with Gemini
    │
    ├─ Guardrail: check output
    ├─ Record metrics
    └─ Cache result → Return response

Public API
----------
    ingest(user_id)          — ingest/re-ingest a user's mind map into Qdrant
    query(question, user_id) — run the full pipeline, return a response dict
"""

import logging
import time
from typing import Any

from . import cache as _cache_module
from . import monitoring
from . import guardrails
from . import web_search
from .ingestion import load_user_documents
from .chunking import chunk_documents
from .embeddings import embed_chunks
from .vector_store import upsert_chunks, delete_user_data, get_existing_hashes, delete_doc_chunks
from .retrieval import retrieve
from .generation import generate_from_context, generate_from_web

logger = logging.getLogger(__name__)

# ── keyword-based fast-path intent detection ──────────────────────────────────
# Used before calling the LLM classifier to save latency on obvious cases.

# Keywords that signal the user explicitly wants a web/internet search
_WEB_SEARCH_TRIGGERS = {
    "search the web", "search online", "search internet",
    "search web for", "google this", "google it",
    "look it up online", "browse the web",
    "find on the internet", "web search", "internet search",
    "find me online", "look up on internet",
}


def _user_wants_web(question: str) -> bool:
    """Return True only when the user explicitly asks for a web/internet search."""
    lower = question.lower()
    return any(kw in lower for kw in _WEB_SEARCH_TRIGGERS)


# ── public API ────────────────────────────────────────────────────────────────

def ingest(user_id: int) -> dict[str, Any]:
    """
    Load the user's mind-map from SQLite, chunk, embed, and store in Qdrant.
    Call this whenever the user saves their map.

    Returns: { "status": "ok"|"empty", "chunks_indexed": int }
    """
    logger.info("Starting ingestion for user_id=%s", user_id)

    docs = load_user_documents(user_id)
    if not docs:
        return {"status": "empty", "chunks_indexed": 0}

    chunks = chunk_documents(docs, strategy="semantic")
    embed_chunks(chunks)

    # Replace all existing data for this user
    delete_user_data(user_id)
    n = upsert_chunks(chunks)

    # Invalidate query cache for this user after re-ingestion
    _cache_module.invalidate_user_cache(user_id)

    logger.info("Ingestion complete for user_id=%s: %d chunks indexed", user_id, n)
    return {"status": "ok", "chunks_indexed": n}


def incremental_ingest(user_id: int) -> dict:
    """
    Only ingest nodes that are new or have changed content since the last ingest.
    Deletes chunks for nodes that were removed from the map.

    Returns: { "status": "ok"|"empty", "added": int, "updated": int, "deleted": int, "unchanged": int }
    """
    logger.info("Starting incremental ingestion for user_id=%s", user_id)

    docs = load_user_documents(user_id)
    if not docs:
        return {"status": "empty", "added": 0, "updated": 0, "deleted": 0, "unchanged": 0}

    # Fetch existing hashes from Qdrant: { doc_id: content_hash }
    existing = get_existing_hashes(user_id)
    current_doc_ids = {d["doc_id"] for d in docs}

    # Nodes removed from the map → delete their chunks
    deleted = 0
    for doc_id in list(existing.keys()):
        if doc_id not in current_doc_ids:
            delete_doc_chunks(doc_id)
            deleted += 1

    # Separate new vs changed vs unchanged
    to_ingest = []
    added = updated = unchanged = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        new_hash = doc["content_hash"]
        old_hash = existing.get(doc_id)

        if old_hash is None:
            added += 1
            to_ingest.append(doc)
        elif old_hash != new_hash:
            updated += 1
            delete_doc_chunks(doc_id)   # remove stale chunks first
            to_ingest.append(doc)
        else:
            unchanged += 1

    # Embed and upsert only changed/new docs
    if to_ingest:
        chunks = chunk_documents(to_ingest, strategy="semantic")
        embed_chunks(chunks)
        upsert_chunks(chunks)

    _cache_module.invalidate_user_cache(user_id)

    logger.info(
        "Incremental ingest done for user_id=%s: added=%d updated=%d deleted=%d unchanged=%d",
        user_id, added, updated, deleted, unchanged,
    )
    return {
        "status": "ok",
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "unchanged": unchanged,
    }


def query(
    question: str,
    user_id: int,
    session_id: str = "default",
    force_mode: str | None = None,   # "rag" | "web" | None
) -> dict[str, Any]:
    """
    Run the full RAG pipeline for *question* scoped to *user_id*.

    Returns:
    {
        "answer":   str,
        "sources":  list[str],
        "mode":     "rag" | "web" | "cache",
        "metrics":  dict,
    }
    """
    t0 = time.monotonic()

    # 1. Input guardrail
    qr = guardrails.validate_query(question)
    if not qr.passed:
        return {
            "answer": qr.reason,
            "sources": [],
            "mode": "blocked",
            "metrics": {},
        }

    # 2. Cache lookup
    cached = _cache_module.get_cached(user_id, question)
    if cached is not None:
        latency = (time.monotonic() - t0) * 1000
        monitoring.record_query(question, "cache", latency, user_id)
        return {**cached, "mode": "cache", "metrics": {"latency_ms": round(latency, 2)}}

    # 3. Route: explicit web request → web search; everything else → RAG notes
    retrieved_chunks: list = []
    if force_mode == "web" or _user_wants_web(question):
        logger.info("Web search triggered explicitly for user_id=%s", user_id)
        search_results = web_search.search(question)
        result = generate_from_web(question, search_results)
    else:
        # Always search notes first
        retrieved_chunks = retrieve(question, user_id=user_id)
        if retrieved_chunks:
            logger.info("RAG hit: %d chunks for user_id=%s", len(retrieved_chunks), user_id)
            result = generate_from_context(question, retrieved_chunks)
        else:
            # Nothing found in notes — tell the user
            logger.info("No RAG results for user_id=%s", user_id)
            result = {
                "answer": "I couldn't find anything about this in your notes. "
                          "Try asking me to 'search the web for ...' if you want an internet search.",
                "sources": [],
                "mode": "rag",
            }

    # 5. Output guardrail
    gr = guardrails.check_output(result["answer"], retrieved_chunks)
    result["answer"] = gr.answer
    result["sources"] = guardrails.sanitise_sources(result.get("sources", []))
    if gr.flags:
        result["guardrail_flags"] = gr.flags

    # 6. Metrics
    latency = (time.monotonic() - t0) * 1000
    mode = result.get("mode", "rag")
    monitoring.record_query(question, mode, latency, user_id)

    result["metrics"] = {
        "latency_ms": round(latency, 2),
        "mode": mode,
    }

    # 7. Cache result
    _cache_module.set_cached(user_id, question, result)

    return result
