"""
Chunking strategies for RAG documents.

Strategies
----------
fixed       — split text every CHUNK_SIZE characters with CHUNK_OVERLAP overlap
overlapping — alias for fixed (explicit naming in callers)
semantic    — split on sentence boundaries, then merge until size limit

Each chunk dict:
    {
        "chunk_id":   str,   # f"{doc_id}_c{index}"
        "doc_id":     str,
        "text":       str,
        "chunk_index": int,
        "metadata":   dict,  # copy of parent doc metadata + chunk_index
    }
"""

import re
import logging
from typing import Any, Literal

from . import config

logger = logging.getLogger(__name__)

Strategy = Literal["fixed", "overlapping", "semantic"]


# ── internal helpers ──────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — handles . ! ? followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += size - overlap
    return [c for c in chunks if c]


def _semantic_chunks(text: str, size: int) -> list[str]:
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > size and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── public API ────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: list[dict[str, Any]],
    strategy: Strategy = "semantic",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """
    Split *documents* into chunks using the chosen *strategy*.

    Returns a flat list of chunk dicts.
    """
    size = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap or config.CHUNK_OVERLAP

    all_chunks: list[dict[str, Any]] = []

    for doc in documents:
        text = doc.get("text", "")
        if not text:
            continue

        if strategy in ("fixed", "overlapping"):
            raw = _fixed_chunks(text, size, overlap)
        else:
            raw = _semantic_chunks(text, size)

        # If the whole text fits in one chunk, keep it as-is
        if not raw:
            raw = [text]

        doc_id = doc["doc_id"]
        base_meta = dict(doc.get("metadata", {}))

        for idx, chunk_text in enumerate(raw):
            meta = {**base_meta, "chunk_index": idx, "strategy": strategy}
            all_chunks.append(
                {
                    "chunk_id": f"{doc_id}_c{idx}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "chunk_index": idx,
                    "metadata": meta,
                }
            )

    logger.info(
        "Chunked %d documents → %d chunks (strategy=%s, size=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        strategy,
        size,
        overlap,
    )
    return all_chunks
