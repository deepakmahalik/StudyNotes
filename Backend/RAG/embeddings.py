"""
Embedding generation using sentence-transformers (free, no API key).

The model is loaded once at module level and reused across calls.
Falls back gracefully if sentence-transformers is not installed.
"""

import logging
from typing import Any

from . import config

logger = logging.getLogger(__name__)

_model: Any = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded (dim=%d)", config.EMBEDDING_DIM)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: uv add sentence-transformers"
            )
    return _model


# ── public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.  Returns a list of float vectors.
    Each vector has length config.EMBEDDING_DIM (384 for all-MiniLM-L6-v2).
    """
    if not texts:
        return []
    model = _get_model()
    # Use a small batch_size (e.g. 8) to reduce peak memory usage and prevent OOM errors
    vectors = model.encode(texts, batch_size=8, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' key to each chunk dict in-place.
    Returns the same list for chaining.
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    logger.info("Embedded %d chunks", len(chunks))
    return chunks
