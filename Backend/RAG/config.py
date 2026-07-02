"""
RAG pipeline configuration.
Reads Config/config.properties and exposes typed constants.
"""

import os

_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Config", "config.properties")
)


def _load() -> dict:
    cfg: dict = {}
    if not os.path.exists(_CONFIG_PATH):
        return cfg
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if val.startswith("="):
                    val = val[1:].strip()
                cfg[key] = val
    return cfg


_cfg = _load()

# ── Gemini ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = _cfg.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
GEMINI_MODEL: str = _cfg.get("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE: float = float(_cfg.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS: int = int(_cfg.get("LLM_MAX_TOKENS", "2048"))

# ── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_URL: str = _cfg.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str | None = _cfg.get("QDRANT_API_KEY") or None
QDRANT_COLLECTION: str = _cfg.get("QDRANT_COLLECTION", "studynotes_nodes")

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = _cfg.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = int(_cfg.get("EMBEDDING_DIM", "384"))

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(_cfg.get("CHUNK_SIZE", "300"))
CHUNK_OVERLAP: int = int(_cfg.get("CHUNK_OVERLAP", "50"))

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K: int = int(_cfg.get("TOP_K", "5"))
HYBRID_ALPHA: float = float(_cfg.get("HYBRID_ALPHA", "0.6"))   # dense weight in RRF
RERANK_MODEL: str = _cfg.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N: int = int(_cfg.get("RERANK_TOP_N", "3"))

# ── Web search ───────────────────────────────────────────────────────────────
WEB_SEARCH_PROVIDER: str = _cfg.get("WEB_SEARCH_PROVIDER", "duckduckgo")
WEB_SEARCH_MAX_RESULTS: int = int(_cfg.get("WEB_SEARCH_MAX_RESULTS", "5"))

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL: int = int(_cfg.get("CACHE_TTL", "300"))
CACHE_MAX_SIZE: int = int(_cfg.get("CACHE_MAX_SIZE", "100"))
