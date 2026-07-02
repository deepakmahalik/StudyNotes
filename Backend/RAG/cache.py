"""
TTL-based in-memory query result cache.

Keys are (user_id, query_text) tuples.
Expired entries are evicted lazily on access and eagerly on every `set` when
the cache is at capacity.
"""

import hashlib
import logging
import time
from threading import Lock
from typing import Any

from . import config

logger = logging.getLogger(__name__)


class QueryCache:
    def __init__(self, ttl: int = config.CACHE_TTL, max_size: int = config.CACHE_MAX_SIZE):
        self._ttl = ttl
        self._max = max_size
        self._store: dict[str, tuple[float, Any]] = {}  # key → (expires_at, value)
        self._lock = Lock()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(user_id: int, query: str) -> str:
        raw = f"{user_id}::{query.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (exp, _) in self._store.items() if exp <= now]
        for k in expired:
            del self._store[k]

    def _evict_oldest(self) -> None:
        if not self._store:
            return
        oldest = min(self._store, key=lambda k: self._store[k][0])
        del self._store[oldest]

    # ── public API ─────────────────────────────────────────────────────────────

    def get(self, user_id: int, query: str) -> Any | None:
        key = self._make_key(user_id, query)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            logger.debug("Cache hit for user_id=%s", user_id)
            return value

    def set(self, user_id: int, query: str, value: Any) -> None:
        key = self._make_key(user_id, query)
        expires_at = time.monotonic() + self._ttl
        with self._lock:
            self._evict_expired()
            while len(self._store) >= self._max:
                self._evict_oldest()
            self._store[key] = (expires_at, value)

    def invalidate_user(self, user_id: int) -> int:
        """Remove all cached results for *user_id* (call after re-ingestion)."""
        prefix = f"{user_id}::"
        with self._lock:
            to_delete = [
                k for k in self._store
                if self._make_key(user_id, "").startswith(
                    hashlib.sha256(prefix.encode()).hexdigest()[:4]
                )
            ]
            # Re-derive properly: rebuild and compare
            to_delete = []
            now = time.monotonic()
            for k, (exp, _) in list(self._store.items()):
                # We cannot reverse the hash, so just clear everything on invalidation
                pass
            # Simple approach: clear the whole cache when a user re-ingests
            removed = len(self._store)
            self._store.clear()
        logger.info("Cache cleared (%d entries removed)", removed)
        return removed

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton
_cache = QueryCache()


def get_cached(user_id: int, query: str) -> Any | None:
    return _cache.get(user_id, query)


def set_cached(user_id: int, query: str, value: Any) -> None:
    _cache.set(user_id, query, value)


def invalidate_user_cache(user_id: int) -> int:
    return _cache.invalidate_user(user_id)
