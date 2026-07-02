"""
Internet search fallback using DuckDuckGo (free, no API key required).

Returns a list of result dicts:
    { "title": str, "href": str, "body": str }
"""

import logging
from typing import Any

from . import config

logger = logging.getLogger(__name__)


def search(query: str, max_results: int | None = None) -> list[dict[str, Any]]:
    """
    Search the web for *query* and return up to *max_results* results.
    Returns an empty list if the package is not installed or the search fails.
    """
    n = max_results or config.WEB_SEARCH_MAX_RESULTS

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning(
            "duckduckgo-search is not installed; web search disabled. "
            "Run: uv add duckduckgo-search"
        )
        return []

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=n))
        results = [
            {
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "body": r.get("body", ""),
            }
            for r in raw
        ]
        logger.info("Web search returned %d results for query: %.60s", len(results), query)
        return results
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return []
