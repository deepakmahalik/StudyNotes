"""
Data ingestion: load mind-map nodes from SQLite, clean and normalise them,
then return a list of document dicts ready for chunking.

Document schema:
    {
        "doc_id":    str,   # unique: f"{user_id}_{node_id}"
        "user_id":   int,
        "node_id":   str,
        "label":     str,
        "parent_id": str | None,
        "text":      str,   # combined label + description
        "source":    str,   # "mindmap"
        "timestamp": str,   # ISO-8601
        "metadata":  dict,
    }
"""

import hashlib
import os
import re
import sys
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Allow importing database.py from Backend/
_backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import database


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Basic normalisation: collapse whitespace, strip control chars."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _get_label(node: dict) -> str:
    """Support both 'label' and 'name' field names."""
    return _clean(node.get("label") or node.get("name", ""))


def _get_desc(node: dict) -> str:
    """Support both 'description' and 'desc' field names."""
    return _clean(node.get("description") or node.get("desc", ""))


def node_content_hash(node: dict) -> str:
    """SHA-256 of the node's searchable content — used to detect changes."""
    label = _get_label(node)
    desc = _get_desc(node)
    raw = f"{label}::{desc}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _node_to_text(node: dict) -> str:
    """Combine label/name and description/desc into a single searchable string."""
    label = _get_label(node)
    desc = _get_desc(node)
    if label and desc:
        return f"{label}: {desc}"
    return label or desc


# ── public API ───────────────────────────────────────────────────────────────

def load_user_documents(user_id: int) -> list[dict[str, Any]]:
    """
    Load all mind-map nodes for *user_id* and return them as documents.

    Three enrichment strategies are applied:
      Level 1 — container nodes (no desc, has children) get children names appended
      Level 2 — leaf nodes (no desc, no children) get their parent path prepended
      Level 3 — each parent node also gets a composite chunk: parent + children
                 with their descriptions, giving hierarchy-aware retrieval
    """
    nodes = database.get_user_maps(user_id)
    if not nodes:
        logger.info("No mind-map nodes found for user_id=%s", user_id)
        return []

    timestamp = datetime.now(timezone.utc).isoformat()
    docs: list[dict[str, Any]] = []

    _PLACEHOLDER_NAMES = {"new node", "node", "untitled", ""}

    # Index nodes by id for quick lookup
    node_by_id: dict[str, dict] = {str(n.get("id", "")): n for n in nodes}

    # Build parent → [child_node] map
    children_map: dict[str, list[dict]] = {}
    for n in nodes:
        pid = str(n.get("parentId") or n.get("pid") or "")
        if pid:
            children_map.setdefault(pid, []).append(n)

    def _parent_path(node: dict, max_depth: int = 3) -> str:
        """Return breadcrumb path e.g. 'Gen AI > RAG > RAG Types'."""
        parts: list[str] = []
        cur = node
        for _ in range(max_depth):
            pid = str(cur.get("parentId") or cur.get("pid") or "")
            if not pid or pid not in node_by_id:
                break
            parent = node_by_id[pid]
            plabel = _get_label(parent)
            if plabel:
                parts.append(plabel)
            cur = parent
        parts.reverse()
        return " > ".join(parts)

    def _make_doc(node_id: str, label: str, text: str,
                  source: str = "mindmap", doc_suffix: str = "") -> dict:
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        did = f"{user_id}_{node_id}{doc_suffix}"
        return {
            "doc_id": did,
            "user_id": user_id,
            "node_id": node_id,
            "label": label,
            "text": text,
            "source": source,
            "timestamp": timestamp,
            "content_hash": h,
            "metadata": {
                "user_id": user_id,
                "node_id": node_id,
                "label": label,
                "source": source,
                "timestamp": timestamp,
                "content_hash": h,
            },
        }

    for node in nodes:
        node_id = str(node.get("id", ""))
        label = _get_label(node)
        desc  = _get_desc(node)

        # Skip pure placeholder nodes
        if label.lower().strip() in _PLACEHOLDER_NAMES and not desc.strip():
            logger.debug("Skipping placeholder node: %r", label)
            continue

        has_children = node_id in children_map
        has_desc     = bool(desc.strip())

        # ── Level 1 & 2: build the node's own text ────────────────────────────
        if has_desc:
            text = f"{label}: {desc}" if label else desc
        elif has_children:
            # Level 1 — container: enrich with direct children names
            child_names = ", ".join(_get_label(c) for c in children_map[node_id] if _get_label(c))
            text = f"{label}: {child_names}" if child_names else label
        else:
            # Level 2 — leaf with no desc: add parent path for context
            path = _parent_path(node)
            text = f"{path} > {label}" if path else label

        if not text.strip():
            continue

        docs.append(_make_doc(node_id, label, text))

        # ── Level 3: composite chunk — parent + all children with descriptions ─
        if has_children:
            child_lines = []
            for child in children_map[node_id]:
                clabel = _get_label(child)
                cdesc  = _get_desc(child)
                if not clabel:
                    continue
                if cdesc:
                    child_lines.append(f"  - {clabel}: {cdesc}")
                else:
                    cid = str(child.get("id", ""))
                    if cid in children_map:
                        gc_names = ", ".join(
                            _get_label(gc) for gc in children_map[cid] if _get_label(gc)
                        )
                        child_lines.append(f"  - {clabel}: {gc_names}" if gc_names else f"  - {clabel}")
                    else:
                        child_lines.append(f"  - {clabel}")

            if child_lines:
                # Header uses natural language so queries like "types of X" or
                # "what are the X" match the composite chunk directly
                child_names_inline = ", ".join(
                    _get_label(c) for c in children_map[node_id] if _get_label(c)
                )
                header = (
                    f"{label} — The {label} include: {child_names_inline}."
                    if child_names_inline else label
                )
                composite_text = header + "\n" + "\n".join(child_lines)
                docs.append(_make_doc(node_id, label, composite_text,
                                      source="composite", doc_suffix="_comp"))

    logger.info("Loaded %d documents (incl. composites) for user_id=%s", len(docs), user_id)
    return docs


def load_all_documents() -> list[dict[str, Any]]:
    """Load mind-map nodes for every registered user (admin/batch ingest)."""
    all_docs: list[dict[str, Any]] = []
    for user in database.get_all_users():
        all_docs.extend(load_user_documents(user["id"]))
    logger.info("Total documents loaded across all users: %d", len(all_docs))
    return all_docs
