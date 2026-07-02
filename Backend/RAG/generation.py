"""
LLM generation layer — wraps Gemini via LangChain.

Two generation modes
--------------------
generate_from_context  — RAG mode: answer strictly from retrieved chunks
generate_from_web      — Web mode: answer from internet search results
"""

import logging
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from . import config
from .prompt_templates import (
    rag_prompt,
    web_prompt,
    intent_prompt,
    build_context,
    build_web_context,
)

logger = logging.getLogger(__name__)

# Inject API key into env so LangChain picks it up
os.environ.setdefault("GOOGLE_API_KEY", config.GEMINI_API_KEY)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_TOKENS,
            google_api_key=config.GEMINI_API_KEY,
        )
        logger.info("Gemini LLM initialised: %s", config.GEMINI_MODEL)
    return _llm


# ── Public API ────────────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities from stored node descriptions."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def generate_from_context(question: str, chunks: list[dict]) -> dict:
    """
    Summarise retrieved chunks using the LLM.
    The context is built from the user's own notes; the LLM synthesises a
    coherent answer while staying strictly within what the notes say.
    """
    if not chunks:
        return {
            "answer": "Your notes do not contain information about this topic.",
            "sources": [],
            "mode": "rag",
        }

    logger.info("generate_from_context: %d chunks received for question=%r", len(chunks), question)
    for i, c in enumerate(chunks):
        logger.info("  chunk[%d] label=%r doc_id=%r text_preview=%r",
                    i, c.get("metadata", {}).get("label"), c.get("doc_id"), c.get("text", "")[:80])

    # Generic placeholder names created by the UI — not meaningful answers
    _PLACEHOLDER_LABELS = {"new node", "node", "untitled", ""}

    clean_chunks: list[dict] = []
    sources: list[str] = []

    for chunk in chunks:
        label = chunk.get("metadata", {}).get("label", "")
        text = _strip_html(chunk.get("text", ""))

        # Only skip when the label is a placeholder AND the text contains
        # no real content beyond the label name itself
        body = (text[len(label):].lstrip(": ").strip()
                if label and text.lower().startswith(label.lower()) else text.strip())
        if label.lower().strip() in _PLACEHOLDER_LABELS and not body:
            continue

        clean_chunks.append({**chunk, "text": text})
        if label and label not in sources:
            sources.append(label)

    if not clean_chunks:
        return {
            "answer": "Your notes do not contain information about this topic.",
            "sources": [],
            "mode": "rag",
        }

    # Deduplicate: drop chunks whose text is >85% similar to an earlier chunk
    seen_texts: list[str] = []
    deduped: list[dict] = []
    for chunk in clean_chunks:
        txt = chunk.get("text", "").strip()
        txt_words = set(txt.lower().split())
        is_dup = any(
            len(txt_words & set(s.lower().split())) / max(len(txt_words | set(s.lower().split())), 1) > 0.85
            for s in seen_texts
        )
        if not is_dup:
            deduped.append(chunk)
            seen_texts.append(txt)
    clean_chunks = deduped

    # Direct-match shortcut: if the top chunk is a composite (_comp) whose label
    # closely matches the question, the items list IS the answer — skip the LLM
    # to prevent the model substituting its training-data knowledge.
    top = clean_chunks[0]
    top_label = top.get("metadata", {}).get("label", "")
    top_doc_id = top.get("doc_id", "")
    if top_doc_id.endswith("_comp") and top_label:
        # Check label words appear in the question (e.g. "RAG Types" in "types of RAG")
        label_words = set(top_label.lower().split())
        q_words = set(question.lower().split())
        if label_words & q_words:  # at least one word in common
            text = top.get("text", "")
            # Header format (after _strip_html collapses newlines):
            # "Label — The Label include: A, B, C. - A - B - C"
            # Extract from the "include:" part
            import re as _re
            m = _re.search(r'include[s]?:\s*(.+?)(?:\.|$)', text, _re.IGNORECASE)
            if m:
                raw = m.group(1).strip().rstrip(".")
                items = [i.strip() for i in raw.split(",") if i.strip()]
            else:
                # Fallback: everything after the first colon
                parts = text.split(":", 1)
                items = [i.strip() for i in parts[1].split(",") if i.strip()] if len(parts) > 1 else []

            if items:
                bullet_list = "\n".join(f"* {item}" for item in items)
                answer = f"According to your notes, the {top_label} are:\n\n{bullet_list}"
                logger.info("Direct composite match: bypassing LLM for label=%r", top_label)
                return {"answer": answer, "sources": [top_label], "mode": "rag"}

    context = build_context(clean_chunks)
    try:
        chain = rag_prompt | _get_llm()
        response = chain.invoke({"context": context, "question": question})
        answer = response.content.strip()
    except Exception as exc:
        logger.warning("LLM summarisation failed (%s); falling back to raw context", exc)
        # Fallback: return formatted raw notes if LLM is unavailable
        lines = []
        for chunk in clean_chunks:
            label = chunk.get("metadata", {}).get("label", "")
            text = chunk.get("text", "")
            if label and text.lower().startswith(label.lower()):
                body = text[len(label):].lstrip(": ").strip()
                lines.append(f"📌 **{label}**\n{body}" if body else f"📌 **{label}**")
            elif label:
                lines.append(f"📌 **{label}**\n{text}")
            else:
                lines.append(text)
        answer = "\n\n".join(lines)

    return {
        "answer": answer,
        "sources": sources,
        "mode": "rag",
    }


def generate_from_web(question: str, search_results: list[dict]) -> dict:
    """
    Generate an answer from web search results.

    Returns:
        {
            "answer":  str,
            "sources": list[str],   # URLs
            "mode":    "web",
        }
    """
    search_str = build_web_context(search_results)
    chain = web_prompt | _get_llm()

    response = chain.invoke({"search_results": search_str, "question": question})
    answer_text: str = response.content.strip()

    urls = [r.get("href", r.get("url", "")) for r in search_results]

    return {
        "answer": answer_text,
        "sources": [u for u in urls if u],
        "mode": "web",
    }


def classify_intent(question: str) -> str:
    """
    Use the LLM to classify query intent.
    Returns "NOTES" or "EXTERNAL".
    Falls back to "NOTES" on any error.
    """
    try:
        chain = intent_prompt | _get_llm()
        response = chain.invoke({"question": question})
        intent = response.content.strip().upper()
        return "NOTES" if "NOTES" in intent else "EXTERNAL"
    except Exception as exc:
        logger.warning("Intent classification failed (%s); defaulting to NOTES", exc)
        return "NOTES"
