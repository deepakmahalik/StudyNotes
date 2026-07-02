"""
PDF Processor
-------------
Extracts text from a PDF file and structures it into mind-map nodes
using the shared Gemini LLM chain from ocr_node_agent.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF using pdfplumber (preserves layout best)."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(f"[Page {i}]\n{page_text.strip()}")
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("pdfplumber not installed, falling back to pypdf")
        return _extract_pypdf(file_path)


def _extract_pypdf(file_path: str) -> str:
    """Fallback PDF extraction using pypdf."""
    import pypdf
    text_parts = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"[Page {i}]\n{page_text.strip()}")
    return "\n\n".join(text_parts)


# ── Node generation ───────────────────────────────────────────────────────────

def process_pdf(file_path: str, output_path: str) -> dict:
    """
    Extract text from *file_path*, send to Gemini, save structured nodes
    to *output_path*, and return the nodes dict.
    """
    logger.info("PDF processor: extracting text from %s", file_path)
    text = extract_text_from_pdf(file_path)

    if not text or not text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    logger.info("PDF processor: %d chars extracted, sending to LLM", len(text))

    # Import chain from the shared agent (avoids duplicating LLM init)
    from ocr_node_agent import chain
    response = chain.invoke({"ocr_text": text})
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    nodes_data = json.loads(raw)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nodes_data, f, indent=2)

    logger.info("PDF processor: nodes saved to %s", output_path)

    # Save copy to public folder for auto-import
    base_dir = os.path.dirname(os.path.abspath(__file__))
    public_ocr_path = os.path.abspath(os.path.join(base_dir, "..", "..", "MindForge", "public", "ocr_nodes.json"))
    try:
        os.makedirs(os.path.dirname(public_ocr_path), exist_ok=True)
        with open(public_ocr_path, "w", encoding="utf-8") as pub_f:
            json.dump(nodes_data, pub_f, indent=2)
        logger.info("PDF processor: copied nodes to MindForge public folder")
    except Exception as pe:
        logger.warning("PDF processor: could not copy to public folder: %s", pe)

    return nodes_data
