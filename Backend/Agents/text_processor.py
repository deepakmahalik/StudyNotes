"""
Text Processor
--------------
Extracts text from a plain text file (or text content) and structures it into mind-map nodes
using the shared Gemini LLM chain from ocr_node_agent.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

def process_text_content(text: str, output_path: str) -> dict:
    """
    Process raw text content directly: send to Gemini, save structured nodes
    to *output_path*, copy to MindForge public folder, and return the nodes dict.
    """
    if not text or not text.strip():
        raise ValueError("No text provided for node generation.")

    logger.info("Text processor: sending %d characters to Gemini", len(text))

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

    logger.info("Text processor: nodes saved to %s", output_path)

    # Save copy to public folder for auto-import
    base_dir = os.path.dirname(os.path.abspath(__file__))
    public_ocr_path = os.path.abspath(os.path.join(base_dir, "..", "..", "MindForge", "public", "ocr_nodes.json"))
    try:
        os.makedirs(os.path.dirname(public_ocr_path), exist_ok=True)
        with open(public_ocr_path, "w", encoding="utf-8") as pub_f:
            json.dump(nodes_data, pub_f, indent=2)
        logger.info("Text processor: copied nodes to MindForge public folder")
    except Exception as pe:
        logger.warning("Text processor: could not copy to public folder: %s", pe)

    return nodes_data


def process_text(file_path: str, output_path: str) -> dict:
    """
    Extract text from *file_path*, send to Gemini, save structured nodes
    to *output_path*, and return the nodes dict.
    """
    logger.info("Text processor: reading text from %s", file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return process_text_content(text, output_path)
