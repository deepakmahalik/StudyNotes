"""
Image Processor
---------------
Extracts text from image files using OCR (Tesseract) and structures it into mind-map nodes
using the shared Gemini LLM chain from ocr_node_agent.
"""

import os
import sys
import json
import logging

logger = logging.getLogger(__name__)

def process_image(file_path: str, output_path: str) -> dict:
    """
    Perform OCR on the image at *file_path*, send extracted text to Gemini,
    save structured nodes to *output_path*, copy to public folder, and return nodes dict.
    """
    logger.info("Image processor: performing OCR on %s", file_path)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys_path = os.path.abspath(os.path.join(base_dir, ".."))
    if sys_path not in sys.path:
        sys.path.append(sys_path)

    try:
        from OCR.OCR import perform_ocr_on_file
    except ImportError:
        # Fallback to direct import if path structure differs
        sys.path.append(os.path.abspath(os.path.join(base_dir, "..", "..")))
        from MM.Backend.OCR.OCR import perform_ocr_on_file

    text = perform_ocr_on_file(file_path)

    if not text or not text.strip():
        raise ValueError("No text could be extracted from the image. Ensure the image has clear, readable text.")

    logger.info("Image processor: %d characters extracted via OCR, sending to Gemini", len(text))

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

    logger.info("Image processor: nodes saved to %s", output_path)

    # Save copy to public folder for auto-import
    public_ocr_path = os.path.abspath(os.path.join(base_dir, "..", "..", "MindForge", "public", "ocr_nodes.json"))
    try:
        os.makedirs(os.path.dirname(public_ocr_path), exist_ok=True)
        with open(public_ocr_path, "w", encoding="utf-8") as pub_f:
            json.dump(nodes_data, pub_f, indent=2)
        logger.info("Image processor: copied nodes to MindForge public folder")
    except Exception as pe:
        logger.warning("Image processor: could not copy to public folder: %s", pe)

    return nodes_data
