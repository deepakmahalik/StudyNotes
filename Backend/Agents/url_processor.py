"""
URL Processor
-------------
Downloads web pages, extracts clean text, and structures it into mind-map nodes
using the shared Gemini LLM chain from ocr_node_agent.
"""

import os
import json
import logging
import requests
import lxml.html

logger = logging.getLogger(__name__)

def extract_text_from_url(url: str) -> str:
    """Download HTML content from a URL and extract text content using lxml."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    logger.info("URL processor: fetching content from %s", url)
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    
    html_content = response.text
    if not html_content or not html_content.strip():
        raise ValueError("URL returned empty content.")
        
    # Parse HTML and extract clean text
    document = lxml.html.fromstring(html_content)
    
    # Remove script, style, head, nav, footer, header elements to clean the text
    for element in document.xpath("//script | //style | //head | //nav | //footer | //header"):
        element.getparent().remove(element)
        
    text = document.text_content()
    
    # Format and clean whitespace
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    cleaned_text = "\n".join(non_empty_lines)
    
    return cleaned_text


def process_url(url: str, output_path: str) -> dict:
    """
    Extract text from a URL, send it to the Gemini LLM chain,
    save the structured nodes to output_path, copy to public folder, and return the nodes.
    """
    text = extract_text_from_url(url)
    
    if not text or not text.strip():
        raise ValueError("No text could be extracted from the URL.")
        
    logger.info("URL processor: %d chars extracted, sending to LLM", len(text))
    
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

    logger.info("URL processor: nodes saved to %s", output_path)

    # Save copy to public folder for auto-import
    base_dir = os.path.dirname(os.path.abspath(__file__))
    public_ocr_path = os.path.abspath(os.path.join(base_dir, "..", "..", "MindForge", "public", "ocr_nodes.json"))
    try:
        os.makedirs(os.path.dirname(public_ocr_path), exist_ok=True)
        with open(public_ocr_path, "w", encoding="utf-8") as pub_f:
            json.dump(nodes_data, pub_f, indent=2)
        logger.info("URL processor: copied nodes to MindForge public folder")
    except Exception as pe:
        logger.warning("URL processor: could not copy to public folder: %s", pe)

    return nodes_data
