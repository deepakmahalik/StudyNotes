import os
import re
import json
import argparse
from logger import log

def update_html_mindmap(html_path, json_path):
    """Injects nodes from the structured JSON file into the standalone HTML mindmap file."""
    if not os.path.exists(html_path):
        log.error(f"HTML mindmap file not found at: {html_path}")
        return False
        
    if not os.path.exists(json_path):
        log.error(f"JSON structure file not found at: {json_path}")
        return False

    try:
        # Load the new nodes JSON
        with open(json_path, "r", encoding="utf-8") as j_file:
            json_data = json.load(j_file)
            
        # Ensure we have a list of nodes
        if "nodes" not in json_data:
            log.error("Invalid JSON structure. Missing 'nodes' key.")
            return False
            
        nodes_list = json_data["nodes"]
        log.info(f"Loaded {len(nodes_list)} nodes from: {json_path}")

        # Read the HTML content
        with open(html_path, "r", encoding="utf-8") as h_file:
            html_content = h_file.read()

        # Regular expression to match the data block:
        # /* --- DATA_START --- */ ... /* --- DATA_END --- */
        data_pattern = re.compile(
            r"(/\*\s*--- DATA_START ---\s*\*/)([\s\S]*?)(/\*\s*--- DATA_END ---\s*\*/)"
        )

        if not data_pattern.search(html_content):
            log.error("Could not find the DATA_START/DATA_END block in the HTML file.")
            return False

        # Format the new data block
        new_data_str = json.dumps({"nodes": nodes_list}, indent=4)
        replacement = f"/* --- DATA_START --- */\n    let mapData = {new_data_str};\n    /* --- DATA_END --- */"

        # Replace the old data block with the new one
        updated_content = data_pattern.sub(replacement, html_content)

        # Write the updated HTML back to the file
        with open(html_path, "w", encoding="utf-8") as h_file:
            h_file.write(updated_content)

        log.info(f"Successfully injected nodes into: {html_path}")
        return True

    except Exception as e:
        log.exception(f"Failed to update mindmap: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Inject OCR structured JSON nodes into a MindForge HTML mindmap.")
    parser.add_argument("--html", help="Path to the standalone HTML mindmap file to update")
    parser.add_argument("--json", help="Path to the structured JSON file containing the nodes")
    
    args = parser.parse_args()
    
    # Setup default paths if not provided
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(base_dir, "..", "OCR", "Processed"))
    
    # Find latest json file in Processed directory as default
    default_json = None
    if os.path.exists(processed_dir):
        json_files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]
        if json_files:
            # Sort by modification time to get the latest
            json_files.sort(key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)), reverse=True)
            default_json = os.path.join(processed_dir, json_files[0])
            
    html_path = args.html
    json_path = args.json or default_json
    
    if not html_path:
        # Prompt for inputs if not supplied
        html_path = input("Enter path to the HTML mindmap file: ").strip()
        
    if not json_path:
        json_path = input("Enter path to the JSON nodes file: ").strip()
        
    update_html_mindmap(html_path, json_path)

if __name__ == "__main__":
    main()
