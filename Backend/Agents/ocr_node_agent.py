import os
import sys
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

# Adjust path so we can import from MM.Backend.OCR.OCR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from MM.Backend.OCR.OCR import perform_ocr_on_file, SUPPORTED_EXTENSIONS
except ImportError:
    # Fallback to local import if structure differs
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from OCR.OCR import perform_ocr_on_file, SUPPORTED_EXTENSIONS

# Path to the config file
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Config", "config.properties"))

def load_config(path):
    config = {}
    if not os.path.exists(path):
        print(f"[WARNING] Config file not found at: {path}")
        return config
    
    with open(path, "r", encoding="utf-8") as f:
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
                config[key] = val
    return config

# Load properties and set environment variables
config = load_config(CONFIG_PATH)
for k, v in config.items():
    os.environ[k] = v

os.environ["LANGCHAIN_TRACING_V2"] = "false"

if "GEMINI_API_KEY" not in os.environ:
    print("[ERROR] GEMINI_API_KEY not found in config.properties or environment.")
    sys.exit(1)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,  # Low temperature for precise structured extraction
)

# Prompt template for structure division
system_prompt = (
    "You are an expert information architect. Your task is to take OCR-extracted text and "
    "organize it into a hierarchical mind map structure consisting of nodes and subnodes.\n\n"
    "Identify the main/central topic as the root node, key concepts as children nodes of the root, "
    "and detailed details as subnodes of those key concepts.\n\n"
    "You must return the result STRICTLY as a JSON object matching this schema:\n"
    "{\n"
    '  "nodes": [\n'
    "    {\n"
    '      "id": "root",\n'
    '      "label": "Central Topic Label",\n'
    '      "description": "Short explanation or description of this topic",\n'
    '      "parentId": null\n'
    "    },\n"
    "    {\n"
    '      "id": "n1",\n'
    '      "label": "Sub-concept A Label",\n'
    '      "description": "Short explanation",\n'
    '      "parentId": "root"\n'
    "    },\n"
    "    {\n"
    '      "id": "n2",\n'
    '      "label": "Detail under A",\n'
    '      "description": "Description of the detail",\n'
    '      "parentId": "n1"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Ensure all node IDs are unique and parentId references match existing node IDs. "
    "Do not include any markup, markdown wrappers, or explanation outside of the valid JSON."
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    ("human", "Analyze this OCR-extracted text and generate the JSON mind map:\n\n{ocr_text}")
])

chain = prompt | llm

def process_images_and_structure(target_filename=None):
    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.abspath(os.path.join(base_dir, "..", "OCR", "Upload"))
    processed_dir = os.path.abspath(os.path.join(base_dir, "..", "OCR", "Processed"))
    
    if not os.path.exists(upload_dir):
        print(f"[ERROR] Upload directory not found at: {upload_dir}")
        return
        
    os.makedirs(processed_dir, exist_ok=True)
    
    files = sorted(os.listdir(upload_dir))
    if target_filename:
        image_files = [f for f in files if f == target_filename]
    else:
        image_files = [f for f in files if os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS or os.path.splitext(f.lower())[1] == '.txt']
    
    if not image_files:
        print(f"[INFO] No image files found in {upload_dir} to process.")
        return

    print(f"[INFO] Found {len(image_files)} image(s) to process.")
    
    for filename in image_files:
        file_path = os.path.join(upload_dir, filename)
        print(f"\nProcessing OCR for: {filename}...")
        
        # 1. Perform OCR or Read Text
        ext = os.path.splitext(filename.lower())[1]
        ocr_text = ""
        if ext == '.txt':
            with open(file_path, "r", encoding="utf-8") as tf:
                ocr_text = tf.read()
        else:
            ocr_text = perform_ocr_on_file(file_path)
            
        if not ocr_text or not ocr_text.strip():
            print(f"[WARNING] No text extracted from {filename}.")
            continue
            
        print("Text successfully extracted. Organizing into nodes and subnodes using Gemini...")
        
        # 2. Call LLM to structure text
        try:
            response = chain.invoke({"ocr_text": ocr_text})
            raw_content = response.content.strip()
            
            # Clean possible markdown code fences from response
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()
            
            # Parse JSON to validate structure
            nodes_data = json.loads(raw_content)
            
            # Save output JSON
            output_name = os.path.splitext(filename)[0] + "_structure.json"
            output_path = os.path.join(processed_dir, output_name)
            
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(nodes_data, out_f, indent=2)
                
            print(f"[SUCCESS] Mind map nodes saved to: {output_path}")

            # Also save a copy to the MindForge public folder for auto-import
            public_ocr_path = os.path.abspath(os.path.join(base_dir, "..", "..", "MindForge", "public", "ocr_nodes.json"))
            try:
                os.makedirs(os.path.dirname(public_ocr_path), exist_ok=True)
                with open(public_ocr_path, "w", encoding="utf-8") as pub_f:
                    json.dump(nodes_data, pub_f, indent=2)
                print(f"[SUCCESS] Copied nodes to MindForge public folder: {public_ocr_path}")
            except Exception as pe:
                print(f"[WARNING] Could not copy to public folder: {pe}")
            print("\nPreview of structured nodes:")
            for node in nodes_data.get("nodes", []):
                depth = 0
                pid = node.get("parentId")
                while pid:
                    depth += 1
                    parent = next((n for n in nodes_data["nodes"] if n["id"] == pid), None)
                    pid = parent.get("parentId") if parent else None
                indent = "  " * depth
                print(f"{indent}- [{node['id']}] {node['label']} ({node.get('description', '')})")
                
            print(f"__OUTPUT_JSON_PATH__={output_path}")
                
        except json.JSONDecodeError as je:
            print(f"[ERROR] LLM response was not valid JSON: {je}")
            print("Raw response content:")
            print(raw_content)
        except Exception as e:
            print(f"[ERROR] Failed to process with Gemini: {e}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    process_images_and_structure(target)
