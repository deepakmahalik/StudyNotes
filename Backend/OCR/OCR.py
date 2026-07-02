import os
from PIL import Image
import pytesseract

# Path to tesseract executable (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}

def perform_ocr_on_file(file_path):
    """Performs OCR on a single image file and returns the extracted text."""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return None

def process_upload_folder(folder_path):
    """Processes all supported image files in the upload folder one by one."""
    if not os.path.exists(folder_path):
        print(f"[WARNING] Upload folder does not exist at: {folder_path}")
        return

    files = sorted(os.listdir(folder_path))
    image_files = [f for f in files if os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS]
    
    if not image_files:
        print(f"[INFO] No supported image files found in: {folder_path}")
        return

    print(f"[INFO] Found {len(image_files)} image(s) to process.\n")

    for filename in image_files:
        file_path = os.path.join(folder_path, filename)
        print("=" * 60)
        print(f"Processing: {filename}")
        print("=" * 60)
        
        extracted_text = perform_ocr_on_file(file_path)
        if extracted_text is not None:
            print("Extracted Text:")
            print(extracted_text)
        else:
            print("Failed to extract text.")
        print("\n")

def main():
    # Resolve absolute path relative to project root or use direct path
    # MM/Backend/OCR/Upload
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(base_dir, "Upload")
    
    process_upload_folder(upload_dir)

if __name__ == "__main__":
    main()
