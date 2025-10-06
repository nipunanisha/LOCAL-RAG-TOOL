import os
from rag.ingest import load_text_from_file

# Test file path for an image containing text
image_path = "test_image.png"

if os.path.exists(image_path):
    extracted_text = load_text_from_file(image_path)
    print("Extracted text from image:")
    print(extracted_text)
else:
    print(f"Test image file {image_path} not found. Please add an image file named 'test_image.png' in the root directory to run this test.")
