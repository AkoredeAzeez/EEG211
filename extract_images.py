import fitz  # PyMuPDF
import os
from PIL import Image
import numpy as np
import cv2

PDF_PATH = "static/dataset.pdf"
OUTPUT_DIR = "static/extracted_images"
FEATURE_DB = "image_features.npz"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_images_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            image_ext = base_image["ext"]
            image_path = os.path.join(output_dir, f"page_{i+1}_img_{img_index+1}.{image_ext}")

            with open(image_path, "wb") as img_file:
                img_file.write(image_data)
            
            image_paths.append(image_path)

    print(f"Extracted {len(image_paths)} images.")
    return image_paths

def compute_image_features(image_paths):
    orb = cv2.ORB_create()
    features = {}

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = orb.detectAndCompute(img, None)

        if descriptors is not None:
            features[img_path] = descriptors

    np.savez_compressed(FEATURE_DB, **features)
    print(f"Saved features for {len(features)} images.")

if __name__ == "__main__":
    print("Extracting images from PDF...")
    image_paths = extract_images_from_pdf(PDF_PATH, OUTPUT_DIR)

    print("Computing image features...")
    compute_image_features(image_paths)

    print("Image extraction and feature processing complete! 🎉")
