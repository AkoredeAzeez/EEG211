from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MATCHED_FOLDER = "static/matched"
FEATURE_DB = "image_features.npz"
EXTRACTED_IMAGES_DIR = "static/extracted_images"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MATCHED_FOLDER"] = MATCHED_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MATCHED_FOLDER, exist_ok=True)

image_features = np.load(FEATURE_DB, allow_pickle=True)
image_features = {key: image_features[key] for key in image_features.files}

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_text(image_path):
    """Detect if image contains text and extract it"""
    try:
        # Read image
        img = Image.open(image_path)
        # Extract text
        text = pytesseract.image_to_string(img)
        # Return True if meaningful text is found
        return len(text.strip()) > 5, text
    except Exception as e:
        print(f"Text detection error: {str(e)}")
        return False, ""

def compute_text_similarity(text1, text2):
    """Compute similarity between two text strings"""
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

def find_best_match(query_img_path):
    """Find best matches using appropriate algorithm based on content"""
    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        return [], []

    # Check if image contains text
    has_text, extracted_text = detect_text(query_img_path)
    
    text_matches = []
    feature_matches = []
    
    if has_text:
        # Use template matching for text images
        for img_path in image_features.keys():
            try:
                dataset_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if dataset_img is None:
                    continue
                
                # Check text similarity with dataset image
                _, dataset_text = detect_text(img_path)
                text_similarity = compute_text_similarity(extracted_text, dataset_text)
                
                if text_similarity > 0.3:  # Minimum text similarity threshold
                    text_matches.append({
                        'path': img_path,
                        'score': float(text_similarity),
                        'text': dataset_text[:100]  # Preview first 100 chars
                    })
                    
                # Also try template matching
                resized_query = cv2.resize(query_img, (dataset_img.shape[1], dataset_img.shape[0]))
                result = cv2.matchTemplate(dataset_img, resized_query, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(result)[1]
                
                if score > 0.5:  # Minimum template matching threshold
                    text_matches.append({
                        'path': img_path,
                        'score': float(score),
                        'method': 'template'
                    })
            except cv2.error:
                continue
    
    # Always try ORB matching as fallback
    query_keypoints, query_descriptors = orb.detectAndCompute(query_img, None)
    if query_descriptors is not None:
        for img_path, descriptors in image_features.items():
            matches = bf.match(query_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            score = len(matches) / max(len(query_descriptors), len(descriptors))
            
            if len(matches) > 10:
                feature_matches.append({
                    'path': img_path,
                    'score': float(score)
                })

    # Sort matches by score and get top 10 matches
    text_matches = sorted(text_matches, key=lambda x: x['score'], reverse=True)[:10]  # Changed from 5 to 10
    feature_matches = sorted(feature_matches, key=lambda x: x['score'], reverse=True)[:10]  # Changed from 5 to 10

    return text_matches, feature_matches

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        text_matches, feature_matches = find_best_match(file_path)
        
        response = {
            "success": True,
            "matches": {
                "text": [
                    {
                        "url": url_for('static', filename=f'extracted_images/{os.path.basename(match["path"])}'),
                        "score": match["score"],
                        "text": match.get("text", "")
                    } for match in text_matches
                ],
                "feature": [
                    {
                        "url": url_for('static', filename=f'extracted_images/{os.path.basename(match["path"])}'),
                        "score": match["score"]
                    } for match in feature_matches
                ]
            }
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/matched/<filename>")
def matched_image(filename):
    return send_from_directory(EXTRACTED_IMAGES_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
