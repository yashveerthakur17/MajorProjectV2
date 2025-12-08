"""
Banana Ripeness Detection - Web Application
============================================
Flask-based web server for real-time banana detection using phone camera.

Usage:
    python web_app.py

Then open http://<your-ip>:5000 on your phone browser.
"""

import base64
import io
import socket
import numpy as np
import torch
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

# ============================================================================
# Configuration
# ============================================================================

CLASSIFIER_PATH = Path("banana_detection/models/banana_classifier.pt")
CLASSIFIER_FALLBACK = Path("banana_detection/runs/classify/weights/best.pt")
DETECTOR_MODEL = "yolov8n.pt"
BANANA_CLASS_ID = 46  # Banana in COCO
FRUIT_CLASSES = [46, 47, 49, 50, 51, 52]  # banana, apple, orange, broccoli, carrot, hot dog (for testing)

DETECTION_CONF = 0.20  # Very low for better detection
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEBUG = True

# Flask app
app = Flask(__name__)
CORS(app)

# Global models (loaded once)
detector = None
classifier = None


def load_models():
    """Load detection and classification models."""
    global detector, classifier
    
    print("Loading models...")
    
    # Detector
    detector = YOLO(DETECTOR_MODEL)
    print(f"✓ Detector loaded: {DETECTOR_MODEL}")
    
    # Classifier
    if CLASSIFIER_PATH.exists():
        classifier = YOLO(str(CLASSIFIER_PATH))
    elif CLASSIFIER_FALLBACK.exists():
        classifier = YOLO(str(CLASSIFIER_FALLBACK))
    else:
        raise FileNotFoundError("Classifier model not found! Run train.py first.")
    
    print(f"✓ Classifier loaded")
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")


def get_local_ip():
    """Get the local IP address for LAN access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def process_image(image_data):
    """Process base64 image - detect objects and classify as bananas."""
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    detections = []
    
    # Detect ALL objects (not just bananas) with low confidence
    results = detector.predict(
        source=img_array,
        conf=DETECTION_CONF,
        device=DEVICE,
        verbose=False
        # No class filter - detect everything
    )
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        if DEBUG:
            detected_names = [detector.names[int(c)] for c in results[0].boxes.cls.cpu().numpy()]
            print(f"Found: {detected_names}")
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            
            if x2c > x1c and y2c > y1c:
                crop = img_array[y1c:y2c, x1c:x2c]
                
                # Classify each detected object
                cls_results = classifier.predict(source=crop, device=DEVICE, verbose=False)
                
                if cls_results[0].probs is not None:
                    probs = cls_results[0].probs
                    class_name = cls_results[0].names[probs.top1]
                    confidence = float(probs.top1conf)
                    
                    # Only keep if classifier is confident it's a banana type
                    if confidence > 0.5:
                        detections.append({
                            "box": [x1, y1, x2, y2],
                            "class": class_name,
                            "confidence": confidence
                        })
    
    # Fallback: If no objects detected, classify entire frame
    # But first check if frame has actual content (not just black/dark)
    if len(detections) == 0:
        # Check average brightness - skip if too dark
        avg_brightness = np.mean(img_array)
        if avg_brightness > 30:  # Only classify if image has content
            cls_results = classifier.predict(source=img_array, device=DEVICE, verbose=False)
            
            if cls_results[0].probs is not None:
                probs = cls_results[0].probs
                class_name = cls_results[0].names[probs.top1]
                confidence = float(probs.top1conf)
                
                # Require higher confidence for full-frame fallback
                if confidence > 0.5:
                    margin = 30
                    detections.append({
                        "box": [margin, margin, w - margin, h - margin],
                        "class": class_name,
                        "confidence": confidence
                    })
    
    return detections


# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Detection API endpoint."""
    try:
        data = request.json
        image_data = data.get("image")
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        detections = process_image(image_data)
        return jsonify({"detections": detections})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    load_models()
    
    ip = get_local_ip()
    port = 5000
    
    print("\n" + "="*60)
    print("BANANA DETECTION WEB SERVER (HTTPS)")
    print("="*60)
    print(f"✓ Local:   https://localhost:{port}")
    print(f"✓ Network: https://{ip}:{port}")
    print("\nOpen the Network URL on your phone!")
    print("Note: Accept the security warning (self-signed certificate)")
    print("="*60 + "\n")
    
    # Run with HTTPS (required for camera access on phones)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, ssl_context='adhoc')
