"""
Banana Ripeness Detection - Live Detection + Classification
============================================================
Hybrid approach:
1. Detect bananas using pre-trained YOLOv8 (COCO dataset includes bananas)
2. Classify each detected banana using your trained ripeness model

This gives you BOUNDING BOXES + RIPENESS CLASSIFICATION for multiple bananas!

Controls:
    q - Quit
    s - Screenshot

Usage: python main.py
"""

import cv2
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

# Your trained ripeness classifier
CLASSIFIER_PATH = Path("banana_detection/models/banana_classifier.pt")
CLASSIFIER_FALLBACK = Path("banana_detection/runs/classify/weights/best.pt")

# Pre-trained detector (COCO has bananas as class 46)
DETECTOR_MODEL = "yolov8n.pt"  # Will download automatically
BANANA_CLASS_ID = 46  # Banana class in COCO dataset

# Detection settings
DETECTION_CONF = 0.4
CLASSIFICATION_CONF = 0.3

# GPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Colors for ripeness (BGR)
COLORS = {
    "unripe": (0, 255, 0),      # Green
    "ripe": (0, 255, 255),      # Yellow
    "overripe": (0, 165, 255),  # Orange
    "rotten": (0, 0, 255),      # Red
}

WINDOW_NAME = "Banana Ripeness Detection"


def load_models():
    """Load detector and classifier models."""
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    
    # Load detector (pre-trained on COCO)
    print("✓ Loading banana detector (YOLOv8 COCO)...")
    detector = YOLO(DETECTOR_MODEL)
    
    # Load classifier (your trained model)
    if CLASSIFIER_PATH.exists():
        classifier_path = CLASSIFIER_PATH
    elif CLASSIFIER_FALLBACK.exists():
        classifier_path = CLASSIFIER_FALLBACK
    else:
        print("✗ Classifier not found! Run 'python train.py' first.")
        print(f"  Expected: {CLASSIFIER_PATH}")
        return None, None
    
    print(f"✓ Loading ripeness classifier: {classifier_path}")
    classifier = YOLO(str(classifier_path))
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    return detector, classifier


def classify_banana(classifier, banana_img):
    """Classify a cropped banana image."""
    results = classifier.predict(source=banana_img, device=DEVICE, verbose=False)
    
    if results[0].probs is not None:
        probs = results[0].probs
        class_idx = probs.top1
        confidence = probs.top1conf.item()
        class_name = results[0].names[class_idx]
        return class_name, confidence
    
    return "unknown", 0.0


def process_frame(frame, detector, classifier):
    """Detect bananas and classify each one."""
    # Detect objects
    results = detector.predict(
        source=frame,
        conf=DETECTION_CONF,
        device=DEVICE,
        verbose=False,
        classes=[BANANA_CLASS_ID]  # Only detect bananas
    )
    
    detections = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        
        for box in boxes:
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            det_conf = float(box.conf[0].cpu().numpy())
            
            # Crop banana region
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                banana_crop = frame[y1:y2, x1:x2]
                
                # Classify ripeness
                ripeness, cls_conf = classify_banana(classifier, banana_crop)
                
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "ripeness": ripeness,
                    "det_conf": det_conf,
                    "cls_conf": cls_conf,
                })
    
    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes and labels."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        ripeness = det["ripeness"]
        cls_conf = det["cls_conf"]
        
        # Get color
        color = COLORS.get(ripeness.lower(), (255, 255, 255))
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label = f"{ripeness}: {cls_conf:.0%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Label background
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame


def draw_overlay(frame, fps, detection_count):
    """Draw FPS and detection info."""
    h, w = frame.shape[:2]
    
    # Top info bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, f"FPS: {fps:.0f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Bananas: {detection_count}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Legend
    legend_y = h - 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, legend_y), (130, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    y = legend_y + 20
    for name, color in COLORS.items():
        cv2.rectangle(frame, (15, y - 10), (30, y + 5), color, -1)
        cv2.putText(frame, name.capitalize(), (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 20
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (w - 150, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return frame


def main():
    """Main detection loop."""
    print("\n" + "="*60)
    print("BANANA RIPENESS DETECTION - LIVE")
    print("="*60)
    
    # Load models
    detector, classifier = load_models()
    if detector is None or classifier is None:
        return
    
    # Open webcam
    print("✓ Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Starting detection...")
    print("\n" + "="*60)
    print("Press 'q' to quit | Press 's' for screenshot")
    print("="*60 + "\n")
    
    # FPS tracking
    fps = 0
    frame_count = 0
    start_time = time.time()
    screenshot_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and classify
        detections = process_frame(frame, detector, classifier)
        
        # Draw results
        frame = draw_detections(frame, detections)
        
        # FPS calculation
        frame_count += 1
        if time.time() - start_time > 0:
            fps = frame_count / (time.time() - start_time)
        if frame_count > 30:
            frame_count = 0
            start_time = time.time()
        
        # Draw overlay
        frame = draw_overlay(frame, fps, len(detections))
        
        # Show
        cv2.imshow(WINDOW_NAME, frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"screenshot_{screenshot_num:03d}.jpg", frame)
            print(f"✓ Saved screenshot_{screenshot_num:03d}.jpg")
            screenshot_num += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
