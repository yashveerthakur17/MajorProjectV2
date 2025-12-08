"""
Banana Ripeness Detection - Training Script
============================================
Trains a YOLOv8 classification model on your folder-organized dataset.
The trained model will be used to classify detected bananas in live video.

Dataset Structure:
    D:/VSCODE/NEWMAJOR/data/banana/
    - train/ (overripe, ripe, rotten, unripe)
    - valid/ (same subfolders)
    - test/  (same subfolders)

Usage: python train.py
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = Path(r"D:\VSCODE\NEWMAJOR\data\banana")

CONFIG = {
    "model": "yolov8n-cls.pt",   # Classification model
    "epochs": 100,
    "batch_size": 32,
    "image_size": 224,
    "patience": 20,
    "device": 0,
    "workers": 4,
    "project": "banana_detection/runs",
    "name": "classify",
}


def check_gpu():
    """Check GPU availability."""
    print("\n" + "="*60)
    print("GPU CHECK")
    print("="*60)
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA: {torch.version.cuda}")
        return True
    print("✗ No GPU - using CPU")
    return False


def validate_dataset():
    """Validate dataset structure."""
    print("\n" + "="*60)
    print("DATASET")
    print("="*60)
    
    classes = ["overripe", "ripe", "rotten", "unripe"]
    for split in ["train", "valid"]:
        split_path = DATASET_PATH / split
        print(f"\n{split.upper()}:")
        for cls in classes:
            cls_path = split_path / cls
            count = len(list(cls_path.glob("*.*"))) if cls_path.exists() else 0
            print(f"  {cls}: {count} images")
    
    return (DATASET_PATH / "train").exists()


def train():
    """Train the classification model."""
    print("\n" + "="*60)
    print("BANANA RIPENESS CLASSIFIER - TRAINING")
    print("="*60)
    
    has_gpu = check_gpu()
    if not validate_dataset():
        print("✗ Dataset not found!")
        return
    
    print(f"\n✓ Loading {CONFIG['model']}...")
    model = YOLO(CONFIG["model"])
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | GPU: {has_gpu}")
    print("="*60 + "\n")
    
    results = model.train(
        data=str(DATASET_PATH),
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch_size"],
        imgsz=CONFIG["image_size"],
        patience=CONFIG["patience"],
        device=CONFIG["device"] if has_gpu else "cpu",
        workers=CONFIG["workers"],
        project=CONFIG["project"],
        name=CONFIG["name"],
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
    )
    
    # Save best model
    best = Path(CONFIG["project"]) / CONFIG["name"] / "weights" / "best.pt"
    if best.exists():
        import shutil
        out_dir = Path("banana_detection/models")
        out_dir.mkdir(exist_ok=True)
        shutil.copy(best, out_dir / "banana_classifier.pt")
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE!")
        print(f"✓ Model saved: banana_detection/models/banana_classifier.pt")
        print("✓ Run 'python main.py' for live detection!")
        print("="*60)
    
    return results


if __name__ == "__main__":
    train()
