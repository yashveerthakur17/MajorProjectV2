# evaluate_metrics.py - Calculate F1, Precision, Recall for the classifier
"""
Calculates detailed classification metrics:
- F1 Score (per class and macro/weighted average)
- Precision, Recall
- Confusion Matrix
- Classification Report

Note: R2 Score is for regression models, not classification.
For classification, we use F1, Precision, Recall instead.
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from collections import defaultdict
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = Path("banana_detection/models/banana_classifier.pt")
DATASET_PATH = Path(r"D:\VSCODE\NEWMAJOR\data\banana\valid")
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']  # Alphabetical order

# ============================================================================
# EVALUATE
# ============================================================================

def calculate_metrics():
    print("\n" + "="*60)
    print("   F1 SCORE, PRECISION, RECALL CALCULATION")
    print("="*60 + "\n")
    
    # Load model
    print("Loading model...")
    model = YOLO(str(MODEL_PATH))
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Collect predictions
    print("Running predictions on validation set...")
    
    y_true = []  # Ground truth
    y_pred = []  # Predictions
    
    total_images = 0
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = DATASET_PATH / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_dir} not found")
            continue
            
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        print(f"  Processing {class_name}: {len(images)} images...")
        
        for img_path in images:
            result = model.predict(str(img_path), verbose=False, device=device)
            if result[0].probs is not None:
                pred_class = result[0].probs.top1
                y_true.append(class_idx)
                y_pred.append(pred_class)
                total_images += 1
    
    print(f"\nTotal images evaluated: {total_images}")
    
    # Convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # ========================================================================
    # CALCULATE METRICS
    # ========================================================================
    
    # Confusion Matrix
    num_classes = len(CLASS_NAMES)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    
    # Per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = confusion[i, i]  # True positives
        fp = confusion[:, i].sum() - tp  # False positives
        fn = confusion[i, :].sum() - tp  # False negatives
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        support[i] = confusion[i, :].sum()
    
    # Macro and Weighted averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    # Accuracy
    accuracy = np.trace(confusion) / confusion.sum()
    
    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    
    print("\n" + "="*60)
    print("                    CONFUSION MATRIX")
    print("="*60)
    print("\n                    Predicted")
    print("              ", end="")
    for name in CLASS_NAMES:
        print(f"{name[:6]:>8}", end="")
    print()
    print("           +" + "-"*35)
    
    for i, name in enumerate(CLASS_NAMES):
        print(f"Actual {name[:7]:>7} |", end="")
        for j in range(num_classes):
            print(f"{confusion[i,j]:>8}", end="")
        print()
    
    print("\n" + "="*60)
    print("              CLASSIFICATION REPORT")
    print("="*60)
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*55)
    
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {int(support[i]):>10}")
    
    print("-"*55)
    print(f"{'Macro Avg':<12} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f} {int(support.sum()):>10}")
    print(f"{'Weighted Avg':<12} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {int(support.sum()):>10}")
    
    print("\n" + "="*60)
    print("                    SUMMARY")
    print("="*60)
    
    print(f"""
    ┌────────────────────────────────────────────────────────┐
    │              KEY CLASSIFICATION METRICS                 │
    ├────────────────────────────────────────────────────────┤
    │                                                        │
    │  Accuracy:           {accuracy*100:>6.2f}%                          │
    │                                                        │
    │  Macro F1-Score:     {macro_f1:>6.4f}                           │
    │  Weighted F1-Score:  {weighted_f1:>6.4f}                           │
    │                                                        │
    │  Macro Precision:    {macro_precision:>6.4f}                           │
    │  Weighted Precision: {weighted_precision:>6.4f}                           │
    │                                                        │
    │  Macro Recall:       {macro_recall:>6.4f}                           │
    │  Weighted Recall:    {weighted_recall:>6.4f}                           │
    │                                                        │
    └────────────────────────────────────────────────────────┘
    
    Note: R² Score is for REGRESSION models, not classification.
    For classification, F1-Score is the standard metric!
    """)
    
    # Per-class summary
    print("\n    PER-CLASS F1 SCORES:")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(f1[i] * 30)
        print(f"    {name:<10}: {f1[i]:.4f} {bar}")
    
    print("\n" + "="*60)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    calculate_metrics()
