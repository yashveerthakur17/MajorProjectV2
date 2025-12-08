# Banana Annotation Guide

This guide explains how to annotate your banana images with bounding boxes for YOLOv8 training.

## 📦 Class Definitions

| Class ID | Class Name | Description | Visual Characteristics |
|----------|------------|-------------|------------------------|
| 0 | Unripe | Green bananas | Mostly or entirely green |
| 1 | Ripe | Ready to eat | Solid yellow, minimal spots |
| 2 | Overripe | Past prime | Yellow with brown spots |
| 3 | Rotten | Inedible | Mostly brown/black |

---

## 🛠️ Annotation Tools

### Option 1: LabelImg (Free, Desktop App)

**Installation:**
```bash
pip install labelImg
```

**Usage:**
1. Run `labelImg` from command line
2. Click "Open Dir" → Select your images folder
3. Click "Change Save Dir" → Select the corresponding labels folder
4. **IMPORTANT**: Click "PascalVOC" button and change to **"YOLO"** format
5. For each image:
   - Press `W` to draw a bounding box
   - Draw a box around each banana
   - Select the class (Unripe/Ripe/Overripe/Rotten)
   - Press `Ctrl+S` to save
   - Press `D` for next image

**Pro Tips:**
- Use hotkeys for speed: W (create box), A/D (prev/next image), Ctrl+S (save)
- Create a `classes.txt` file in your labels folder with:
  ```
  Unripe
  Ripe
  Overripe
  Rotten
  ```

---

### Option 2: Roboflow (Free Tier, Web-Based)

**Website:** [https://roboflow.com](https://roboflow.com)

**Steps:**
1. Create a free account
2. Create a new project → Select "Object Detection"
3. Upload your images
4. Annotate in the browser (draw boxes, assign classes)
5. Export dataset → Select "YOLOv8" format
6. Download and extract to your dataset folder

**Pros:**
- Auto-augmentation options
- Team collaboration
- Cloud storage

---

## 📁 File Structure After Annotation

```
banana_detection/
└── dataset/
    ├── images/
    │   ├── train/
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   └── val/
    │       ├── img101.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── img001.txt    ← Same name as image!
        │   ├── img002.txt
        │   └── ...
        └── val/
            ├── img101.txt
            └── ...
```

**⚠️ IMPORTANT:** Each label file must have the **exact same name** as its image (just different extension).

---

## 📝 YOLO Label Format

Each `.txt` file contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** (0.0 to 1.0):
- `x_center` = box center X / image width
- `y_center` = box center Y / image height  
- `width` = box width / image width
- `height` = box height / image height

**Example:** `img001.txt`
```
1 0.5 0.5 0.3 0.15
0 0.25 0.7 0.2 0.1
```
This means:
- A **Ripe** banana (class 1) centered at (50%, 50%) with size 30% x 15%
- An **Unripe** banana (class 0) at (25%, 70%) with size 20% x 10%

---

## ✅ Annotation Best Practices

1. **Draw tight boxes** - Minimize background in the box
2. **Include partial bananas** - If 50%+ visible, annotate it
3. **Handle bunches** - Draw one box around the entire bunch if bananas are the same ripeness, or individual boxes if different
4. **Variety is key** - Include different:
   - Lighting conditions (natural, artificial, shadows)
   - Backgrounds (kitchen, store, outdoor)
   - Angles (top-down, side view, tilted)
   - Distances (close-up, medium, far)
5. **Minimum 100 images per class** for good results

---

## 🔢 Recommended Dataset Size

| Quality | Images per Class | Total Images |
|---------|------------------|--------------|
| Minimum | 50 | 200 |
| Good | 100-200 | 400-800 |
| Excellent | 500+ | 2000+ |

---

## 🚀 After Annotation

1. Split your data: **80% train, 20% validation**
2. Run `python train.py` to start training
3. Run `python main.py` for live detection

Good luck with your banana detector! 🍌
