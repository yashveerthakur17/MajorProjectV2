# Banana Ripeness Detection System
## Complete Project Documentation

---

## 📋 Project Overview

### Objective
Develop an AI-powered computer vision system that detects bananas in real-time video feeds and classifies them into 4 ripeness stages:
- **Unripe** (Green)
- **Ripe** (Yellow)
- **Overripe** (Yellow with brown spots)
- **Rotten** (Mostly brown/black)

### Key Features
- Real-time detection using webcam or phone camera (DroidCam)
- GPU-accelerated inference (NVIDIA RTX 3050)
- Web-based interface accessible over network
- Multi-banana detection and classification
- Live counters for each ripeness category

---

## 🏗️ System Architecture

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ Phone/Webcam│───►│ Web Browser │───►│ Counter Dashboard       │  │
│  │   Camera    │    │ (localhost) │    │ Unripe|Ripe|Overripe|Rot│  │
│  └─────────────┘    └──────┬──────┘    └─────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────────┘
                             │ HTTPS (Base64 JPEG)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FLASK WEB SERVER                               │
│                        (web_app.py)                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Detection Pipeline                         │   │
│  │  ┌──────────────┐    ┌─────────────────┐    ┌─────────────┐  │   │
│  │  │ YOLOv8 COCO  │───►│ Object Cropping │───►│ YOLOv8-cls  │  │   │
│  │  │  (Detector)  │    │                 │    │(Classifier) │  │   │
│  │  └──────────────┘    └─────────────────┘    └──────┬──────┘  │   │
│  └───────────────────────────────────────────────────▼──────────┘   │
│                                                      │               │
│                              JSON Response: [{box, class, conf}]     │
└──────────────────────────────────────────────────────┼──────────────┘
                                                       │
                             ┌─────────────────────────▼─────────────┐
                             │           GPU PROCESSING              │
                             │      NVIDIA RTX 3050 (CUDA)          │
                             │  - PyTorch 2.5.1+cu121               │
                             │  - Ultralytics YOLOv8                │
                             └───────────────────────────────────────┘
```

### Detailed Processing Pipeline

```
Input Frame
     │
     ▼
┌────────────────────────────────────────┐
│ 1. IMAGE PREPROCESSING                  │
│    - Decode Base64 to RGB array        │
│    - Resize to 640px width (web)       │
│    - Normalize for model input         │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│ 2. OBJECT DETECTION (YOLOv8n)          │
│    - Model: yolov8n.pt (COCO)          │
│    - Confidence threshold: 0.20        │
│    - Detects all 80 COCO classes       │
│    - Returns bounding boxes            │
└────────────────┬───────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
   Objects Found     No Objects
        │                 │
        ▼                 ▼
┌───────────────┐  ┌───────────────────┐
│ 3A. CROP EACH │  │ 3B. BRIGHTNESS    │
│    DETECTED   │  │     CHECK         │
│    OBJECT     │  │  avg > 30?        │
└───────┬───────┘  └─────────┬─────────┘
        │                    │
        ▼              Yes   │   No
┌───────────────┐      ┌─────┴─────┐
│ 4. CLASSIFY   │      │           │
│   EACH CROP   │      ▼           ▼
│ YOLOv8n-cls   │ ┌─────────┐ ┌────────┐
│ (Your Model)  │ │Classify │ │Return  │
└───────┬───────┘ │Full     │ │Empty   │
        │         │Frame    │ │Array   │
        ▼         └────┬────┘ └────────┘
┌───────────────┐      │
│ 5. FILTER     │◄─────┘
│  conf > 0.5   │
└───────┬───────┘
        │
        ▼
┌────────────────────────────────────────┐
│ 6. OUTPUT RESPONSE                      │
│    [{                                   │
│      "box": [x1, y1, x2, y2],          │
│      "class": "ripe",                  │
│      "confidence": 0.87                │
│    }, ...]                             │
└────────────────────────────────────────┘
```

---

## 📊 Model Performance Statistics

### Training Results

| Metric | Value |
|--------|-------|
| **Model Architecture** | YOLOv8n-cls (Classification) |
| **Training Epochs** | 32 (Early stopping triggered) |
| **Top-1 Accuracy** | **98.7%** |
| **Top-5 Accuracy** | 100% |
| **Training Time** | ~15 minutes |
| **GPU Used** | NVIDIA RTX 3050 Laptop GPU |
| **GPU Memory Used** | ~500 MB |

### Dataset Statistics

| Class | Training Images | Validation Images | Total |
|-------|-----------------|-------------------|-------|
| Overripe | 2,349 | 229 | 2,578 |
| Ripe | 3,522 | 339 | 3,861 |
| Rotten | 4,020 | 388 | 4,408 |
| Unripe | 1,902 | 167 | 2,069 |
| **TOTAL** | **11,793** | **1,123** | **12,916** |

### Inference Performance

| Metric | Value |
|--------|-------|
| **Preprocessing Time** | ~0.1 ms |
| **Detection Inference** | ~5-10 ms |
| **Classification Inference** | ~2-5 ms |
| **Total Pipeline (per frame)** | ~15-25 ms |
| **Web App FPS** | 6-10 FPS (network dependent) |
| **Local FPS (main.py)** | 25-40 FPS |

---

## 🔧 Technical Stack

### Hardware Requirements
- **GPU**: NVIDIA RTX 3050 (4GB VRAM) or better
- **RAM**: 8GB minimum
- **Storage**: 5GB for models and dataset

### Software Dependencies

```python
# Core ML/DL
torch==2.5.1+cu121          # PyTorch with CUDA 12.1
torchvision==0.20.1+cu121   # Vision utilities
ultralytics>=8.0.0          # YOLOv8 framework

# Computer Vision
opencv-python>=4.8.0        # Image processing
Pillow>=9.0.0               # Image handling
numpy>=1.24.0               # Numerical operations

# Web Framework
flask>=3.0.0                # Web server
flask-cors>=6.0.0           # Cross-origin support
pyopenssl>=24.0.0           # HTTPS support

# Utilities
pyyaml>=6.0                 # Configuration files
matplotlib>=3.7.0           # Visualization
```

### Installation Commands

```bash
# 1. Create virtual environment (optional)
python -m venv venv
.\venv\Scripts\Activate  # Windows

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
pip install ultralytics opencv-python flask flask-cors pyopenssl pyyaml
```

---

## 📁 Project Structure

```
MajorProjectV2/
│
├── banana_detection/
│   ├── dataset/            # (Optional local copy)
│   ├── models/
│   │   └── banana_classifier.pt    # Trained classification model
│   └── runs/
│       └── classify/
│           └── weights/
│               └── best.pt         # Best model checkpoint
│
├── templates/
│   └── index.html          # Web interface
│
├── static/
│   └── style.css           # Styling
│
├── train.py                # Training script
├── main.py                 # Local webcam detection
├── web_app.py              # Network-accessible web server
├── setup_project.py        # Project folder setup
├── requirements.txt        # Dependencies
├── ANNOTATION_GUIDE.md     # How to annotate data
└── PROJECT_DOCUMENTATION.md # This file
```

---

## 💻 Key Code Components

### 1. Training Script (train.py)

```python
# Configuration
DATASET_PATH = Path(r"D:\VSCODE\NEWMAJOR\data\banana")
CONFIG = {
    "model": "yolov8n-cls.pt",   # Pre-trained classification model
    "epochs": 100,
    "batch_size": 32,
    "image_size": 224,
    "patience": 20,              # Early stopping
    "device": 0,                 # GPU index
}

# Training call
model = YOLO(CONFIG["model"])
results = model.train(
    data=str(DATASET_PATH),
    epochs=CONFIG["epochs"],
    batch=CONFIG["batch_size"],
    imgsz=CONFIG["image_size"],
    patience=CONFIG["patience"],
    device=CONFIG["device"],
    pretrained=True,
    verbose=True,
    plots=True,
)
```

### 2. Detection Pipeline (web_app.py)

```python
def process_image(image_data):
    """Process base64 image - detect objects and classify as bananas."""
    # Decode image
    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    detections = []
    
    # Step 1: Detect ALL objects
    results = detector.predict(
        source=img_array,
        conf=DETECTION_CONF,  # 0.20
        device=DEVICE,
        verbose=False
    )
    
    # Step 2: Classify each detected object
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            crop = img_array[y1:y2, x1:x2]
            
            # Classify the crop
            cls_results = classifier.predict(source=crop, device=DEVICE, verbose=False)
            
            if cls_results[0].probs is not None:
                probs = cls_results[0].probs
                class_name = cls_results[0].names[probs.top1]
                confidence = float(probs.top1conf)
                
                # Keep only confident banana detections
                if confidence > 0.5:
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class": class_name,
                        "confidence": confidence
                    })
    
    return detections
```

### 3. Frontend Detection Loop (index.html)

```javascript
async function detectLoop() {
    if (isProcessing) {
        requestAnimationFrame(detectLoop);
        return;
    }
    isProcessing = true;

    try {
        // Capture and resize frame
        const maxWidth = 640;
        const scale = Math.min(1, maxWidth / video.videoWidth);
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth * scale;
        canvas.height = video.videoHeight * scale;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg', 0.6);

        // Send to server
        const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const result = await response.json();
        
        // Draw results
        if (result.detections) {
            drawDetections(result.detections);
            updateCounters(result.detections);
        }
    } catch (err) {
        console.error(err);
    }

    isProcessing = false;
    requestAnimationFrame(detectLoop);
}
```

---

## 🎨 UI Color Coding

| Ripeness Stage | Color (RGB) | Color (Hex) | Visual |
|----------------|-------------|-------------|--------|
| Unripe | (0, 255, 0) | #00FF00 | 🟢 Green |
| Ripe | (255, 255, 0) | #FFFF00 | 🟡 Yellow |
| Overripe | (255, 165, 0) | #FFA500 | 🟠 Orange |
| Rotten | (255, 0, 0) | #FF0000 | 🔴 Red |

---

## 🚀 Usage Instructions

### Option 1: Local Webcam Detection

```bash
python main.py
```
- Uses default webcam
- 25-40 FPS
- Press 'q' to quit, 's' for screenshot

### Option 2: Web Interface (Network Access)

```bash
python web_app.py
```
- Open `https://localhost:5000` in browser
- Select camera from dropdown (including DroidCam)
- Access from phone: `https://<your-ip>:5000`
- Accept security warning (self-signed certificate)

---

## 🎓 Easy Model Training Guide (For Beginners)

This section explains how to train your own banana classification model **without any coding knowledge**. Just organize your photos into folders and run a single command!

### Step 1: Prepare Your Photos

Collect photos of bananas at different ripeness stages. **The more photos, the better!**

**Recommended:**
- Minimum 100 photos per category
- Ideal: 500+ photos per category
- Use different lighting, angles, and backgrounds

### Step 2: Create Folder Structure

Create folders on your computer like this:

```
📁 my_banana_dataset/
│
├── 📁 train/                    ← 80% of your photos go here
│   ├── 📁 unripe/              ← Green bananas
│   │   ├── 🖼️ green1.jpg
│   │   ├── 🖼️ green2.jpg
│   │   └── ... (more green banana photos)
│   │
│   ├── 📁 ripe/                ← Yellow bananas
│   │   ├── 🖼️ yellow1.jpg
│   │   ├── 🖼️ yellow2.jpg
│   │   └── ... (more yellow banana photos)
│   │
│   ├── 📁 overripe/            ← Yellow with brown spots
│   │   ├── 🖼️ spotty1.jpg
│   │   ├── 🖼️ spotty2.jpg
│   │   └── ... (more overripe banana photos)
│   │
│   └── 📁 rotten/              ← Brown/black bananas
│       ├── 🖼️ brown1.jpg
│       ├── 🖼️ brown2.jpg
│       └── ... (more rotten banana photos)
│
└── 📁 valid/                    ← 20% of your photos go here (for testing)
    ├── 📁 unripe/
    ├── 📁 ripe/
    ├── 📁 overripe/
    └── 📁 rotten/
```

### Step 3: Sorting Your Photos

**Simple Rule:**
| Banana Appearance | Put in Folder |
|-------------------|---------------|
| 🟢 Green, hard, not ready to eat | `unripe/` |
| 🟡 Yellow, perfect for eating | `ripe/` |
| 🟡🟤 Yellow with brown spots | `overripe/` |
| 🟤 Mostly brown or black | `rotten/` |

**Tips for Better Results:**
- ✅ One banana per photo is best
- ✅ Include close-up shots
- ✅ Use photos from different angles
- ✅ Mix indoor and outdoor lighting
- ❌ Avoid blurry photos
- ❌ Avoid photos with multiple different stages

### Step 4: Update the Training Script

Open `train.py` and change ONE line - the path to your dataset:

```python
# Find this line (around line 25):
DATASET_PATH = Path(r"D:\VSCODE\NEWMAJOR\data\banana")

# Change it to YOUR folder path:
DATASET_PATH = Path(r"C:\Users\YourName\my_banana_dataset")
```

**⚠️ Important:** 
- Use the FULL path to your folder
- Use `r"..."` format (with the r before the quotes)
- Use backslashes `\` on Windows

### Step 5: Run Training

Open a command prompt/terminal in your project folder and type:

```bash
python train.py
```

**What happens:**
1. ✅ Script loads your photos automatically
2. ✅ Counts photos in each category
3. ✅ Starts training (you'll see progress)
4. ✅ Saves the trained model when done

**Training takes:** 15-30 minutes (depending on your GPU and dataset size)

### Step 6: Training Output

When training finishes, you'll see:
- **Accuracy score** (higher is better, aim for 90%+)
- **Model saved to:** `banana_detection/models/banana_classifier.pt`

The system will automatically use your new model!

### Common Problems & Solutions

| Problem | Solution |
|---------|----------|
| "Dataset not found" | Check your folder path is correct |
| "No images in folder" | Make sure photos are .jpg or .png format |
| Training is slow | Use a GPU (NVIDIA graphics card) |
| Low accuracy | Add more photos, ensure correct labeling |
| "CUDA out of memory" | Reduce batch_size in train.py (change 32 to 16) |

### Quick Reference: File Formats Supported

- ✅ `.jpg` / `.jpeg`
- ✅ `.png`
- ✅ `.bmp`
- ✅ `.webp`

### Example Folder After Setup

```
my_banana_dataset/
├── train/
│   ├── unripe/       → 150 photos ✅
│   ├── ripe/         → 200 photos ✅
│   ├── overripe/     → 180 photos ✅
│   └── rotten/       → 170 photos ✅
└── valid/
    ├── unripe/       → 30 photos ✅
    ├── ripe/         → 40 photos ✅
    ├── overripe/     → 36 photos ✅
    └── rotten/       → 34 photos ✅

Total: 840 photos → Ready to train! 🚀
```

### After Training: Test Your Model

1. Run: `python web_app.py`
2. Open: `https://localhost:5000`
3. Point camera at bananas
4. See your model in action!

---

## 📈 PowerPoint Presentation Outline

### Slide 1: Title
- **Banana Ripeness Detection using Deep Learning**
- Your Name, Institution, Date

### Slide 2: Problem Statement
- Manual inspection is time-consuming
- Inconsistent human judgment
- Need for automated quality control

### Slide 3: Objectives
- Detect bananas in real-time video
- Classify into 4 ripeness categories
- Provide web-accessible interface

### Slide 4: System Architecture
- Include the flow diagram from this document
- Highlight two-stage approach (Detection → Classification)

### Slide 5: Dataset
- Include dataset statistics table
- Show sample images from each class
- Source: Roboflow (mention if applicable)

### Slide 6: Model Architecture
- YOLOv8n-cls for classification
- Pre-trained on ImageNet, fine-tuned on banana dataset
- Transfer learning approach

### Slide 7: Training Process
- Hardware used (RTX 3050)
- Training configuration (epochs, batch size, etc.)
- Early stopping mechanism

### Slide 8: Results
- **98.7% accuracy**
- Confusion matrix (from training outputs)
- Loss curves (from training plots)

### Slide 9: Demo Screenshots
- Web interface with detection
- Multiple bananas with different classifications
- Counter dashboard

### Slide 10: Technical Stack
- Python, PyTorch, YOLOv8, Flask
- CUDA/GPU acceleration
- Web technologies (HTML, CSS, JavaScript)

### Slide 11: Future Improvements
- Train custom YOLOv8 detection model with bounding boxes
- Mobile app development
- Integration with conveyor belt systems

### Slide 12: Conclusion
- Successfully achieved 98.7% accuracy
- Real-time detection at 6-10 FPS over network
- Practical application for fruit quality assessment

### Slide 13: References
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Dataset source (if applicable)
- Research papers on fruit ripeness detection

---

## 📑 Report Structure

### Chapter 1: Introduction
1.1 Background
1.2 Problem Statement
1.3 Objectives
1.4 Scope and Limitations

### Chapter 2: Literature Review
2.1 Object Detection Methods
2.2 YOLO Architecture Evolution
2.3 Fruit Ripeness Detection Studies
2.4 Transfer Learning Approaches

### Chapter 3: Methodology
3.1 Dataset Collection and Preparation
3.2 Model Selection (YOLOv8)
3.3 Training Procedure
3.4 Evaluation Metrics
3.5 Web Application Development

### Chapter 4: Implementation
4.1 Hardware Setup
4.2 Software Environment
4.3 Training Pipeline (include code)
4.4 Inference Pipeline (include code)
4.5 Web Interface Design

### Chapter 5: Results and Analysis
5.1 Training Results (accuracy, loss curves)
5.2 Confusion Matrix Analysis
5.3 Inference Speed Benchmarks
5.4 Real-world Testing

### Chapter 6: Conclusion and Future Work
6.1 Summary of Achievements
6.2 Limitations
6.3 Future Improvements
6.4 Practical Applications

### Appendices
A. Full Source Code
B. Dataset Statistics
C. Training Logs
D. Installation Guide

---

## 🔮 Future Improvements

1. **Train Custom Detection Model**: Create bounding box annotations and train YOLOv8 detection model specifically for bananas
2. **Mobile App**: Develop native Android/iOS apps for offline detection
3. **Conveyor Integration**: Deploy on industrial conveyor belt systems for automated sorting
4. **Edge Deployment**: Optimize for Raspberry Pi or Jetson Nano
5. **Multi-Fruit Support**: Extend to other fruits (apples, oranges, etc.)

---

## 📞 Contact / Credits

- **Developer**: [Your Name]
- **Institution**: [Your Institution]
- **Date**: December 2025
- **Technologies**: Python, PyTorch, YOLOv8, Flask, OpenCV

---

*This documentation was generated for the Banana Ripeness Detection project.*
*Last Updated: December 8, 2025*
