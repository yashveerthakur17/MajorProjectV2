# Arduino Fruit Sorting Integration Plan

## 🎯 Updated Project Scope

**The System:**
1. Fruits move on a **conveyor belt in a line** under a camera
2. Camera detects and **tracks each fruit** with position
3. If fruit is **ROTTEN** → mark it and add to sorting queue
4. When fruit reaches **servo arm position** → trigger Arduino
5. Handle **multiple consecutive rotten** fruits with continuous payloads
6. **Reset quickly** for next fruit

---

## 🏗️ System Flow with Tracking

```
CAMERA ZONE                                      SORTING ZONE
     │                                                │
     ▼                                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONVEYOR BELT                                                  │
│                                                                 │
│  [🍌1]──────►[🍌2]──────►[🍌3]──────►[📍ARM]──────►             │
│   ↓           ↓           ↓            ↓                        │
│  DETECT     DETECT      DETECT    TRIGGER IF ROTTEN             │
│  +TRACK     +TRACK      +TRACK                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
    ┌──────────┐                 ┌──────────────┐
    │ TRACKING │                 │   ARDUINO    │
    │  QUEUE   │ ──────────────► │   TRIGGER    │
    │          │  When X pos     │              │
    │ [🍌1: R] │  reaches arm    │  Servo push  │
    │ [🍌2: G] │                 │  + reset     │
    │ [🍌3: R] │                 └──────────────┘
    └──────────┘
```

---

## 📋 Core Logic

### Tracking Queue System

```python
class FruitTracker:
    """
    Each fruit has:
    - ID (unique)
    - Position X (horizontal position in frame)
    - Classification (rotten/good)
    - Timestamp (when detected)
    """
    
    fruits = []  # Active tracked fruits
    
    # When X position exceeds ARM_TRIGGER_X → send to Arduino
    ARM_TRIGGER_X = 500  # pixels from left edge (configurable)
```

### Detection + Tracking Flow

```
Frame N:
  └─► Detect fruits in frame
  └─► For each detection:
        ├─► New fruit? → Add to tracking queue with classification
        └─► Existing fruit? → Update position (X coordinate)
  └─► Check queue: Any fruit X > ARM_TRIGGER_X?
        ├─► Yes + Rotten → Send "ROTTEN" to Arduino
        ├─► Yes + Good → Send nothing (or "GOOD" for logging)
        └─► Remove from queue (passed arm)
```

---

## 💻 Python Code: Tracking Sorter

```python
# sorting_app.py - Full tracking-based sorting system

import cv2
import time
import serial
import numpy as np
from collections import deque
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

SERIAL_PORT = 'COM3'           # Arduino port
BAUD_RATE = 9600
ARM_TRIGGER_X = 500            # X position where servo triggers (pixels)
DETECTION_CONF = 0.5           # Min confidence for detection
CONVEYOR_SPEED = 50            # Estimated pixels/second (calibrate this!)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrackedFruit:
    id: int
    x: float                   # Current X position
    y: float                   # Current Y position  
    classification: str        # 'rotten', 'ripe', 'overripe', 'unripe'
    confidence: float
    first_seen: float          # Timestamp
    triggered: bool = False    # Already sent to Arduino?

# ============================================================================
# FRUIT TRACKER
# ============================================================================

class FruitTracker:
    def __init__(self):
        self.fruits: list[TrackedFruit] = []
        self.next_id = 0
        self.match_threshold = 100  # Max distance to match same fruit
    
    def update(self, detections: list[dict]) -> list[TrackedFruit]:
        """Update tracking with new detections"""
        current_time = time.time()
        matched_ids = set()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cx = (x1 + x2) / 2  # Center X
            cy = (y1 + y2) / 2  # Center Y
            
            # Try to match with existing fruit (by proximity)
            matched = False
            for fruit in self.fruits:
                if fruit.id in matched_ids:
                    continue
                    
                dist = abs(fruit.x - cx) + abs(fruit.y - cy)
                if dist < self.match_threshold:
                    # Update existing fruit position
                    fruit.x = cx
                    fruit.y = cy
                    matched_ids.add(fruit.id)
                    matched = True
                    break
            
            if not matched:
                # New fruit detected
                self.fruits.append(TrackedFruit(
                    id=self.next_id,
                    x=cx, y=cy,
                    classification=det['class'],
                    confidence=det['confidence'],
                    first_seen=current_time
                ))
                self.next_id += 1
        
        # Remove fruits that have moved off screen (X > frame width + margin)
        self.fruits = [f for f in self.fruits if f.x < 800]
        
        return self.fruits
    
    def get_fruits_at_arm(self, arm_x: float) -> list[TrackedFruit]:
        """Get fruits that have reached the arm position"""
        ready = []
        for fruit in self.fruits:
            if fruit.x >= arm_x and not fruit.triggered:
                fruit.triggered = True
                ready.append(fruit)
        return ready

# ============================================================================
# ARDUINO BRIDGE
# ============================================================================

class ArduinoBridge:
    def __init__(self, port: str, baudrate: int = 9600):
        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            self.connected = True
            print(f"✓ Arduino connected on {port}")
        except Exception as e:
            print(f"✗ Arduino not connected: {e}")
            self.connected = False
            self.serial = None
    
    def trigger_sort(self):
        """Send ROTTEN signal to trigger servo"""
        if self.connected and self.serial:
            self.serial.write(b'SORT\n')
            print("→ Sent SORT command to Arduino")
    
    def send_good(self):
        """Optional: Log good fruit passing"""
        if self.connected and self.serial:
            self.serial.write(b'GOOD\n')
    
    def close(self):
        if self.serial:
            self.serial.close()

# ============================================================================
# MAIN SORTING APPLICATION
# ============================================================================

class FruitSortingSystem:
    def __init__(self):
        # Load models
        print("Loading models...")
        self.detector = YOLO('yolov8n.pt')
        self.classifier = YOLO('banana_detection/models/banana_classifier.pt')
        
        # Initialize tracker and Arduino
        self.tracker = FruitTracker()
        self.arduino = ArduinoBridge(SERIAL_PORT, BAUD_RATE)
        
        # Stats
        self.total_processed = 0
        self.total_rotten = 0
        self.total_good = 0
    
    def process_frame(self, frame):
        """Process a single frame"""
        h, w = frame.shape[:2]
        
        # Detect objects
        results = self.detector.predict(frame, conf=0.3, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # Classify
                    cls_result = self.classifier.predict(crop, verbose=False)
                    if cls_result[0].probs is not None:
                        probs = cls_result[0].probs
                        class_name = cls_result[0].names[probs.top1]
                        confidence = float(probs.top1conf)
                        
                        if confidence > DETECTION_CONF:
                            detections.append({
                                'box': [x1, y1, x2, y2],
                                'class': class_name,
                                'confidence': confidence
                            })
        
        # Update tracker
        tracked_fruits = self.tracker.update(detections)
        
        # Check for fruits at arm position
        fruits_at_arm = self.tracker.get_fruits_at_arm(ARM_TRIGGER_X)
        
        for fruit in fruits_at_arm:
            self.total_processed += 1
            if fruit.classification.lower() == 'rotten':
                self.arduino.trigger_sort()
                self.total_rotten += 1
                print(f"🔴 ROTTEN fruit #{fruit.id} → SORTED")
            else:
                self.arduino.send_good()
                self.total_good += 1
                print(f"🟢 GOOD fruit #{fruit.id} → PASSED")
        
        # Draw visualization
        self.draw_visualization(frame, tracked_fruits)
        
        return frame
    
    def draw_visualization(self, frame, fruits):
        """Draw tracking visualization"""
        h, w = frame.shape[:2]
        
        # Draw arm trigger line
        cv2.line(frame, (ARM_TRIGGER_X, 0), (ARM_TRIGGER_X, h), (0, 0, 255), 2)
        cv2.putText(frame, "ARM", (ARM_TRIGGER_X + 5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Color map
        colors = {
            'unripe': (0, 255, 0),    # Green
            'ripe': (0, 255, 255),    # Yellow
            'overripe': (0, 165, 255), # Orange
            'rotten': (0, 0, 255)      # Red
        }
        
        # Draw tracked fruits
        for fruit in fruits:
            color = colors.get(fruit.classification.lower(), (255, 255, 255))
            
            # Draw circle at fruit center
            cv2.circle(frame, (int(fruit.x), int(fruit.y)), 30, color, 3)
            
            # Draw label
            label = f"#{fruit.id} {fruit.classification}"
            cv2.putText(frame, label, (int(fruit.x) - 50, int(fruit.y) - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw triggered indicator
            if fruit.triggered:
                cv2.putText(frame, "SORTED!", (int(fruit.x) - 30, int(fruit.y) + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw stats
        cv2.putText(frame, f"Processed: {self.total_processed}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Rotten: {self.total_rotten}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Good: {self.total_good}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def run(self, camera_id=0):
        """Main loop"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*50)
        print("FRUIT SORTING SYSTEM RUNNING")
        print("="*50)
        print(f"Arm trigger position: X = {ARM_TRIGGER_X}")
        print("Press 'q' to quit, 's' to save screenshot")
        print("="*50 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('Fruit Sorting System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'screenshot_{int(time.time())}.jpg', frame)
        
        cap.release()
        cv2.destroyAllWindows()
        self.arduino.close()

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    system = FruitSortingSystem()
    system.run(camera_id=0)  # Use 0 for default webcam, or 1 for DroidCam
```

---

## 🔧 Arduino Code (Updated)

```cpp
// fruit_sorter.ino - Handles rapid consecutive triggers

#include <Servo.h>

Servo sorterServo;

const int SERVO_PIN = 9;
const int LED_ROTTEN = 10;  // Red LED
const int LED_GOOD = 11;    // Green LED

const int REST_ANGLE = 0;
const int PUSH_ANGLE = 90;
const int PUSH_DURATION = 300;  // ms - time to hold push position

void setup() {
    Serial.begin(9600);
    
    sorterServo.attach(SERVO_PIN);
    sorterServo.write(REST_ANGLE);
    
    pinMode(LED_ROTTEN, OUTPUT);
    pinMode(LED_GOOD, OUTPUT);
    
    Serial.println("READY");
}

void loop() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        
        if (command == "SORT") {
            // Trigger sorting mechanism
            digitalWrite(LED_ROTTEN, HIGH);
            
            sorterServo.write(PUSH_ANGLE);
            delay(PUSH_DURATION);
            sorterServo.write(REST_ANGLE);
            
            digitalWrite(LED_ROTTEN, LOW);
            Serial.println("SORTED");
        }
        else if (command == "GOOD") {
            // Just blink green LED
            digitalWrite(LED_GOOD, HIGH);
            delay(100);
            digitalWrite(LED_GOOD, LOW);
            Serial.println("PASSED");
        }
    }
}
```

---

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ARM_TRIGGER_X` | 500 | X pixel position where arm triggers |
| `SERIAL_PORT` | COM3 | Arduino COM port |
| `DETECTION_CONF` | 0.5 | Min confidence to track fruit |
| `PUSH_DURATION` | 300ms | How long servo stays extended |
| `match_threshold` | 100px | Max distance to match same fruit |

**To calibrate:** Run the system and adjust `ARM_TRIGGER_X` based on where you want the arm to trigger.

---

## 🎬 How It Handles Multiple Consecutive Rotten Fruits

```
Time T1: [🍌R] detected at X=100
Time T2: [🍌R] moves to X=300, [🍌R2] detected at X=100
Time T3: [🍌R] reaches X=500 (ARM) → TRIGGER! 
Time T4: [🍌R2] reaches X=500 (ARM) → TRIGGER AGAIN!
         (Continuous sorting for consecutive rotten)
```

The queue system ensures each fruit triggers independently when it reaches the arm position.

---

## 📊 Visualization Output

```
┌────────────────────────────────────────────────────────────┐
│  Processed: 15                           │ARM│            │
│  Rotten: 4                              ┌┴──┴┐            │
│  Good: 11                               │    │            │
│                                         │    │            │
│  ┌──────┐        ┌──────┐               │    │            │
│  │ #3   │───────►│ #2   │──────────────►│    │            │
│  │ROTTEN│        │ RIPE │               │    │            │
│  └──────┘        └──────┘               └────┘            │
│                                                           │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 To Run

1. **Install dependencies:**
   ```bash
   pip install pyserial
   ```

2. **Upload Arduino code** via Arduino IDE

3. **Run Python:**
   ```bash
   python sorting_app.py
   ```

4. **Calibrate** `ARM_TRIGGER_X` based on your setup

---

*Branch: `feature/arduinoconnect`*
*Updated: December 9, 2025*
