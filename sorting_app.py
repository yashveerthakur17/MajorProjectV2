# sorting_app.py - Full tracking-based sorting system with Arduino integration
"""
Automated Fruit Sorting System
==============================
Detects fruits on a conveyor belt, tracks their position,
and triggers Arduino servo to sort rotten fruits.

Usage:
    python sorting_app.py

Requirements:
    - Arduino connected with servo motor
    - Webcam or DroidCam for video input
"""

import cv2
import time
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO

# Try to import serial, handle if not connected
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. Run: pip install pyserial")

# ============================================================================
# CONFIGURATION - ADJUST THESE VALUES
# ============================================================================

SERIAL_PORT = 'COM3'           # Arduino port (check Device Manager)
BAUD_RATE = 9600
ARM_TRIGGER_X = 500            # X position where servo triggers (pixels)
DETECTION_CONF = 0.5           # Min confidence for detection
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Model paths
DETECTOR_MODEL = 'yolov8n.pt'
CLASSIFIER_MODEL = 'banana_detection/models/banana_classifier.pt'

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
        self.fruits: list = []
        self.next_id = 0
        self.match_threshold = 100  # Max distance to match same fruit
    
    def clear(self):
        """Clear all tracked fruits"""
        self.fruits = []
        print("🗑️ Trackers cleared!")
    
    def update(self, detections: list) -> list:
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
        
        # Remove fruits that have moved off screen OR processed more than 2 seconds ago
        self.fruits = [
            f for f in self.fruits 
            if f.x < FRAME_WIDTH + 100 and 
               (not f.triggered or (current_time - f.first_seen) < 2.0)
        ]
        
        return self.fruits
    
    def get_fruits_at_arm(self, arm_x: float) -> list:
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
        self.connected = False
        self.serial = None
        
        if not SERIAL_AVAILABLE:
            print("✗ Serial library not available")
            return
            
        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            self.connected = True
            print(f"✓ Arduino connected on {port}")
        except Exception as e:
            print(f"✗ Arduino not connected: {e}")
            print("  Running in SIMULATION mode (no actual sorting)")
    
    def trigger_sort(self):
        """Send SORT signal to trigger servo"""
        if self.connected and self.serial:
            self.serial.write(b'SORT\n')
            return True
        return False
    
    def send_good(self):
        """Optional: Log good fruit passing"""
        if self.connected and self.serial:
            self.serial.write(b'GOOD\n')
            return True
        return False
    
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
        self.detector = YOLO(DETECTOR_MODEL)
        
        try:
            self.classifier = YOLO(CLASSIFIER_MODEL)
            print("✓ Classification model loaded")
        except:
            print("✗ Classification model not found, using detection only")
            self.classifier = None
        
        # Initialize tracker and Arduino
        self.tracker = FruitTracker()
        self.arduino = ArduinoBridge(SERIAL_PORT, BAUD_RATE)
        
        # Stats
        self.total_processed = 0
        self.total_rotten = 0
        self.total_good = 0
        self.start_time = time.time()
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
    
    def process_frame(self, frame):
        """Process a single frame"""
        h, w = frame.shape[:2]
        
        # Detect objects
        results = self.detector.predict(frame, conf=0.25, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Ensure valid crop
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0 and self.classifier:
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
                    else:
                        # No classifier - use detector class
                        det_class = self.detector.names[int(box.cls[0])]
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'class': det_class,
                            'confidence': float(box.conf[0])
                        })
        
        # Update tracker
        tracked_fruits = self.tracker.update(detections)
        
        # Check for fruits at arm position
        fruits_at_arm = self.tracker.get_fruits_at_arm(ARM_TRIGGER_X)
        
        for fruit in fruits_at_arm:
            self.total_processed += 1
            if fruit.classification.lower() == 'rotten':
                success = self.arduino.trigger_sort()
                self.total_rotten += 1
                status = "SORTED" if success else "SIMULATED"
                print(f"🔴 ROTTEN fruit #{fruit.id} → {status}")
            else:
                self.arduino.send_good()
                self.total_good += 1
                print(f"🟢 GOOD fruit #{fruit.id} ({fruit.classification}) → PASSED")
        
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
            'unripe': (0, 255, 0),     # Green
            'ripe': (0, 255, 255),     # Yellow
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
                cv2.putText(frame, "PROCESSED", (int(fruit.x) - 40, int(fruit.y) + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw stats panel (expanded for FPS)
        cv2.rectangle(frame, (5, 5), (150, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (150, 100), (255, 255, 255), 1)
        
        cv2.putText(frame, f"Processed: {self.total_processed}", (10, 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Rotten: {self.total_rotten}", (10, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, f"Good: {self.total_good}", (10, 58),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw Arduino status
        status = "CONNECTED" if self.arduino.connected else "SIM"
        status_color = (0, 255, 0) if self.arduino.connected else (0, 165, 255)
        cv2.putText(frame, f"Arduino: {status}", (10, 76),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, status_color, 1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 94),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)
        
        # Draw instructions
        cv2.putText(frame, "q=quit | s=screenshot | c=clear | r=reset | +/- arm", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self, camera_id=0):
        """Main loop"""
        global ARM_TRIGGER_X
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n" + "="*60)
        print("       FRUIT SORTING SYSTEM RUNNING")
        print("="*60)
        print(f"  Camera: {camera_id}")
        print(f"  Arm trigger position: X = {ARM_TRIGGER_X}")
        print(f"  Arduino: {'CONNECTED' if self.arduino.connected else 'SIMULATION MODE'}")
        print("-"*60)
        print("  Controls:")
        print("    q     - Quit")
        print("    s     - Save screenshot")
        print("    +/-   - Adjust arm position")
        print("    r     - Reset stats")
        print("    c     - Clear trackers")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Calculate FPS
            self.frame_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start_time = time.time()
            
            # Display
            cv2.imshow('Fruit Sorting System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'sorting_screenshot_{int(time.time())}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📷 Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                ARM_TRIGGER_X = min(FRAME_WIDTH - 50, ARM_TRIGGER_X + 20)
                print(f"Arm position: {ARM_TRIGGER_X}")
            elif key == ord('-'):
                ARM_TRIGGER_X = max(50, ARM_TRIGGER_X - 20)
                print(f"Arm position: {ARM_TRIGGER_X}")
            elif key == ord('r'):
                self.total_processed = 0
                self.total_rotten = 0
                self.total_good = 0
                print("Stats reset!")
            elif key == ord('c'):
                self.tracker.clear()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.arduino.close()
        
        # Print final stats
        elapsed = time.time() - self.start_time
        print("\n" + "="*60)
        print("       SESSION COMPLETE")
        print("="*60)
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Total processed: {self.total_processed}")
        print(f"  Rotten sorted: {self.total_rotten}")
        print(f"  Good passed: {self.total_good}")
        print("="*60)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fruit Sorting System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (0=default, 1=DroidCam)')
    parser.add_argument('--port', type=str, default='COM3', help='Arduino COM port')
    parser.add_argument('--arm', type=int, default=500, help='Arm trigger X position')
    
    args = parser.parse_args()
    
    # Update globals
    SERIAL_PORT = args.port
    ARM_TRIGGER_X = args.arm
    
    # Run system
    system = FruitSortingSystem()
    system.run(camera_id=args.camera)
