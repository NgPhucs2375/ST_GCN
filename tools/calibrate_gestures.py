"""
Gesture Calibration Tool — Record hand landmarks for each gesture
and build a template database for KNN-based recognition.

Usage:
    python tools/calibrate_gestures.py

Flow:
    1. Show gesture name
    2. Press SPACE to start recording
    3. Record ~20 frames of MediaPipe landmarks
    4. Press SPACE to finish
    5. Compute mean & std of landmarks
    6. Save template
    7. Move to next gesture
"""

import os
import json
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
)

GESTURES = [
    ("D0X", "Không cử chỉ (Nghỉ)"),
    ("B0A", "Chỉ 1 ngón tay"),
    ("B0B", "Chỉ 2 ngón tay"),
    ("G01", "Click 1 ngón"),
    ("G02", "Click 2 ngón"),
    ("G03", "Hất lên (Throw up)"),
    ("G04", "Hất xuống (Throw down)"),
    ("G05", "Hất trái (Throw left)"),
    ("G06", "Hất phải (Throw right)"),
    ("G07", "Mở 2 lần (Open twice)"),
    ("G08", "Nhấp đúp 1 ngón"),
    ("G09", "Nhấp đúp 2 ngón"),
    ("G10", "Phóng to (Zoom in)"),
    ("G11", "Thu nhỏ (Zoom out)"),
]

OUTPUT_FILE = PROJECT_ROOT / "data" / "gesture_templates.json"

def ensure_task_model(path: Path) -> Path:
    if path.exists(): return path
    print(f"📥 Downloading hand_landmarker.task...")
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(TASK_MODEL_URL, path)
    return path

def landmarks_to_array(landmarks) -> np.ndarray:
    if isinstance(landmarks, np.ndarray):
        return landmarks.astype(np.float32, copy=False)
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]
    arr = landmarks_to_array(landmarks)
    pts = []
    for p in arr:
        x, y = int(float(p[0]) * w), int(float(p[1]) * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 3, (70, 220, 255), -1)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (90, 160, 255), 1)

def create_detector(task_model_path: Path):
    model_path = ensure_task_model(task_model_path)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.35,
    )
    return HandLandmarker.create_from_options(options)

def detect_landmarks(detector, frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    if result.hand_landmarks:
        return result.hand_landmarks[0]
    return None

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmarks relative to wrist position and palm scale."""
    wrist = landmarks[0:1, :]
    landmarks = landmarks - wrist
    palm = landmarks[9:10, :]
    scale = np.linalg.norm(palm, axis=-1, keepdims=True)
    scale[scale == 0] = 1.0
    return landmarks / scale

def calibrate_gesture(gesture_id: str, gesture_name: str, detector, cap):
    """Calibrate a single gesture: record landmarks, compute mean/std."""
    print(f"\n{'='*60}")
    print(f"📋 Gesture: {gesture_id} — {gesture_name}")
    print(f"{'='*60}")
    print("🎥 Instructions:")
    print("  1. Position your hand in front of camera")
    print("  2. Press SPACE to START recording")
    print("  3. Hold the gesture for ~20 frames")
    print("  4. Press SPACE to STOP recording")
    print("  5. System calculates mean/std and saves template")
    print("  6. Press 'N' for next gesture or 'Q' to quit")

    landmarks_buffer = []
    recording = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect landmarks
        hand_lm = detect_landmarks(detector, frame)

        # Draw UI
        status_text = f"{'🔴 RECORDING' if recording else '⚫ IDLE'} | Frames: {len(landmarks_buffer)}/20"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255 if recording else 100, 255), 2)
        cv2.putText(frame, f"Gesture: {gesture_id} - {gesture_name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        hint_color = (0, 255, 0) if recording else (255, 200, 0)
        cv2.putText(frame, "SPACE=Toggle  Q=Quit  N=Next", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hint_color, 2)

        # Draw landmarks if detected
        if hand_lm is not None:
            draw_landmarks(frame, hand_lm)
            if recording:
                lm_array = landmarks_to_array(hand_lm)
                lm_normalized = normalize_landmarks(lm_array.copy())
                landmarks_buffer.append(lm_normalized)

        cv2.imshow("📺 Calibration Mode", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE = toggle recording
            recording = not recording
            if recording:
                landmarks_buffer = []
                print(f"  ✓ Recording started...")
            else:
                if len(landmarks_buffer) > 0:
                    print(f"  ✓ Recording stopped. Collected {len(landmarks_buffer)} frames.")
                else:
                    print(f"  ⚠️  No landmarks recorded. Try again.")
                    recording = False
                    continue
                break  # Exit calibration for this gesture

        elif key == ord('n') or key == ord('N'):
            print(f"  ⏭️  Skipping gesture {gesture_id}")
            return None  # Skip this gesture

        elif key == ord('q') or key == ord('Q'):
            print(f"  ❌ Calibration cancelled by user")
            return None  # Exit entire calibration

    if len(landmarks_buffer) == 0:
        print(f"  ⚠️  No valid landmarks recorded for {gesture_id}. Skipping.")
        return None

    # Compute statistics
    landmarks_array = np.array(landmarks_buffer)  # (N, 21, 3)
    mean_landmarks = np.mean(landmarks_array, axis=0)  # (21, 3)
    std_landmarks = np.std(landmarks_array, axis=0)  # (21, 3)

    template = {
        "gesture_id": gesture_id,
        "gesture_name": gesture_name,
        "num_samples": len(landmarks_buffer),
        "mean": mean_landmarks.tolist(),
        "std": std_landmarks.tolist(),
    }

    print(f"  ✅ Template saved: {len(landmarks_buffer)} samples, mean shape {mean_landmarks.shape}, std shape {std_landmarks.shape}")
    return template

def main():
    print("\n" + "="*60)
    print("🎮 ST-GCN GESTURE CALIBRATION TOOL")
    print("="*60)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Gestures to calibrate: {len(GESTURES)}")

    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Setup detector
    task_model_path = PROJECT_ROOT / "tools" / "assets" / "hand_landmarker.task"
    detector = create_detector(task_model_path)

    # Calibrate each gesture
    templates = {}
    for gesture_id, gesture_name in GESTURES:
        template = calibrate_gesture(gesture_id, gesture_name, detector, cap)
        if template is not None:
            templates[gesture_id] = template
        else:
            print(f"  ⏭️  Skipped {gesture_id}")

    cv2.destroyAllWindows()
    cap.release()

    # Save templates
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ Calibration complete!")
    print(f"   Calibrated {len(templates)}/{len(GESTURES)} gestures")
    print(f"   Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
