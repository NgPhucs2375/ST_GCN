#!/usr/bin/env python3
"""
Extract hand landmarks from AVI videos using MediaPipe.
Simple, robust version.
"""
import json
import sys
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

# Import MediaPipe components
import mediapipe as mp
from mediapipe.tasks import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Setup
VIDEO_DIR = Path("data/videos/videos")
OUTPUT_DIR = Path("data/raw_ipn_new")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def map_class(filename: str) -> Optional[str]:
    """Map video filename to class ID"""
    name = filename.split("__")[0] if "__" in filename else filename.split("_1_")[0]
    
    if "1CM1" in name:
        return "B0A"
    elif "1CM2" in name or "1CM42" in name:
        return "B0B"
    elif "D0X" in name:
        return "D0X"
    elif "G01" in name:
        return "G01"
    elif "G02" in name:
        return "G02"
    elif "G03" in name:
        return "G03"
    elif "G04" in name:
        return "G04"
    elif "G05" in name:
        return "G05"
    elif "G06" in name:
        return "G06"
    elif "G07" in name:
        return "G07"
    elif "G08" in name:
        return "G08"
    elif "G09" in name:
        return "G09"
    elif "G10" in name:
        return "G10"
    elif "G11" in name:
        return "G11"
    return None

def process_video_new_api(video_path: Path, class_id: str) -> Tuple[Optional[str], int]:
    """Process using new MediaPipe Tasks API"""
    try:
        # Initialize detector
        base_options = BaseOptions(model_asset_path="tools/assets/hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"    ERROR: Detector failed: {e}")
        return None, 0
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0
    
    frames_data = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        try:
            # Convert BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect
            results = detector.detect(mp_image)
            
            # Extract landmarks
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                lm_list = results.hand_landmarks[0]
                frame_lms = [[lm.x, lm.y, lm.z] for lm in lm_list]
                frames_data.append(frame_lms)
        except Exception as e:
            pass
    
    cap.release()
    
    # Save if enough frames
    if len(frames_data) >= 10:
        outfile = f"{video_path.stem}__{class_id}.json"
        outpath = OUTPUT_DIR / outfile
        with open(outpath, "w") as f:
            json.dump({"class": class_id, "frames": frames_data}, f)
        print(f"  ✅ {outfile:<50} {len(frames_data):4d} frames")
        return outfile, 1
    else:
        print(f"  ⚠️  {video_path.name:<50} {len(frames_data):4d} frames (too short)")
        return None, 0

def main():
    print("="*80)
    print("📹 EXTRACT LANDMARKS FROM 200 VIDEOS")
    print("="*80)
    
    videos = sorted(list(VIDEO_DIR.glob("*.avi")) + list(VIDEO_DIR.glob("*.mp4")))
    print(f"\nFound {len(videos)} videos\n")
    
    success = 0
    failed = 0
    
    for i, vpath in enumerate(videos, 1):
        sys.stdout.write(f"\r[{i:3d}/{len(videos)}] Processing...")
        sys.stdout.flush()
        
        class_id = map_class(vpath.name)
        if not class_id:
            failed += 1
            continue
        
        # Clear line and print
        sys.stdout.write(f"\r[{i:3d}/{len(videos)}] {vpath.name:<50}")
        sys.stdout.flush()
        
        _, num_saved = process_video_new_api(vpath, class_id)
        if num_saved > 0:
            success += 1
    
    print("\n" + "="*80)
    print(f"✅ EXTRACTION COMPLETE!")
    print(f"   Successful: {success}/{len(videos)} videos")
    print(f"   Output: {OUTPUT_DIR}/")
    print("="*80)

if __name__ == "__main__":
    main()
