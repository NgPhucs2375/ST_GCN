"""
Gesture Calibration & Template Learning System
Tính toán gesture templates từ webcam recordings cho real-time KNN matching.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time


@dataclass
class GestureTemplate:
    """Template cho 1 gesture: mean + std dev của landmarks qua nhiều recordings."""
    gesture_id: str
    count: int  # số lần đã record
    mean_landmarks: List[List[float]]  # shape: (21, 3)
    std_landmarks: List[List[float]]   # shape: (21, 3)
    timestamp: float


class GestureCalibrator:
    """Quản lý calibration gestures từ camera."""
    
    def __init__(self, save_path: Optional[Path] = None):
        self.save_path = save_path or Path("Gan_nut") / "gesture_templates.json"
        self.templates: Dict[str, GestureTemplate] = {}
        self.current_gesture: Optional[str] = None
        self.current_buffer: List[np.ndarray] = []  # buffer landmarks cho gesture hiện tại
        self.load_templates()
    
    def load_templates(self) -> None:
        """Load gesture templates từ file."""
        if self.save_path.exists():
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for gid, tmpl_dict in data.items():
                    self.templates[gid] = GestureTemplate(
                        gesture_id=gid,
                        count=tmpl_dict["count"],
                        mean_landmarks=tmpl_dict["mean_landmarks"],
                        std_landmarks=tmpl_dict["std_landmarks"],
                        timestamp=tmpl_dict.get("timestamp", time.time())
                    )
                print(f"✅ Loaded {len(self.templates)} gesture templates from {self.save_path}")
            except Exception as e:
                print(f"⚠️ Failed to load templates: {e}")
    
    def save_templates(self) -> None:
        """Save gesture templates to file."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for gid, tmpl in self.templates.items():
            data[gid] = {
                "count": tmpl.count,
                "mean_landmarks": tmpl.mean_landmarks,
                "std_landmarks": tmpl.std_landmarks,
                "timestamp": tmpl.timestamp,
            }
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(self.templates)} gesture templates to {self.save_path}")
    
    def start_calibration(self, gesture_id: str) -> None:
        """Bắt đầu thu thập dữ liệu cho gesture."""
        self.current_gesture = gesture_id
        self.current_buffer = []
        print(f"📹 Started calibration for gesture: {gesture_id}")
    
    def add_landmarks_frame(self, landmarks: np.ndarray) -> None:
        """Thêm 1 frame landmarks (shape: 21x3) vào buffer."""
        if self.current_gesture is None:
            return
        if landmarks is not None and landmarks.shape == (21, 3):
            self.current_buffer.append(landmarks.copy())
    
    def finish_calibration(self, min_frames: int = 15) -> bool:
        """Kết thúc calibration & tính template. Return True nếu thành công."""
        if self.current_gesture is None:
            return False
        
        if len(self.current_buffer) < min_frames:
            print(f"⚠️ Not enough frames ({len(self.current_buffer)}/{min_frames}) for {self.current_gesture}")
            self.current_gesture = None
            self.current_buffer = []
            return False
        
        # Stack all frames: (N, 21, 3)
        frames_array = np.stack(self.current_buffer, axis=0)
        
        # Compute mean & std per landmark
        mean_lm = np.mean(frames_array, axis=0)  # (21, 3)
        std_lm = np.std(frames_array, axis=0)    # (21, 3)
        
        # Create template
        template = GestureTemplate(
            gesture_id=self.current_gesture,
            count=len(self.current_buffer),
            mean_landmarks=mean_lm.tolist(),
            std_landmarks=std_lm.tolist(),
            timestamp=time.time(),
        )
        
        self.templates[self.current_gesture] = template
        print(f"✅ Calibrated {self.current_gesture}: {len(self.current_buffer)} frames")
        
        self.current_gesture = None
        self.current_buffer = []
        return True
    
    def get_calibration_progress(self) -> Tuple[int, int]:
        """Return (current_frames_collected, target_frames)."""
        return len(self.current_buffer), 20
    
    def get_gesture_counts(self) -> Dict[str, int]:
        """Return số frames cho mỗi gesture đã calibrate."""
        return {gid: t.count for gid, t in self.templates.items()}
    
    def distance_to_template(self, landmarks: np.ndarray, gesture_id: str) -> Optional[float]:
        """Compute Euclidean distance từ landmarks đến template."""
        if gesture_id not in self.templates:
            return None
        
        template = self.templates[gesture_id]
        mean_array = np.array(template.mean_landmarks, dtype=np.float32)
        
        if landmarks.shape != mean_array.shape:
            return None
        
        # Normalized distance (chia cho std để normalize by variation)
        std_array = np.array(template.std_landmarks, dtype=np.float32)
        std_array[std_array < 1e-6] = 1.0  # avoid division by zero
        
        dist = np.linalg.norm((landmarks - mean_array) / std_array)
        return float(dist)
    
    def find_closest_gesture(self, landmarks: np.ndarray, 
                           threshold: float = 2.0) -> Optional[Tuple[str, float]]:
        """Find closest gesture template. Return (gesture_id, distance) or None."""
        if not self.templates:
            return None
        
        min_dist = float('inf')
        closest_gesture = None
        
        for gid in self.templates:
            dist = self.distance_to_template(landmarks, gid)
            if dist is not None and dist < min_dist:
                min_dist = dist
                closest_gesture = gid
        
        if closest_gesture and min_dist <= threshold:
            return closest_gesture, min_dist
        
        return None
