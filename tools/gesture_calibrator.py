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

# Độ dài chuỗi cố định cho mỗi mẫu cử chỉ động/tĩnh
SEQUENCE_LENGTH = 10

@dataclass
class GestureTemplate:
    """Template cho 1 gesture: một chuỗi landmarks đại diện."""
    gesture_id: str
    count: int  # số lần đã record
    sequence_landmarks: List[List[List[float]]] # shape: (SEQUENCE_LENGTH, 21, 3)
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
                    # Tương thích ngược với template cũ (mean/std)
                    if "mean_landmarks" in tmpl_dict:
                        print(f"⚠️  Skipping old format template for {gid}. Please re-calibrate.")
                        continue
                    self.templates[gid] = GestureTemplate(
                        gesture_id=gid,
                        count=tmpl_dict["count"],
                        sequence_landmarks=tmpl_dict["sequence_landmarks"],
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
                "sequence_landmarks": tmpl.sequence_landmarks,
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
    
    def finish_calibration(self) -> bool:
        """Kết thúc calibration & tính template. Return True nếu thành công."""
        if self.current_gesture is None:
            return False
        
        if len(self.current_buffer) < SEQUENCE_LENGTH:
            print(f"⚠️ Not enough frames ({len(self.current_buffer)}/{SEQUENCE_LENGTH}) for {self.current_gesture}. Please hold gesture longer.")
            self.current_gesture = None
            self.current_buffer = []
            return False
        
        # Lấy SEQUENCE_LENGTH frames cuối cùng từ buffer
        sequence_frames = self.current_buffer[-SEQUENCE_LENGTH:]
        frames_array = np.stack(sequence_frames, axis=0) # (SEQUENCE_LENGTH, 21, 3)

        # Chuẩn hóa chuỗi: center và scale dựa trên frame ĐẦU TIÊN của chuỗi
        wrist = frames_array[0:1, 0:1, :] # (1, 1, 3)
        frames_array = frames_array - wrist

        palm = frames_array[0:1, 9:10, :] # (1, 1, 3)
        scale = np.linalg.norm(palm, axis=-1, keepdims=True)
        scale[scale < 1e-6] = 1.0
        normalized_array = frames_array / scale
        
        # Create template
        template = GestureTemplate(
            gesture_id=self.current_gesture,
            count=len(self.current_buffer),
            sequence_landmarks=normalized_array.tolist(),
            timestamp=time.time(),
        )
        
        self.templates[self.current_gesture] = template
        print(f"✅ Calibrated {self.current_gesture}: captured a {SEQUENCE_LENGTH}-frame sequence.")
        
        self.current_gesture = None
        self.current_buffer = []
        return True
    
    def get_calibration_progress(self) -> Tuple[int, int]:
        """Return (current_frames_collected, target_frames)."""
        return len(self.current_buffer), 20 # Vẫn giữ target 20 để có buffer rộng
    
    def get_gesture_counts(self) -> Dict[str, int]:
        """Return số frames cho mỗi gesture đã calibrate."""
        return {gid: t.count for gid, t in self.templates.items()}
