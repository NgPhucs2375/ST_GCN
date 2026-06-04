"""
Gesture Calibration UI - Interactive interface for recording gesture templates.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from gesture_calibrator import GestureCalibrator, GESTURE_ORDER


VI_LABELS: Dict[str, str] = {
    "D0X": "Khong cu chi", "B0A": "Chi 1 ngon", "B0B": "Chi 2 ngon",
    "G01": "Click 1 ngon", "G02": "Click 2 ngon", "G03": "Hat len",
    "G04": "Hat xuong", "G05": "Hat trai", "G06": "Hat phai",
    "G07": "Mo 2 lan", "G08": "Double click 1 ngon",
    "G09": "Double click 2 ngon", "G10": "Phong to", "G11": "Thu nho",
}

GESTURE_ORDER = ["D0X","B0A","B0B","G01","G02","G03","G04","G05","G06","G07","G08","G09","G10","G11"]


class CalibrationUI:
    """UI cho Gesture Calibration mode."""
    
    def __init__(self, calibrator: GestureCalibrator, num_samples: int = 3):
        self.calibrator = calibrator
        self.current_gesture_idx = 0
        self.current_recording_idx = 0
        self.recordings_per_gesture = max(1, num_samples)
        self.target_frames = 20  # mỗi gesture phải record 20 lần
        self.is_recording = False
        self.recording_started_at_frame = -1
        self.last_spacebar_frame = -1
    
    def get_current_gesture(self) -> str:
        """Get gesture ID hiện tại."""
        if self.current_gesture_idx < len(GESTURE_ORDER):
            return GESTURE_ORDER[self.current_gesture_idx]
        return None
    
    def get_current_gesture_vi(self) -> str:
        """Get Vietnamese name của gesture hiện tại."""
        gid = self.get_current_gesture()
        return VI_LABELS.get(gid, gid) if gid else ""
    
    def start_recording_current(self, frame_idx: int) -> None:
        """Bắt đầu record gesture hiện tại."""
        gid = self.get_current_gesture()
        if gid:
            self.calibrator.start_calibration(gid)
            self.is_recording = True
            self.recording_started_at_frame = frame_idx
            print(f"🎬 Recording started for {gid}")
    
    def finish_recording(self) -> bool:
        """Kết thúc record & move to next gesture. Return True nếu hoàn tất."""
        if self.is_recording:
            success = self.calibrator.finish_calibration()
            self.is_recording = False
            
            if success:
                self.current_recording_idx += 1
                self.calibrator.save_templates()
                
                if self.current_recording_idx >= self.recordings_per_gesture:
                    # Đã ghi đủ mẫu cho cử chỉ này, chuyển sang cử chỉ tiếp theo
                    self.current_recording_idx = 0
                    self.current_gesture_idx += 1
                    
                    if self.current_gesture_idx >= len(GESTURE_ORDER):
                        print("✅ Calibration completed!")
                        return True
                    else:
                        print(f"📝 Move to next gesture: {self.get_current_gesture()}")
        
        return False
    
    def is_complete(self) -> bool:
        """Check if all gestures đã calibrate."""
        return self.current_gesture_idx >= len(GESTURE_ORDER)
    
    def draw_overlay(self, frame: np.ndarray, frame_idx: int) -> None:
        """Draw calibration UI lên frame."""
        h, w = frame.shape[:2]
        
        # Background tối để dễ đọc
        cv2.rectangle(frame, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 150), (200, 200, 0), 2)
        
        # Title
        cv2.putText(frame, "GESTURE CALIBRATION MODE", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Current gesture
        gid = self.get_current_gesture()
        if gid:
            gvi = self.get_current_gesture_vi()
            gesture_progress = f"{self.current_gesture_idx + 1}/{len(GESTURE_ORDER)}"
            sample_progress = f"Sample {self.current_recording_idx + 1}/{self.recordings_per_gesture}"
            
            cv2.putText(frame, f"Gesture: {gid} - {gvi} ({gesture_progress})", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (74, 222, 128), 2)
            cv2.putText(frame, sample_progress, (w - 250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN, 2)
        
        # Recording status
        if self.is_recording:
            current_count, target = self.calibrator.get_calibration_progress()
            pct = int(100 * current_count / target)
            cv2.putText(frame, f"REC: {current_count}/{target} frames ({pct}%)", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Progress bar
            bar_w = 300
            bar_filled = int(bar_w * current_count / target)
            cv2.rectangle(frame, (w - 330, 110), (w - 30, 135), (100, 100, 100), -1)
            cv2.rectangle(frame, (w - 330, 110), (w - 330 + bar_filled, 135), (0, 255, 0), -1)
        else:
            cv2.putText(frame, "READY - Press SPACE to start", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        # Instructions at bottom
        cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (100, 100, 150), 1)
        cv2.putText(frame, "SPACE: Start/Stop recording  |  Q: Quit  |  S: Skip gesture", 
                   (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 200, 255), 1)
        total_samples = sum(len(v) for v in self.calibrator.templates.values())
        cv2.putText(frame, f"Saved gestures: {len(self.calibrator.templates)} ({total_samples} samples)", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 200, 255), 1)
    
    def handle_keypress(self, key: int, frame_idx: int) -> str:
        """
        Handle keyboard input.
        Return: "continue", "stop_recording", "finish_calibration", or "quit"
        """
        if key == ord('q') or key == ord('Q'):
            return "quit"
        
        if key == ord(' '):
            # Spacebar: toggle recording
            if self.is_recording and (frame_idx - self.recording_started_at_frame > 5):
                self.finish_recording()
                return "stop_recording"
            else:
                self.start_recording_current(frame_idx)
                return "continue"
        
        if key == ord('s') or key == ord('S'):
            # Skip current gesture
            if self.is_recording:
                self.calibrator.current_gesture = None
                self.calibrator.current_buffer = []
                self.is_recording = False
            
            self.current_recording_idx = 0 # Reset sample count khi skip
            
            self.current_gesture_idx += 1
            if self.current_gesture_idx >= len(GESTURE_ORDER):
                return "finish_calibration"
        
        return "continue"
    
    def get_calibration_summary(self) -> str:
        """Return summary của calibration results."""
        counts = self.calibrator.get_gesture_counts()
        lines = ["📊 Calibration Summary:"]
        for gid in GESTURE_ORDER:
            num_samples = counts.get(gid, 0)
            gvi = VI_LABELS.get(gid, gid)
            status = "✅" if num_samples >= self.recordings_per_gesture else "⏳" if num_samples > 0 else "❌"
            lines.append(f"  {status} {gid:5s} ({gvi:20s}): {num_samples}/{self.recordings_per_gesture} samples")
        return "\n".join(lines)
