import os
# luồng CPU
os.environ["OMP_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)

import argparse
import json
import sys
import time
import urllib.request
import subprocess
import threading
from collections import deque
from math import pi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_OK = True
except ImportError:
    PYAUTOGUI_OK = False
    print("⚠️  pyautogui chưa cài. Chạy: pip install pyautogui")

# Allow running as "python tools/demo_webcam.py" from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stgcn import STGCN, build_hand_edge_index
from gesture_calibrator import GestureCalibrator
from knn_matcher import KNNGestureMatcher
from calibration_ui import CalibrationUI

TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
)

VI_LABELS: Dict[str, str] = {
    "D0X": "Khong cu chi", "B0A": "Chi 1 ngon", "B0B": "Chi 2 ngon",
    "G01": "Click 1 ngon", "G02": "Click 2 ngon", "G03": "Hat len",
    "G04": "Hat xuong", "G05": "Hat trai", "G06": "Hat phai",
    "G07": "Mo 2 lan", "G08": "Double click 1 ngon",
    "G09": "Double click 2 ngon", "G10": "Phong to", "G11": "Thu nho",
}

GESTURE_ORDER = ["D0X","B0A","B0B","G01","G02","G03","G04","G05","G06","G07","G08","G09","G10","G11"]

# Cử chỉ tĩnh, sẽ được ưu tiên bởi KNN trong hybrid mode
STATIC_GESTURES = {"B0A", "B0B", "D0X"}

# Màu sắc UI
C_CYAN      = (255, 210, 60)     # vàng nhấn
C_MUTED     = (100, 116, 139)    # chữ mờ
C_GREEN     = (74, 222, 128)     # xanh lá (đã gán)

# ════════════════════════════════════════════════════════════════════════════
#  GESTURE CONFIG
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "Gan_nut" / "gesture_config.json"

def load_gesture_config(path: Path) -> Dict[str, str]:
    if not path.exists():
        config = {g: "" for g in GESTURE_ORDER}
        save_gesture_config(path, config)
        return config
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_gesture_config(path: Path, config: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def send_key(key: str) -> None:
    if not PYAUTOGUI_OK or not key:
        return
    try:
        pyautogui.press(key)
        # Small console log to help debug focus/permission issues
        print(f"→ send_key: pressed '{key}'")
    except Exception as e:
        print(f"⚠️  Không gửi được phím '{key}': {e}")


def send_action(action: str, frame_idx: int = 0) -> None:
    if not action:
        return
    action = action.strip()
    print(f"→ send_action: {action}")
    global MOUSE_FOLLOW_ENABLED, MOUSE_POS_EMA

    # support repeat: and instant: prefixes
    if action.lower().startswith("repeat:"):
        try:
            rest = action[7:]
            n_str, cmd = rest.split(":", 1)
            n = int(n_str)
            def runner():
                for _ in range(n):
                    send_action(cmd, frame_idx)
                    time.sleep(0.08)
            threading.Thread(target=runner, daemon=True).start()
        except Exception:
            pass
        return

    # instant: prefix - caller can bypass debounce/stability
    instant = False
    if action.lower().startswith("instant:"):
        instant = True
        action = action[8:]

    # toggle mouse follow mode (with debounce)
    if action == "mouse:follow_on":
        toggle_mouse_follow(True, frame_idx)
        return
    if action == "mouse:follow_off":
        toggle_mouse_follow(False, frame_idx)
        return

    # type text
    if action.lower().startswith("type:"):
        text = action[5:]
        try:
            pyautogui.typewrite(text)
            print(f"→ typed: {text}")
        except Exception as e:
            print(f"⚠️ type failed: {e}")
        return

    if action.lower().startswith("run:"):
        cmd = action[4:].strip()
        if cmd:
            subprocess.Popen(cmd, shell=True)
        return

    if action == "hotkey:ctrl+plus":
        pyautogui.hotkey('ctrl', 'shift', '=')
        return

    if action == "hotkey:ctrl+minus":
        pyautogui.hotkey('ctrl', '-')
        return

    if action.lower().startswith("hotkey:"):
        keys = action[7:].split("+")
        pyautogui.hotkey(*keys)
        return

    if action == "mouse:left":
        pyautogui.click(button="left")
    elif action == "mouse:right":
        pyautogui.click(button="right")
    elif action == "mouse:middle":
        pyautogui.click(button="middle")
    elif action == "mouse:double":
        pyautogui.doubleClick()
    elif action == "mouse:scroll_up":
        pyautogui.scroll(3)
    elif action == "mouse:scroll_down":
        pyautogui.scroll(-3)
    else:
        pyautogui.press(action)


# Global state for mouse follow mode
MOUSE_FOLLOW_ENABLED = False
MOUSE_POS_EMA = None
# Mutex groups last-used timestamps
MUTEX_LAST = {}
# Motion prediction for mouse follow
MOTION_HISTORY = deque(maxlen=5)  # Track fingertip positions for velocity estimation
FOLLOW_TOGGLE_FRAME = -999  # Track when follow was last toggled to debounce
FOLLOW_TOGGLE_COOLDOWN = 10  # Frames to wait before accepting new toggle



# ════════════════════════════════════════════════════════════════════════════
#  MODEL & MEDIAPIPE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def ensure_task_model(path: Path) -> Path:
    if path.exists(): return path
    print(f"📥 Đang tải hand_landmarker.task...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TASK_MODEL_URL, path)
    return path

def load_label_map(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return {v: k for k, v in class_to_idx.items()}

def index_to_label(idx_to_class_map: Dict[int, str], num_classes: int) -> List[str]:
    labels = ["" for _ in range(num_classes)]
    for idx, label in idx_to_class_map.items():
        if 0 <= idx < num_classes: labels[idx] = label
    for i, label in enumerate(labels):
        if not label: labels[i] = f"class_{i}"
    return labels

def normalize_frames(frames: np.ndarray) -> np.ndarray:
    wrist = frames[:, 0:1, :]
    frames = frames - wrist
    palm  = frames[:, 9:10, :]
    scale = np.linalg.norm(palm, axis=-1, keepdims=True)
    scale[scale == 0] = 1.0
    return frames / scale

def add_velocity(frames: np.ndarray) -> np.ndarray:
    velocity = np.diff(frames, axis=0, prepend=frames[:1])
    return np.concatenate([frames, velocity], axis=-1)

def add_acceleration(frames: np.ndarray) -> np.ndarray:
    velocity     = np.diff(frames, axis=0, prepend=frames[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    return np.concatenate([frames, velocity, acceleration], axis=-1)

def infer_in_channels_from_state(state: Dict[str, torch.Tensor]) -> int:
    return int(state["data_bn.weight"].numel()) // 21

def infer_feature_config(in_channels: int) -> Tuple[bool, bool, bool]:
    if in_channels == 2: return False, False, False
    if in_channels == 3: return True,  False, False
    if in_channels == 4: return False, True,  False
    if in_channels == 6: return True,  True,  False
    if in_channels == 9: return True,  True,  True
    raise ValueError(f"Unsupported in_channels={in_channels}.")

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists(): return p
    return None

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def format_label(label: str, overlay_lang: str) -> str:
    vi = VI_LABELS.get(label, label)
    if overlay_lang == "code": return label
    if overlay_lang == "vi":   return vi
    return f"{label} ({vi})"

def create_detector(task_model_path: Path, det_conf: float, track_conf: float):
    model_path = ensure_task_model(task_model_path)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_hands=1,
        min_hand_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )
    return HandLandmarker.create_from_options(options)

def detect_landmarks(detector, frame_bgr: np.ndarray):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    if result.hand_landmarks: return result.hand_landmarks[0]
    return None

def landmarks_to_array(landmarks) -> np.ndarray:
    if isinstance(landmarks, np.ndarray):
        return landmarks.astype(np.float32, copy=False)
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    h, w  = frame.shape[:2]
    arr   = landmarks_to_array(landmarks)
    pts   = []
    for p in arr:
        x, y = int(float(p[0]) * w), int(float(p[1]) * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 3, (70, 220, 255), -1)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (90, 160, 255), 1)


# ════════════════════════════════════════════════════════════════════════════
#  FILTERING, QUALITY GATES, ROI HELPERS
# ════════════════════════════════════════════════════════════════════════════

class OneEuroFilter:
    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.0, d_cutoff: float = 1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.last_ts = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * pi * cutoff)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def _lowpass(self, x, x_prev, alpha: float):
        if x_prev is None:
            return x
        return alpha * x + (1.0 - alpha) * x_prev

    def filter(self, x: np.ndarray, ts: Optional[float] = None) -> np.ndarray:
        if ts is None:
            ts = time.perf_counter()
        if self.last_ts is not None:
            dt = max(ts - self.last_ts, 1e-6)
            self.freq = 1.0 / dt
        self.last_ts = ts

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        dx = (x - self.x_prev) * self.freq
        dx_hat = self._lowpass(dx, self.dx_prev, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        x_hat = self._lowpass(x, self.x_prev, self._alpha(cutoff))
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def compute_blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_hand_bbox(landmarks: np.ndarray, w: int, h: int,
                      margin: float = 0.25, min_size: int = 80) -> Tuple[int, int, int, int]:
    xs = landmarks[:, 0] * w
    ys = landmarks[:, 1] * h
    x1, x2 = float(np.min(xs)), float(np.max(xs))
    y1, y2 = float(np.min(ys)), float(np.max(ys))
    bw = max(x2 - x1, min_size)
    bh = max(y2 - y1, min_size)
    pad_w = bw * margin
    pad_h = bh * margin
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    x1 = int(max(cx - bw * 0.5 - pad_w, 0))
    y1 = int(max(cy - bh * 0.5 - pad_h, 0))
    x2 = int(min(cx + bw * 0.5 + pad_w, w - 1))
    y2 = int(min(cy + bh * 0.5 + pad_h, h - 1))
    return x1, y1, x2, y2


def map_landmarks_to_full(landmarks: np.ndarray, roi: Tuple[int, int, int, int],
                          frame_w: int, frame_h: int) -> np.ndarray:
    x1, y1, x2, y2 = roi
    rw = max(x2 - x1, 1)
    rh = max(y2 - y1, 1)
    out = landmarks.copy()
    out[:, 0] = (out[:, 0] * rw + x1) / max(frame_w, 1)
    out[:, 1] = (out[:, 1] * rh + y1) / max(frame_h, 1)
    return out


def detect_landmarks_with_roi(detector, frame_bgr: np.ndarray,
                               roi: Optional[Tuple[int, int, int, int]] = None):
    if roi is None:
        lm = detect_landmarks(detector, frame_bgr)
        if lm is None:
            return None, None
        return landmarks_to_array(lm), None
    x1, y1, x2, y2 = roi
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return detect_landmarks(detector, frame_bgr), None
    lm = detect_landmarks(detector, crop)
    if lm is None:
        return None, None
    arr = landmarks_to_array(lm)
    return arr, roi


# ════════════════════════════════════════════════════════════════════════════
#  BLUR DETECTION & MOTION PREDICTION HELPERS
# ════════════════════════════════════════════════════════════════════════════

def is_frame_blurry(frame: np.ndarray, threshold: float) -> bool:
    """Check if frame is blurry using Laplacian variance."""
    if frame is None or frame.size == 0:
        return True
    try:
        return compute_blur_score(frame) < threshold
    except Exception:
        return False

def predict_fingertip_position(current_pos: Tuple[float, float], factor: float = 0.3) -> Tuple[int, int]:
    """Predict next fingertip position based on motion history."""
    if len(MOTION_HISTORY) < 2:
        return tuple(map(int, current_pos))
    
    # Compute velocity from last 2 positions
    positions = list(MOTION_HISTORY)
    prev_pos = np.array(positions[-2], dtype=float)
    last_pos = np.array(positions[-1], dtype=float)
    velocity = last_pos - prev_pos
    
    # Predict future position
    predicted = np.array(current_pos, dtype=float) + velocity * factor
    
    # Clamp to screen bounds
    screen_w, screen_h = pyautogui.size()
    predicted[0] = np.clip(predicted[0], 0, screen_w - 1)
    predicted[1] = np.clip(predicted[1], 0, screen_h - 1)
    
    return tuple(predicted.astype(int))

def toggle_mouse_follow(enable: bool, frame_idx: int, force: bool = False) -> bool:
    """Toggle mouse follow with debounce to prevent rapid on/off flickering.
    
    Returns True if toggle was accepted, False if still in cooldown.
    """
    global MOUSE_FOLLOW_ENABLED, FOLLOW_TOGGLE_FRAME, MOTION_HISTORY
    
    if not force and (frame_idx - FOLLOW_TOGGLE_FRAME) < FOLLOW_TOGGLE_COOLDOWN:
        return False  # Still in cooldown, ignore
    
    MOUSE_FOLLOW_ENABLED = enable
    FOLLOW_TOGGLE_FRAME = frame_idx
    MOTION_HISTORY.clear()  # Reset motion history when toggling
    
    if enable:
        print(f"[follow_toggle] ON @ frame {frame_idx} (debounced)")
    else:
        print(f"[follow_toggle] OFF @ frame {frame_idx} (debounced)")
    
    return True


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def run_calibration_mode(args) -> None:
    """Run gesture calibration mode (record templates for KNN matching)."""
    print("🎓 Gesture Calibration Mode")
    print("=" * 60)
    
    # Setup camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera id {args.camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS,          args.camera_fps)
    
    # Setup detector
    detector = create_detector(Path(args.task_model), args.det_conf, args.track_conf)
    
    # Setup calibrator & UI
    calibrator = GestureCalibrator()
    ui = CalibrationUI(calibrator)
    
    print(f"📷 Camera: {args.camera_width}x{args.camera_height} @ {args.camera_fps} FPS")
    print(f"🎨 Gesture mode: Press SPACE to record, S to skip, Q to exit\n")
    
    frame_idx = 0
    landmark_filtered = None
    one_euro = OneEuroFilter(min_cutoff=args.oneeuro_min_cutoff,
                            beta=args.oneeuro_beta,
                            d_cutoff=args.oneeuro_d_cutoff)
    missing_count = 0
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            
            frame_idx += 1
            
            # Detect landmarks
            raw_landmarks = detect_landmarks(detector, frame)
            
            if raw_landmarks is not None:
                missing_count = 0
                raw_landmarks = landmarks_to_array(raw_landmarks)
                
                # Filter
                if args.landmark_filter == "oneeuro":
                    raw_landmarks = one_euro.filter(raw_landmarks, time.perf_counter())
                else:
                    if landmark_filtered is not None:
                        alpha = float(np.clip(args.landmark_ema_alpha, 0.0, 0.99))
                        raw_landmarks = alpha * landmark_filtered + (1.0 - alpha) * raw_landmarks
                
                landmark_filtered = raw_landmarks
                
                # Add to calibrator buffer if recording
                if ui.is_recording:
                    calibrator.add_landmarks_frame(landmark_filtered)
                
                # Draw landmarks
                draw_landmarks(frame, landmark_filtered)
            else:
                missing_count += 1
                if missing_count >= args.missing_reset_frames:
                    landmark_filtered = None
            
            # Draw UI
            ui.draw_overlay(frame, frame_idx)
            
            # Show frame
            cv2.imshow("Gesture Calibration", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            action = ui.handle_keypress(key, frame_idx)
            
            if action == "quit":
                break
            elif action == "finish_calibration":
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(detector, "close"):
            detector.close()
        
        # Summary
        print("\n" + ui.get_calibration_summary())
        print("\n✅ Calibration completed! Templates saved to:", calibrator.save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="")
    parser.add_argument("--labels",      default="")
    parser.add_argument("--config",      default="", help="Path to gesture_config.json")
    parser.add_argument("--task-model",  default="tools/assets/hand_landmarker.task")
    parser.add_argument("--camera-id",   type=int,   default=0)
    parser.add_argument("--camera-width",type=int,   default=640)
    parser.add_argument("--camera-height",type=int,  default=480)
    parser.add_argument("--camera-fps",  type=int,   default=30)
    parser.add_argument("--length",      type=int,   default=30)
    parser.add_argument("--device",      default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--det-conf",    type=float, default=0.6)
    parser.add_argument("--track-conf",  type=float, default=0.35)
    parser.add_argument("--ema-alpha",   type=float, default=0.7)
    parser.add_argument("--landmark-ema-alpha", type=float, default=0.08)
    parser.add_argument("--landmark-filter", choices=["ema", "oneeuro"], default="ema")
    parser.add_argument("--oneeuro-min-cutoff", type=float, default=1.2)
    parser.add_argument("--oneeuro-beta", type=float, default=0.01)
    parser.add_argument("--oneeuro-d-cutoff", type=float, default=1.0)
    parser.add_argument("--max-hand-jump",      type=float, default=0.12)
    parser.add_argument("--missing-reset-frames", type=int, default=30)
    parser.add_argument("--topk",        type=int,   default=3)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--brightness-min", type=float, default=35.0)
    parser.add_argument("--brightness-max", type=float, default=230.0)
    parser.add_argument("--min-hand-size", type=float, default=0.02,
                        help="Min hand bbox area ratio to accept frame (0..1)")
    parser.add_argument("--quality-min-ok", type=int, default=3,
                        help="Require this many consecutive quality-ok frames before action")
    parser.add_argument("--send-cooldown",  type=int,   default=15,
                        help="Số frame chờ giữa 2 lần gửi phím (default 15 ~ 0.5s)")
    parser.add_argument("--stable-count", type=int, default=3,
                        help="Số lần cùng 1 label phải xuất hiện liên tiếp trước khi thực hiện action (giảm false positives)")
    parser.add_argument("--action-delay",   type=float, default=0.4,
                        help="Số giây tối thiểu giữa 2 action thực sự (debounce) - prevents spurious sequential actions")
    parser.add_argument("--stability-threshold", type=float, default=0.03,
                        help="Max allowed wrist movement (normalized) during sequence to consider landmarks stable")
    parser.add_argument("--mouse-follow-smooth", type=float, default=0.6,
                        help="EMA alpha for smoothing mouse movement (0..1), higher = smoother")
    parser.add_argument("--min-action-frames", type=int, default=8,
                        help="Minimum number of frames buffered before attempting an action (for responsiveness). Default 8")
    parser.add_argument("--early-conf", type=float, default=0.85,
                        help="High-confidence threshold to allow early triggering when reached (0..1). Default 0.85")
    parser.add_argument("--early-frames", type=int, default=2,
                        help="Number of consecutive high-confidence frames required for early trigger. Default 2")
    parser.add_argument("--show-fps",    action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay-lang", choices=["vi","code","both"], default="vi")
    parser.add_argument("--use-z",       action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-velocity",action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--roi-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roi-margin", type=float, default=0.25)
    parser.add_argument("--roi-min-size", type=int, default=100)
    parser.add_argument("--roi-max-miss", type=int, default=8)
    parser.add_argument("--calibration-mode", action="store_true", default=False,
                        help="Enable gesture calibration mode (record templates)")
    parser.add_argument("--knn-mode", action="store_true", default=False,
                        help="Use KNN matching instead of STGCN for faster gesture recognition")
    parser.add_argument("--hybrid-mode", action="store_true", default=False,
                        help="Enable Hybrid mode: KNN for static gestures, ST-GCN for dynamic.")
    parser.add_argument("--knn-templates", type=str, default="Gan_nut/gesture_templates.json",
                        help="Path to gesture_templates.json for KNN/Hybrid mode.")
    parser.add_argument("--knn-threshold", type=float, default=2.5,
                        help="Distance threshold for a valid KNN match.")
    parser.add_argument("--knn-k", type=int, default=3, help="Number of neighbors for KNN.")
    args = parser.parse_args()

    # ── Calibration mode ─────────────────────────────────────────────────────
    if args.calibration_mode:
        run_calibration_mode(args)
        return

    # --- KNN/Hybrid Mode Setup ---
    knn_matcher = None
    if args.hybrid_mode or args.knn_mode:
        templates_path = Path(args.knn_templates)
        if not templates_path.exists():
            print(f"⚠️  KNN/Hybrid mode: Template file not found at {templates_path}")
            print(f"⚠️  Run calibration first: python tools/demo_webcam.py --calibration-mode")
            if args.knn_mode: # knn-mode cannot function without templates
                return
        else:
            # Note: GestureCalibrator is used here just to load templates for the matcher
            calibrator = GestureCalibrator(save_path=templates_path)
            knn_matcher = KNNGestureMatcher(calibrator, k=args.knn_k)
            print(f"✅ KNN/Hybrid mode: Loaded {len(calibrator.templates)} templates from {templates_path}.")

    # ── Tìm file model & labels ──────────────────────────────────────────────
    model_candidates = [
        Path("Gan_nut\stgcn_best.pt"),
        Path("outputs_resume/stgcn_trained_6ch.pt"),
        Path("outputs_resume2/stgcn_best.pt"), Path("outputs_resume/stgcn_best.pt"),
        Path("outputs/outputs/stgcn_best.pt"),  Path("outputs/stgcn_best.pt"),
        Path("data/stgcn_best.pt"),
    ]
    labels_candidates = [
        Path("Gan_nut\labels.json"), Path("Gan_nut\labels.json"),
        Path("outputs/outputs/labels.json"),  Path("outputs/labels.json"),
        Path("data/labels.json"),
    ]
    model_path  = Path(args.model)  if args.model  else first_existing(model_candidates)
    labels_path = Path(args.labels) if args.labels else first_existing(labels_candidates)
    
    # ST-GCN is not needed for pure KNN mode
    if not args.knn_mode and (model_path is None or labels_path is None):
        raise FileNotFoundError("Không tìm thấy checkpoint hoặc labels.json!")
    elif args.knn_mode:
        print("🏃 Chạy chế độ KNN. Model ST-GCN sẽ không được tải.")

    # ── Load gesture config ──────────────────────────────────────────────────
    config_path    = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    gesture_config = load_gesture_config(config_path)
    print(f"⌨️  Gesture config loaded from: {config_path}")
    active_keys = {k: v for k, v in gesture_config.items() if v}
    for gid, key in active_keys.items():
        print(f"   {gid:5s} → [{key}]")

    # ── Load model ───────────────────────────────────────────────────────────
    model, labels, device, use_z, use_velocity, use_acceleration = None, [], "cpu", False, False, False
    if not args.knn_mode:
        label_map = load_label_map(labels_path)
        labels    = index_to_label(label_map, len(label_map))
        device    = resolve_device(args.device)
        state     = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        in_channels = infer_in_channels_from_state(state)
        inferred_use_z, inferred_use_velocity, inferred_use_acceleration = infer_feature_config(in_channels)
        use_z            = inferred_use_z            if args.use_z       is None else args.use_z
        use_velocity     = inferred_use_velocity     if args.use_velocity is None else args.use_velocity
        use_acceleration = inferred_use_acceleration

        edge_index = build_hand_edge_index()
        model      = STGCN(in_channels=in_channels, num_classes=len(labels), edge_index=edge_index)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        print(f"✅ Model: {model_path} | Channels: {in_channels} | Device: {device}")
    detector = create_detector(Path(args.task_model), args.det_conf, args.track_conf)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera id {args.camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS,          args.camera_fps)

    frame_lock = threading.Lock()
    latest_frame = {"frame": None, "ts": 0.0}
    stop_event = threading.Event()

    def capture_loop():
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with frame_lock:
                latest_frame["frame"] = frame
                latest_frame["ts"] = time.perf_counter()

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    frame_buffer = deque(maxlen=args.length)
    prob_ema, landmark_filtered = None, None
    one_euro = OneEuroFilter(min_cutoff=args.oneeuro_min_cutoff,
                             beta=args.oneeuro_beta,
                             d_cutoff=args.oneeuro_d_cutoff)
    missing_count  = 0
    fps_ema, prev_ts = None, time.perf_counter()
    quality_ok_count = 0
    roi_bbox = None
    roi_miss = 0

    # Biến gửi phím
    last_sent_label = None
    send_cooldown   = 0
    last_action_ts = 0.0
    frame_idx = 0  # Track frame number for blur detection and toggle debounce
    # label stability history (keep larger buffer so per-gesture stable_count can be higher)
    # 📌 PATCH: Biến lưu trạng thái Click siêu tốc
    PINCH_CLICKED = False
    LAST_PINCH_TS = 0.0
    from collections import deque as _dq
    # label history sized based on stable_count and early_frames (avoid waiting for a large fixed buffer)
    label_history = _dq(maxlen=max(args.stable_count, args.early_frames, 8))
    # early high-confidence buffer for faster triggers
    early_history = _dq(maxlen=args.early_frames)

    def perform_mapped_action(raw_mapping: str, seq: np.ndarray, now: float, frame_idx: int = 0):
        """Handle mapping options and perform action accordingly.

        raw_mapping format: base_action[|opt1|opt2=val|...]
        Supported opts:
          - still : require tighter wrist stability before action
          - hold=<seconds> : for mouse clicks, hold mouse down for given seconds
          - stable_count=<N> : override stable_count for this gesture
        """
        parts = raw_mapping.split("|") if raw_mapping else [""]
        base = parts[0]
        opts = {}
        for p in parts[1:]:
            if '=' in p:
                k, v = p.split('=', 1)
                opts[k.strip()] = v.strip()
            else:
                opts[p.strip()] = True

        # compute effective stable count
        eff_stable = int(opts.get('stable_count', args.stable_count))
        # check label stability using last entries of label_history
        lh = list(label_history)
        if eff_stable > 0 and len(lh) < eff_stable:
            return False
        stable_label = True
        if eff_stable > 0:
            tail = lh[-eff_stable:]
            stable_label = all(l == top_label for l in tail)
        if not stable_label:
            return False

        # wrist still requirement
        if 'still' in opts:
            try:
                seq_wrist = seq[:, 0:1, :2]
                disp = np.linalg.norm(seq_wrist - seq_wrist[0:1], axis=-1)
                max_disp = float(np.max(disp))
                if max_disp > (args.stability_threshold * 0.5):
                    return False
            except Exception:
                pass

        # hold option for mouse clicks
        hold = float(opts.get('hold', 0.0)) if 'hold' in opts else 0.0

        # mutex handling: ensure group isn't recently used
        mutex = opts.get('mutex', None)
        if mutex:
            last = MUTEX_LAST.get(mutex, 0.0)
            if now - last < 0.6:
                return False
            # reserve it
            MUTEX_LAST[mutex] = now

        # perform action (support base being mouse:left etc.)
        if base.startswith('mouse:') and hold > 0.0:
            # mouseDown/Up with hold in background
            def click_hold():
                try:
                    btn = base.split(':',1)[1]
                    if btn == 'left':
                        pyautogui.mouseDown(button='left')
                        time.sleep(hold)
                        pyautogui.mouseUp(button='left')
                    elif btn == 'right':
                        pyautogui.mouseDown(button='right')
                        time.sleep(hold)
                        pyautogui.mouseUp(button='right')
                except Exception:
                    pass
            threading.Thread(target=click_hold, daemon=True).start()
            return True

        # otherwise fallback to normal send_action (may be hotkey/run/type/mouse ops)
        send_action(base, frame_idx)
        return True

    # mouse follow state
    global MOUSE_FOLLOW_ENABLED, MOUSE_POS_EMA
    MOUSE_FOLLOW_ENABLED = False
    MOUSE_POS_EMA = None

    print("🎮 Hệ thống đang chạy cử chỉ... Nhấn Q trong cửa sổ camera để thoát")

    try:
        while True:
            frame = None
            with frame_lock:
                if latest_frame["frame"] is not None:
                    frame = latest_frame["frame"].copy() # Copy ngay trong lock ngắn để giải phóng sớm
            if frame is None:
                time.sleep(0.002)
                continue
            frame_idx += 1  # Increment frame counter

            # FPS
            now_ts = time.perf_counter()
            dt     = max(now_ts - prev_ts, 1e-6)
            prev_ts = now_ts
            fps_ema = (1/dt) if fps_ema is None else (0.9*fps_ema + 0.1/dt)

            # ── Quality metrics ─────────────────────────────────────────────
            blur_score = compute_blur_score(frame)
            brightness = compute_brightness(frame)
            quality_reasons = []
            if blur_score < args.blur_threshold:
                quality_reasons.append("blurry")
            if brightness < args.brightness_min:
                quality_reasons.append("dark")
            elif brightness > args.brightness_max:
                quality_reasons.append("bright")

            # ── Landmark detection with ROI ─────────────────────────────────
            frame_h, frame_w = frame.shape[:2]
            use_roi = args.roi_enable and roi_bbox is not None
            raw_landmarks, roi_used = detect_landmarks_with_roi(detector, frame, roi_bbox if use_roi else None)
            if raw_landmarks is not None and roi_used is not None:
                raw_landmarks = map_landmarks_to_full(raw_landmarks, roi_used, frame_w, frame_h)

            if raw_landmarks is not None:
                missing_count = 0
                roi_miss = 0

                # reject large wrist jumps
                if landmark_filtered is not None:
                    prev_wrist = landmark_filtered[0]
                    curr_wrist = raw_landmarks[0]
                    if np.linalg.norm(curr_wrist[:2] - prev_wrist[:2]) > args.max_hand_jump:
                        raw_landmarks = landmark_filtered

                if args.landmark_filter == "oneeuro":
                    raw_landmarks = one_euro.filter(raw_landmarks, now_ts)
                else:
                    if landmark_filtered is not None:
                        alpha = float(np.clip(args.landmark_ema_alpha, 0.0, 0.99))
                        raw_landmarks = alpha * landmark_filtered + (1.0 - alpha) * raw_landmarks

                landmark_filtered = raw_landmarks
                frame_buffer.append(landmark_filtered)
                # # ==============================================================
                # # ⚡ PATCH: HYBRID CLICK TỐC ĐỘ CAO (QUA MẶT ST-GCN)
                # # ==============================================================
                # if landmark_filtered is not None:
                #     # Điểm số 4 là đầu ngón cái, điểm số 8 là đầu ngón trỏ
                #     thumb_tip = landmark_filtered[4][:2] 
                #     index_tip = landmark_filtered[8][:2]
                    
                #     # Tính khoảng cách Euclidean giữa 2 ngón
                #     pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
                    
                #     # Nếu khoảng cách < 0.04 (Hai ngón chạm nhau)
                #     if pinch_dist < 0.04:
                #         if not PINCH_CLICKED and (now_ts - LAST_PINCH_TS > 0.3): # Cooldown 0.3s
                #             print("⚡ BẮT CLICK SIÊU TỐC TỪ KHỚP XƯƠNG!")
                            
                #             # Chạy click chuột khác luồng để không làm giật camera
                #             threading.Thread(target=lambda: pyautogui.click(button='left'), daemon=True).start()
                            
                #             PINCH_CLICKED = True
                #             LAST_PINCH_TS = now_ts
                            
                #             # Hiển thị UI
                #             cv2.putText(frame, ">> ACTION: [INSTANT CLICK]", (20, 130), 
                #                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                #     else:
                #         # Thả tay ra thì reset lại trạng thái
                #         PINCH_CLICKED = False
                # # ==============================================================

                if args.roi_enable:
                    roi_bbox = compute_hand_bbox(landmark_filtered, frame_w, frame_h,
                                                args.roi_margin, args.roi_min_size)
            else:
                missing_count += 1
                roi_miss += 1
                if roi_miss >= args.roi_max_miss:
                    roi_bbox = None
                if missing_count >= args.missing_reset_frames:
                    frame_buffer.clear()
                    prob_ema, landmark_filtered = None, None
                elif landmark_filtered is not None:
                    frame_buffer.append(landmark_filtered)

            # hand size quality check
            hand_size_ratio = 0.0
            if landmark_filtered is not None:
                xs = landmark_filtered[:, 0]
                ys = landmark_filtered[:, 1]
                hand_size_ratio = float((np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys)))
                if hand_size_ratio < args.min_hand_size:
                    quality_reasons.append("hand_small")

            quality_ok = len(quality_reasons) == 0 and landmark_filtered is not None
            if quality_ok:
                quality_ok_count += 1
            else:
                quality_ok_count = 0

            if landmark_filtered is not None:
                draw_landmarks(frame, landmark_filtered)

            # ── Status bar ───────────────────────────────────────────────────
            cv2.putText(frame, f"Frames: {len(frame_buffer)}/{args.length}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)
            if args.show_fps:
                cv2.putText(frame, f"FPS: {fps_ema:.1f}",
                            (frame.shape[1]-120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_MUTED, 1)

            # quality overlay
            if quality_reasons:
                cv2.putText(frame, f"Quality: {', '.join(quality_reasons)}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # --- KNN Prediction (for KNN and Hybrid modes) ---
            knn_prediction = None
            if (args.knn_mode or args.hybrid_mode) and knn_matcher and landmark_filtered is not None:
                knn_prediction = knn_matcher.predict_knn(landmark_filtered, threshold=args.knn_threshold)
                if knn_prediction:
                    gid, dist, _ = knn_prediction
                    # Display KNN's raw prediction
                    cv2.putText(frame, f"KNN: {gid} ({dist:.2f})", (frame.shape[1] - 200, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN, 1)

            # ── Inference ────────────────────────────────────────────────────
            top_label, top_conf = "D0X", 0.0

            if args.knn_mode:
                if knn_prediction:
                    top_label = knn_prediction[0]
                    top_conf = max(0.0, 1.0 - (knn_prediction[1] / args.knn_threshold))
                else:
                    top_label, top_conf = "D0X", 0.0
                prob_ema = None # In KNN mode, we don't use EMA

                # Display result for KNN mode
                top_text = format_label(top_label, args.overlay_lang)
                if top_conf > 0.1:
                    display, color = f"{top_text} ({top_conf:.2f})", C_GREEN
                else:
                    display, color = f"Khong chac ({top_conf:.2f})", (80, 170, 250)
                cv2.putText(frame, display, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            elif len(frame_buffer) >= args.min_action_frames and landmark_filtered is not None:
                # Use available frames (can be shorter than args.length for responsiveness)
                seq = np.stack(list(frame_buffer), axis=0)
                
                if not use_z: seq = seq[:, :, :2]
                seq = normalize_frames(seq)
                if use_acceleration: seq = add_acceleration(seq)
                elif use_velocity:   seq = add_velocity(seq)

                x = torch.from_numpy(seq).float().unsqueeze(0).permute(0,3,1,2).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1).squeeze(0).cpu()

                prob_ema = probs if prob_ema is None else (
                    args.ema_alpha * prob_ema + (1.0 - args.ema_alpha) * probs)

                scores, indices = torch.topk(prob_ema, min(args.topk, prob_ema.numel()))
                stgcn_label = labels[int(indices[0].item())]
                stgcn_conf  = float(scores[0].item())

                # --- Hybrid Logic: Override with KNN for static gestures ---
                if args.hybrid_mode and knn_prediction:
                    knn_label, knn_dist, _ = knn_prediction
                    if knn_label in STATIC_GESTURES:
                        # Static gesture detected by KNN, let's override ST-GCN
                        top_label = knn_label
                        top_conf = max(0.0, 1.0 - (knn_dist / args.knn_threshold))
                        # Reset EMA and history to make the override more responsive
                        # prob_ema = None
                        # label_history.clear()
                        # Add a visual indicator for the override
                        cv2.putText(frame, "KNN OVERRIDE", (frame.shape[1] - 150, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # KNN detected a dynamic gesture, let ST-GCN handle it
                        top_label, top_conf = stgcn_label, stgcn_conf
                else:
                    # Default ST-GCN behavior
                    top_label, top_conf = stgcn_label, stgcn_conf

                top_text  = format_label(top_label, args.overlay_lang)
                
                if top_conf >= args.min_confidence:
                    display, color = f"{top_text} ({top_conf:.2f})", C_GREEN
                else:
                    display, color = f"Khong chac ({top_conf:.2f})", (80, 170, 250)
                cv2.putText(frame, display, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # ── Gửi phím dựa trên file JSON (yêu cầu độ ổn định label) ─────
                if top_conf >= args.min_confidence and (top_label != "D0X" or args.knn_mode):
                    raw_mapping = gesture_config.get(top_label, "")
                    mapped_key = raw_mapping
                    # If mapping requests follow_hold, enable follow while this gesture is active
                    try:
                        if raw_mapping and ("follow_hold" in raw_mapping or "follow=hold" in raw_mapping):
                            # enable follow only when label is stable (or early trigger)
                            # we'll set MOUSE_FOLLOW_ENABLED below after computing stable/early
                            follow_hold_requested = True
                        else:
                            follow_hold_requested = False
                    except Exception:
                        follow_hold_requested = False
                    # detect instant prefix (bypass debounce/stability)
                    is_instant = False
                    if mapped_key and mapped_key.lower().startswith("instant:"):
                        is_instant = True
                        mapped_key = mapped_key[8:]
                    # track label history
                    try:
                        label_history.append(top_label)
                    except Exception:
                        pass

                    # early high-confidence path
                    early_trigger = False
                    try:
                        if top_conf >= args.early_conf:
                            early_history.append(top_label)
                        else:
                            early_history.clear()
                        if len(early_history) == early_history.maxlen and all(l == top_label for l in early_history):
                            early_trigger = True
                    except Exception:
                        early_trigger = False

                    # check last N entries for stability (N = args.stable_count)
                    lh = list(label_history)
                    tail = lh[-args.stable_count:] if args.stable_count > 0 else lh
                    stable_label = (len(tail) == args.stable_count and all(l == top_label for l in tail))

                    if not stable_label:
                        # show waiting overlay with progress towards stable_count
                        prog = len([l for l in tail if l == top_label])
                        cv2.putText(frame, f"Waiting stable ({prog}/{args.stable_count})...",
                                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,80), 2)

                    # If this mapping is follow_hold, enable/disable follow based on stability
                    if follow_hold_requested:
                        if stable_label or early_trigger or is_instant:
                            MOUSE_FOLLOW_ENABLED = True
                            cv2.putText(frame, f"MOUSE FOLLOW (hold)",
                                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN, 2)
                        else:
                            MOUSE_FOLLOW_ENABLED = False
                    else:
                        # 📌 VÁ LỖI 1: Tắt chuột khi chuyển sang cử chỉ khác (ví dụ: đang rê chuột thì chuyển sang nắm tay click)
                        MOUSE_FOLLOW_ENABLED = False       

                    if mapped_key and (stable_label or is_instant):
                        # Check wrist stability over the buffered frames
                        stable = True
                        try:
                            seq_wrist = seq[:, 0:1, :2]  # frames x 1 x 2
                            # compute max displacement across frames
                            disp = np.linalg.norm(seq_wrist - seq_wrist[0:1], axis=-1)
                            max_disp = float(np.max(disp))
                            # relax stability threshold slightly for early triggers
                            thr = args.stability_threshold * (1.5 if early_trigger else 1.0)
                            if max_disp > thr:
                                stable = False
                        except Exception:
                            stable = True

                        # Debounce by time as well
                        now = time.perf_counter()
                        if not stable and not is_instant:
                            # skip spurious action when hand moved
                            pass
                        elif not is_instant and now - last_action_ts < args.action_delay:
                            pass
                        elif quality_ok_count < args.quality_min_ok and not is_instant:
                            pass
                        elif top_label != last_sent_label or send_cooldown <= 0:
                            performed = False
                            try:
                                performed = perform_mapped_action(mapped_key, seq, now, frame_idx)
                            except Exception:
                                performed = False
                            if performed:
                                last_sent_label = top_label
                                send_cooldown   = args.send_cooldown
                                last_action_ts  = now
                                # Hiển thị hành động vừa thực hiện lên góc màn hình camera
                                disp = mapped_key
                                # Shorten display for run: commands
                                if mapped_key.lower().startswith("run:"):
                                    disp = f"run:{mapped_key[4:].strip()}"
                                cv2.putText(frame, f">> ACTION: [{disp}]",
                                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.65, C_CYAN, 2)
                else:
                    last_sent_label = None
                    # reset label history when no stable detection
                    MOUSE_FOLLOW_ENABLED = False
                    try:
                        label_history.clear()
                    except Exception:
                        pass

                if send_cooldown > 0:
                    send_cooldown -= 1

                # Mouse follow: if enabled, move mouse to index fingertip (landmark idx 8)
                if MOUSE_FOLLOW_ENABLED and len(frame_buffer) >= args.min_action_frames and landmark_filtered is not None:
                    try:
                        # use the latest landmarks (landmark_ema or current)
                        lm = landmark_filtered
                        idx8 = lm[8]  # x,y
                        # idx8 coords are normalized (0..1) relative to image
                        fx, fy = float(idx8[0]), float(idx8[1])
                        # map to screen coordinates
                        screen_w, screen_h = pyautogui.size()
                        sx = int(fx * screen_w)
                        sy = int(fy * screen_h)
                        
                        # Add current position to motion history for prediction
                        MOTION_HISTORY.append((sx, sy))
                        
                        # Predict next position based on velocity
                        predicted_sx, predicted_sy = predict_fingertip_position((sx, sy), factor=0.3)
                        
                        # smooth with EMA
                        if MOUSE_POS_EMA is None:
                            MOUSE_POS_EMA = np.array([predicted_sx, predicted_sy], dtype=float)
                        else:
                            alpha = float(np.clip(args.mouse_follow_smooth, 0.0, 0.99))
                            MOUSE_POS_EMA = alpha * MOUSE_POS_EMA + (1.0 - alpha) * np.array([predicted_sx, predicted_sy], dtype=float)
                        # move mouse (non-blocking)
                        tx, ty = int(MOUSE_POS_EMA[0]), int(MOUSE_POS_EMA[1])
                        pyautogui.moveTo(tx, ty)
                    except Exception:
                        pass

            cv2.imshow("ST-GCN Hand Gesture Demo", frame)

            # ── Chỉ bắt duy nhất phím Q để thoát ─────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(detector, "close"):
            detector.close()
        print("👋 Đã đóng camera và thoát chương trình.")


if __name__ == "__main__":
    main()