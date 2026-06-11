import os
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
BLUR_THRESHOLD = 50.0  # Laplacian variance threshold for blur detection
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
    if device_arg == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        num_hands=1, min_hand_detection_confidence=det_conf, min_tracking_confidence=track_conf,
    )
    return HandLandmarker.create_from_options(options)


def detect_landmarks(detector, frame_bgr: np.ndarray):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    if result.hand_landmarks: return result.hand_landmarks[0]
    return None

def landmarks_to_array(landmarks) -> np.ndarray:
    if isinstance(landmarks, np.ndarray): return landmarks.astype(np.float32, copy=False)
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
#  BLUR DETECTION & MOTION PREDICTION HELPERS
# ════════════════════════════════════════════════════════════════════════════

def is_frame_blurry(frame: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    """Check if frame is blurry using Laplacian variance."""
    if frame is None or frame.size == 0:
        return True
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="")
    parser.add_argument("--labels",      default="")
    parser.add_argument("--config",      default="", help="Path to gesture_config.json")
    parser.add_argument("--task-model",  default="tools/assets/hand_landmarker.task")
    parser.add_argument("--camera-id",   type=int,   default=0)
    parser.add_argument("--camera-width",type=int,   default=700) #chiều rộng camera
    parser.add_argument("--camera-height",type=int,  default=500) #chiều cao camera
    parser.add_argument("--camera-fps",  type=int,   default=40) #FPS
    parser.add_argument("--length",      type=int,   default=30) #chinh frame 
    parser.add_argument("--device",      default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--det-conf",    type=float, default=0.6)
    parser.add_argument("--track-conf",  type=float, default=0.35)
    parser.add_argument("--ema-alpha",   type=float, default=0.7)
    parser.add_argument("--landmark-ema-alpha", type=float, default=0.08)
    parser.add_argument("--max-hand-jump",      type=float, default=0.12)
    parser.add_argument("--missing-reset-frames", type=int, default=30)
    parser.add_argument("--topk",        type=int,   default=3)
    parser.add_argument("--min-confidence", type=float, default=0.35)
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
    parser.add_argument("--use-knn", action="store_true",
                        help="Use KNN-based matching instead of ST-GCN model")
    parser.add_argument("--knn-threshold", type=float, default=5.0,
                        help="Distance threshold for KNN match (lower = stricter). Default 5.0")
    parser.add_argument("--knn-k", type=int, default=3,
                        help="Number of neighbors for KNN voting. Default 3")
    parser.add_argument("--template-file", default="data/gesture_templates.json",
                        help="Path to gesture templates JSON. Default data/gesture_templates.json")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Skip N frames between inference runs (0 = every frame). Useful for 60+ FPS on slower GPUs")
    parser.add_argument("--optimize-240p", action="store_true",
                        help="Downscale input to 240p for faster inference (for weak GPUs). Improves FPS but may reduce accuracy")
    args = parser.parse_args()

    # ── Tìm file model & labels ──────────────────────────────────────────────
    model_candidates = [
        Path("ST_GCN\Gan_nut\stgcn_best.pt"), Path("outputs_resume/stgcn_best.pt"),
        Path("outputs/outputs/stgcn_best.pt"),  Path("outputs/stgcn_best.pt"),
        Path("data/stgcn_best.pt"),
    ]
    labels_candidates = [
        Path("ST_GCN\Gan_nut\labels.json"), Path("outputs_resume/labels.json"),
        Path("outputs/outputs/labels.json"),  Path("outputs/labels.json"),
        Path("data/labels.json"),
    ]
    model_path  = Path(args.model)  if args.model  else first_existing(model_candidates)

    labels_path = Path(args.labels) if args.labels else first_existing(labels_candidates)
    if model_path is None or labels_path is None:
        raise FileNotFoundError("Không tìm thấy checkpoint hoặc labels.json!")


    # ── Load gesture config ──────────────────────────────────────────────────
    config_path    = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    gesture_config = load_gesture_config(config_path)
    print(f"⌨️  Gesture config loaded from: {config_path}")
    active_keys = {k: v for k, v in gesture_config.items() if v}
    for gid, key in active_keys.items():
        print(f"   {gid:5s} → [{key}]")

    # ── Load model ───────────────────────────────────────────────────────────
    label_map = load_label_map(labels_path)

    labels    = index_to_label(label_map, len(label_map))
    device    = resolve_device(args.device)
    state     = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    in_channels = infer_in_channels_from_state(state)
    inferred_use_z, inferred_use_velocity, inferred_use_acceleration = infer_feature_config(in_channels)
    # Force 9-channel mode (z + velocity + acceleration)
    use_z            = True
    use_velocity     = True
    use_acceleration = True

    edge_index = build_hand_edge_index()
    model      = STGCN(in_channels=in_channels, num_classes=len(labels), edge_index=edge_index)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # ── Initialize KNN matcher if --use-knn flag set ──────────────────────────
    knn_matcher = None
    use_knn = False
    if args.use_knn:
        try:
            from tools.knn_matcher import KNNGestureMatcher
            template_path = Path(args.template_file)
            if not template_path.exists():
                print(f"⚠️  Template file not found: {template_path}")
                print(f"   Please run: python tools/calibrate_gestures.py")
                print(f"   Falling back to ST-GCN model...")
            else:
                knn_matcher = KNNGestureMatcher(template_path, k=args.knn_k)
                use_knn = True
                print(f"✅ KNN matcher initialized with threshold={args.knn_threshold}, k={args.knn_k}")
        except Exception as e:
            print(f"❌ Failed to load KNN matcher: {e}")
            print(f"   Falling back to ST-GCN model...")

    detector = create_detector(Path(args.task_model), args.det_conf, args.track_conf)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera id {args.camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS,          args.camera_fps)

    frame_buffer = deque(maxlen=args.length)
    prob_ema, landmark_ema = None, None
    missing_count  = 0
    fps_ema, prev_ts = None, time.perf_counter()

    # Biến gửi phím
    last_sent_label = None
    send_cooldown   = 0
    last_action_ts = 0.0
    frame_idx = 0  # Track frame number for blur detection and toggle debounce
    # label stability history (keep larger buffer so per-gesture stable_count can be higher)
    from collections import deque as _dq
    # label history sized based on stable_count and early_frames (avoid waiting for a large fixed buffer)
    label_history = _dq(maxlen=max(args.stable_count, args.early_frames, 8))
    # early high-confidence buffer for faster triggers
    early_history = _dq(maxlen=args.early_frames)

    def perform_mapped_action(raw_mapping: str, seq: np.ndarray, now: float, frame_idx: int = 0, is_instant: bool = False):
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
        # If caller requested instant, bypass the minimum-length check to allow immediate action
        if eff_stable > 0 and not is_instant and len(lh) < eff_stable:
            return False
        stable_label = True
        if eff_stable > 0:
            tail = lh[-eff_stable:]
            stable_label = all(l == top_label for l in tail)
        # If not instant and not stable, reject
        if not is_instant and not stable_label:
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

    print(f"✅ Model: {model_path} | Channels: {in_channels} | Device: {device}")
    print("🎮 Hệ thống đang chạy cử chỉ... Nhấn Q trong cửa sổ camera để thoát")
    print("Demo started thành công rực rỡ bro ơi!")
    try:
        inference_frame = 0  # Counter for inference skipping
        while True:
            ok, frame = cap.read()
            if not ok: break

            frame_idx += 1  # Increment frame counter

            # ── Fast input preprocessing for 60+ FPS ────────────────────────
            if args.optimize_240p:
                # Downscale to 240p for faster detection
                frame = cv2.resize(frame, (426, 240), interpolation=cv2.INTER_LINEAR)
                # Will be upscaled back for display

            # ── Blur detection ───────────────────────────────────────────────
            if is_frame_blurry(frame, BLUR_THRESHOLD):
                cv2.putText(frame, "⚠️ Frame blurry (skip inference)",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.imshow("ST-GCN Hand Gesture Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                continue  # Skip this frame

            # FPS
            now_ts = time.perf_counter()
            dt     = max(now_ts - prev_ts, 1e-6)
            prev_ts = now_ts
            fps_ema = (1/dt) if fps_ema is None else (0.9*fps_ema + 0.1/dt)

            # ── Landmark detection ───────────────────────────────────────────
            # Optional frame skipping for higher FPS (use interpolation for buffering)
            should_detect = (args.skip_frames == 0) or (frame_idx % (args.skip_frames + 1) == 1)
            
            if should_detect:
                landmarks = detect_landmarks(detector, frame)
            else:
                landmarks = None  # Skip detection, reuse previous landmark_ema

            if landmarks is not None:
                missing_count = 0
                current = landmarks_to_array(landmarks)
                if landmark_ema is not None:
                    alpha   = float(np.clip(args.landmark_ema_alpha, 0.0, 0.99))
                    current = alpha * landmark_ema + (1.0 - alpha) * current
                landmark_ema = current
                draw_landmarks(frame, current)
                frame_buffer.append(current)
            else:
                missing_count += 1
                if missing_count >= args.missing_reset_frames:
                    frame_buffer.clear()
                    prob_ema, landmark_ema = None, None
                elif landmark_ema is not None:
                    frame_buffer.append(landmark_ema)
                    draw_landmarks(frame, landmark_ema)

            # ── Status bar ───────────────────────────────────────────────────
            cv2.putText(frame, f"Frames: {len(frame_buffer)}/{args.length}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)
            if args.show_fps:
                cv2.putText(frame, f"FPS: {fps_ema:.1f}",
                            (frame.shape[1]-120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_MUTED, 1)

            # ── Inference ────────────────────────────────────────────────────
            top_label, top_conf = "D0X", 0.0
            if len(frame_buffer) >= args.min_action_frames:
                # Use available frames (can be shorter than args.length for responsiveness)
                seq = np.stack(list(frame_buffer), axis=0)
                
                if use_knn and knn_matcher is not None and landmark_ema is not None:
                    # ── KNN-based matching ───────────────────────────────────
                    # For KNN: normalize single frame (latest landmark)
                    lm_norm = normalize_frames(np.array([landmark_ema]))[0]  # (21, 3) or (21, 2)
                    top_label, top_conf = knn_matcher.match(lm_norm, threshold=args.knn_threshold)
                else:
                    # ── ST-GCN model inference ────────────────────────────────
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
                    top_label = labels[int(indices[0].item())]
                    top_conf  = float(scores[0].item())
                
                top_text  = format_label(top_label, args.overlay_lang)

                if top_conf >= args.min_confidence:
                    display, color = f"{top_text} ({top_conf:.2f})", C_GREEN
                else:
                    display, color = f"Khong chac ({top_conf:.2f})", (80, 170, 250)
                cv2.putText(frame, display, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # ── Gửi phím dựa trên file JSON (yêu cầu độ ổn định label) ─────
                if top_conf >= args.min_confidence and top_label != "D0X":
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
                        elif top_label != last_sent_label or send_cooldown <= 0:
                            performed = False
                            try:
                                performed = perform_mapped_action(mapped_key, seq, now, frame_idx, is_instant=is_instant)
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
                    try:
                        label_history.clear()
                    except Exception:
                        pass

                if send_cooldown > 0:
                    send_cooldown -= 1

                # Mouse follow: if enabled, move mouse to index fingertip (landmark idx 8)
                if MOUSE_FOLLOW_ENABLED and len(frame_buffer) >= args.min_action_frames and landmark_ema is not None:
                    try:
                        # use the latest landmarks (landmark_ema or current)
                        lm = landmark_ema if landmark_ema is not None else current
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
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(detector, "close"):
            detector.close()
        print("👋 Đã đóng camera và thoát chương trình.")


if __name__ == "__main__":
    main()