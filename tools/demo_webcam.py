import os
# 🔥 VẮC-XIN CHỐNG ĐƠ CAM CHO WINDOWS 🔥
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

import argparse
import json
import sys
import time
import urllib.request
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

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "gesture_config.json"

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
    except Exception as e:
        print(f"⚠️  Không gửi được phím '{key}': {e}")


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
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

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
    parser.add_argument("--max-hand-jump",      type=float, default=0.12)
    parser.add_argument("--missing-reset-frames", type=int, default=30)
    parser.add_argument("--topk",        type=int,   default=3)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--send-cooldown",  type=int,   default=15,
                        help="Số frame chờ giữa 2 lần gửi phím (default 15 ~ 0.5s)")
    parser.add_argument("--show-fps",    action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay-lang", choices=["vi","code","both"], default="vi")
    parser.add_argument("--use-z",       action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-velocity",action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    # ── Tìm file model & labels ──────────────────────────────────────────────
    model_candidates = [
        Path("outputs_resume2/stgcn_best.pt"), Path("outputs_resume/stgcn_best.pt"),
        Path("outputs/outputs/stgcn_best.pt"),  Path("outputs/stgcn_best.pt"),
        Path("data/stgcn_best.pt"),
    ]
    labels_candidates = [
        Path("outputs_resume2/labels.json"), Path("outputs_resume/labels.json"),
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
    use_z            = inferred_use_z            if args.use_z       is None else args.use_z
    use_velocity     = inferred_use_velocity     if args.use_velocity is None else args.use_velocity
    use_acceleration = inferred_use_acceleration

    edge_index = build_hand_edge_index()
    model      = STGCN(in_channels=in_channels, num_classes=len(labels), edge_index=edge_index)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

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

    print(f"✅ Model: {model_path} | Channels: {in_channels} | Device: {device}")
    print("🎮 Hệ thống đang chạy cử chỉ... Nhấn Q trong cửa sổ camera để thoát")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            # FPS
            now_ts = time.perf_counter()
            dt     = max(now_ts - prev_ts, 1e-6)
            prev_ts = now_ts
            fps_ema = (1/dt) if fps_ema is None else (0.9*fps_ema + 0.1/dt)

            # ── Landmark detection ───────────────────────────────────────────
            landmarks = detect_landmarks(detector, frame)
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
            if len(frame_buffer) == args.length:
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
                top_label = labels[int(indices[0].item())]
                top_conf  = float(scores[0].item())
                top_text  = format_label(top_label, args.overlay_lang)

                if top_conf >= args.min_confidence:
                    display, color = f"{top_text} ({top_conf:.2f})", C_GREEN
                else:
                    display, color = f"Khong chac ({top_conf:.2f})", (80, 170, 250)
                cv2.putText(frame, display, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # ── Gửi phím dựa trên file JSON ──────────────────────────────
                if top_conf >= args.min_confidence and top_label != "D0X":
                    mapped_key = gesture_config.get(top_label, "")
                    if mapped_key:
                        if top_label != last_sent_label or send_cooldown <= 0:
                            send_key(mapped_key)
                            last_sent_label = top_label
                            send_cooldown   = args.send_cooldown
                            # Hiển thị phím vừa gõ lên góc màn hình camera
                            cv2.putText(frame, f">> TOGGLE KEY: [{mapped_key.upper()}]",
                                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65, C_CYAN, 2)
                else:
                    last_sent_label = None

                if send_cooldown > 0:
                    send_cooldown -= 1

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