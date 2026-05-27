import os
# 🌟 TIÊM VẮC-XIN KHÓA LUỒNG TRÁNH DEADLOCK TRÊN WINDOWS CPU
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

# Allow running as "python tools/demo_webcam.py" from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stgcn import STGCN, build_hand_edge_index


TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

VI_LABELS: Dict[str, str] = {
    "D0X": "Khong cu chi", "B0A": "Chi 1 ngon", "B0B": "Chi 2 ngon",
    "G01": "Click 1 ngon", "G02": "Click 2 ngon", "G03": "Hat len",
    "G04": "Hat xuong", "G05": "Hat trai", "G06": "Hat phai",
    "G07": "Mo 2 lan", "G08": "Double click 1 ngon",
    "G09": "Double click 2 ngon", "G10": "Phong to", "G11": "Thu nho",
}

def ensure_task_model(path: Path) -> Path:
    if path.exists(): return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TASK_MODEL_URL, path)
    return path

def load_label_map(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as f: class_to_idx = json.load(f)
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
    palm = frames[:, 9:10, :]
    scale = np.linalg.norm(palm, axis=-1, keepdims=True)
    scale[scale == 0] = 1.0
    frames = frames / scale
    return frames

def add_velocity(frames: np.ndarray) -> np.ndarray:
    velocity = np.diff(frames, axis=0, prepend=frames[:1])
    return np.concatenate([frames, velocity], axis=-1)

def add_acceleration(frames: np.ndarray) -> np.ndarray:
    velocity = np.diff(frames, axis=0, prepend=frames[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    return np.concatenate([frames, velocity, acceleration], axis=-1)

def infer_in_channels_from_state(state: Dict[str, torch.Tensor]) -> int:
    if "data_bn.weight" not in state: raise ValueError("Cannot infer in_channels: key 'data_bn.weight' is missing in checkpoint")
    bn_size = int(state["data_bn.weight"].numel())
    if bn_size % 21 != 0: raise ValueError(f"Invalid data_bn size {bn_size}; expected multiple of 21")
    return bn_size // 21

def infer_feature_config(in_channels: int) -> Tuple[bool, bool, bool]:
    if in_channels == 2: return False, False, False
    if in_channels == 3: return True, False, False
    if in_channels == 4: return False, True, False
    if in_channels == 6: return True, True, False
    if in_channels == 9: return True, True, True
    raise ValueError(f"Unsupported in_channels={in_channels}.")

def feature_channels(use_z: bool, use_velocity: bool, use_acceleration: bool = False) -> int:
    if use_acceleration: return (3 if use_z else 2) * 3
    elif use_velocity: return (3 if use_z else 2) * 2
    else: return 3 if use_z else 2

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
    if overlay_lang == "vi": return vi
    return f"{label} ({vi})"

def create_detector(task_model_path: Path, det_conf: float, track_conf: float):
    if hasattr(mp, "solutions"):
        return mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=1,
            min_detection_confidence=det_conf, min_tracking_confidence=track_conf,
        ), False
    model_path = ensure_task_model(task_model_path)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_hands=1, min_hand_detection_confidence=det_conf, min_tracking_confidence=track_conf,
    )
    return HandLandmarker.create_from_options(options), True

def detect_landmarks(detector, frame_bgr: np.ndarray, use_tasks: bool):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if use_tasks:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(image)
        if result.hand_landmarks: return result.hand_landmarks[0]
        return None
    result = detector.process(rgb)
    if result.multi_hand_landmarks: return result.multi_hand_landmarks[0].landmark
    return None

def landmarks_to_array(landmarks) -> np.ndarray:
    if isinstance(landmarks, np.ndarray): return landmarks.astype(np.float32, copy=False)
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]
    arr = landmarks_to_array(landmarks)
    pts = []
    for p in arr:
        x, y = int(float(p[0]) * w), int(float(p[1]) * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 3, (70, 220, 255), -1)
    for a, b in HAND_CONNECTIONS: cv2.line(frame, pts[a], pts[b], (90, 160, 255), 1)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Path to checkpoint (.pt).")
    parser.add_argument("--labels", default="", help="Path to labels.json.")
    parser.add_argument("--task-model", default="tools/assets/hand_landmarker.task")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--length", type=int, default=30)
    
    # 🔥 ĐÃ GẮN CỨNG THÔNG SỐ SIÊU NHẠY VÀO CODE 🔥
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"]) # Mặc định chạy CPU
    parser.add_argument("--det-conf", type=float, default=0.3) # Giảm để bắt tay cực nhạy
    parser.add_argument("--track-conf", type=float, default=0.3) # Giảm để bám xương chặt
    parser.add_argument("--ema-alpha", type=float, default=0.3) # Tăng độ mượt nhảy số
    parser.add_argument("--landmark-ema-alpha", type=float, default=0.4)
    parser.add_argument("--max-hand-jump", type=float, default=0)
    parser.add_argument("--missing-reset-frames", type=int, default=30) # Ghi nhớ tay khi bị nắm lại (tàng hình)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.25) # Chỉ cần 25% là cho hiển thị
    parser.add_argument("--show-fps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay-lang", choices=["vi", "code", "both"], default="vi")
    parser.add_argument("--use-z", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-velocity", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    model_candidates = [Path("data/stgcn_best.pt"), Path("outputs_resume2/stgcn_best.pt")]
    labels_candidates = [Path("data/labels.json"), Path("outputs_resume2/labels.json")]

    model_path = Path(args.model) if args.model else first_existing(model_candidates)
    labels_path = Path(args.labels) if args.labels else first_existing(labels_candidates)

    if model_path is None or labels_path is None: raise FileNotFoundError("Thiếu file checkpoint .pt hoặc labels.json rồi bro ơi!")

    label_map = load_label_map(labels_path)
    labels = index_to_label(label_map, len(label_map))

    device = resolve_device(args.device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]

    in_channels = infer_in_channels_from_state(state)
    inferred_use_z, inferred_use_velocity, inferred_use_acceleration = infer_feature_config(in_channels)

    use_z = inferred_use_z if args.use_z is None else args.use_z
    use_velocity = inferred_use_velocity if args.use_velocity is None else args.use_velocity
    use_acceleration = inferred_use_acceleration

    edge_index = build_hand_edge_index()
    
    # 🌟 RADAR ÉP CHANNELS CHO 9 NÒNG
    try:
        model = STGCN(in_channels=in_channels, num_classes=len(labels), edge_index=edge_index, hidden_channels=64)
    except TypeError:
        try: model = STGCN(in_channels=in_channels, num_classes=len(labels), edge_index=edge_index, c_hid=64)
        except TypeError: model = STGCN(in_channels=in_channels, num_classes=len(labels), edge_index=edge_index)

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    detector, use_tasks = create_detector(Path(args.task_model), args.det_conf, args.track_conf)
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open camera id {args.camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    frame_buffer = deque(maxlen=args.length)
    prob_ema, landmark_ema = None, None
    missing_count, dropped_jump_frames = 0, 0
    fps_ema, prev_ts = None, time.perf_counter()
    double_click_cooldown = 0
    last_double_type = None

    print("Demo started thành công rực rỡ bro ơi!")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            now_ts = time.perf_counter()
            dt = max(now_ts - prev_ts, 1e-6)
            prev_ts, fps_inst = now_ts, 1.0 / dt
            fps_ema = fps_inst if fps_ema is None else (0.9 * fps_ema + 0.1 * fps_inst)

            landmarks = detect_landmarks(detector, frame, use_tasks)
            if landmarks is not None:
                missing_count = 0
                current = landmarks_to_array(landmarks)
                if landmark_ema is not None:
                    alpha = float(np.clip(args.landmark_ema_alpha, 0.0, 0.99))
                    current = alpha * landmark_ema + (1.0 - alpha) * current
                landmark_ema = current
                draw_landmarks(frame, current)
                frame_buffer.append(current)
            else:
                missing_count += 1
                if missing_count >= args.missing_reset_frames:
                    frame_buffer.clear()
                    prob_ema, landmark_ema = None, None

            status = f"Khung hinh: {len(frame_buffer)}/{args.length}"
            cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)

            if len(frame_buffer) == args.length:
                seq = np.stack(list(frame_buffer), axis=0)
                if not use_z: seq = seq[:, :, :2]
                
                # Vẫn tắt Normalize vì file gốc của bro không dùng
                # seq = normalize_frames(seq)
                
                if use_acceleration: seq = add_acceleration(seq)
                elif use_velocity: seq = add_velocity(seq)

                x = torch.from_numpy(seq).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
                with torch.no_grad(): probs = torch.softmax(model(x), dim=1).squeeze(0).cpu()

                prob_ema = probs if prob_ema is None else args.ema_alpha * prob_ema + (1.0 - args.ema_alpha) * probs
                scores, indices = torch.topk(prob_ema, min(args.topk, prob_ema.numel()))

                top_label = labels[int(indices[0].item())]
                top_conf = float(scores[0].item())
                
                # 🚀 LOGIC HACK: ÉP CUNG D0X 🚀
                # Nếu nó báo "Khong cu chi" nhưng cái nhãn Top 2 lại có Confidence > 0.25 -> Lấy luôn nhãn Top 2!
                if top_label == "D0X" and len(scores) > 1 and float(scores[1].item()) >= 0.25:
                    top_label = labels[int(indices[1].item())]
                    top_conf = float(scores[1].item())
                    
                top_text = format_label(top_label, args.overlay_lang)

                if top_label in ["G08", "G09"] and top_conf >= args.min_confidence:
                    double_click_cooldown = 15
                    last_double_type = top_label

                if double_click_cooldown > 0:
                    if last_double_type == "G08" and top_label == "G01": 
                        top_label, top_text = "B0A", format_label("B0A", args.overlay_lang)
                    elif last_double_type == "G09" and top_label == "G02": 
                        top_label, top_text = "B0B", format_label("B0B", args.overlay_lang)
                    double_click_cooldown -= 1

                display = f"{top_text} ({top_conf:.2f})" if top_conf >= args.min_confidence else f"Khong chac ({top_conf:.2f})"
                color = (30, 230, 30) if top_conf >= args.min_confidence else (40, 170, 250)
                
                cv2.putText(frame, display, (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                y = 112
                for score, idx in zip(scores.tolist(), indices.tolist()):
                    label_text = format_label(labels[idx], args.overlay_lang)
                    cv2.putText(frame, f"{label_text}: {score:.2f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    y += 22

            cv2.imshow("ST-GCN Hand Gesture Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            if cv2.waitKey(1) & 0xFF == ord("c"): frame_buffer.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(detector, "close"): detector.close()

if __name__ == "__main__":
    main()