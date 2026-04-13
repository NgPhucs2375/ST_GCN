# phần 2: chuẩn hóa dữ liệu

import argparse # thư viện dùng để phân tích cú pháp các đối số dòng lệnh
import json # thư viện dùng để làm việc với dữ liệu JSON
from pathlib import Path # thư viện dùng để làm việc với hệ thống tệp

import numpy as np # thư viện dùng để làm việc với mảng và tính toán khoa học

# Định nghĩa các kết nối giữa các khớp tay (21 khớp)
# ngón trỏ : 0-4
# ngón giữa : 0-8
# ngón áp út : 0-12
# ngón út : 0-16
# ngón cái : 0-20
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

# Hàm để tải chuỗi từ tệp JSON
def load_sequence(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def frames_to_array(frames) -> np.ndarray:
    if not frames:
        return np.zeros((0, 0, 0), dtype=np.float32)

    first = frames[0][0] if frames[0] else None
    if isinstance(first, dict):
        t = len(frames)
        v = len(frames[0]) if frames[0] else 0
        arr = np.zeros((t, v, 3), dtype=np.float32)
        for i, frame in enumerate(frames):
            for j, p in enumerate(frame):
                arr[i, j, 0] = float(p.get("x", 0.0))
                arr[i, j, 1] = float(p.get("y", 0.0))
                arr[i, j, 2] = float(p.get("z", 0.0))
        return arr

    return np.array(frames, dtype=np.float32)

# Hàm để chuẩn hóa các khung hình
def normalize_frames(frames: np.ndarray) -> np.ndarray:
    # frames: (T, V, C)
    wrist = frames[:, 0:1, :]
    frames = frames - wrist

    palm = frames[:, 9:10, :]
    scale = np.linalg.norm(palm, axis=-1, keepdims=True)
    scale[scale == 0] = 1.0
    frames = frames / scale
    return frames

# Hàm để thêm vận tốc vào các khung hình
def add_velocity(frames: np.ndarray) -> np.ndarray:
    """Add velocity (1st derivative) and acceleration (2nd derivative)"""
    # frames: (T, V, C) - normalized coordinates
    # velocity: difference between consecutive frames
    velocity = np.diff(frames, axis=0, prepend=frames[:1])
    
    # acceleration: difference of velocity (2nd derivative)
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    
    # Concatenate: [position, velocity, acceleration]
    # Output shape: (T, V, C*3) e.g., (T, 21, 6) if input is (T, 21, 2)
    result = np.concatenate([frames, velocity, acceleration], axis=-1)
    return result

# Hàm để đệm hoặc cắt các khung hình về độ dài mục tiêu
def pad_or_trim(frames: np.ndarray, target_len: int) -> np.ndarray:
    length = frames.shape[0]
    if length == target_len:
        return frames
    if length > target_len:
        return frames[:target_len]
    pad = np.repeat(frames[-1:], target_len - length, axis=0)
    return np.concatenate([frames, pad], axis=0)

# Hàm chính để chuyển đổi chuỗi từ JSON sang NPZ
# NPZ : Định dạng tệp nén của NumPy để lưu trữ nhiều mảng
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder with JSON sequences")
    parser.add_argument("--output", required=True, help="Output npz file")
    parser.add_argument("--length", type=int, default=30)
    parser.add_argument("--use-z", action="store_true")
    parser.add_argument("--use-velocity", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input)
    sequences = []
    labels = []

    for path in sorted(input_dir.glob("*.json")):
        payload = load_sequence(path)
        frames = payload["frames"]
        label = payload.get("label", "unknown")

        array = frames_to_array(frames)
        if not args.use_z:
            array = array[:, :, :2]

        array = normalize_frames(array)
        if args.use_velocity:
            array = add_velocity(array)

        array = pad_or_trim(array, args.length)

        sequences.append(array)
        labels.append(label)

    if not sequences:
        raise RuntimeError("No JSON files found")

    np.savez(
        args.output,
        sequences=np.stack(sequences, axis=0),
        labels=np.array(labels),
    )

# Chạy hàm chính khi tập lệnh được thực thi trực tiếp
if __name__ == "__main__":
    main()
