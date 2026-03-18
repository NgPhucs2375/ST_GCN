import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Allow running as "python tools/infer.py" from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stgcn import STGCN, build_hand_edge_index


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_label_map(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_to_label(label_map: Dict[str, int], num_classes: int) -> List[str]:
    labels = ["" for _ in range(num_classes)]
    for label, idx in label_map.items():
        if 0 <= idx < num_classes:
            labels[idx] = label
    for i, label in enumerate(labels):
        if not label:
            labels[i] = f"class_{i}"
    return labels


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


def pad_or_trim(frames: np.ndarray, target_len: int) -> np.ndarray:
    length = frames.shape[0]
    if length == target_len:
        return frames
    if length > target_len:
        return frames[:target_len]
    pad = np.repeat(frames[-1:], target_len - length, axis=0)
    return np.concatenate([frames, pad], axis=0)


def load_sequence_from_json(path: Path, use_z: bool, use_velocity: bool, length: int) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload.get("frames", [])
    array = frames_to_array(frames)
    if array.size == 0:
        raise ValueError("No frames found in JSON")

    if not use_z:
        array = array[:, :, :2]

    array = normalize_frames(array)
    if use_velocity:
        array = add_velocity(array)

    return pad_or_trim(array, length)


def load_sequence_from_npz(path: Path, index: int) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    sequences = data["sequences"]
    if index < 0 or index >= len(sequences):
        raise IndexError("Index out of range")
    return sequences[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to stgcn_best.pt")
    parser.add_argument("--labels", required=True, help="Path to labels.json")
    parser.add_argument("--json", help="Path to a JSON sequence")
    parser.add_argument("--npz", help="Path to an NPZ file")
    parser.add_argument("--index", type=int, default=0, help="Index for NPZ inference")
    parser.add_argument("--length", type=int, default=30)
    parser.add_argument("--use-z", action="store_true")
    parser.add_argument("--use-velocity", action="store_true")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if not args.json and not args.npz:
        raise SystemExit("Provide --json or --npz")

    if args.json:
        array = load_sequence_from_json(Path(args.json), args.use_z, args.use_velocity, args.length)
    else:
        array = load_sequence_from_npz(Path(args.npz), args.index)

    num_channels = array.shape[-1]

    label_map = load_label_map(Path(args.labels))
    labels = index_to_label(label_map, len(label_map))

    edge_index = build_hand_edge_index()
    model = STGCN(in_channels=num_channels, num_classes=len(labels), edge_index=edge_index)

    device = resolve_device(args.device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    x = torch.from_numpy(array).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)

    topk = min(args.topk, probs.numel())
    values, indices = torch.topk(probs, k=topk)

    for score, idx in zip(values.tolist(), indices.tolist()):
        print(f"{labels[idx]}\t{score:.4f}")


if __name__ == "__main__":
    main()
