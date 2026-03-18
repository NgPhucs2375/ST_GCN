import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_ADD = "G04=78,G05=58,G06=55,G03=49,G01=37,G09=37,B0B=35,G02=32"


def parse_add_map(text: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    text = text.strip()
    if not text:
        return mapping
    for item in text.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid add item: {item}")
        label, count = item.split("=", 1)
        label = label.strip()
        count = int(count.strip())
        if count < 0:
            raise ValueError(f"Add count must be >= 0: {item}")
        mapping[label] = count
    return mapping


def detect_velocity_channels(num_channels: int) -> bool:
    return num_channels in (4, 6)


def recompute_velocity(pos: torch.Tensor) -> torch.Tensor:
    # pos: (T, V, C)
    return torch.diff(pos, dim=0, prepend=pos[:1])


def time_warp_sequence(pos: torch.Tensor, time_warp: float, gen: torch.Generator) -> torch.Tensor:
    if time_warp <= 0:
        return pos
    t, v, c = pos.shape
    scale = float((1 - time_warp) + (2 * time_warp) * torch.rand((), generator=gen).item())
    new_len = max(2, int(round(t * scale)))

    xt = pos.permute(2, 1, 0).unsqueeze(0)  # (1, C, V, T)
    xt = F.interpolate(xt, size=(v, new_len), mode="bilinear", align_corners=False)
    pos = xt.squeeze(0).permute(2, 1, 0).contiguous()  # (new_len, V, C)

    if pos.size(0) > t:
        pos = pos[:t]
    elif pos.size(0) < t:
        pad = pos[-1:].repeat(t - pos.size(0), 1, 1)
        pos = torch.cat([pos, pad], dim=0)

    return pos


def augment_sequence(
    x: torch.Tensor,
    jitter_std: float,
    time_warp: float,
    flip_prob: float,
    drop_frames: int,
    gen: torch.Generator,
) -> torch.Tensor:
    # x: (T, V, C) on CPU
    t, v, c = x.shape
    has_vel = detect_velocity_channels(c)
    pos_c = c // 2 if has_vel else c

    pos = x[:, :, :pos_c].clone()

    # 1) Jitter
    if jitter_std > 0:
        noise = torch.randn(pos.shape, generator=gen) * jitter_std
        pos = pos + noise

    # 2) Flip on x-axis
    if flip_prob > 0 and torch.rand((), generator=gen).item() < flip_prob:
        pos[:, :, 0] = -pos[:, :, 0]

    # 3) Time-warp
    pos = time_warp_sequence(pos, time_warp, gen)

    # 4) Drop frames
    if drop_frames > 0 and t > 2:
        k = min(drop_frames, t - 1)
        idx = torch.randperm(t - 1, generator=gen)[:k] + 1
        pos[idx] = pos[idx - 1]

    if has_vel:
        vel = recompute_velocity(pos)
        x = torch.cat([pos, vel], dim=-1)
    else:
        x = pos

    return x


def build_class_index(labels: List[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        mapping[label].append(i)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input npz file")
    parser.add_argument("--output", required=True, help="Output npz file")
    parser.add_argument("--add", default=DEFAULT_ADD, help="Add map like G04=78,G05=58")
    parser.add_argument("--seed", type=int, default=42)

    # Augmentation params
    parser.add_argument("--aug-jitter-std", type=float, default=0.0)
    parser.add_argument("--aug-time-warp", type=float, default=0.0)
    parser.add_argument("--aug-flip-prob", type=float, default=0.0)
    parser.add_argument("--aug-drop-frames", type=int, default=0)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    add_map = parse_add_map(args.add)
    if not add_map:
        raise ValueError("Add map is empty; provide --add")

    data = np.load(input_path, allow_pickle=True)
    sequences = data["sequences"]
    labels = data["labels"].tolist()

    class_index = build_class_index(labels)
    counts_before = Counter(labels)

    rng = np.random.default_rng(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    new_sequences = []
    new_labels: List[str] = []

    for label, add_count in add_map.items():
        if add_count <= 0:
            continue
        if label not in class_index or not class_index[label]:
            raise ValueError(f"Label not found in input data: {label}")

        indices = class_index[label]
        for _ in range(add_count):
            src_idx = int(rng.choice(indices))
            x = torch.from_numpy(sequences[src_idx]).float()
            x_aug = augment_sequence(
                x,
                jitter_std=args.aug_jitter_std,
                time_warp=args.aug_time_warp,
                flip_prob=args.aug_flip_prob,
                drop_frames=args.aug_drop_frames,
                gen=gen,
            )
            new_sequences.append(x_aug.numpy())
            new_labels.append(label)

    if new_sequences:
        new_sequences_arr = np.stack(new_sequences, axis=0)
        sequences_out = np.concatenate([sequences, new_sequences_arr], axis=0)
        labels_out = np.array(labels + new_labels)
    else:
        sequences_out = sequences
        labels_out = np.array(labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, sequences=sequences_out, labels=labels_out)

    counts_after = Counter(labels_out.tolist())

    print("Input:", input_path)
    print("Output:", output_path)
    print("Added:", add_map)
    print("Counts before:", {k: counts_before.get(k, 0) for k in sorted(add_map)})
    print("Counts after:", {k: counts_after.get(k, 0) for k in sorted(add_map)})
    print("Total before:", len(labels))
    print("Total after:", len(labels_out))


if __name__ == "__main__":
    main()
