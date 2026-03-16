"""Data quality checks for MediaPipe hand landmark sequences.

Input:
- Folder containing JSON sequences saved from web UI.

Each JSON is expected to have:
- label: string
- frames: list[frame]
  - frame: list[point] length ~21
  - point: {x, y, z}

Output:
- CSV report with per-file stats
- JSON summary with per-class counts and reasons for failures
- Optional: copy OK files into a clean folder

This script intentionally uses simple heuristics (no ML) to help you spot:
- too short sequences
- wrong landmark count
- out-of-range coordinates
- tracking glitches (large jumps between consecutive frames)
- duplicates
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


EXPECTED_LANDMARKS = 21


@dataclass
class FileStats:
    path: Path
    label: str
    num_frames: int
    landmarks_ok: bool
    out_of_range_ratio: float
    max_wrist_jump: float
    max_mean_jump: float
    duplicate_of: Optional[str]
    ok: bool
    reasons: List[str]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def frames_to_array(frames: List[List[Dict[str, Any]]], use_z: bool = True) -> np.ndarray:
    # Return array shape (T, V, C)
    if not frames:
        return np.zeros((0, 0, 0), dtype=np.float32)

    channels = 3 if use_z else 2
    t = len(frames)
    v = len(frames[0]) if frames[0] else 0
    arr = np.zeros((t, v, channels), dtype=np.float32)

    for i, frame in enumerate(frames):
        for j, p in enumerate(frame):
            arr[i, j, 0] = float(p.get("x", 0.0))
            arr[i, j, 1] = float(p.get("y", 0.0))
            if use_z and channels == 3:
                arr[i, j, 2] = float(p.get("z", 0.0))

    return arr


def sequence_hash(arr: np.ndarray, decimals: int = 3) -> str:
    # Hash rounded coordinates to detect exact/near-exact duplicates.
    rounded = np.round(arr.astype(np.float32), decimals=decimals)
    digest = hashlib.sha1(rounded.tobytes()).hexdigest()
    return digest


def compute_jumps(arr: np.ndarray) -> Tuple[float, float]:
    # arr: (T, V, C)
    if arr.shape[0] < 2:
        return 0.0, 0.0

    diffs = np.linalg.norm(arr[1:, :, :2] - arr[:-1, :, :2], axis=-1)  # (T-1, V)
    wrist = diffs[:, 0]  # landmark 0
    max_wrist_jump = float(np.max(wrist))
    mean_per_frame = np.mean(diffs, axis=1)  # (T-1,)
    max_mean_jump = float(np.max(mean_per_frame))
    return max_wrist_jump, max_mean_jump


def out_of_range_ratio(arr: np.ndarray, tol: float) -> float:
    # x,y should be roughly in [0,1] for MediaPipe normalized coords.
    if arr.size == 0:
        return 1.0

    xy = arr[:, :, :2]
    bad = (xy < -tol) | (xy > (1.0 + tol))
    return float(np.mean(bad))


def is_finite(arr: np.ndarray) -> bool:
    return bool(np.isfinite(arr).all())


def check_file(
    path: Path,
    min_frames: int,
    max_frames: int,
    tol_xy: float,
    max_wrist_jump: float,
    max_mean_jump: float,
    expected_landmarks: int,
    dedup_map: Dict[str, str],
    dedup_decimals: int,
) -> FileStats:
    payload = load_json(path)
    label = str(payload.get("label", "unknown"))
    frames = payload.get("frames", [])

    reasons: List[str] = []

    # Basic shape checks
    num_frames = len(frames)
    if num_frames < min_frames:
        reasons.append(f"too_short(<{min_frames})")
    if max_frames > 0 and num_frames > max_frames:
        reasons.append(f"too_long(>{max_frames})")

    landmarks_ok = True
    for f in frames:
        if not isinstance(f, list) or len(f) != expected_landmarks:
            landmarks_ok = False
            break
    if not landmarks_ok:
        reasons.append("bad_landmark_count")

    arr = frames_to_array(frames, use_z=True) if landmarks_ok else np.zeros((num_frames, 0, 0), dtype=np.float32)

    if landmarks_ok:
        if not is_finite(arr):
            reasons.append("non_finite")

    oor = out_of_range_ratio(arr, tol_xy) if landmarks_ok else 1.0
    if landmarks_ok and oor > 0.02:
        # more than 2% points are outside [0,1] range (with tolerance)
        reasons.append("out_of_range")

    mwj, mmj = compute_jumps(arr) if landmarks_ok else (math.inf, math.inf)
    if landmarks_ok and mwj > max_wrist_jump:
        reasons.append(f"wrist_jump(>{max_wrist_jump})")
    if landmarks_ok and mmj > max_mean_jump:
        reasons.append(f"mean_jump(>{max_mean_jump})")

    dup_of: Optional[str] = None
    if landmarks_ok and num_frames > 0:
        h = sequence_hash(arr, decimals=dedup_decimals)
        if h in dedup_map:
            dup_of = dedup_map[h]
            reasons.append("duplicate")
        else:
            dedup_map[h] = path.name

    ok = len(reasons) == 0
    return FileStats(
        path=path,
        label=label,
        num_frames=num_frames,
        landmarks_ok=landmarks_ok,
        out_of_range_ratio=oor,
        max_wrist_jump=mwj,
        max_mean_jump=mmj,
        duplicate_of=dup_of,
        ok=ok,
        reasons=reasons,
    )


def write_csv(stats: List[FileStats], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "label",
                "num_frames",
                "ok",
                "reasons",
                "out_of_range_ratio",
                "max_wrist_jump",
                "max_mean_jump",
                "duplicate_of",
            ]
        )
        for s in stats:
            w.writerow(
                [
                    s.path.name,
                    s.label,
                    s.num_frames,
                    int(s.ok),
                    ";".join(s.reasons),
                    f"{s.out_of_range_ratio:.6f}",
                    f"{s.max_wrist_jump:.6f}" if math.isfinite(s.max_wrist_jump) else "inf",
                    f"{s.max_mean_jump:.6f}" if math.isfinite(s.max_mean_jump) else "inf",
                    s.duplicate_of or "",
                ]
            )


def write_summary(stats: List[FileStats], out_json: Path) -> None:
    total = len(stats)
    ok_count = sum(1 for s in stats if s.ok)
    reason_counts = Counter(r for s in stats for r in s.reasons)

    per_class_total = Counter(s.label for s in stats)
    per_class_ok = Counter(s.label for s in stats if s.ok)

    summary = {
        "total_files": total,
        "ok_files": ok_count,
        "bad_files": total - ok_count,
        "reason_counts": dict(reason_counts),
        "per_class_total": dict(per_class_total),
        "per_class_ok": dict(per_class_ok),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def copy_ok_files(stats: List[FileStats], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for s in stats:
        if s.ok:
            shutil.copy2(s.path, out_dir / s.path.name)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="Folder with raw JSON sequences")
    parser.add_argument("--report-csv", default="outputs/data_quality_report.csv")
    parser.add_argument("--report-json", default="outputs/data_quality_summary.json")
    parser.add_argument("--copy-ok-to", default="", help="If set, copy OK JSON files to this folder")

    # Heuristics
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    parser.add_argument("--expected-landmarks", type=int, default=EXPECTED_LANDMARKS)
    parser.add_argument("--tol-xy", type=float, default=0.05, help="Allowed xy tolerance outside [0,1]")
    parser.add_argument("--max-wrist-jump", type=float, default=0.25)
    parser.add_argument("--max-mean-jump", type=float, default=0.15)

    # Dedup
    parser.add_argument("--dedup-decimals", type=int, default=3)

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    dedup_map: Dict[str, str] = {}
    stats: List[FileStats] = []

    for path in sorted(input_dir.glob("*.json")):
        stats.append(
            check_file(
                path=path,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                tol_xy=args.tol_xy,
                max_wrist_jump=args.max_wrist_jump,
                max_mean_jump=args.max_mean_jump,
                expected_landmarks=args.expected_landmarks,
                dedup_map=dedup_map,
                dedup_decimals=args.dedup_decimals,
            )
        )

    if not stats:
        print(f"No JSON files found in {input_dir}")
        return

    out_csv = Path(args.report_csv)
    out_json = Path(args.report_json)
    write_csv(stats, out_csv)
    write_summary(stats, out_json)

    ok_count = sum(1 for s in stats if s.ok)
    print(f"Checked {len(stats)} files | OK: {ok_count} | Bad: {len(stats) - ok_count}")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")

    if args.copy_ok_to:
        copied = copy_ok_files(stats, Path(args.copy_ok_to))
        print(f"Copied {copied} OK files to {args.copy_ok_to}")


if __name__ == "__main__":
    main()
