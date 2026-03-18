import argparse
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"


@dataclass
class Annotation:
    video: str
    class_id: str
    start_frame: int
    end_frame: int
    num_frames: int


def parse_class_details(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def parse_annotations(path: Path) -> List[Annotation]:
    rows: List[Annotation] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            if parts[0].lower() == "video":
                continue
            video = parts[0]
            class_id = parts[1]
            try:
                start = int(float(parts[3]))
                end = int(float(parts[4]))
                num_frames = int(float(parts[5]))
            except ValueError:
                continue
            if end <= start:
                continue
            rows.append(Annotation(video, class_id, start, end, num_frames))
    return rows


def find_video_path(videos_dir: Path, video_name: str) -> Optional[Path]:
    direct = videos_dir / video_name
    if direct.exists():
        return direct

    for ext in VIDEO_EXTS:
        cand = videos_dir / f"{video_name}{ext}"
        if cand.exists():
            return cand

    matches = list(videos_dir.glob(f"{video_name}.*"))
    if matches:
        return matches[0]

    return None


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")
    return cleaned or "unknown"


def ensure_task_model(path: Path) -> Path:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TASK_MODEL_URL, path)
    return path


def extract_landmarks(detector, rgb, use_tasks: bool):
    if use_tasks:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(image)
        if result.hand_landmarks:
            return result.hand_landmarks[0]
        return None

    result = detector.process(rgb)
    if result.multi_hand_landmarks:
        return result.multi_hand_landmarks[0].landmark
    return None


def extract_sequence(
    cap: cv2.VideoCapture,
    detector,
    start_frame: int,
    end_frame: int,
    frame_step: int,
    use_tasks: bool,
) -> List[List[Dict[str, float]]]:
    frames: List[List[Dict[str, float]]] = []
    start_idx = max(0, start_frame - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frame_no = start_frame

    while frame_no < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_no - start_frame) % frame_step != 0:
            frame_no += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = extract_landmarks(detector, rgb, use_tasks)
        if landmarks:
            frame_points = [
                {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
                for p in landmarks
            ]
            frames.append(frame_points)

        frame_no += 1

    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipn-root", required=True, help="Root of IPN dataset")
    parser.add_argument("--annotations", default="annotations/annotations/Annot_List.txt")
    parser.add_argument("--class-details", default="annotations/annotations/class_details.txt")
    parser.add_argument("--videos", default="videos/videos")
    parser.add_argument("--output", required=True, help="Output folder for JSON sequences")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--det-conf", type=float, default=0.6)
    parser.add_argument("--track-conf", type=float, default=0.6)
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument(
        "--task-model",
        default="tools/assets/hand_landmarker.task",
        help="Path to MediaPipe task model (used when mp.solutions is unavailable)",
    )
    args = parser.parse_args()

    ipn_root = Path(args.ipn_root)
    ann_path = ipn_root / args.annotations
    class_path = ipn_root / args.class_details
    videos_dir = ipn_root / args.videos
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_map = parse_class_details(class_path)
    annotations = parse_annotations(ann_path)
    if not annotations:
        raise RuntimeError("No annotations found")

    use_tasks = not hasattr(mp, "solutions")
    if use_tasks:
        model_path = ensure_task_model(Path(args.task_model))
        base_options = BaseOptions(model_asset_path=str(model_path))
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=args.det_conf,
            min_tracking_confidence=args.track_conf,
        )
        detector = HandLandmarker.create_from_options(options)
    else:
        detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.det_conf,
            min_tracking_confidence=args.track_conf,
        )

    per_class_count: Dict[str, int] = Counter()
    total_written = 0
    skipped = 0

    for ann in annotations:
        if args.max_samples > 0 and total_written >= args.max_samples:
            break

        label = class_map.get(ann.class_id, ann.class_id)
        safe_label = sanitize_label(label)

        if args.max_per_class > 0 and per_class_count[safe_label] >= args.max_per_class:
            continue

        video_path = find_video_path(videos_dir, ann.video)
        if not video_path:
            skipped += 1
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            skipped += 1
            continue

        frames = extract_sequence(
            cap,
            detector,
            ann.start_frame,
            ann.end_frame,
            args.frame_step,
            use_tasks,
        )
        cap.release()

        if len(frames) < args.min_frames:
            skipped += 1
            continue

        payload = {
            "label": safe_label,
            "frames": frames,
            "source": {
                "video": video_path.name,
                "start_frame": ann.start_frame,
                "end_frame": ann.end_frame,
                "frame_step": args.frame_step,
            },
        }

        out_name = f"{safe_label}_{ann.video}_{ann.start_frame}_{ann.end_frame}.json"
        out_path = out_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        per_class_count[safe_label] += 1
        total_written += 1

    if hasattr(detector, "close"):
        detector.close()
    print(f"Written: {total_written} JSON files")
    print(f"Skipped: {skipped}")
    print(f"Classes: {len(per_class_count)}")


if __name__ == "__main__":
    main()
