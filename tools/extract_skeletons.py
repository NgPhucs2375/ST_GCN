
import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from tqdm import tqdm

# Allow running as "python tools/extract_skeletons.py" from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.demo_webcam import ensure_task_model


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand skeletons from a directory of videos."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/videos_augmented",
        help="Directory containing the augmented videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_ipn_augmented",
        help="Directory to save the raw skeleton JSON files.",
    )
    parser.add_argument("--det-conf", type=float, default=0.5)
    parser.add_argument("--track-conf", type=float, default=0.5)
    parser.add_argument("--task-model", default="tools/assets/hand_landmarker.task")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Initialize MediaPipe Hand Landmarker ---
    task_model_path = ensure_task_model(Path(args.task_model))
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_model_path)),
        num_hands=1,
        min_hand_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
    )
    detector = HandLandmarker.create_from_options(options)

    video_files = sorted(list(input_path.glob("*.avi")))
    if not video_files:
        print(f"Error: No .avi files found in {input_path}")
        return

    print(f"Found {len(video_files)} videos. Starting skeleton extraction...")

    for video_file in tqdm(video_files, desc="Extracting Skeletons"):
        output_json_path = output_path / f"{video_file.stem}.json"

        # Bỏ qua nếu file JSON đã tồn tại (Giúp resume tiến trình)
        if output_json_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_file}. Skipping.")
            continue

        all_frames_data = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)

            frame_landmarks = []
            if result.hand_landmarks:
                # Get the first detected hand
                hand_landmarks = result.hand_landmarks[0]
                for landmark in hand_landmarks:
                    frame_landmarks.append([landmark.x, landmark.y, landmark.z])
            else:
                # If no hand is detected, add a frame of zeros (21 landmarks, 3 coords)
                frame_landmarks = [[0.0, 0.0, 0.0]] * 21
            
            all_frames_data.append({"landmarks": frame_landmarks})

        cap.release()

        # Save all frames for this video to a single JSON file
        with open(output_json_path, "w") as f:
            json.dump({"frames": all_frames_data}, f)

    detector.close()
    print("\n-----------------------------------------")
    print("✅ Skeleton extraction complete!")
    print(f"New JSON files are saved in: {output_path}")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
