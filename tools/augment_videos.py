
import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm

# Define the set of augmentations to apply to each frame
# p=0.5 means each augmentation has a 50% chance of being applied
AUGMENTATION_PIPELINE = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
    ]
)

# A separate pipeline for the horizontally flipped version
FLIP_PIPELINE = A.Compose(
    [
        A.HorizontalFlip(always_apply=True),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
    ]
)


def augment_video(video_path: Path, output_dir: Path, pipeline: A.Compose, suffix: str):
    """
    Reads a video, applies augmentations to each frame, and saves the new video.
    """
    # Create the output path
    output_filename = f"{video_path.stem}_{suffix}.avi"
    output_path = output_dir / output_filename

    # Open the source video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Create video writer
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply augmentation
        augmented_frame = pipeline(image=frame)["image"]

        # Write the frame
        out.write(augmented_frame)

    # Release everything
    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(
        description="Augment video data for hand gesture recognition."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/videos/videos",
        help="Directory containing the original videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/videos_augmented",
        help="Directory to save the augmented videos.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of all .avi files
    video_files = list(input_path.glob("*.avi"))

    if not video_files:
        print(f"Error: No .avi files found in {input_path}")
        return

    print(
        f"Found {len(video_files)} videos. Starting augmentation..."
        f"\nEach video will generate 2 new versions (normal + flipped)."
        f"\nTotal videos to be created: {len(video_files) * 2}"
    )

    # Process each video with a progress bar
    for video_file in tqdm(video_files, desc="Augmenting Videos"):
        # Create a normally augmented version
        augment_video(video_file, output_path, AUGMENTATION_PIPELINE, "aug")
        # Create a flipped and augmented version
        augment_video(video_file, output_path, FLIP_PIPELINE, "aug_flip")

    print("\n-----------------------------------------")
    print("✅ Video augmentation complete!")
    print(f"New videos are saved in: {output_path}")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
