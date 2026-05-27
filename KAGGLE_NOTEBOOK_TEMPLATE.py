"""
Kaggle notebook template for the refactored single-entry training workflow.

Usage:
1. Upload a ZIP that contains this project (or at least the code + data you need).
2. Unzip it in Kaggle.
3. Run `train.py` directly with command-line arguments.
"""

from pathlib import Path
import subprocess


ARCHIVE = "/kaggle/input/<your-dataset>/<your-project>.zip"
WORKDIR = Path("/kaggle/working/dl_demo")

print("Step 1: install dependencies if needed")
print("!pip install torch-geometric -q")

print("\nStep 2: unpack the project archive")
WORKDIR.mkdir(parents=True, exist_ok=True)
subprocess.run(["unzip", "-q", ARCHIVE, "-d", str(WORKDIR)], check=True)

print("\nStep 3: run training directly")
subprocess.run(
    [
        "python",
        "train.py",
        "--data",
        "data/processed/train_data_4ch.npz",
        "--channels",
        "4",
        "--batch-size",
        "32",
        "--class-weighted-loss",
        "--patience",
        "20",
        "--scheduler-patience",
        "5",
        "--scheduler",
        "plateau",
        "--scheduler-monitor",
        "val_loss",
        "--out",
        "outputs_kaggle",
    ],
    check=True,
)

print("\nDone. Check outputs_kaggle/stgcn_best.pt after training.")
