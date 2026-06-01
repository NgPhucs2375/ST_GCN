#!/usr/bin/env python
"""
Quick test: import gesture calibration modules
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

print("Testing imports...")

try:
    from tools.gesture_calibrator import GestureCalibrator, GestureTemplate
    print("✅ gesture_calibrator imported successfully")
except Exception as e:
    print(f"❌ gesture_calibrator import failed: {e}")
    sys.exit(1)

try:
    from tools.knn_matcher import KNNGestureMatcher
    print("✅ knn_matcher imported successfully")
except Exception as e:
    print(f"❌ knn_matcher import failed: {e}")
    sys.exit(1)

try:
    from tools.calibration_ui import CalibrationUI
    print("✅ calibration_ui imported successfully")
except Exception as e:
    print(f"❌ calibration_ui import failed: {e}")
    sys.exit(1)

# Test basic functionality
print("\nTesting basic functionality...")

try:
    calibrator = GestureCalibrator()
    print(f"✅ GestureCalibrator created. Templates: {len(calibrator.templates)}")
except Exception as e:
    print(f"❌ GestureCalibrator creation failed: {e}")
    sys.exit(1)

try:
    ui = CalibrationUI(calibrator)
    print(f"✅ CalibrationUI created. Current gesture: {ui.get_current_gesture()}")
except Exception as e:
    print(f"❌ CalibrationUI creation failed: {e}")
    sys.exit(1)

try:
    matcher = KNNGestureMatcher(calibrator, k=3)
    print(f"✅ KNNGestureMatcher created")
except Exception as e:
    print(f"❌ KNNGestureMatcher creation failed: {e}")
    sys.exit(1)

print("\n✅ All imports and basic functionality tests passed!")
print("\nYou can now run:")
print("  python tools/demo_webcam.py --calibration-mode")
