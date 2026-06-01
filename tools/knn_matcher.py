"""
KNN-based gesture matching using pre-calibrated gesture templates.

Usage:
    matcher = KNNGestureMatcher(template_file)
    gesture_id, confidence = matcher.match(landmarks_array)
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

class KNNGestureMatcher:
    """KNN gesture matcher using Euclidean distance on normalized landmarks."""

    def __init__(self, template_file: Path, k: int = 3):
        """
        Args:
            template_file: Path to gesture_templates.json
            k: Number of nearest neighbors to use
        """
        self.k = k
        self.templates = {}
        self.load_templates(template_file)

    def load_templates(self, template_file: Path):
        """Load gesture templates from JSON file."""
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")

        with open(template_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for gesture_id, template_info in data.items():
            self.templates[gesture_id] = {
                "gesture_name": template_info.get("gesture_name", gesture_id),
                "mean": np.array(template_info["mean"], dtype=np.float32),
                "std": np.array(template_info["std"], dtype=np.float32),
            }

        print(f"✅ Loaded {len(self.templates)} gesture templates")

    def compute_distance(self, landmarks: np.ndarray, template_mean: np.ndarray, template_std: np.ndarray) -> float:
        """Compute Mahalanobis-like distance between landmarks and template."""
        # Normalize by template std (avoid division by zero)
        std_safe = np.where(template_std > 1e-6, template_std, 1.0)
        
        # Compute normalized difference
        diff = (landmarks - template_mean) / std_safe
        
        # Euclidean distance on normalized space
        distance = float(np.linalg.norm(diff))
        return distance

    def match(self, landmarks: np.ndarray, threshold: float = 5.0) -> Tuple[str, float]:
        """
        Match landmarks to closest gesture using KNN.

        Args:
            landmarks: (21, 3) normalized landmarks array
            threshold: Max distance to consider a match (higher = more permissive)

        Returns:
            (gesture_id, confidence) where confidence = 1 - (distance / threshold)
        """
        if landmarks.shape != (21, 3):
            raise ValueError(f"Expected landmarks shape (21, 3), got {landmarks.shape}")

        distances = []
        for gesture_id, template_info in self.templates.items():
            dist = self.compute_distance(
                landmarks,
                template_info["mean"],
                template_info["std"]
            )
            distances.append((gesture_id, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Get closest gesture
        closest_id, closest_dist = distances[0]

        # Compute confidence (inverse of normalized distance)
        confidence = max(0.0, 1.0 - (closest_dist / threshold))

        return closest_id, confidence

    def match_with_knn(self, landmarks: np.ndarray, k: Optional[int] = None) -> Tuple[str, float]:
        """
        Match using KNN voting (return most common gesture among k nearest).

        Args:
            landmarks: (21, 3) normalized landmarks
            k: Number of neighbors (uses self.k if None)

        Returns:
            (gesture_id, confidence)
        """
        if k is None:
            k = self.k

        distances = []
        for gesture_id, template_info in self.templates.items():
            dist = self.compute_distance(
                landmarks,
                template_info["mean"],
                template_info["std"]
            )
            distances.append((gesture_id, dist))

        # Sort and get top k
        distances.sort(key=lambda x: x[1])
        top_k = distances[:k]

        # Vote
        votes = {}
        for gesture_id, dist in top_k:
            votes[gesture_id] = votes.get(gesture_id, 0) + 1

        best_gesture = max(votes, key=votes.get)
        confidence = votes[best_gesture] / k

        return best_gesture, confidence
