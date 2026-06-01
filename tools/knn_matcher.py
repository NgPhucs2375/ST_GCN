"""
KNN Gesture Matcher - Fast real-time gesture matching using precomputed templates.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from gesture_calibrator import GestureCalibrator


class KNNGestureMatcher:
    """K-Nearest Neighbors matcher cho gesture recognition."""
    
    def __init__(self, calibrator: GestureCalibrator, k: int = 3):
        self.calibrator = calibrator
        self.k = k
        self.template_buffer: List[Tuple[str, np.ndarray]] = []
        self._build_buffer()
    
    def _build_buffer(self) -> None:
        """Build cached buffer của mean landmarks cho mỗi gesture."""
        self.template_buffer = []
        for gid, template in self.calibrator.templates.items():
            mean_lm = np.array(template.mean_landmarks, dtype=np.float32)
            self.template_buffer.append((gid, mean_lm))
    
    def predict_knn(self, landmarks: np.ndarray, 
                    threshold: float = 2.0) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
        """
        KNN prediction. Return (top_gesture, top_distance, neighbors_list) or None if no match.
        """
        if not self.template_buffer:
            return None
        
        if landmarks.shape != (21, 3):
            return None
        
        # Compute distances to all templates
        distances: Dict[str, float] = {}
        for gid, mean_lm in self.template_buffer:
            template = self.calibrator.templates[gid]
            std_array = np.array(template.std_landmarks, dtype=np.float32)
            std_array[std_array < 1e-6] = 1.0
            
            dist = np.linalg.norm((landmarks - mean_lm) / std_array)
            distances[gid] = float(dist)
        
        # Sort by distance
        sorted_gestures = sorted(distances.items(), key=lambda x: x[1])
        
        # Top gesture
        top_gid, top_dist = sorted_gestures[0]
        
        if top_dist > threshold:
            return None
        
        # K neighbors
        k_neighbors = sorted_gestures[:self.k]
        
        return top_gid, top_dist, k_neighbors
    
    def predict_voting(self, landmarks_sequence: List[np.ndarray],
                       threshold: float = 2.0) -> Optional[Tuple[str, float]]:
        """
        Predict gesture từ sequence of frames using voting (majority).
        Return (gesture_id, confidence) or None.
        """
        predictions = []
        for lm in landmarks_sequence:
            result = self.predict_knn(lm, threshold=threshold * 1.5)  # relax threshold for sequence
            if result:
                gid, dist, _ = result
                predictions.append((gid, 1.0 / (1.0 + dist)))  # confidence = 1/(1+distance)
        
        if not predictions:
            return None
        
        # Voting
        gesture_votes: Dict[str, float] = {}
        for gid, conf in predictions:
            gesture_votes[gid] = gesture_votes.get(gid, 0) + conf
        
        best_gid = max(gesture_votes, key=gesture_votes.get)
        best_score = gesture_votes[best_gid]
        
        # Normalize confidence
        total_vote = sum(gesture_votes.values())
        confidence = best_score / total_vote if total_vote > 0 else 0.0
        
        return best_gid, confidence
