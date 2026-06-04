"""
KNN Gesture Matcher - Fast real-time gesture matching using precomputed templates.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from gesture_calibrator import GestureCalibrator, SEQUENCE_LENGTH


class KNNGestureMatcher:
    """K-Nearest Neighbors matcher cho gesture recognition dựa trên chuỗi."""
    
    def __init__(self, calibrator: GestureCalibrator, k: int = 3):
        self.calibrator = calibrator
        self.k = k
        self.template_buffer: List[Tuple[str, np.ndarray]] = []
        self._build_buffer()
    
    def _build_buffer(self) -> None:
        """Build cached buffer của các chuỗi landmark đã được phẳng hóa."""
        self.template_buffer = []
        for gid, template_list in self.calibrator.templates.items():
            for template in template_list:
                if not hasattr(template, 'sequence_landmarks'):
                    continue
                seq_lm = np.array(template.sequence_landmarks, dtype=np.float32)
                if seq_lm.shape == (SEQUENCE_LENGTH, 21, 3):
                    flattened_seq = seq_lm.flatten()
                    self.template_buffer.append((gid, flattened_seq))
    
    def predict_sequence_knn(self, landmarks_sequence: np.ndarray, 
                             threshold: float = 15.0) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
        """
        KNN prediction trên một chuỗi landmark.
        Return (top_gesture, top_distance, neighbors_list) or None.
        
        Args:
            landmarks_sequence: np.ndarray with shape (SEQUENCE_LENGTH, 21, 3)
            threshold: Ngưỡng khoảng cách để chấp nhận một kết quả.
        """
        if not self.template_buffer or landmarks_sequence.shape != (SEQUENCE_LENGTH, 21, 3):
            return None
        
        # 1. Chuẩn hóa chuỗi đầu vào (giống hệt lúc calibrate)
        wrist = landmarks_sequence[0:1, 0:1, :]
        normalized_sequence = landmarks_sequence - wrist
        palm = normalized_sequence[0:1, 9:10, :]
        scale = np.linalg.norm(palm, axis=-1, keepdims=True)
        scale[scale < 1e-6] = 1.0
        normalized_sequence = normalized_sequence / scale
        
        # 2. Phẳng hóa thành vector dài
        query_vector = normalized_sequence.flatten()
        
        # 3. Tính khoảng cách tới tất cả các template
        distances = []
        for gid, template_vector in self.template_buffer:
            # Dùng khoảng cách Manhattan (L1) vì nó nhanh hơn và thường hiệu quả
            # cho dữ liệu nhiều chiều.
            dist = np.sum(np.abs(query_vector - template_vector))
            distances.append((gid, float(dist)))
            
        # 4. Sort và tìm k-neighbors
        if not distances:
            return None
            
        sorted_gestures = sorted(distances, key=lambda x: x[1])
        
        top_gid, top_dist = sorted_gestures[0]
        
        if top_dist > threshold:
            return None
        
        # 5. Lấy k neighbors và thực hiện voting (nếu cần)
        # Hiện tại, chỉ trả về kết quả gần nhất (1-NN)
        k_neighbors = sorted_gestures[:self.k]
        
        return top_gid, top_dist, k_neighbors
