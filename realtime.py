import cv2
import mediapipe as mp
import torch
import numpy as np
import json
from collections import deque
from models.stgcn import STGCN, build_hand_edge_index

# ---- Load labels ----
with open('outputs/labels.json') as f:
    labels = json.load(f)

# ---- Build hand graph ----
edge_index = build_hand_edge_index()  # tạo đồ thị tay cho STGCN

# ---- Load model ----
model = STGCN(in_channels=4, num_classes=len(labels), edge_index=edge_index)
model.load_state_dict(torch.load('outputs/stgcn_best.pt', map_location='cpu'))
model.eval()

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---- Sequence buffer ----
T = 30    # số frame liên tiếp
V = 21    # số landmark
C = 4     # x,y + velocity
buffer = deque(maxlen=T)
prev_frame = None

# ---- OpenCV webcam ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks)

        # ---- normalize ----
        wrist = landmarks[0]
        landmarks = landmarks - wrist
        palm = np.linalg.norm(landmarks[9])
        if palm > 1e-6:
            landmarks /= palm

        # ---- velocity ----
        if prev_frame is None:
            velocity = np.zeros_like(landmarks[:, :2])
        else:
            velocity = landmarks[:, :2] - prev_frame[:, :2]

        frame_feature = np.concatenate([landmarks[:, :2], velocity], axis=1)  # C=4
        buffer.append(frame_feature)
        prev_frame = landmarks

        # ---- draw landmarks ----
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # ---- predict if buffer full ----
    if len(buffer) == T:
        x = np.array(buffer)  # (T,V,C)
        x = torch.tensor(x).unsqueeze(0).permute(0, 3, 1, 2).float()  # (1,C,T,V)
        with torch.no_grad():
            logits = model(x)
            pred_id = logits.argmax(dim=1).item()
            pred_label = [k for k, v in labels.items() if v == pred_id][0]

        cv2.putText(frame, f'Gesture: {pred_label}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()