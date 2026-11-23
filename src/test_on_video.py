import os, json, cv2, numpy as np, torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic_solution
from collections import deque

BASE       = r"C:\Users\JACOB\Documents\HAND GESTURE SIGN LANGUAGE PROJECT\ASL_Project"
MODELS_DIR = os.path.join(BASE, "models")
VIDEOS     = os.path.join(BASE, "data", "videos")
JSON_PATH  = os.path.join(BASE, "data", "nslt_100.json")
LABELS     = os.path.join(BASE, "data", "wlasl_class_list.txt")

# ── Load label map ─────────────────────────────────────────────────────────
with open(LABELS) as f:
    lines = [l.strip() for l in f.readlines()]
idx_to_word_data = {}
for line in lines:
    parts = line.split("\t")
    if len(parts) == 2:
        idx_to_word_data[int(parts[0])] = parts[1]

with open(JSON_PATH) as f:
    data = json.load(f)

with open(os.path.join(MODELS_DIR, "idx_to_word.json")) as f:
    idx_to_word = json.load(f)

# ── Model ──────────────────────────────────────────────────────────────────
class ASLModel(nn.Module):
    def __init__(self, input_size=201, hidden=128, num_classes=100):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True,
                            num_layers=2, dropout=0.3)
        self.gru  = nn.GRU(hidden, 64, batch_first=True)
        self.fc   = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Dropout(0.3),    nn.Linear(128, num_classes))
    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.gru(x)
        return self.fc(x[:, -1, :])

model = ASLModel(num_classes=100)
model.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, "asl_model_best.pth"), map_location="cpu"))
model.eval()
print("✅ Model loaded!\n")

# ── Keypoint extraction ────────────────────────────────────────────────────
def extract_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z]
                     for lm in results.pose_landmarks.landmark[:25]]).flatten() \
           if results.pose_landmarks else np.zeros(75)
    lh   = np.array([[lm.x, lm.y, lm.z]
                     for lm in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(63)
    rh   = np.array([[lm.x, lm.y, lm.z]
                     for lm in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

# ── Pick test videos ───────────────────────────────────────────────────────
test_videos = []
for vid_id, info in data.items():
    path = os.path.join(VIDEOS, f"{vid_id}.mp4")
    if os.path.exists(path):
        word = idx_to_word_data.get(info["action"][0], None)
        if word:
            test_videos.append((vid_id, word, path,
                                info["action"][1], info["action"][2]))
    if len(test_videos) >= 20:
        break

# ── Test each video ────────────────────────────────────────────────────────
correct = 0
total   = 0

with mp_holistic_solution.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3) as holistic:

    for vid_id, true_word, path, start_f, end_f in test_videos:
        cap          = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_f      = max(0, min(start_f, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        frames_kp = []
        while cap.isOpened() and len(frames_kp) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            frames_kp.append(extract_keypoints(results))
        cap.release()

        if len(frames_kp) == 0:
            continue

        # Pad if needed
        while len(frames_kp) < 30:
            frames_kp.append(np.zeros(201))
        frames_kp = frames_kp[:30]

        # Predict
        x = torch.tensor(np.array(frames_kp),
                         dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out   = model(x)
            probs = torch.softmax(out, dim=1)[0]
            idx   = int(probs.argmax())
            conf  = float(probs[idx])
            pred  = idx_to_word.get(str(idx), "?")

        status = "✅" if pred == true_word else "❌"
        if pred == true_word:
            correct += 1
        total += 1
        print(f"{status} True: {true_word:15s} "
              f"Pred: {pred:15s} Conf: {conf:.2f}")

print(f"\n📊 Real-time accuracy: {correct}/{total} "
      f"= {correct/total*100:.1f}%")
