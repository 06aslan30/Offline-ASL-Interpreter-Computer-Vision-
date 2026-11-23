import json, os, cv2, numpy as np
import mediapipe as mp
from tqdm import tqdm

# ── MediaPipe setup ────────────────────────────────────────────────────────
mp_holistic_solution = mp.solutions.holistic

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = r"C:\Users\JACOB\Documents\HAND GESTURE SIGN LANGUAGE PROJECT\ASL_Project"
DATA    = os.path.join(BASE, "data")
VIDEOS  = os.path.join(DATA, "videos")
OUT_DIR = os.path.join(DATA, "keypoints")
JSON    = os.path.join(DATA, "nslt_100.json")
LABELS  = os.path.join(DATA, "wlasl_class_list.txt")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load class list ────────────────────────────────────────────────────────
with open(LABELS, "r") as f:
    lines = [l.strip() for l in f.readlines()]

idx_to_word = {}
for line in lines:
    parts = line.split("\t")
    if len(parts) == 2:
        idx_to_word[int(parts[0])] = parts[1]

# ── Load JSON ──────────────────────────────────────────────────────────────
with open(JSON, "r") as f:
    data = json.load(f)

# ── Config ─────────────────────────────────────────────────────────────────
SEQUENCE_LEN = 30
NUM_FEATURES = 201  # 75 pose + 63 left hand + 63 right hand

# ── Keypoint extraction ────────────────────────────────────────────────────
def extract_keypoints(results):
    """Flatten pose + both hands into 201 numbers."""
    
    # Pose (25 landmarks * 3 = 75)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark[:25]]).flatten()
    else:
        pose = np.zeros(75)

    # Left hand (21 * 3 = 63)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)

    # Right hand (21 * 3 = 63)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)

    return np.concatenate([pose, lh, rh])  # total = 201

# ── Main extraction loop ───────────────────────────────────────────────────
skipped   = 0
processed = 0
label_map = {}

print("Starting keypoint extraction...")
print(f"Output folder: {OUT_DIR}\n")

with mp_holistic_solution.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3) as holistic:

    for video_id, info in tqdm(data.items(), total=len(data)):

        video_path = os.path.join(VIDEOS, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            skipped += 1
            continue

        class_idx   = info["action"][0]
        start_frame = info["action"][1]
        end_frame   = info["action"][2]

        word = idx_to_word.get(class_idx, None)
        if word is None:
            skipped += 1
            continue

        if word not in label_map:
            label_map[word] = len(label_map)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Safe bounds
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame   = max(start_frame + 1, min(end_frame, total_frames))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_kp = []

        while cap.isOpened() and len(frames_kp) < SEQUENCE_LEN:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            frames_kp.append(extract_keypoints(results))

        cap.release()

        if len(frames_kp) == 0:
            skipped += 1
            continue

        # Padding / trimming
        if len(frames_kp) < SEQUENCE_LEN:
            pad = [np.zeros(NUM_FEATURES)] * (SEQUENCE_LEN - len(frames_kp))
            frames_kp.extend(pad)
        else:
            frames_kp = frames_kp[:SEQUENCE_LEN]

        sequence = np.array(frames_kp)  # shape: (30, 201)

        out_path = os.path.join(OUT_DIR, f"{word}_{video_id}.npy")
        np.save(out_path, sequence)

        processed += 1

# ── Save label map ─────────────────────────────────────────────────────────
label_map_path = os.path.join(DATA, "label_map.json")

with open(label_map_path, "w") as f:
    json.dump(label_map, f, indent=2)

# ── Final logs ─────────────────────────────────────────────────────────────
print(f"\n✅ Done!")
print(f"   Processed : {processed}")
print(f"   Skipped   : {skipped}")
print(f"   Words     : {len(label_map)}")
print(f"   Label map saved to: {label_map_path}")
print(f"   Keypoints saved to: {OUT_DIR}")