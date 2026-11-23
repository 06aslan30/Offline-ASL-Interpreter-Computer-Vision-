import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
import json
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic_solution
from collections import deque
import sys
import time

sys.path.append(os.path.dirname(__file__))
from temporal_smoother import TemporalSmoother
from grammar_fix import GrammarFix
from tts_kokoro import TTS

# ── Paths ──────────────────────────────────────────────────────────────────
BASE       = r"C:\Users\JACOB\Documents\HAND GESTURE SIGN LANGUAGE PROJECT\ASL_Project"
MODELS_DIR = os.path.join(BASE, "models")

# ── Model definition ───────────────────────────────────────────────────────
class ASLModel(nn.Module):
    def __init__(self, input_size=201, hidden=128, num_classes=100):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True,
                            num_layers=2, dropout=0.3)
        self.gru  = nn.GRU(hidden, 64, batch_first=True)
        self.fc   = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.gru(x)
        x    = x[:, -1, :]
        return self.fc(x)

# ── Load model ─────────────────────────────────────────────────────────────
print("Loading model...")
with open(os.path.join(MODELS_DIR, "idx_to_word.json"), "r") as f:
    idx_to_word = json.load(f)

NUM_CLASSES = len(idx_to_word)
model = ASLModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, "asl_model_best.pth"),
    map_location="cpu"))
model.eval()
print(f"✅ Model loaded! {NUM_CLASSES} words")

# ── Init pipeline ──────────────────────────────────────────────────────────
smoother = TemporalSmoother(window_size=10, threshold=0.7, cooldown=20)
grammar  = GrammarFix(use_llm=False)
tts      = TTS()

SEQUENCE_LEN  = 30
NUM_FEATURES  = 201
WORD_DELAY    = 5.0   # seconds between words

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

# ── Predict ────────────────────────────────────────────────────────────────
def predict(sequence):
    x = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1)[0]
        idx   = int(probs.argmax())
        conf  = float(probs[idx])
    return idx_to_word.get(str(idx), "?"), conf

# ── UI ─────────────────────────────────────────────────────────────────────
def draw_ui(frame, seq_len, current_word, confidence,
            sentence_words, last_sentence, cooldown_remaining):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, "ASL Sign Language Translator",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 150), 2)

    # Last spoken sentence
    if last_sentence:
        cv2.rectangle(frame, (0, 60), (w, 95), (20, 60, 20), -1)
        cv2.putText(frame, f"Said: {last_sentence}",
                    (10, 83), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (100, 255, 100), 1)

    # Controls hint
    cv2.putText(frame, "SPACE=speak  C=clear  Q=quit",
                (w - 320, 83), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1)

    # Cooldown timer — shown when waiting between words
    if cooldown_remaining > 0:
        cv2.rectangle(frame, (0, 95), (w, 130), (60, 20, 20), -1)
        cv2.putText(frame, f"⏳ Next word in: {cooldown_remaining:.1f}s",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2)

    # Progress bar
    bar_w = int((seq_len / SEQUENCE_LEN) * (w - 20))
    cv2.rectangle(frame, (10, h - 90), (w - 10, h - 70), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, h - 90), (10 + bar_w, h - 70),
                  (0, 255, 100), -1)
    cv2.putText(frame, f"Frames: {seq_len}/{SEQUENCE_LEN}",
                (10, h - 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    # Current prediction
    if current_word:
        conf_pct = int(confidence * 100)
        color    = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
        cv2.putText(frame, f"Detecting: {current_word} ({conf_pct}%)",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Accumulated words
    words_str = "  |  ".join(sentence_words) if sentence_words else "..."
    cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Words: {words_str}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 100), 1)

    return frame

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sequence          = deque(maxlen=SEQUENCE_LEN)
    sentence_words    = []
    last_sentence     = ""
    current_word      = ""
    confidence        = 0.0
    last_commit_time  = 0.0   # ← tracks when last word was committed

    print("\n✅ App started!")
    print("SPACE = speak sentence | C = clear | Q = quit\n")

    with mp_holistic_solution.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks,
                mp_holistic_solution.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp_holistic_solution.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp_holistic_solution.HAND_CONNECTIONS)

            # Collect keypoints
            kp = extract_keypoints(results)
            sequence.append(kp)

            # Predict
            if len(sequence) == SEQUENCE_LEN:
                current_word, confidence = predict(sequence)
                committed = smoother.update(current_word, confidence)
                if committed:
                    now               = time.time()
                    cooldown_left     = WORD_DELAY - (now - last_commit_time)
                    if cooldown_left <= 0:
                        print(f"✅ Word: {committed}")
                        sentence_words.append(committed)
                        last_commit_time = now

            # Calculate remaining cooldown for UI
            now               = time.time()
            cooldown_remaining = max(0.0, WORD_DELAY - (now - last_commit_time))

            # Draw UI
            frame = draw_ui(frame, len(sequence), current_word,
                            confidence, sentence_words,
                            last_sentence, cooldown_remaining)
            cv2.imshow("ASL Translator", frame)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                sentence_words.clear()
                smoother.reset()
                last_commit_time = 0.0
                print("🗑️  Cleared")
            elif key == ord(" ") and sentence_words:
                last_sentence = grammar.fix(sentence_words)
                print(f"🗣️  {last_sentence}")
                tts.speak(last_sentence)
                sentence_words.clear()
                smoother.reset()
                last_commit_time = 0.0

    cap.release()
    cv2.destroyAllWindows()
    print("👋 App closed!")

if __name__ == "__main__":
    main()