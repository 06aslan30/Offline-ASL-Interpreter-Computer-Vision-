import os, json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
BASE       = r"C:\Users\JACOB\Documents\HAND GESTURE SIGN LANGUAGE PROJECT\ASL_Project"
DATA       = os.path.join(BASE, "data")
KEYPOINTS  = os.path.join(DATA, "keypoints_augmented")
LABEL_MAP  = os.path.join(DATA, "label_map.json")
MODELS_DIR = os.path.join(BASE, "models")
LOGS_DIR   = os.path.join(BASE, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

# ── Load label map ─────────────────────────────────────────────────────────
with open(LABEL_MAP, "r") as f:
    label_map = json.load(f)
word_to_idx = label_map
idx_to_word = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)
print(f"Number of classes: {NUM_CLASSES}")

# ── Load keypoints ─────────────────────────────────────────────────────────
X, y = [], []
skipped = 0

for fname in os.listdir(KEYPOINTS):
    if not fname.endswith(".npy"):
        continue
    word = fname.rsplit("_", 2)[0]   # e.g. "book" from "book_05237.npy"
    if word not in word_to_idx:
        skipped += 1
        continue
    data = np.load(os.path.join(KEYPOINTS, fname))
    if data.shape != (30, 201):
        skipped += 1
        continue
    X.append(data)
    y.append(word_to_idx[word])

print(f"Loaded  : {len(X)} samples")
print(f"Skipped : {skipped}")

X = np.array(X, dtype=np.float32)   # (N, 30, 201)
y = np.array(y, dtype=np.int32)

# ── Train / val split ──────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat   = to_categorical(y_val,   NUM_CLASSES)

print(f"Train   : {X_train.shape}")
print(f"Val     : {X_val.shape}")

# ── Build LSTM-GRU model ───────────────────────────────────────────────────
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 201)),
    BatchNormalization(),
    Dropout(0.3),

    GRU(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    GRU(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    os.path.join(MODELS_DIR, "asl_model_best.keras"),
    monitor="val_accuracy", save_best_only=True, verbose=1)

early_stop = EarlyStopping(
    monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# ── Train ──────────────────────────────────────────────────────────────────
print("\nStarting training...\n")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ── Save final model ───────────────────────────────────────────────────────
model.save(os.path.join(MODELS_DIR, "asl_model_final.keras"))
print("\n✅ Model saved!")

# ── Save label map for inference ───────────────────────────────────────────
inference_map = {str(v): k for k, v in label_map.items()}
with open(os.path.join(MODELS_DIR, "idx_to_word.json"), "w") as f:
    json.dump(inference_map, f, indent=2)
print("✅ idx_to_word.json saved!")

# ── Plot training curves ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["accuracy"],     label="Train Acc")
ax1.plot(history.history["val_accuracy"], label="Val Acc")
ax1.set_title("Accuracy")
ax1.legend()

ax2.plot(history.history["loss"],     label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Loss")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(LOGS_DIR, "training_curves.png"))
print("✅ Training curves saved to logs/training_curves.png")
plt.show()