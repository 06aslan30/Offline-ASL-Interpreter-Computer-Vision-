import os, json, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
BASE       = r"C:\Users\JACOB\Documents\HAND GESTURE SIGN LANGUAGE PROJECT\ASL_Project"
DATA       = os.path.join(BASE, "data")
KEYPOINTS  = os.path.join(DATA, "keypoints_augmented")
LABEL_MAP  = os.path.join(DATA, "label_map.json")
MODELS_DIR = os.path.join(BASE, "models")
LOGS_DIR   = os.path.join(BASE, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load label map ─────────────────────────────────────────────────────────
with open(LABEL_MAP, "r") as f:
    label_map = json.load(f)
NUM_CLASSES = len(label_map)
print(f"Classes: {NUM_CLASSES}")

# ── Load keypoints ─────────────────────────────────────────────────────────
X, y = [], []
for fname in os.listdir(KEYPOINTS):
    if not fname.endswith(".npy"):
        continue
    word = fname.rsplit("_", 2)[0]
    if word not in label_map:
        continue
    data = np.load(os.path.join(KEYPOINTS, fname))
    if data.shape != (30, 201):
        continue
    X.append(data)
    y.append(label_map[word])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
print(f"Samples: {len(X)}")

# ── Dataset ────────────────────────────────────────────────────────────────
class ASLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

train_loader = DataLoader(ASLDataset(X_train, y_train),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(ASLDataset(X_val, y_val),
                          batch_size=32, shuffle=False)

# ── Model ──────────────────────────────────────────────────────────────────
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
        x    = x[:, -1, :]   # last timestep
        return self.fc(x)

device = torch.device("cpu")
model  = ASLModel(num_classes=NUM_CLASSES).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training ───────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
patience_counter = 0
PATIENCE = 15
history  = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

print("\nStarting training...\n")

for epoch in range(100):
    # ── Train ──
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (out.argmax(1) == yb).sum().item()
        train_total   += len(yb)

    # ── Validate ──
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)
            val_loss    += loss.item()
            val_correct += (out.argmax(1) == yb).sum().item()
            val_total   += len(yb)

    train_acc = train_correct / train_total
    val_acc   = val_correct   / val_total
    t_loss    = train_loss    / len(train_loader)
    v_loss    = val_loss      / len(val_loader)

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)

    scheduler.step(v_loss)

    print(f"Epoch {epoch+1:3d} | "
          f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
          f"Train Loss: {t_loss:.3f} | Val Loss: {v_loss:.3f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(MODELS_DIR, "asl_model_best.pth"))
        print(f"  ✅ Best model saved! Val Acc: {val_acc:.3f}")

    # Early stopping
    if val_acc >= best_val_acc:
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# ── Save final + label map ─────────────────────────────────────────────────
torch.save(model.state_dict(),
           os.path.join(MODELS_DIR, "asl_model_final.pth"))

# Save idx_to_word for inference
idx_to_word = {str(v): k for k, v in label_map.items()}
with open(os.path.join(MODELS_DIR, "idx_to_word.json"), "w") as f:
    json.dump(idx_to_word, f, indent=2)

print(f"\n✅ Training done! Best Val Acc: {best_val_acc:.3f}")
print(f"✅ Model saved to models/asl_model_best.pth")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_acc"], label="Train Acc")
ax1.plot(history["val_acc"],   label="Val Acc")
ax1.set_title("Accuracy"); ax1.legend()
ax2.plot(history["train_loss"], label="Train Loss")
ax2.plot(history["val_loss"],   label="Val Loss")
ax2.set_title("Loss"); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOGS_DIR, "training_curves_pytorch.png"))
print("✅ Training curves saved!")
plt.show()