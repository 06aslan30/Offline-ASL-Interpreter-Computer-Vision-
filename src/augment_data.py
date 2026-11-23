import os, numpy as np
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = r"C:\Users\JACOB\Documents\HAND GESTURE SIGN LANGUAGE PROJECT\ASL_Project"
KP_DIR    = os.path.join(BASE, "data", "keypoints")
AUG_DIR   = os.path.join(BASE, "data", "keypoints_augmented")
os.makedirs(AUG_DIR, exist_ok=True)

# ── Augmentation functions ─────────────────────────────────────────────────
def add_noise(seq, sigma=0.01):
    """Add small random noise to each keypoint."""
    return seq + np.random.normal(0, sigma, seq.shape)

def flip_horizontal(seq):
    """Mirror the gesture left-right."""
    flipped = seq.copy()
    flipped[:, 0::3] = 1.0 - flipped[:, 0::3]  # flip x coordinates
    return flipped

def time_stretch(seq, rate=None):
    """Speed up or slow down the sequence."""
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    length = seq.shape[0]
    new_len = int(length * rate)
    if new_len < 2:
        return seq
    indices = np.linspace(0, length - 1, new_len)
    stretched = np.array([seq[int(i)] for i in indices])
    # Pad or trim back to original length
    if len(stretched) < length:
        pad = np.zeros((length - len(stretched), seq.shape[1]))
        stretched = np.vstack([stretched, pad])
    else:
        stretched = stretched[:length]
    return stretched.astype(np.float32)

def scale(seq, factor=None):
    """Scale the gesture bigger or smaller."""
    if factor is None:
        factor = np.random.uniform(0.8, 1.2)
    return seq * factor

# ── Copy originals + generate augmented copies ────────────────────────────
files = [f for f in os.listdir(KP_DIR) if f.endswith(".npy")]
print(f"Original samples : {len(files)}")
print(f"Augmenting to    : {len(files) * 5} samples (×5)\n")

for fname in tqdm(files):
    src = os.path.join(KP_DIR, fname)
    seq = np.load(src).astype(np.float32)
    base = fname.replace(".npy", "")

    # 1. Save original
    np.save(os.path.join(AUG_DIR, f"{base}_orig.npy"), seq)

    # 2. Noise
    np.save(os.path.join(AUG_DIR, f"{base}_noise.npy"),
            add_noise(seq))

    # 3. Flip
    np.save(os.path.join(AUG_DIR, f"{base}_flip.npy"),
            flip_horizontal(seq))

    # 4. Time stretch
    np.save(os.path.join(AUG_DIR, f"{base}_stretch.npy"),
            time_stretch(seq))

    # 5. Scale
    np.save(os.path.join(AUG_DIR, f"{base}_scale.npy"),
            scale(seq))

total = len(os.listdir(AUG_DIR))
print(f"\n✅ Done! Total augmented samples: {total}")
print(f"   Saved to: {AUG_DIR}")