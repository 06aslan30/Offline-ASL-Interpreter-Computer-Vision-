import json
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = r"C:\Users\JACOB\Documents\HAND GESTURE SIGN LANGUAGE PROJECT\ASL_Project"
DATA   = os.path.join(BASE, "data")
JSON   = os.path.join(DATA, "nslt_100.json")   # We use 100-word version
VIDEOS = os.path.join(DATA, "videos")
LABELS = os.path.join(DATA, "wlasl_class_list.txt")

# ── 1. Load the class list ─────────────────────────────────────────────────
print("=" * 50)
print("CLASS LIST (first 10 words):")
with open(LABELS, "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes[:10])
print(f"Total classes in file: {len(classes)}")

# ── 2. Load nslt_100.json ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("nslt_100.json STRUCTURE:")
with open(JSON, "r") as f:
    data = json.load(f)

print(f"Type: {type(data)}")
if isinstance(data, dict):
    print(f"Keys (first 5): {list(data.keys())[:5]}")
    first_key = list(data.keys())[0]
    print(f"\nFirst entry key: {first_key}")
    print(f"First entry value: {data[first_key]}")
elif isinstance(data, list):
    print(f"Total entries: {len(data)}")
    print(f"\nFirst entry: {data[0]}")

# ── 3. Check videos folder ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("VIDEOS FOLDER:")
video_files = os.listdir(VIDEOS)
print(f"Total videos: {len(video_files)}")
print(f"First 5 filenames: {video_files[:5]}")
print(f"Video format: {os.path.splitext(video_files[0])[1]}")

# ── 4. Cross-check: how many WLASL100 videos exist on disk ────────────────
print("\n" + "=" * 50)
print("CROSS-CHECK (nslt_100 videos vs disk):")
if isinstance(data, dict):
    ids_in_json = list(data.keys())
else:
    ids_in_json = [str(entry.get("video_id", i)) for i, entry in enumerate(data)]

video_names = set(os.path.splitext(v)[0] for v in video_files)
found    = [v for v in ids_in_json if v in video_names]
missing  = [v for v in ids_in_json if v not in video_names]
print(f"Videos listed in nslt_100.json : {len(ids_in_json)}")
print(f"Found on disk                  : {len(found)}")
print(f"Missing from disk              : {len(missing)}")

print("\n✅ Explore done!")