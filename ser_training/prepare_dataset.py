"""
prepare_dataset.py
==================
Prepares a unified CSV dataset from:
  - RAVDESS (Actor_XX/filename.wav)
  - Kaggle Audio Speech Sentiment (flat folder or RAVDESS-style naming)
  - Custom voice notes (named like: sad_001.wav, angry_note.wav)

Output: ser_training/data/dataset.csv
Columns: path, emotion, app_label
"""

import os
import re
import csv
import librosa
import soundfile as sf
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
RAVDESS_DIR = os.path.join(BASE_DIR, "data", "ravdess")
KAGGLE_DIR  = os.path.join(BASE_DIR, "data", "kaggle")
CUSTOM_DIR  = os.path.join(BASE_DIR, "data", "custom")
OUTPUT_CSV  = os.path.join(BASE_DIR, "data", "dataset.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─── RAVDESS emotion code → label ─────────────────────────────────────────────
# Filename format: 03-01-[emotion]-01-01-01-01.wav
# emotion codes: 01=neutral,02=calm,03=happy,04=sad,05=angry,06=fearful,07=disgust,08=surprised
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# ─── App-level label mapping ───────────────────────────────────────────────────
# Maps fine-grained emotion → your 3 app buckets
# NEW
APP_LABEL_MAP = {
    "neutral":   "neutral",
    "calm":      "neutral",    # calm → neutral (merged)
    "happy":     "happy",
    "surprised": "happy",      # surprised → happy (closest)
    "sad":       "sad",
    "fearful":   "fearful",
    "angry":     "angry",
    "disgust":   "angry",      # disgust → angry (closest)
}

LABEL_TO_ID = {
    "neutral": 0,
    "happy":   1,
    "sad":     2,
    "angry":   3,
    "fearful": 4,
}
# APP_LABEL_MAP = {
#     "neutral":   "neutral",
#     "calm":      "neutral",
#     "happy":     "neutral",
#     "surprised": "neutral",
#     "angry":     "mild_distress",
#     "disgust":   "mild_distress",
#     "sad":       "low_energy",
#     "fearful":   "low_energy",
# }

# # Numeric label for model training (3-class)
# LABEL_TO_ID = {
#     "neutral":       0,
#     "mild_distress": 1,
#     "low_energy":    2,
# }


def preprocess_audio(input_path: str, output_path: str, target_sr: int = 16000) -> bool:
    """Convert to mono, resample to 16kHz, normalize amplitude, save."""
    try:
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
        # Amplitude normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        sf.write(output_path, audio, target_sr)
        return True
    except Exception as e:
        print(f"  [WARN] Could not process {input_path}: {e}")
        return False


def parse_ravdess(folder: str) -> list:
    """Walk RAVDESS Actor_XX folders and extract emotion from filename."""
    records = []
    if not os.path.exists(folder):
        print(f"[SKIP] RAVDESS folder not found: {folder}")
        return records

    for actor_dir in sorted(os.listdir(folder)):
        actor_path = os.path.join(folder, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
            if emotion is None:
                continue
            app_label = APP_LABEL_MAP.get(emotion, "neutral")
            full_path = os.path.join(actor_path, fname)
            records.append((full_path, emotion, app_label))

    print(f"[RAVDESS] Found {len(records)} files.")
    return records


def parse_kaggle(folder: str) -> list:
    """
    Kaggle dataset — supports two layouts:
      1. RAVDESS-style filenames (03-01-XX-...) in flat or subfolders
      2. Files named with emotion word anywhere: sad_001.wav, happy-note.wav
    """
    records = []
    if not os.path.exists(folder):
        print(f"[SKIP] Kaggle folder not found: {folder}")
        return records

    ravdess_pattern = re.compile(r"^\d{2}-\d{2}-(\d{2})-")

    for root, _, files in os.walk(folder):
        for fname in files:
            if not fname.endswith(".wav"):
                continue
            full_path = os.path.join(root, fname)
            emotion = None

            # Try RAVDESS-style filename first
            m = ravdess_pattern.match(fname)
            if m:
                emotion = RAVDESS_EMOTION_MAP.get(m.group(1))
            else:
                # Try keyword match in filename
                lower = fname.lower()
                for kw in APP_LABEL_MAP:
                    if kw in lower:
                        emotion = kw
                        break

            if emotion is None:
                continue
            app_label = APP_LABEL_MAP.get(emotion, "neutral")
            records.append((full_path, emotion, app_label))

    print(f"[KAGGLE] Found {len(records)} files.")
    return records


def parse_custom(folder: str) -> list:
    """
    Custom voice notes — emotion keyword must appear in filename.
    Examples: sad_001.wav, angry_note.wav, neutral_chat.wav
    """
    records = []
    if not os.path.exists(folder):
        print(f"[SKIP] Custom folder not found: {folder}")
        return records

    for fname in os.listdir(folder):
        if not fname.endswith(".wav"):
            continue
        lower = fname.lower()
        emotion = None
        for kw in APP_LABEL_MAP:
            if kw in lower:
                emotion = kw
                break
        if emotion is None:
            print(f"  [WARN] Cannot infer emotion from filename: {fname} — skipping.")
            continue
        app_label = APP_LABEL_MAP.get(emotion, "neutral")
        full_path = os.path.join(folder, fname)
        records.append((full_path, emotion, app_label))

    print(f"[CUSTOM] Found {len(records)} files.")
    return records


def build_dataset():
    all_records = []
    all_records.extend(parse_ravdess(RAVDESS_DIR))
    all_records.extend(parse_kaggle(KAGGLE_DIR))
    all_records.extend(parse_custom(CUSTOM_DIR))

    print(f"\n[TOTAL] {len(all_records)} files before preprocessing.")

    processed_records = []
    for i, (src_path, emotion, app_label) in enumerate(all_records):
        fname = f"{i:05d}_{os.path.basename(src_path)}"
        out_path = os.path.join(PROCESSED_DIR, fname)
        success = preprocess_audio(src_path, out_path)
        if success:
            processed_records.append((out_path, emotion, app_label, LABEL_TO_ID[app_label]))

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "emotion", "app_label", "label_id"])
        writer.writerows(processed_records)

    print(f"\n[DONE] Dataset CSV written to: {OUTPUT_CSV}")
    print(f"       Total usable files: {len(processed_records)}")

    # Print class distribution
    from collections import Counter
    dist = Counter(r[2] for r in processed_records)
    print("\nClass distribution:")
    for label, count in dist.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    build_dataset()