"""
train.py
========
Fine-tunes facebook/wav2vec2-base for Speech Emotion Recognition (SER).
Uses MPS (Apple Silicon) if available, else CPU.

Run after prepare_dataset.py has created data/dataset.csv.
"""

import os
import csv
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report
from tqdm import tqdm
import soundfile as sf
import librosa

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
CSV_PATH     = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_OUTPUT = os.path.join(BASE_DIR, "model_output")
os.makedirs(MODEL_OUTPUT, exist_ok=True)

PRETRAINED_MODEL = "facebook/wav2vec2-base"
NUM_LABELS       = 5          # neutral, happy, sad, angry, fearful
MAX_DURATION_SEC = 5          # clip/pad audio to 5 seconds
TARGET_SR        = 16000
MAX_LENGTH       = TARGET_SR * MAX_DURATION_SEC  # samples

BATCH_SIZE       = 4          # keep small for M4 RAM
EPOCHS           = 7
LR               = 1e-4
WARMUP_STEPS     = 50
FREEZE_BACKBONE  = True       # freeze wav2vec2 encoder, train head only
UNFREEZE_EPOCH   = 4          # unfreeze backbone from this epoch onward (set to EPOCHS+1 to never unfreeze)
VAL_SPLIT        = 0.15
SEED             = 42

# LABEL_NAMES = ["neutral", "mild_distress", "low_energy"]
LABEL_NAMES = ["neutral", "happy", "sad", "angry", "fearful"]

# ─── ETA helper ───────────────────────────────────────────────────────────────
def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    elif m:
        return f"{m}m {s}s"
    return f"{s}s"

# ─── Device ───────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("[DEVICE] Using Apple MPS (Metal)")
        return torch.device("mps")
    print("[DEVICE] Using CPU")
    return torch.device("cpu")


# ─── Dataset ──────────────────────────────────────────────────────────────────
class SpeechEmotionDataset(Dataset):
    def __init__(self, records, feature_extractor, max_length=MAX_LENGTH):
        self.records = records
        self.feature_extractor = feature_extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label_id = self.records[idx]

        # Load audio (already preprocessed to 16kHz mono)
        audio, sr = sf.read(path)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        audio = audio.astype(np.float32)

        # Clip or pad to MAX_LENGTH
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)))

        # Feature extraction
        inputs = self.feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=False,
        )

        input_values = inputs["input_values"].squeeze(0)  # (max_length,)
        return {
            "input_values": input_values,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


def collate_fn(batch):
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_values": input_values, "labels": labels}


# ─── Load CSV ─────────────────────────────────────────────────────────────────
def load_records(csv_path):
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["path"]
            label_id = int(row["label_id"])
            if os.path.exists(path):
                records.append((path, label_id))
            else:
                print(f"[WARN] File missing: {path}")
    return records


def split_records(records, val_split=VAL_SPLIT, seed=SEED):
    random.seed(seed)
    shuffled = records[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_split))
    return shuffled[:split], shuffled[split:]


# ─── Training ─────────────────────────────────────────────────────────────────
def set_backbone_frozen(model, frozen: bool):
    for param in model.wav2vec2.parameters():
        param.requires_grad = not frozen
    status = "FROZEN" if frozen else "UNFROZEN"
    print(f"[BACKBONE] wav2vec2 encoder is {status}")


def train():
    device = get_device()

    print(f"[INFO] Loading records from {CSV_PATH}")
    records = load_records(CSV_PATH)
    if not records:
        print("[ERROR] No records found. Run prepare_dataset.py first.")
        return

    train_records, val_records = split_records(records)
    print(f"[INFO] Train: {len(train_records)} | Val: {len(val_records)}")

    # Feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PRETRAINED_MODEL)

    # Datasets & loaders
    train_ds = SpeechEmotionDataset(train_records, feature_extractor)
    val_ds   = SpeechEmotionDataset(val_records, feature_extractor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model
    print(f"[INFO] Loading {PRETRAINED_MODEL} with {NUM_LABELS} output labels")
    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    if FREEZE_BACKBONE:
        set_backbone_frozen(model, frozen=True)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    training_start = time.time()
    epoch_times = []

    # Overall epoch-level progress bar
    epoch_bar = tqdm(
        range(1, EPOCHS + 1),
        desc="Overall",
        unit="epoch",
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]",
    )

    for epoch in epoch_bar:
        epoch_start = time.time()

        epoch_bar.set_description(f"Epoch {epoch}/{EPOCHS}")
        print(f"\n{'='*55}")
        print(f"  EPOCH {epoch}/{EPOCHS}"
              + (f"  [backbone FROZEN]" if FREEZE_BACKBONE and epoch < UNFREEZE_EPOCH else "  [backbone UNFROZEN]"))
        print(f"{'='*55}")

        # Unfreeze backbone from UNFREEZE_EPOCH onward
        if FREEZE_BACKBONE and epoch == UNFREEZE_EPOCH:
            set_backbone_frozen(model, frozen=False)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LR * 0.1,
                weight_decay=0.01,
            )

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(
            train_loader,
            desc="  Training",
            unit="batch",
            position=1,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] loss={postfix}",
            postfix="-.----",
        )

        for batch in train_bar:
            input_values = batch["input_values"].to(device)
            labels       = batch["labels"].to(device)

            outputs = model(input_values=input_values, labels=labels)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            # Live loss update in bar
            train_bar.set_postfix_str(f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        print(f"\n  [TRAIN] Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} ({correct}/{total})")

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        val_bar = tqdm(
            val_loader,
            desc="  Validating",
            unit="batch",
            position=1,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        with torch.no_grad():
            for batch in val_bar:
                input_values = batch["input_values"].to(device)
                labels       = batch["labels"].to(device)

                outputs = model(input_values=input_values, labels=labels)
                val_loss += outputs.loss.item()

                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        # ── Epoch summary ──────────────────────────────────────────────────
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = EPOCHS - epoch
        eta = avg_epoch_time * remaining_epochs

        print(f"  [VAL]   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"  [TIME]  Epoch took {format_time(epoch_elapsed)} | "
              f"ETA for remaining {remaining_epochs} epoch(s): {format_time(eta)}")
        print(f"\n  Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0))

        # ── Save best model ────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(MODEL_OUTPUT)
            feature_extractor.save_pretrained(MODEL_OUTPUT)
            print(f"  ✅ Best model saved! val_acc={best_val_acc:.4f}\n")
        else:
            print(f"  — No improvement (best so far: {best_val_acc:.4f})\n")

        # Update overall bar with key stats
        epoch_bar.set_postfix({
            "train_acc": f"{train_acc:.3f}",
            "val_acc":   f"{val_acc:.3f}",
            "best":      f"{best_val_acc:.3f}",
        })

    total_time = time.time() - training_start
    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Total time        : {format_time(total_time)}")
    print(f"  Model saved to    : {MODEL_OUTPUT}")
    print(f"{'='*55}")


if __name__ == "__main__":
    train()