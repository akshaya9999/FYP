"""
train_reward.py
===============
Trains a DistilBERT-based reward model on human preference feedback
collected from Redis.

Architecture:
    Input  : "[CLS] query [SEP] response [SEP]"
    Output : scalar score ∈ [0, 1]  (1 = good response, 0 = bad)

Bradley-Terry preference model — as used in InstructGPT (Ouyang et al. 2022).

Run this periodically (e.g. once you have 50+ feedback entries):
    python reward_model/train_reward.py
"""

import os
import json
import redis
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
MODEL_OUTPUT   = os.path.join(os.path.dirname(__file__), "reward_model_output")
PRETRAINED     = "distilbert-base-uncased"
MAX_LENGTH     = 512
BATCH_SIZE     = 8
EPOCHS         = 4
LR             = 2e-5
VAL_SPLIT      = 0.15
MIN_SAMPLES    = 20      # minimum feedback entries required to train
SEED           = 42

os.makedirs(MODEL_OUTPUT, exist_ok=True)

# ─── Device ───────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("[DEVICE] Using Apple MPS")
        return torch.device("mps")
    print("[DEVICE] Using CPU")
    return torch.device("cpu")


# ─── Reward Model Architecture ────────────────────────────────────────────────
class RewardModel(nn.Module):
    """
    DistilBERT encoder + scalar regression head.
    Follows Bradley-Terry reward model formulation.
    """
    def __init__(self, pretrained: str = PRETRAINED):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(pretrained)
        hidden_size  = self.encoder.config.hidden_size  # 768

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),   # output ∈ [0, 1]
        )

    def forward(self, input_ids, attention_mask):
        outputs     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output  = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        score       = self.reward_head(cls_output).squeeze(-1)
        return score


# ─── Dataset ──────────────────────────────────────────────────────────────────
class PreferenceDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=MAX_LENGTH):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        query, response, label = self.records[idx]

        # Format: "[CLS] query [SEP] response [SEP]"
        encoding = self.tokenizer(
            query,
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.float32),
        }


# ─── Load feedback from Redis ─────────────────────────────────────────────────
def load_feedback_from_redis():
    """
    Reads all feedback entries from Redis.
    Key pattern: feedback:{email}:* → {query, response, rating}
    rating: 1 = thumbs up, 0 = thumbs down
    """
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

    records = []
    keys = r.keys("feedback:*")

    print(f"[REDIS] Found {len(keys)} feedback entries.")

    for key in keys:
        try:
            data     = json.loads(r.get(key))
            query    = data.get("query", "").strip()
            response = data.get("response", "").strip()
            rating   = int(data.get("rating", -1))

            if not query or not response or rating not in (0, 1):
                continue

            records.append((query, response, float(rating)))
        except Exception as e:
            print(f"[WARN] Skipping key {key}: {e}")

    return records


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    device = get_device()

    # Load data
    records = load_feedback_from_redis()

    if len(records) < MIN_SAMPLES:
        print(f"[ERROR] Not enough feedback to train. Have {len(records)}, need {MIN_SAMPLES}.")
        print("        Keep collecting user feedback and run this again later.")
        return

    print(f"[INFO] Total feedback records: {len(records)}")
    pos = sum(1 for _, _, r in records if r == 1.0)
    neg = len(records) - pos
    print(f"[INFO] Positive (👍): {pos} | Negative (👎): {neg}")

    # Split
    train_records, val_records = train_test_split(
        records, test_size=VAL_SPLIT, random_state=SEED, stratify=[r for _, _, r in records]
    )

    # Tokenizer & model
    tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED)
    model     = RewardModel(PRETRAINED).to(device)

    train_ds = PreferenceDataset(train_records, tokenizer)
    val_ds   = PreferenceDataset(val_records, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler  = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    criterion  = nn.BCELoss()
    best_auc   = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"  EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*50}")

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc="  Training", unit="batch", leave=False)
        for batch in train_bar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            scores = model(input_ids, attention_mask)
            loss   = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"\n  [TRAIN] Loss: {avg_train_loss:.4f}")

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Validating", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                scores = model(input_ids, attention_mask)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        preds      = (all_scores >= 0.5).astype(int)
        acc        = accuracy_score(all_labels, preds)

        try:
            auc = roc_auc_score(all_labels, all_scores)
        except Exception:
            auc = 0.0   # only one class present

        print(f"  [VAL]  Acc: {acc:.4f} | AUC: {auc:.4f}")

        # ── Save best ──────────────────────────────────────────────────────
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT, "reward_model.pt"))
            tokenizer.save_pretrained(MODEL_OUTPUT)
            # Save config for inference
            import json
            with open(os.path.join(MODEL_OUTPUT, "config.json"), "w") as f:
                json.dump({"pretrained": PRETRAINED, "max_length": MAX_LENGTH}, f)
            print(f"  ✅ Best reward model saved (AUC={best_auc:.4f})")

    print(f"\n[DONE] Reward model training complete.")
    print(f"       Best AUC: {best_auc:.4f}")
    print(f"       Model saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    train()