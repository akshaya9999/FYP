"""
rlhf_inference.py
=================
Loads the trained reward model and scores candidate responses.
Used by app.py for Best-of-N response selection at inference time.

If no reward model exists yet, falls back to returning the first candidate
(standard RAG behavior) so the app never breaks.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

REWARD_MODEL_DIR = os.path.join(os.path.dirname(__file__), "reward_model", "reward_model_output")


# ─── Model definition (must match train_reward.py) ────────────────────────────
class RewardModel(nn.Module):
    def __init__(self, pretrained: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(pretrained)
        hidden_size  = self.encoder.config.hidden_size

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        score      = self.reward_head(cls_output).squeeze(-1)
        return score


# ─── Loader ───────────────────────────────────────────────────────────────────
class RLHFRanker:
    """
    Loads reward model once at startup.
    Scores (query, response) pairs and selects the best candidate.
    """

    def __init__(self, model_dir: str = REWARD_MODEL_DIR):
        self.model     = None
        self.tokenizer = None
        self.device    = self._get_device()
        self.max_length = 512

        self._load(model_dir)

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self, model_dir: str):
        pt_path     = os.path.join(model_dir, "reward_model.pt")
        config_path = os.path.join(model_dir, "config.json")

        if not os.path.exists(pt_path):
            print(f"[RLHF] No reward model found at {model_dir}. Will use first candidate (no ranking).")
            return

        try:
            with open(config_path) as f:
                config = json.load(f)

            pretrained      = config.get("pretrained", "distilbert-base-uncased")
            self.max_length = config.get("max_length", 512)

            self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            model          = RewardModel(pretrained)
            model.load_state_dict(torch.load(pt_path, map_location=self.device))
            model.eval()
            model.to(self.device)
            self.model = model

            print(f"[RLHF] Reward model loaded from {model_dir}")
        except Exception as e:
            print(f"[RLHF] Failed to load reward model: {e}. Falling back to first candidate.")

    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def score(self, query: str, response: str) -> float:
        """Score a single (query, response) pair. Returns float in [0, 1]."""
        if not self.is_ready():
            return 0.0

        encoding = self.tokenizer(
            query,
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            score = self.model(input_ids, attention_mask)

        return score.item()

    def best_of_n(self, query: str, candidates: list) -> tuple:
        """
        Given a query and N candidate responses, return the best one
        according to the reward model, along with its score.

        Falls back to first candidate if model not ready.
        """
        if not self.is_ready() or not candidates:
            return candidates[0] if candidates else "", 0.0

        scores = [self.score(query, c) for c in candidates]
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))

        print(f"[RLHF] Candidate scores: {[f'{s:.3f}' for s in scores]}")
        print(f"[RLHF] Selected candidate {best_idx + 1}/{len(candidates)} (score={scores[best_idx]:.3f})")

        return candidates[best_idx], scores[best_idx]