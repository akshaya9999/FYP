# ─────────────────────────────────────────────────────────────────────────────
# ADD THIS to the top of app.py (with your other imports)
# ─────────────────────────────────────────────────────────────────────────────

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
import numpy as np
import soundfile as sf
import librosa

# Path to your fine-tuned model (after training)
SER_MODEL_PATH = "ser_model"   # folder containing config.json, model.safetensors, etc.
SER_CONFIDENCE_THRESHOLD = 0.55   # if model confidence < this, fall back to energy heuristic

# Label mapping must match what was used in training
SER_ID_TO_LABEL = {
    0: "neutral",
    1: "mild_distress",
    2: "low_energy",
}

def load_ser_model(model_path: str):
    """
    Load the fine-tuned SER model and feature extractor.
    Returns (feature_extractor, model) or (None, None) if not found.
    """
    if not os.path.exists(model_path):
        print(f"[SER] Model not found at {model_path}. Will use energy heuristic only.")
        return None, None

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print(f"[SER] Loaded fine-tuned model from {model_path}")
    return feature_extractor, model


# ─── Load once at startup (put this near your whisper_model init) ─────────────
ser_feature_extractor, ser_model = load_ser_model(SER_MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# REPLACE your existing detect_audio_emotion() with this:
# ─────────────────────────────────────────────────────────────────────────────

def energy_heuristic(audio: np.ndarray) -> tuple[str, float]:
    """
    Baseline energy-based emotion detection.
    Returns (label, pseudo_confidence).
    """
    energy = np.mean(audio ** 2)
    if energy < 0.001:
        return "low_energy", 0.9
    elif energy < 0.005:
        return "mild_distress", 0.7
    else:
        return "neutral", 0.8


def detect_audio_emotion(file_path: str) -> str:
    """
    Two-method emotion detection:
      1. Primary:  fine-tuned wav2vec2 SER model
      2. Fallback: energy heuristic (if model absent or low confidence)

    Returns one of: 'neutral', 'mild_distress', 'low_energy'
    """
    TARGET_SR   = 16000
    MAX_SAMPLES = TARGET_SR * 5   # 5-second clip

    # ── Convert webm → wav ──────────────────────────────────────────────────
    wav_path = "converted_audio.wav"
    subprocess.run(
        ["ffmpeg", "-i", file_path, "-ac", "1", "-ar", str(TARGET_SR), wav_path, "-y"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # ── Load audio ──────────────────────────────────────────────────────────
    try:
        audio, sr = sf.read(wav_path)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        audio = audio.astype(np.float32)
        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    except Exception as e:
        print(f"[SER] Audio load error: {e}. Defaulting to neutral.")
        return "neutral"

    # ── Energy heuristic (always compute as fallback) ────────────────────────
    energy_label, energy_conf = energy_heuristic(audio)

    # ── Model inference ──────────────────────────────────────────────────────
    if ser_model is None or ser_feature_extractor is None:
        print("[SER] No model loaded — using energy heuristic.")
        return energy_label

    try:
        # Clip or pad to MAX_SAMPLES
        if len(audio) > MAX_SAMPLES:
            audio_input = audio[:MAX_SAMPLES]
        else:
            audio_input = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

        inputs = ser_feature_extractor(
            audio_input,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=False,
        )

        with torch.no_grad():
            outputs = ser_model(input_values=inputs["input_values"])

        logits = outputs.logits[0]
        probs  = torch.softmax(logits, dim=-1)
        confidence, predicted_id = probs.max(dim=-1)

        confidence   = confidence.item()
        predicted_id = predicted_id.item()
        model_label  = SER_ID_TO_LABEL.get(predicted_id, "neutral")

        print(f"[SER] Model → {model_label} (conf={confidence:.2f}) | Energy → {energy_label}")

        # ── Fusion: trust model if confidence is high enough ─────────────────
        if confidence >= SER_CONFIDENCE_THRESHOLD:
            return model_label
        else:
            print(f"[SER] Low confidence ({confidence:.2f}) — falling back to energy heuristic.")
            return energy_label

    except Exception as e:
        print(f"[SER] Model inference error: {e}. Using energy heuristic.")
        return energy_label