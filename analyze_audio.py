"""
analyze.py
Minimal AASIST inference — no training code, no dataset pipelines.
Run: python analyze.py path/to/audio.wav
"""

import sys
import os
import json
import torch
import numpy as np
import soundfile as sf
import librosa


# ── Point Python at the submodule so we can import its code ──────────────────
AASIST_DIR = os.path.join(os.path.dirname(__file__), "AASIST")
sys.path.insert(0, AASIST_DIR)

from models.AASIST import Model  # AASIST's own Model class

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "AASIST.pth")
CONFIG_PATH  = os.path.join(AASIST_DIR, "config", "AASIST.conf")
SAMPLE_RATE  = 16000      # AASIST was trained on 16 kHz
MAX_SAMPLES  = 64600      # ~4 sec at 16 kHz — matches training padding
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> torch.nn.Module:
    """Load AASIST model + pretrained weights. Call once, reuse."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model_config = config["model_config"]
    model = Model(model_config).to(DEVICE)

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


import librosa

def preprocess(file_path: str) -> torch.Tensor:
    """
    Load a .wav file and prepare it for AASIST:
    - Mono
    - 16 kHz (auto resample)
    - Fixed length: pad or truncate to MAX_SAMPLES
    """
    # 🔥 Carga y re-muestrea automáticamente a 16kHz
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Pad or truncate to fixed length
    if len(audio) < MAX_SAMPLES:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
    else:
        audio = audio[:MAX_SAMPLES]

    # Shape: (1, samples)
    tensor = torch.FloatTensor(audio).unsqueeze(0).to(DEVICE)
    return tensor


def analyze_audio(file_path: str, model: torch.nn.Module = None) -> dict:
    """
    Main inference function.

    Returns:
        {
            "probability_real": float,   # 0.0–1.0, higher = more likely real
            "probability_fake": float,
            "result": "real" | "fake",
        }
    """
    if model is None:
        model = load_model()

    audio_tensor = preprocess(file_path)

    with torch.no_grad():
        # AASIST forward: returns (frame_level_logits, utterance_level_logits)
        _, logits = model(audio_tensor)
        # logits shape: (batch, 2) — class 0 = fake, class 1 = real
        probs = torch.softmax(logits, dim=1).squeeze()

    prob_real = probs[0].item()
    prob_fake = probs[1].item()

    return {
        "probability_real": round(prob_real, 4),
        "probability_fake": round(prob_fake, 4),
        "result": "real" if prob_real > prob_fake else "fake",
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py path/to/audio.wav")
        sys.exit(1)

    wav_file = sys.argv[1]
    if not os.path.exists(wav_file):
        print(f"File not found: {wav_file}")
        sys.exit(1)

    print(f"Loading model from {WEIGHTS_PATH} ...")
    _model = load_model()

    print(f"Analyzing: {wav_file}")
    result = analyze_audio(wav_file, _model)

    print("\n─── Result ───────────────────────────────")
    print(f"  Real probability : {result['probability_real']:.2%}")
    print(f"  Fake probability : {result['probability_fake']:.2%}")
    print(f"  Verdict          : {result['result'].upper()}")
    print("──────────────────────────────────────────")