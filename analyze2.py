"""
analyze.py
Audio deepfake detection using wav2vec2 (MelodyMachine/Deepfake-audio-detection-V2)
Much more robust for real-world mic recordings than AASIST.
Run: python analyze.py path/to/audio.wav
"""

import sys
import torch
import librosa
import numpy as np
from transformers import pipeline

SAMPLE_RATE = 16000
MODEL_ID    = "MelodyMachine/Deepfake-audio-detection-V2"

# Load once, reuse
def load_model():
    return pipeline(
        "audio-classification",
        model=MODEL_ID,
        device=0 if torch.cuda.is_available() else -1,
    )

def preprocess(file_path: str) -> np.ndarray:
    """Load any audio file, resample to 16kHz mono, truncate to 10s."""
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio[:SAMPLE_RATE * 10]  # max 10 seconds

def analyze_audio(file_path: str, model=None) -> dict:
    if model is None:
        model = load_model()

    audio = preprocess(file_path)
    results = model(audio, sampling_rate=SAMPLE_RATE)

    # results = [{"label": "fake", "score": 0.99}, {"label": "real", "score": 0.01}]
    scores = {r["label"].lower(): round(r["score"], 4) for r in results}

    return {
        "probability_real": scores.get("real", 0.0),
        "probability_fake": scores.get("fake", 0.0),
        "result": "real" if scores.get("real", 0) > scores.get("fake", 0) else "fake",
    }

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py path/to/audio.wav")
        sys.exit(1)

    wav_file = sys.argv[1]

    print(f"Loading model (downloads once, ~370MB)...")
    _model = load_model()

    print(f"Analyzing: {wav_file}")
    result = analyze_audio(wav_file, _model)

    print("\n─── Result ───────────────────────────────")
    print(f"  Real probability : {result['probability_real']:.2%}")
    print(f"  Fake probability : {result['probability_fake']:.2%}")
    print(f"  Verdict          : {result['result'].upper()}")
    print("──────────────────────────────────────────")