"""
analyze_audio_svm.py
Audio deepfake detection using AASIST + SVM pipeline.
Run: python analyze_audio_svm.py path/to/audio.wav
"""

import sys
import os
import json
import torch
import numpy as np
import librosa
import joblib

# ── AASIST setup ──────────────────────────────────────────────────
AASIST_DIR   = os.path.join(os.path.dirname(__file__), "AASIST")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "AASIST.pth")
CONFIG_PATH  = os.path.join(AASIST_DIR, "config", "AASIST.conf")
sys.path.insert(0, AASIST_DIR)
from models.AASIST import Model

# ── SVM setup ─────────────────────────────────────────────────────
SVM_PATH    = os.path.join(os.path.dirname(__file__), "svm", "best_svm_v3.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "svm", "scaler_v3.pkl")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
MAX_SAMPLES = 64600

SUPPORTED_FORMATS = {".wav", ".ogg", ".mp3", ".flac", ".m4a"}

# ── Load models ───────────────────────────────────────────────────
def load_models():
    # AASIST
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    aasist = Model(config["model_config"]).to(DEVICE)
    aasist.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    aasist.eval()

    # SVM + scaler
    svm    = joblib.load(SVM_PATH)
    scaler = joblib.load(SCALER_PATH)

    return aasist, svm, scaler

# ── Preprocess ────────────────────────────────────────────────────
def preprocess(file_path: str) -> torch.Tensor:
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < MAX_SAMPLES:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
    else:
        audio = audio[:MAX_SAMPLES]
    return torch.FloatTensor(audio).unsqueeze(0).to(DEVICE)

# ── Main inference ────────────────────────────────────────────────
def analyze_audio(file_path: str, aasist=None, svm=None, scaler=None) -> dict:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}'")

    if aasist is None or svm is None or scaler is None:
        aasist, svm, scaler = load_models()

    # Step 1 — extract embedding with AASIST
    tensor = preprocess(file_path)
    with torch.no_grad():
        embedding, _ = aasist(tensor)
    embedding = embedding.squeeze().cpu().numpy().reshape(1, -1)

    # Step 2 — normalize
    embedding = scaler.transform(embedding)

    # Step 3 — classify with SVM
    prob      = svm.predict_proba(embedding)[0]  # [fake_prob, real_prob]
    prob_fake = prob[0]
    prob_real = prob[1]

    llr             = float(np.log(prob_real / (prob_fake + 1e-9)))
    decision_margin = round(abs(prob_real - 0.5), 4)
    suspicion_score = round(prob_fake * 100, 2)
    entropy         = -sum(p * np.log(p + 1e-9) for p in [prob_real, prob_fake])

    return {
        "result"               : "real" if prob_real > prob_fake else "fake",
        "probability_real"     : round(prob_real, 4),
        "probability_fake"     : round(prob_fake, 4),
        "log_likelihood_ratio" : round(llr, 4),
        "decision_margin"      : decision_margin,
        "suspicion_score"      : suspicion_score,
        "entropy"              : round(entropy, 4),
    }

# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio_svm.py path/to/audio.wav")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    print("Loading models...")
    _aasist, _svm, _scaler = load_models()

    print(f"Analyzing: {file_path}")
    result = analyze_audio(file_path, _aasist, _svm, _scaler)

    print("\n─── Result ───────────────────────────────")
    print(f"  Real probability : {result['probability_real']:.2%}")
    print(f"  Fake probability : {result['probability_fake']:.2%}")
    print(f"  Verdict          : {result['result'].upper()}")
    print(f"\n─── Metrics ──────────────────────────────")
    print(f"  Log-likelihood   : {result['log_likelihood_ratio']:+.4f}  (+ real, - fake)")
    print(f"  Decision margin  : {result['decision_margin']:.4f}  (0=borderline, 0.5=certain)")
    print(f"  Suspicion score  : {result['suspicion_score']:.1f}/100")
    print(f"  Entropy          : {result['entropy']:.4f}")
    print("──────────────────────────────────────────")