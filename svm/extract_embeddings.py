"""
extract_embeddings.py
Extrae embeddings de todos los audios del dataset ASVspoof 2019 LA
usando AASIST como extractor de features (congelado).
Run: python extract_embeddings.py
"""

import os
import sys
import json
import torch
import numpy as np
import librosa
from tqdm import tqdm

# ── AASIST setup ──────────────────────────────────────────────────
AASIST_DIR   = os.path.join(os.path.dirname(__file__), "..", "AASIST")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "pretrained", "AASIST.pth")
CONFIG_PATH  = os.path.join(AASIST_DIR, "config", "AASIST.conf")
sys.path.insert(0, AASIST_DIR)
from models.AASIST import Model

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
MAX_SAMPLES = 64600

# ── Dataset paths ─────────────────────────────────────────────────
LA_DIR = os.path.join(os.path.dirname(__file__), "..", "LA")
PROTOCOL_DIR = os.path.join(LA_DIR, "ASVspoof2019_LA_cm_protocols")

SPLITS = {
    "train": {
        "audio_dir" : os.path.join(LA_DIR, "ASVspoof2019_LA_train", "flac"),
        "protocol"  : os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt"),
    },
    "dev": {
        "audio_dir" : os.path.join(LA_DIR, "ASVspoof2019_LA_dev", "flac"),
        "protocol"  : os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt"),
    },
    "eval": {
        "audio_dir" : os.path.join(LA_DIR, "ASVspoof2019_LA_eval", "flac"),
        "protocol"  : os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.eval.trl.txt"),
    },
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "embeddings")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load AASIST ───────────────────────────────────────────────────
def load_aasist() -> torch.nn.Module:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model = Model(config["model_config"]).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    return model


# ── Extract embedding from a single audio ─────────────────────────
def get_embedding(model: torch.nn.Module, file_path: str) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < MAX_SAMPLES:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
    else:
        audio = audio[:MAX_SAMPLES]

    tensor = torch.FloatTensor(audio).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # We grab the graph-level output (first return value) as embedding
        embedding, _ = model(tensor)

    return embedding.squeeze().cpu().numpy()


# ── Parse protocol file ───────────────────────────────────────────
def parse_protocol(protocol_path: str) -> list:
    """
    Returns list of (filename, label) where label is 0=fake, 1=real
    Protocol format: SPEAKER_ID FILENAME - SYSTEM_ID LABEL
    """
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[1]          # e.g. LA_T_1000137
            label_str = parts[4]         # "bonafide" or "spoof"
            label = 1 if label_str == "bonafide" else 0
            entries.append((filename, label))
    return entries


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading AASIST...")
    model = load_aasist()

    for split, paths in SPLITS.items():
        print(f"\nProcessing {split} split...")
        entries = parse_protocol(paths["protocol"])

        embeddings = []
        labels     = []
        skipped    = 0

        for filename, label in tqdm(entries):
            file_path = os.path.join(paths["audio_dir"], filename + ".flac")

            if not os.path.exists(file_path):
                skipped += 1
                continue

            try:
                emb = get_embedding(model, file_path)
                embeddings.append(emb)
                labels.append(label)
            except Exception as e:
                skipped += 1
                continue

        embeddings = np.array(embeddings)
        labels     = np.array(labels)

        np.save(os.path.join(OUTPUT_DIR, f"{split}_embeddings.npy"), embeddings)
        np.save(os.path.join(OUTPUT_DIR, f"{split}_labels.npy"),     labels)

        print(f"  Saved {len(embeddings)} embeddings — skipped {skipped}")
        print(f"  Real: {labels.sum()} — Fake: {(labels == 0).sum()}")

    print("\nDone! Embeddings saved in /embeddings")