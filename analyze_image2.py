"""
analyze_image.py
Deepfake detection for images with face preprocessing.
Run: python analyze_image.py path/to/image.jpg
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from facenet_pytorch import MTCNN

# ── Models ────────────────────────────────────────────────────────
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

mtcnn = MTCNN(keep_all=False, post_fix=None, device="cpu")  # face detector

REAL_IDX = 1
FAKE_IDX = 0


# ── Face extraction ───────────────────────────────────────────────
def extract_face(image: Image.Image) -> Image.Image:
    """
    Detect and crop the face from the image.
    Falls back to the full image if no face is detected.
    """
    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        print("  [!] No face detected — using full image")
        return image

    # Take the first (most confident) face
    x1, y1, x2, y2 = [int(b) for b in boxes[0]]

    # Add 20% padding around the face
    w, h    = image.size
    pad_x   = int((x2 - x1) * 0.2)
    pad_y   = int((y2 - y1) * 0.2)
    x1      = max(0, x1 - pad_x)
    y1      = max(0, y1 - pad_y)
    x2      = min(w, x2 + pad_x)
    y2      = min(h, y2 + pad_y)

    return image.crop((x1, y1, x2, y2))


# ── Inference ─────────────────────────────────────────────────────
def analyze_image(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    image      = Image.open(file_path).convert("RGB")
    face       = extract_face(image)
    inputs     = processor(images=face, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1).squeeze()

    prob_real  = probs[REAL_IDX].item()
    prob_fake  = probs[FAKE_IDX].item()
    confidence = max(prob_real, prob_fake)
    entropy    = -sum(p * np.log(p + 1e-9) for p in [prob_real, prob_fake])

    return {
        "result"           : "real" if prob_real > prob_fake else "fake",
        "probability_real" : round(prob_real,  4),
        "probability_fake" : round(prob_fake,  4),
        "confidence"       : round(confidence, 4),
        "entropy"          : round(entropy,    4),
    }


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_image.py path/to/image.jpg")
        sys.exit(1)

    result = analyze_image(sys.argv[1])

    print("\n─── Result ───────────────────────────────")
    print(f"  Real probability : {result['probability_real']:.2%}")
    print(f"  Fake probability : {result['probability_fake']:.2%}")
    print(f"  Verdict          : {result['result'].upper()}")
    print(f"\n─── Metrics ──────────────────────────────")
    print(f"  Confidence       : {result['confidence']:.2%}")
    print(f"  Entropy          : {result['entropy']:.4f}")
    print("──────────────────────────────────────────")