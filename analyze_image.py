"""
analyze_image.py
Deepfake detection for images with face preprocessing using OpenCV.
Run: python analyze_image.py path/to/image.jpg
"""

import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ── Models ────────────────────────────────────────────────────────
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

# OpenCV's built-in face detector (no install needed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

REAL_IDX = 1
FAKE_IDX = 0


# ── Face extraction ───────────────────────────────────────────────
def extract_face(image: Image.Image) -> Image.Image:
    img_cv  = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces   = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("  [!] No face detected — using full image")
        return image

    # Take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add 20% padding
    pad_x  = int(w * 0.2)
    pad_y  = int(h * 0.2)
    ih, iw = img_cv.shape[:2]
    x1     = max(0, x - pad_x)
    y1     = max(0, y - pad_y)
    x2     = min(iw, x + w + pad_x)
    y2     = min(ih, y + h + pad_y)

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