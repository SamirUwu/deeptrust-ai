import sys
import cv2
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
FRAMES_TO_SAMPLE = 20

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

# Resolve which index = "real" from model's own labels
id2label  = model.config.id2label
real_idx  = next(i for i, l in id2label.items() if "real" in l.lower())
fake_idx  = next(i for i, l in id2label.items() if "fake" in l.lower() or "deep" in l.lower())

def extract_frames(video_path: str, n: int) -> list:
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames  = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def analyze_video(video_path: str) -> dict:
    frames       = extract_frames(video_path, FRAMES_TO_SAMPLE)
    scores_real  = []

    for frame in frames:
        inputs = processor(images=frame, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).squeeze()
        scores_real.append(probs[real_idx].item())

    prob_real  = float(np.mean(scores_real))
    prob_fake  = 1.0 - prob_real
    confidence = max(prob_real, prob_fake)
    entropy    = -sum(p * np.log(p + 1e-9) for p in [prob_real, prob_fake])

    return {
        "result"           : "real" if prob_real > 0.5 else "fake",
        "probability_real" : round(prob_real,  4),
        "probability_fake" : round(prob_fake,  4),
        "confidence"       : round(confidence, 4),  # how sure the model is
        "entropy"          : round(entropy,    4),  # how uncertain the model is
    }

if __name__ == "__main__":
    result = analyze_video(sys.argv[1])
    print("\n─── Result ───────────────────────────────")
    print(f"  Real probability : {result['probability_real']:.2%}")
    print(f"  Fake probability : {result['probability_fake']:.2%}")
    print(f"  Verdict          : {result['result'].upper()}")
    print(f"\n─── Metrics ──────────────────────────────")
    print(f"  Confidence       : {result['confidence']:.2%}")
    print(f"  Entropy          : {result['entropy']:.4f}")
    print("──────────────────────────────────────────")
    #print(json.dumps(result, indent=2))