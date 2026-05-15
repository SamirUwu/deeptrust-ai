"""
api.py
Flask backend for DeepTrust AI — audio + video deepfake detection.
Run: python api.py
"""

import os
import sys
import json
import warnings
import tempfile
warnings.filterwarnings("ignore")

import torch
import numpy as np
import librosa
import joblib
import cv2
import timm
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── AASIST setup ──────────────────────────────────────────────────
AASIST_DIR   = os.path.join(os.path.dirname(__file__), "AASIST")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "AASIST.pth")
CONFIG_PATH  = os.path.join(AASIST_DIR, "config", "AASIST.conf")
sys.path.insert(0, AASIST_DIR)
from models.AASIST import Model as AASISTModel

SVM_PATH    = os.path.join(os.path.dirname(__file__), "svm", "best_svm_v3.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "svm", "scaler_v3.pkl")

# ── EfficientNet setup ────────────────────────────────────────────
VIDEO_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "efficientnet_b0_v5.pt")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
MAX_SAMPLES = 64600

app = Flask(__name__)
CORS(app)

# ── Load all models once at startup ───────────────────────────────
print("Loading models...")

# AASIST
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
aasist = AASISTModel(config["model_config"]).to(DEVICE)
aasist.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
aasist.eval()

# SVM
svm    = joblib.load(SVM_PATH)
scaler = joblib.load(SCALER_PATH)

# EfficientNet
effnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1, drop_rate=0.4)
effnet.load_state_dict(torch.load(VIDEO_WEIGHTS, map_location=DEVICE))
effnet = effnet.to(DEVICE)
effnet.eval()

# MTCNN face detector
mtcnn = MTCNN(min_face_size=80, thresholds=[0.6, 0.7, 0.9], keep_all=False, device=DEVICE)

# Video transform
video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("All models loaded!")

# ── Audio inference ───────────────────────────────────────────────
def analyze_audio_file(file_path: str) -> dict:
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < MAX_SAMPLES:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
    else:
        audio = audio[:MAX_SAMPLES]

    tensor = torch.FloatTensor(audio).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding, _ = aasist(tensor)
    embedding = embedding.squeeze().cpu().numpy().reshape(1, -1)
    embedding = scaler.transform(embedding)

    prob      = svm.predict_proba(embedding)[0]
    prob_fake = float(prob[0])
    prob_real = float(prob[1])

    confidence = "High" if max(prob_real, prob_fake) > 0.80 else \
                 "Medium" if max(prob_real, prob_fake) > 0.60 else "Low"

    return {
        "type"        : "audio",
        "probability" : round(prob_real, 4),
        "label"       : "Authentic" if prob_real > prob_fake else "Potential Deepfake",
        "confidence"  : confidence,
    }

# ── Video inference ───────────────────────────────────────────────
def analyze_video_file(file_path: str, frames_per_video: int = 12) -> dict:
    cap          = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices      = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

    probs = []
    faces_detected = 0  # ← añade esto
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img             = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, conf     = mtcnn.detect(img)
        if boxes is None or conf[0] < 0.90:
            continue
        faces_detected += 1  # ← añade esto
        x1, y1, x2, y2 = boxes[0]
        w, h            = x2 - x1, y2 - y1
        x1, y1          = max(0, x1 - 0.15*w), max(0, y1 - 0.15*h)
        x2, y2          = min(img.width, x2 + 0.15*w), min(img.height, y2 + 0.15*h)
        face            = img.crop((x1, y1, x2, y2))
        tensor          = video_transform(face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(effnet(tensor)).item()
        probs.append(prob)
        print(f"  Frame {idx}: face_prob={prob:.4f}")  # ← añade esto

    cap.release()
    print(f"  Faces detected: {faces_detected}/{frames_per_video}")
    print(f"  Raw probs: {probs}")

    # ← esto faltaba
    if not probs:
        return None

    fake_probability = float(np.mean(probs))
    prob_real        = 1.0 - fake_probability
    confidence       = "High"   if abs(fake_probability - 0.5) > 0.30 else \
                       "Medium" if abs(fake_probability - 0.5) > 0.15 else "Low"

    return {
        "type"           : "video",
        "probability"    : round(prob_real, 4),
        "label"          : "Potential Deepfake" if fake_probability > 0.5 else "Authentic",
        "confidence"     : confidence,
        "frames_analyzed": len(probs),
    }

# ── Routes ────────────────────────────────────────────────────────
@app.route("/api/analyze/audio", methods=["POST"])
def route_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        try:
            result = analyze_audio_file(tmp.name)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(tmp.name)

    return jsonify(result)

@app.route("/api/analyze/video", methods=["POST"])
def route_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    
    # Windows fix: no delete=True, borramos manualmente después
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(file.read())
    tmp.close()  # cerrar antes de que cv2 lo abra
    
    try:
        result = analyze_video_file(tmp.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)  # borrar después de que cv2 terminó
        except:
            pass

    if result is None:
        return jsonify({"error": "No face detected in video"}), 422

    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)