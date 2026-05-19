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
import torch.nn as nn
import torchaudio
import numpy as np
import cv2
import timm
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from transformers import Wav2Vec2Model
from flask import Flask, request, jsonify
from flask_cors import CORS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Wav2Vec2 architecture ─────────────────────────────────────────
class Wav2Vec2Classifier(nn.Module):
    def __init__(self, dropout: float = 0.3, freeze_backbone: bool = True):
        super().__init__()
        self.backbone  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden         = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),    nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(64, 1)
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out    = self.backbone(input_values)
        hidden = out.last_hidden_state.mean(dim=1)
        return self.classifier(hidden).squeeze(1)

# ── EfficientNet setup ────────────────────────────────────────────
VIDEO_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "efficientnet_b0_v5.pt")
AUDIO_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "mejor_modelo.pt")
UMBRAL_PATH   = os.path.join(os.path.dirname(__file__), "weights", "umbral.json")

app = Flask(__name__)
CORS(app)

# ── Load all models once at startup ───────────────────────────────
print("Loading models...")

# Wav2Vec2 audio model
with open(UMBRAL_PATH, "r") as f:
    umbral = json.load(f)["umbral"]

audio_model = Wav2Vec2Classifier(freeze_backbone=True)
audio_model.load_state_dict(torch.load(AUDIO_WEIGHTS, map_location=DEVICE))
audio_model = audio_model.to(DEVICE)
audio_model.eval()
print(f"  Audio model loaded — umbral: {umbral}")

# EfficientNet video model
effnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1, drop_rate=0.4)
effnet.load_state_dict(torch.load(VIDEO_WEIGHTS, map_location=DEVICE))
effnet = effnet.to(DEVICE)
effnet.eval()
print("  Video model loaded")

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
    import librosa
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    wav = torch.FloatTensor(audio).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits    = audio_model(wav)
        prob_fake = torch.sigmoid(logits).item()

    es_fake       = prob_fake >= umbral
    confianza     = (prob_fake - umbral) / (1.0 - umbral) if es_fake else (umbral - prob_fake) / umbral
    confianza_pct = round(confianza * 100, 2)
    prob_fake_pct = round(prob_fake * 100, 2)

    explicacion = (
        f"El sistema analizó la firma acústica mediante Wav2Vec2. "
        f"Se detectaron {'anomalías sintéticas y artefactos de clonación' if es_fake else 'patrones vocales naturales y respiración coherente'} "
        f"en el espectro de frecuencias. "
        f"El umbral de decisión del sistema es {round(umbral * 100, 1)}%."
    )

    return {
        "type"                  : "audio",
        "probability"           : round(1.0 - prob_fake, 4),
        "label": "Authentic" if es_fake else "Potential Deepfake", "probability": round(prob_fake, 4),
        "confidence"            : "High" if confianza_pct > 60 else "Medium" if confianza_pct > 30 else "Low",
        "probabilidad_deepfake" : f"{prob_fake_pct}%",
        "nivel_confianza"       : f"{confianza_pct}%",
        "explicacion"           : explicacion,
    }

def analyze_image_file(file_path: str) -> dict:
    image  = Image.open(file_path).convert("RGB")

    # Detect face with MTCNN
    boxes, conf = mtcnn.detect(image)
    if boxes is not None and conf[0] >= 0.70:
        x1, y1, x2, y2 = boxes[0]
        w, h = x2 - x1, y2 - y1
        x1, y1 = max(0, x1 - 0.15*w), max(0, y1 - 0.15*h)
        x2, y2 = min(image.width, x2 + 0.15*w), min(image.height, y2 + 0.15*h)
        image = image.crop((x1, y1, x2, y2))

    tensor = video_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(effnet(tensor)).item()

    fake_probability = float(prob)
    prob_real        = 1.0 - fake_probability
    confidence       = "High"   if abs(fake_probability - 0.5) > 0.30 else \
                       "Medium" if abs(fake_probability - 0.5) > 0.15 else "Low"

    return {
        "type"      : "video",   # usa "video" para que el frontend lo renderice igual
        "probability": round(prob_real, 4),
        "label"     : "Authentic" if fake_probability > 0.5 else "Potential Deepfake",
        "confidence": confidence,
        "frames_analyzed": 1,
    }

# ── Video inference ───────────────────────────────────────────────
def analyze_video_file(file_path: str, frames_per_video: int = 12) -> dict:
    cap          = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices      = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

    probs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img         = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, conf = mtcnn.detect(img)
        if boxes is None or conf[0] < 0.70:
            continue    
        x1, y1, x2, y2 = boxes[0]
        w, h            = x2 - x1, y2 - y1
        x1, y1          = max(0, x1 - 0.15*w), max(0, y1 - 0.15*h)
        x2, y2          = min(img.width, x2 + 0.15*w), min(img.height, y2 + 0.15*h)
        face            = img.crop((x1, y1, x2, y2))
        tensor          = video_transform(face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(effnet(tensor)).item()
        probs.append(prob)

    cap.release()

    if not probs:
        return None

    fake_probability = float(np.mean(probs))
    prob_real        = 1.0 - fake_probability
    confidence       = "High"   if abs(fake_probability - 0.5) > 0.30 else \
                       "Medium" if abs(fake_probability - 0.5) > 0.15 else "Low"

    return {
        "type"           : "video",
        "probability"    : round(prob_real, 4),
        "label": "Authentic" if fake_probability > 0.5 else "Potential Deepfake",
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

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(file.read())
    tmp.close()

    try:
        result = analyze_audio_file(tmp.name)
    except Exception as e:
        import traceback
        traceback.print_exc()  # ← esto imprime el error completo en la terminal
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass

    return jsonify(result)

@app.route("/api/analyze/image", methods=["POST"])
def route_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower() or ".jpg"

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(file.read())
    tmp.close()

    try:
        result = analyze_image_file(tmp.name)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass

    return jsonify(result)

@app.route("/api/analyze/video", methods=["POST"])
def route_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    
    # Guardar con extensión correcta según el tipo
    ext = os.path.splitext(file.filename)[1].lower() or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(file.read())
    tmp.close()

    try:
        result = analyze_video_file(tmp.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
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