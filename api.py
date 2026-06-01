"""
api.py
Flask backend for DeepTrust AI — audio + video + image deepfake detection.
Run: python api.py
"""

import os
import json
import warnings
import tempfile
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
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

# ── EfficientNet wrapper (face y image detectors) ─────────────────
class EfficientNetClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone   = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),      # índice 0
            nn.Linear(1280, 256),     # índice 1
            nn.ReLU(),                # índice 2
            nn.Dropout(dropout),      # índice 3
            nn.Linear(256, 1)         # índice 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x).squeeze(1)

# ── CNN-LSTM architecture ─────────────────────────────────────────
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn        = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.lstm       = nn.LSTM(1280, 512, num_layers=2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),        # índice 0
            nn.Linear(512, 128),    # índice 1  ← 128 no 256
            nn.ReLU(),              # índice 2
            nn.Dropout(0.3),        # índice 3
            nn.Linear(128, 1)       # índice 4  ← 128 no 256
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x).view(B, T, -1)
        _, (h, _) = self.lstm(x)
        return self.classifier(h[-1]).squeeze(1)

# ── Paths ─────────────────────────────────────────────────────────
WEIGHTS_DIR     = os.path.join(os.path.dirname(__file__), "weights")
VIDEO_WEIGHTS   = os.path.join(WEIGHTS_DIR, "efficientnet_b0_v5.pt")
CNNLSTM_WEIGHTS = os.path.join(WEIGHTS_DIR, "cnnlstm_v2.pt")
FACE_WEIGHTS    = os.path.join(WEIGHTS_DIR, "face_detector_v1.pt")
IMAGE_WEIGHTS   = os.path.join(WEIGHTS_DIR, "image_detector_v1.pt")
AUDIO_WEIGHTS   = os.path.join(WEIGHTS_DIR, "mejor_modelo.pt")
UMBRAL_PATH     = os.path.join(WEIGHTS_DIR, "umbral.json")

app = Flask(__name__)
CORS(app)

# ── Load models ───────────────────────────────────────────────────
print("Loading models...")

# Audio
with open(UMBRAL_PATH, "r") as f:
    umbral = json.load(f)["umbral"]

audio_model = Wav2Vec2Classifier(freeze_backbone=True)
audio_model.load_state_dict(torch.load(AUDIO_WEIGHTS, map_location=DEVICE))
audio_model = audio_model.to(DEVICE)
audio_model.eval()
print(f"  Audio (Wav2Vec2) loaded — umbral: {umbral}")

# Video — EfficientNet
effnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1, drop_rate=0.4)
effnet.load_state_dict(torch.load(VIDEO_WEIGHTS, map_location=DEVICE))
effnet = effnet.to(DEVICE)
effnet.eval()
print("  Video (EfficientNet-B0) loaded")

# Video — CNN-LSTM
cnnlstm = CNNLSTM()
checkpoint = torch.load(CNNLSTM_WEIGHTS, map_location=DEVICE, weights_only=False)
cnnlstm.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
cnnlstm = cnnlstm.to(DEVICE)
cnnlstm.eval()
print("  Video (CNN-LSTM) loaded")

# Image — Face detector
face_model = EfficientNetClassifier()
checkpoint = torch.load(FACE_WEIGHTS, map_location=DEVICE, weights_only=False)
face_model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
face_model = face_model.to(DEVICE)
face_model.eval()
print("  Image (Face Detector) loaded")

# Image — General detector
image_model = EfficientNetClassifier()
checkpoint = torch.load(IMAGE_WEIGHTS, map_location=DEVICE, weights_only=False)
image_model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
image_model = image_model.to(DEVICE)
image_model.eval()
print("  Image (Image Detector) loaded")

# MTCNN
mtcnn = MTCNN(min_face_size=80, thresholds=[0.6, 0.7, 0.9], keep_all=False, device=DEVICE)

# Transforms
video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("All models loaded!")

# ── Helpers ───────────────────────────────────────────────────────
def confidence_level(fake_prob: float) -> str:
    return "High"   if abs(fake_prob - 0.5) > 0.30 else \
           "Medium" if abs(fake_prob - 0.5) > 0.15 else "Low"

def crop_face(image: Image.Image) -> Image.Image:
    boxes, conf = mtcnn.detect(image)
    if boxes is None or conf[0] < 0.70:
        return image
    x1, y1, x2, y2 = boxes[0]
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - 0.15*w), max(0, y1 - 0.15*h)
    x2, y2 = min(image.width, x2 + 0.15*w), min(image.height, y2 + 0.15*h)
    return image.crop((x1, y1, x2, y2))

# ── Audio inference ───────────────────────────────────────────────
def analyze_audio_file(file_path: str) -> dict:
    import librosa
    audio, _ = librosa.load(file_path, sr=16000, mono=True)

    # ── NORMALIZACIÓN (igual que en entrenamiento) ──
    mean, std = audio.mean(), audio.std()
    audio = (audio - mean) / (std + 1e-8)
    
    # ── TRUNCADO / PADDING a 6 segundos ──
    MAX_SAMPLES = 16000 * 6  # 96_000
    if len(audio) > MAX_SAMPLES:
        s = (len(audio) - MAX_SAMPLES) // 2
        audio = audio[s:s + MAX_SAMPLES]
    else:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

    wav = torch.FloatTensor(audio).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob_fake = torch.sigmoid(audio_model(wav)).item()

    es_fake       = prob_fake >= umbral
    confianza     = (prob_fake - umbral) / (1.0 - umbral) if es_fake else (umbral - prob_fake) / umbral
    confianza_pct = round(confianza * 100, 2)

    return {
        "type"                  : "audio",
        "probability"           : round(prob_fake if es_fake else 1.0 - prob_fake, 4),
        "label"                 : "Potential Deepfake" if es_fake else "Authentic",
        "confidence"            : "High" if confianza_pct > 60 else "Medium" if confianza_pct > 30 else "Low",
        "probabilidad_deepfake" : f"{round(prob_fake * 100, 2)}%",
        "nivel_confianza"       : f"{confianza_pct}%",
        "explicacion"           : (
            f"El sistema analizó la firma acústica mediante Wav2Vec2. "
            f"Se detectaron {'anomalías sintéticas' if es_fake else 'patrones vocales naturales'} "
            f"en el espectro de frecuencias. Umbral: {round(umbral * 100, 1)}%."
        ),
    }

# ── Video inference — EfficientNet ────────────────────────────────
def analyze_video_efficientnet(file_path: str, frames_per_video: int = 12) -> dict:
    cap          = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices      = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

    probs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face   = crop_face(img)
        tensor = video_transform(face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs.append(torch.sigmoid(effnet(tensor)).item())

    cap.release()
    if not probs:
        return None

    fake_probability = float(np.mean(probs))
    return {
        "type"           : "video",
        "probability"    : round(1.0 - fake_probability, 4),
        "label"          : "Authentic" if fake_probability < 0.5 else "Potential Deepfake",
        "confidence"     : confidence_level(fake_probability),
        "frames_analyzed": len(probs),
        "model"          : "efficientnet_b0_v5",
    }

# ── Video inference — CNN-LSTM ────────────────────────────────────
def analyze_video_cnnlstm(file_path: str, frames_per_video: int = 16) -> dict:
    cap          = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices      = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face   = crop_face(img)
        tensor = video_transform(face)
        frames.append(tensor)

    cap.release()
    if not frames:
        return None

    sequence = torch.stack(frames).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_probability = torch.sigmoid(cnnlstm(sequence)).item()

    return {
        "type"           : "video",
        "probability"    : round(1.0 - fake_probability, 4),
        "label"          : "Authentic" if fake_probability < 0.5 else "Potential Deepfake",
        "confidence"     : confidence_level(fake_probability),
        "frames_analyzed": len(frames),
        "model"          : "cnnlstm_v2",
    }

# ── Image inference — Face detector ──────────────────────────────
def analyze_image_face(file_path: str) -> dict:
    img    = Image.open(file_path).convert("RGB")
    face   = crop_face(img)
    tensor = image_transform(face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_probability = torch.sigmoid(face_model(tensor)).item()

    return {
        "type"       : "image",
        "probability": round(1.0 - fake_probability, 4),
        "label"      : "Authentic" if fake_probability < 0.5 else "Potential Deepfake",
        "confidence" : confidence_level(fake_probability),
        "model"      : "face_detector_v1",
    }

# ── Image inference — General detector ───────────────────────────
def analyze_image_general(file_path: str) -> dict:
    img    = Image.open(file_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_probability = torch.sigmoid(image_model(tensor)).item()

    return {
        "type"       : "image",
        "probability": round(1.0 - fake_probability, 4),
        "label"      : "Authentic" if fake_probability < 0.5 else "Potential Deepfake",
        "confidence" : confidence_level(fake_probability),
        "model"      : "image_detector_v1",
    }

# ── Routes ────────────────────────────────────────────────────────

# Audio — única ruta
@app.route("/api/analyze/audio", methods=["POST"])
def route_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower()
    tmp  = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(file.read()); tmp.close()
    try:
        result = analyze_audio_file(tmp.name)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.unlink(tmp.name)
        except: pass
    return jsonify(result)

# Video — /api/analyze/video es el default (EfficientNet)
@app.route("/api/analyze/video", methods=["POST"])
@app.route("/api/analyze/video/efficientnet", methods=["POST"])
def route_video_efficientnet():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower() or ".webm"
    tmp  = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(file.read()); tmp.close()
    try:
        result = analyze_video_efficientnet(tmp.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.unlink(tmp.name)
        except: pass
    if result is None:
        return jsonify({"error": "No face detected in video"}), 422
    return jsonify(result)

@app.route("/api/analyze/video/cnnlstm", methods=["POST"])
def route_video_cnnlstm():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower() or ".webm"
    tmp  = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(file.read()); tmp.close()
    try:
        result = analyze_video_cnnlstm(tmp.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.unlink(tmp.name)
        except: pass
    if result is None:
        return jsonify({"error": "No face detected in video"}), 422
    return jsonify(result)

# Image — /api/analyze/image es el default (face detector)
@app.route("/api/analyze/image", methods=["POST"])
@app.route("/api/analyze/image/face", methods=["POST"])
def route_image_face():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower()
    tmp  = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(file.read()); tmp.close()
    try:
        result = analyze_image_face(tmp.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.unlink(tmp.name)
        except: pass
    return jsonify(result)

@app.route("/api/analyze/image/general", methods=["POST"])
def route_image_general():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower()
    tmp  = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(file.read()); tmp.close()
    try:
        result = analyze_image_general(tmp.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.unlink(tmp.name)
        except: pass
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)