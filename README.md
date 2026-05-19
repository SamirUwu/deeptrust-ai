# DeepTrust AI

Web platform for detecting deepfake audio and video using pretrained AI models.

---

## Features

- Upload audio or video files for deepfake analysis
- Real-time probability and confidence scores
- Basic history tracking
- REST API backend (Flask) + React frontend (Next.js)

---

## Tech Stack

- **Frontend**: Next.js + Tailwind + shadcn/ui
- **Backend**: Flask (Python)
- **AI — Audio**: Wav2Vec2 (`facebook/wav2vec2-base`) fine-tuned for deepfake detection
- **AI — Video**: EfficientNet-B0 trained on FaceForensics++ c23 + Celeb-DF v2

---

## Project Structure

```
DeepTrustAI/
├── AASIST/                          ← git submodule (not used in web pipeline)
├── components/deep-trust/           ← Next.js frontend components
├── pretrained/
│   └── AASIST.pth                   ← AASIST weights (standalone scripts only)
├── weights/
│   ├── mejor_modelo.pt              ← Wav2Vec2 audio model weights
│   ├── umbral.json                  ← optimal decision threshold (0.41)
│   └── efficientnet_b0_v5.pt        ← EfficientNet video model weights
├── svm/
│   ├── best_svm_v3.pkl              ← trained SVM (AASIST pipeline, research only)
│   └── scaler_v3.pkl                ← StandardScaler for SVM embeddings
├── api.py                           ← Flask backend (main entry point)
├── next.config.mjs                  ← Next.js config
└── package.json
```

---

## Requirements

- Python **3.10 or 3.11** (not 3.12+)
- Node.js 18+
- Git
- FFmpeg (required for audio/video processing)

---

## Setup

### 1. Clone the repo

```bash
git clone --recurse-submodules https://github.com/you/your-repo.git
cd DeepTrustAI
```

### 2. Install FFmpeg (Windows)

FFmpeg is required to process audio and video files recorded from the browser.

1. Go to https://ffmpeg.org/download.html
2. Click **Windows** → **Windows builds from gyan.dev**
3. Download `ffmpeg-release-essentials.zip`
4. Extract the zip and open the `bin/` folder
5. Copy `ffmpeg.exe`, `ffplay.exe`, and `ffprobe.exe` to `C:\Windows\System32\`
6. Open a new terminal and verify the installation:

```bash
ffmpeg -version
```

### 3. Create a Python virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install flask flask-cors
pip install transformers librosa soundfile
pip install timm facenet-pytorch
pip install opencv-python Pillow
pip install scikit-learn joblib
```

### 5. Install frontend dependencies

```bash
npm install
```

### 6. Add model weights

Place the following files in the `weights/` folder (request from the team):

```
weights/
├── mejor_modelo.pt          ← Wav2Vec2 audio model (~360MB)
├── umbral.json              ← decision threshold
└── efficientnet_b0_v5.pt   ← EfficientNet video model (~20MB)
```

`umbral.json` format:
```json
{ "umbral": 0.409912109375 }
```

---

## Running the Project

You need **two terminals** running at the same time.

### Terminal 1 — Python backend

```bash
python api.py
```

You should see:
```
Loading models...
  Audio model loaded — umbral: 0.4099
  Video model loaded
All models loaded!
Running on http://127.0.0.1:8000
```

### Terminal 2 — Next.js frontend

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

---

## API Endpoints

### POST `/api/analyze/audio`
Accepts `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.webm`.

```json
{
  "type": "audio",
  "label": "Authentic",
  "probability": 0.87,
  "confidence": "High",
  "probabilidad_deepfake": "13.0%",
  "nivel_confianza": "78.4%",
  "explicacion": "El sistema analizó la firma acústica..."
}
```

### POST `/api/analyze/video`
Accepts `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.

```json
{
  "type": "video",
  "label": "Potential Deepfake",
  "probability": 0.23,
  "confidence": "High",
  "frames_analyzed": 12
}
```

### GET `/health`
Returns `{"status": "ok"}` — use to verify the backend is running.

---

## Models

### Audio — Wav2Vec2 Classifier
- Base: `facebook/wav2vec2-base` (frozen backbone)
- Head: 3-layer MLP (768 → 256 → 64 → 1)
- Threshold: 0.4099 (optimized on validation set)
- Input: 16kHz mono audio

### Video — EfficientNet-B0
- Architecture: EfficientNet-B0 (ImageNet pretrained, fine-tuned)
- Training data: FaceForensics++ c23 + Celeb-DF v2
- Face detection: MTCNN (min confidence 0.70)
- Input: 12 sampled frames per video, 224×224 face crops

| Dataset | AUC-ROC |
|---|---|
| Celeb-DF v2 (cross-validation) | 0.9999 |
| DFDC (in-the-wild test) | 0.7498 |

---

## Known Limitations

- Audio model works best with clean 16kHz recordings — compressed formats (OGG, MP3 from messaging apps) may affect accuracy
- Video model may struggle with low-quality or heavily compressed footage
- Both models were trained on specific datasets and may not generalize perfectly to all real-world media
- No GPU required but inference will be slower on CPU 

---

## Team

- Juan David Barcelo Barraza — Video module (EfficientNet-B0)
- Amir Orozco / Elkin Pulgar — Audio module (Wav2Vec2 + AASIST research)
- Samir Barcelo — Frontend & backend integration