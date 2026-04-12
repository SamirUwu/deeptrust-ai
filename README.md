# DeepTrust AI

Web platform for detecting deepfake audio and video.

## Features
- Upload audio/video files
- Analyze authenticity using AI models
- Display probability and confidence
- Basic history tracking

## Tech Stack
- Frontend: React + Tailwind
- Backend: Flask (Python)
- AI: PyTorch (pretrained models)

## Status
MVP in development

------------------------------------------------------------------------------------------------------------------------------------

# Deepfake Detection

A collection of scripts to detect deepfakes in **audio**, **images**, and **video** using pretrained models.

---

## Project Structure

```
your-repo/
├── AASIST/               ← git submodule (audio model architecture)
├── pretrained/
│   └── AASIST.pth        ← pretrained weights (copy manually, ~85MB)
├── analyze_audio.py      ← audio detection via AASIST
├── analyze_audio2.py     ← audio detection via HuggingFace (easier)
├── analyze_image.py      ← image detection
├── analyze_video.py      ← video detection
└── README.md
```

---

## Requirements

- Python **3.10 or 3.11** (not 3.12+)
- Git with submodule support

---

## Setup

### 1. Clone the repo

```bash
git clone --recurse-submodules https://github.com/you/your-repo.git
cd your-repo
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers librosa soundfile opencv-python Pillow
```

For `analyze_audio.py` (AASIST) only:
```bash
pip install -r AASIST/requirements.txt
```

### 4. Pretrained weights (AASIST only)

Copy `AASIST.pth` into the `pretrained/` folder manually.  
The HuggingFace models used by the other scripts **download automatically** on first run and are cached locally.

---

## Usage

### Audio — AASIST (higher accuracy, needs weights file)
```bash
python analyze_audio.py path/to/audio.wav
```

### Audio — HuggingFace (easier, no setup)
```bash
python analyze_audio2.py path/to/audio.wav
```

### Image
```bash
python analyze_image.py path/to/image.jpg
```

### Video
```bash
python analyze_video.py path/to/video.mp4
```

---

## Output

All scripts return the same format:

```
─── Result ───────────────────────────────
  Real probability : 82.34%
  Fake probability : 17.66%
  Verdict          : REAL

─── Metrics ──────────────────────────────
  Confidence       : 82.34%  ← how sure the model is
  Entropy          : 0.4321  ← lower = more decisive
──────────────────────────────────────────
```

| Metric | What it means |
|---|---|
| `probability_real` | Likelihood the media is genuine |
| `probability_fake` | Likelihood the media is AI-generated or manipulated |
| `confidence` | How certain the model is (either way) |
| `entropy` | Uncertainty — near 0 = very sure, near 0.693 = total coin flip |

---

## Models Used

| Script | Model | Type |
|---|---|---|
| `analyze_audio.py` | AASIST | Graph Attention Network (custom weights) |
| `analyze_audio2.py` | `MelodyMachine/Deepfake-audio-detection-V2` | wav2vec2 |
| `analyze_image.py` | `prithivMLmods/deepfake-detector-model-v1` | SigLIP ViT |
| `analyze_video.py` | `prithivMLmods/deepfake-detector-model-v1` | SigLIP ViT (per-frame) |

---

## Known Limitations

- **Low quality / old videos and images** may be misclassified — compression artifacts can resemble deepfake artifacts
- **Audio models** work best with clean 16kHz recordings
- **Video detection** samples 20 frames and averages scores — it does not analyze temporal consistency
- All models were trained on specific datasets (FaceForensics++, ASVspoof 2019) and may not generalize perfectly to all real-world media