
import torch
import torch.nn as nn
import timm

class EfficientNetImageDetector(nn.Module):
    """Detector de imagenes AI-generadas (objetos) - CIFAKE"""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))


class FaceDetector(nn.Module):
    """Detector de caras AI-generadas - 140k Real and Fake Faces (StyleGAN)"""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))


class CNNLSTM(nn.Module):
    """Detector de video deepfake - FF++ + Celeb-DF"""
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.lstm = nn.LSTM(1280, 512, num_layers=2, batch_first=True, dropout=0.4)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])
