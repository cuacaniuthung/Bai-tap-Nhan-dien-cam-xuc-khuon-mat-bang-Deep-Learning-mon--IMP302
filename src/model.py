"""
src/model.py
============
Định nghĩa kiến trúc 2 model:
  - EmotionCNN             : M1 – CNN tự thiết kế (input: 1×48×48)
  - EmotionEfficientNetCBAM: M2 – EfficientNet-B0 + CBAM Attention (input: 3×224×224)

QUAN TRỌNG: Kiến trúc này khớp 100% với notebook đã train trên Kaggle.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────
# M1: CNN tự thiết kế
# Input : (B, 1, 48, 48)  – grayscale
# Output: (B, 7)
# ─────────────────────────────────────────────
class EmotionCNN(nn.Module):
    """
    3 Conv Block (Conv–BN–ReLU–Pooling–Dropout) + 2-layer Classifier.
    Số tham số: ~1.5M
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            # ── Block 1: 1→32 ch, 48→24 ──────────────────
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # ── Block 2: 32→64 ch, 24→12 ─────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # ── Block 3: 64→128 ch, 12→6 ─────────────────
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# CBAM (Channel & Spatial Attention Module)
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ─────────────────────────────────────────────
# M2: EfficientNet-B0 + CBAM Attention
# Input : (B, 3, 224, 224) – 3-channel (grayscale repeated)
# Output: (B, 7)
# ─────────────────────────────────────────────
class EmotionEfficientNetCBAM(nn.Module):
    """
    EfficientNet-B0 + CBAM Attention Module + Custom Classifier.
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.cbam = CBAM(in_planes=in_features)

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.classifier(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_stage(self, stage_idx: int = 6):
        for param in self.backbone.features[stage_idx:].parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


# Alias tương thích ngược
EmotionMobileNet = EmotionEfficientNetCBAM


# ─────────────────────────────────────────────
# Factory – load model từ weights file
# ─────────────────────────────────────────────
def load_model(
    model_type: str,
    weights_path: str,
    device: torch.device,
    num_classes: int = 7,
) -> nn.Module:
    """
    Load model đã train từ file weights.

    Args:
        model_type  : 'cnn' hoặc 'transfer'
        weights_path: đường dẫn tới file .pt
        device      : torch.device('cpu') hoặc cuda
        num_classes : số lớp cảm xúc (mặc định 7)

    Returns:
        model đã load weights, ở chế độ eval()
    """
    if model_type == "cnn":
        model = EmotionCNN(num_classes=num_classes)
    elif model_type in ["transfer", "efficientnet", "mobilenet"]:
        model = EmotionEfficientNetCBAM(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(
            f"model_type phải là 'cnn' hoặc 'transfer', nhận được: '{model_type}'"
        )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    dummy_cnn = torch.randn(2, 1, 48, 48)
    dummy_eff = torch.randn(2, 3, 224, 224)

    cnn = EmotionCNN()
    eff = EmotionEfficientNetCBAM(pretrained=False)

    out_cnn = cnn(dummy_cnn)
    out_eff = eff(dummy_eff)

    print(f"EmotionCNN              output shape: {out_cnn.shape}")   # (2, 7)
    print(f"EmotionEfficientNetCBAM output shape: {out_eff.shape}")   # (2, 7)
    print(f"EmotionCNN              params: {sum(p.numel() for p in cnn.parameters()):,}")
    print(f"EmotionEfficientNetCBAM params: {sum(p.numel() for p in eff.parameters()):,}")

