"""
src/predict.py
==============
Inference module – dự đoán cảm xúc cho 1 vùng khuôn mặt.

QUAN TRỌNG: Transform phải khớp HOÀN TOÀN với transform lúc train Kaggle.
  - CNN      : grayscale, 48×48,  Normalize(0.5, 0.5)
  - MobileNet: 3-channel, 224×224, Normalize(ImageNet stats)
"""

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────────
# Transforms – phải KHỚP với Kaggle notebook!
# ─────────────────────────────────────────────

CNN_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

MOBILENET_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),   # repeat grayscale → 3 ch
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],                # ImageNet mean
            std=[0.229, 0.224, 0.225],                 # ImageNet std
        ),
    ]
)

TRANSFORM_MAP = {
    "cnn": CNN_TRANSFORM,
    "transfer": MOBILENET_TRANSFORM,
}


def predict_emotion(
    face_bgr: np.ndarray,
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
    class_names: list,
    conf_threshold: float = 0.4,
) -> tuple | None:
    """
    Dự đoán cảm xúc từ vùng khuôn mặt đã crop.

    Args:
        face_bgr      : numpy array BGR, vùng mặt crop từ frame
        model         : model đã load (eval mode)
        model_type    : 'cnn' hoặc 'transfer'
        device        : torch.device
        class_names   : danh sách tên lớp (7 lớp)
        conf_threshold: ngưỡng confidence tối thiểu

    Returns:
        (label: str, confidence: float) nếu đủ confidence
        None nếu dưới ngưỡng hoặc face_bgr rỗng
    """
    if face_bgr is None or face_bgr.size == 0:
        return None

    # BGR → RGB → PIL Image
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    # Lấy transform tương ứng
    transform = TRANSFORM_MAP.get(model_type)
    if transform is None:
        raise ValueError(f"model_type không hợp lệ: '{model_type}'")

    # Inference
    tensor = transform(face_pil).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    confidence = conf.item()
    if confidence < conf_threshold:
        return None

    label = class_names[idx.item()]
    return label, confidence


def get_all_probs(
    face_bgr: np.ndarray,
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
    class_names: list,
) -> dict:
    """
    Trả về xác suất đầy đủ cho tất cả 7 lớp (dùng để debug / visualize).

    Returns:
        dict {class_name: probability}
    """
    if face_bgr is None or face_bgr.size == 0:
        return {}

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    transform = TRANSFORM_MAP.get(model_type, CNN_TRANSFORM)
    tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().tolist()

    return {cls: round(p, 4) for cls, p in zip(class_names, probs)}
