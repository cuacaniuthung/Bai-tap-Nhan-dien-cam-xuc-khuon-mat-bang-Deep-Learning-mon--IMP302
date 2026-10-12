# 🧠 Hướng dẫn Phát triển Lab 1 – Nhận diện Cảm xúc Khuôn mặt

> **Chiến lược:** Train trên Kaggle (GPU miễn phí) → Export weights → Deploy local (CPU inference)  
> **Model Transfer Learning:** MobileNetV2  
> **Ngôn ngữ:** Python 3.9+  
> **Thời gian ước tính:** 5–6 ngày

---

## 📌 Tổng quan kiến trúc hệ thống

```
[KAGGLE - Phase 1 & 2]                     [LOCAL - Phase 3 & 4]
┌─────────────────────────────┐             ┌──────────────────────────────┐
│  FER-2013 Dataset           │             │  weights/                    │
│       ↓                     │  download   │    model_cnn.pt              │
│  Tiền xử lý & Augmentation  │ ──────────► │    model_mobilenet.pt        │
│       ↓                     │             │         ↓                    │
│  Train M1 (CNN)             │             │  src/app.py                  │
│  Train M2 (MobileNetV2)     │             │    ↓ load model              │
│       ↓                     │             │    ↓ IP Camera (điện thoại)  │
│  Đánh giá, So sánh, Phân    │             │    ↓ Face Detection          │
│  tích lỗi, Đồ thị           │             │    ↓ Preprocess              │
│       ↓                     │             │    ↓ Inference               │
│  Export weights (.pt)       │             │    ↓ Hiển thị kết quả        │
└─────────────────────────────┘             └──────────────────────────────┘
```

---

## 📁 Cấu trúc thư mục cuối cùng

```
MSSV_HoTen_Lab1/
├── DEVELOPMENT_GUIDE.md        ← file này
├── README.md                   ← hướng dẫn cài đặt & chạy
├── requirements.txt            ← thư viện cần cài
├── config.yaml                 ← URL camera, path model, confidence threshold
│
├── notebooks/
│   └── train.ipynb             ← toàn bộ quá trình train (chạy trên Kaggle)
│
├── src/
│   ├── model.py                ← định nghĩa kiến trúc M1 và M2
│   ├── detector.py             ← face detection
│   ├── predict.py              ← inference cho 1 ảnh khuôn mặt
│   └── app.py                  ← ứng dụng realtime chính
│
├── weights/
│   ├── model_cnn.pt            ← weights M1 (download từ Kaggle)
│   └── model_mobilenet.pt      ← weights M2 (download từ Kaggle)
│
└── report.pdf                  ← báo cáo 4–6 trang
```

---

## 🗓️ Lộ trình theo ngày

| Ngày | Giai đoạn | Nơi làm | Ước tính thời gian |
|------|-----------|---------|-------------------|
| 1 | Phase 1: Chuẩn bị & EDA | Kaggle | 2–3 giờ |
| 2 | Phase 2a: Train M1 (CNN) | Kaggle | 3–4 giờ |
| 3 | Phase 2b: Train M2 (MobileNetV2) | Kaggle | 3–4 giờ |
| 4 | Phase 2c: Đánh giá, phân tích, vẽ đồ thị | Kaggle | 2 giờ |
| 5 | Phase 3: Xây dựng local app | Local | 3–4 giờ |
| 6 | Phase 4: Test, viết báo cáo, đóng gói | Local | 2–3 giờ |

---

## ⚙️ Phase 0 – Chuẩn bị môi trường (Local)

### Mục tiêu
Cài đặt đầy đủ thư viện trên máy local để chạy inference (không cần GPU).

### Các bước

**0.1. Tạo virtual environment**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

**0.2. Tạo `requirements.txt`**
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=9.0.0
PyYAML>=6.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```
> ⚠️ Trên máy local, cài `torch` CPU-only để nhẹ hơn:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

**0.3. Cài đặt**
```bash
pip install -r requirements.txt
```

### Kết quả kỳ vọng
- ✅ `import torch; import cv2; import torchvision` chạy không lỗi
- ✅ `torch.cuda.is_available()` có thể trả về `False` — không sao, local chỉ dùng CPU

---

## 🏋️ Phase 1 – Chuẩn bị dữ liệu (Kaggle)

### Mục tiêu
Hiểu dữ liệu FER-2013, xử lý mất cân bằng lớp, thiết lập pipeline augmentation.

### Các bước trên Kaggle Notebook

**1.1. Thêm FER-2013 Dataset vào Kaggle Notebook**
- Vào Kaggle → New Notebook → Add Data → tìm `fer2013`
- Dataset: `msambare/fer2013` (phổ biến nhất)
- Cấu trúc dataset:
  ```
  /kaggle/input/fer2013/
  ├── train/
  │   ├── angry/     (3995 ảnh)
  │   ├── disgust/   (436 ảnh)  ← lớp thiểu số nhất!
  │   ├── fear/      (4097 ảnh)
  │   ├── happy/     (7215 ảnh) ← lớp đa số
  │   ├── neutral/   (4965 ảnh)
  │   ├── sad/       (4830 ảnh)
  │   └── surprise/  (3171 ảnh)
  └── test/
      └── ...
  ```

**1.2. Exploratory Data Analysis (EDA)**
```python
import os, matplotlib.pyplot as plt
import numpy as np

# Thống kê phân bố lớp
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']
train_counts = [len(os.listdir(f'/kaggle/input/fer2013/train/{c}')) for c in classes]

plt.figure(figsize=(10, 5))
plt.bar(classes, train_counts, color='steelblue')
plt.title('Phân bố lớp trong tập Train FER-2013')
plt.ylabel('Số lượng ảnh')
plt.savefig('class_distribution.png')
plt.show()
```

**1.3. Tính class weights để xử lý mất cân bằng**
```python
from sklearn.utils.class_weight import compute_class_weight

labels_list = []
for i, cls in enumerate(classes):
    labels_list.extend([i] * train_counts[i])

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(7),
    y=labels_list
)
# class_weights là array 7 phần tử → dùng trong loss function
```

**1.4. Định nghĩa Data Augmentation**
```python
from torchvision import transforms

# Transform cho tập TRAIN (có augmentation)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # FER-2013 là ảnh xám
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Transform cho tập TEST/VAL (không augmentation)
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

> ⚠️ **Quan trọng:** Lưu lại `mean=[0.5], std=[0.5]` vì local inference phải dùng **đúng** giá trị này.

**1.5. Tạo DataLoader**
```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

train_dataset = ImageFolder('/kaggle/input/fer2013/train', transform=train_transform)
test_dataset  = ImageFolder('/kaggle/input/fer2013/test',  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2)
```

### Kết quả kỳ vọng
- ✅ Biểu đồ phân bố lớp hiển thị rõ sự mất cân bằng (disgust << happy)
- ✅ `class_weights` là array 7 phần tử (lớp thiểu số có weight cao hơn)
- ✅ DataLoader hoạt động: `next(iter(train_loader))` trả về `(batch_images, batch_labels)`

---

## 🔬 Phase 2a – Train M1: CNN tự thiết kế (Kaggle)

### Mục tiêu
Xây dựng và huấn luyện CNN từ đầu, đạt accuracy ≥ 60% trên test FER-2013.

### Kiến trúc M1

```python
# src/model.py — phần M1
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 48→24
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 24→12
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 12→6
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### Huấn luyện M1

```python
import torch, torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cnn = EmotionCNN().to(device)

# Dùng class weights để xử lý mất cân bằng
weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model_cnn.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop (25–30 epochs)
history_cnn = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(30):
    # ... (training + validation loop)
    pass

# Lưu weights
torch.save(model_cnn.state_dict(), 'model_cnn.pt')
```

### Kết quả kỳ vọng
- ✅ Training loss giảm đều qua các epoch
- ✅ **Test accuracy ≥ 60%** (thường đạt 62–66% với kiến trúc này)
- ✅ File `model_cnn.pt` được tạo ra (~2–5 MB)
- ✅ Đồ thị loss/accuracy train vs validation không bị overfit quá mức

---

## 🚀 Phase 2b – Train M2: MobileNetV2 Transfer Learning (Kaggle)

### Mục tiêu
Fine-tune MobileNetV2 pretrained trên ImageNet, đạt accuracy ≥ 65%.

### Lưu ý quan trọng về input MobileNetV2
MobileNetV2 pretrained trên ImageNet nhận **ảnh RGB 3 kênh**, nhưng FER-2013 là **grayscale 1 kênh**.

**Giải pháp:** Chuyển grayscale → RGB bằng cách repeat 3 kênh.

```python
# Transform cho MobileNetV2 (khác M1!)
mobilenet_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ← 3 kênh!
    transforms.Resize((224, 224)),                 # MobileNetV2 cần 224×224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])
```

### Kiến trúc M2

```python
# src/model.py — phần M2
import torchvision.models as models

class EmotionMobileNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Freeze backbone (phase 1: chỉ train classifier)
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Thay thế classifier cuối
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Gọi sau vài epoch để fine-tune toàn bộ model"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
```

### Chiến lược huấn luyện 2 giai đoạn (Progressive Fine-tuning)

```python
model_mobilenet = EmotionMobileNet().to(device)

# Giai đoạn 1 (10 epoch): Chỉ train classifier, backbone đóng băng
optimizer_phase1 = optim.Adam(
    filter(lambda p: p.requires_grad, model_mobilenet.parameters()),
    lr=1e-3
)
# ... train 10 epochs

# Giai đoạn 2 (20 epoch): Unfreeze backbone, fine-tune toàn bộ với LR nhỏ
model_mobilenet.unfreeze_backbone()
optimizer_phase2 = optim.Adam(model_mobilenet.parameters(), lr=1e-4)
# ... train 20 epochs thêm

# Lưu weights
torch.save(model_mobilenet.state_dict(), 'model_mobilenet.pt')
```

### Kết quả kỳ vọng
- ✅ Sau giai đoạn 1 (10 epoch): accuracy ~55–60%
- ✅ Sau giai đoạn 2 (30 epoch tổng): **Test accuracy ≥ 65%** (thường 67–72%)
- ✅ File `model_mobilenet.pt` (~9–12 MB)
- ✅ MobileNetV2 accuracy cao hơn CNN tự thiết kế

---

## 📊 Phase 2c – Đánh giá & Phân tích (Kaggle)

### Mục tiêu
Hoàn thành M3 (so sánh), M4 (phân tích lỗi), M5 (đồ thị).

### M3 – Bảng so sánh

```python
import time
from sklearn.metrics import f1_score

def evaluate_model(model, loader, device, model_name):
    model.eval()
    all_preds, all_labels = [], []

    start = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    elapsed = time.time() - start

    fps      = len(loader.dataset) / elapsed
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1       = f1_score(all_labels, all_preds, average='macro')
    params   = sum(p.numel() for p in model.parameters())

    print(f"=== {model_name} ===")
    print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1 : {f1:.4f}")
    print(f"Params   : {params:,}")
    print(f"FPS      : {fps:.1f}")
    return all_preds, all_labels
```

**Kết quả kỳ vọng bảng M3:**

| Metric | M1 (CNN tự thiết kế) | M2 (MobileNetV2) |
|--------|----------------------|------------------|
| Accuracy | ~62–66% | ~67–72% |
| Macro F1 | ~0.58–0.63 | ~0.64–0.70 |
| Số tham số | ~1.5M | ~2.2M |
| FPS (CPU inference) | cao hơn | thấp hơn một chút |

### M4 – Confusion Matrix & Phân tích lỗi

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix – MobileNetV2')
plt.savefig('confusion_matrix.png', dpi=150)
```

**Nhận xét kỳ vọng:**
- `disgust` hay bị nhầm sang `angry` (biểu cảm tương đồng)
- `fear` hay bị nhầm sang `sad` hoặc `surprise`
- `happy` có accuracy cao nhất (đặc trưng nụ cười rõ ràng)

### M5 – Đồ thị huấn luyện

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
ax1.set_title('Loss theo Epoch'); ax1.legend()
ax2.plot(history['train_acc'], label='Train'); ax2.plot(history['val_acc'], label='Val')
ax2.set_title('Accuracy theo Epoch'); ax2.legend()
plt.savefig('training_curves.png', dpi=150)
```

### Download về máy
- `model_cnn.pt` → `weights/model_cnn.pt`
- `model_mobilenet.pt` → `weights/model_mobilenet.pt`
- `train.ipynb` (còn nguyên output cells) → `notebooks/train.ipynb`

---

## 💻 Phase 3 – Xây dựng Local App

### Mục tiêu
Xây dựng ứng dụng realtime chạy trên CPU, đạt ≥ 10 FPS.

### 3.1. `config.yaml`

```yaml
camera:
  url: "http://192.168.1.15:8080/video"   # thay bằng IP thực tế lúc chạy
  buffer_size: 1
  width: 640
  height: 480

model:
  cnn_path: "weights/model_cnn.pt"
  mobilenet_path: "weights/model_mobilenet.pt"
  confidence_threshold: 0.4
  num_classes: 7
  class_names: ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

inference:
  face_detector: "haar"
  target_fps: 15
```

### 3.2. `src/model.py`
Định nghĩa cả 2 class — **phải giống hệt** kiến trúc trên Kaggle.

```python
import torch, torch.nn as nn
import torchvision.models as models

class EmotionCNN(nn.Module):
    # ... (copy y chang từ Kaggle)

class EmotionMobileNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=False):  # pretrained=False khi load local
        # ... (copy y chang từ Kaggle)

def load_model(model_type: str, weights_path: str, device: torch.device):
    if model_type == 'cnn':
        model = EmotionCNN(num_classes=7)
    elif model_type == 'transfer':
        model = EmotionMobileNet(num_classes=7, pretrained=False)
    else:
        raise ValueError(f"model_type không hợp lệ: {model_type}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model
```

### 3.3. `src/detector.py`

```python
import cv2

class FaceDetector:
    def __init__(self, method='haar'):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect(self, frame):
        """Trả về list (x, y, w, h) của các khuôn mặt."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces if len(faces) > 0 else []
```

### 3.4. `src/predict.py`

```python
import torch, cv2, numpy as np
from PIL import Image
from torchvision import transforms

# ⚠️ PHẢI KHỚP với transform lúc train trên Kaggle!
CNN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

MOBILENET_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_emotion(face_bgr, model, model_type, device, class_names, conf_threshold=0.4):
    face_pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    transform = CNN_TRANSFORM if model_type == 'cnn' else MOBILENET_TRANSFORM
    tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        conf, idx = probs.max(dim=1)

    if conf.item() < conf_threshold:
        return None
    return class_names[idx.item()], conf.item()
```

### 3.5. `src/app.py`

```python
import cv2, time, argparse, yaml, torch
from detector import FaceDetector
from predict  import predict_emotion
from model    import load_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--model',  type=str, default='transfer',
                        choices=['cnn', 'transfer'])
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()

def run(args, cfg):
    device = torch.device('cpu')
    weights_path = cfg['model']['cnn_path'] if args.model == 'cnn' \
                   else cfg['model']['mobilenet_path']
    model    = load_model(args.model, weights_path, device)
    detector = FaceDetector(cfg['inference']['face_detector'])
    class_names   = cfg['model']['class_names']
    conf_threshold = cfg['model']['confidence_threshold']
    source = args.source

    # Ảnh tĩnh
    if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        frame = cv2.imread(source)
        for (x, y, w, h) in detector.detect(frame):
            result = predict_emotion(frame[y:y+h, x:x+w], model, args.model,
                                     device, class_names, conf_threshold)
            label, conf = result if result else ("Unknown", 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label}: {conf:.0%}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('Emotion Detection', frame)
        cv2.waitKey(0)
    else:
        # Realtime stream
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg['camera']['buffer_size'])
        prev_time = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ Mất kết nối camera, đang thử lại...")
                cap.release(); cap = cv2.VideoCapture(source)
                continue
            for (x, y, w, h) in detector.detect(frame):
                result = predict_emotion(frame[y:y+h, x:x+w], model, args.model,
                                         device, class_names, conf_threshold)
                label, conf = result if result else ("Unknown", 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"{label}: {conf:.0%}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            fps = 1 / (time.time() - prev_time + 1e-6)
            prev_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,200), 2)
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    run(args, cfg)
```

### Kết quả kỳ vọng Phase 3
- ✅ `python src/app.py --source test.jpg --model transfer` → hiển thị ảnh với bounding box
- ✅ `python src/app.py --source http://<ip>:8080/video --model cnn` → realtime ≥ 10 FPS
- ✅ Không crash khi mất kết nối camera
- ✅ Hiển thị FPS, nhãn cảm xúc và confidence trên frame

---

## 🧪 Phase 4 – Test, Hoàn thiện & Đóng gói

### 4.1. Test 2 lệnh bắt buộc của đề
```bash
python src/app.py --source http://<ip>:8080/video --model cnn
python src/app.py --source test.jpg --model transfer
```

### 4.2. Kết nối IP Camera điện thoại
1. Cài **IP Webcam** (Android) hoặc **DroidCam** (Android/iOS)
2. Điện thoại và laptop cùng mạng Wi-Fi
3. Mở app → Start Server → ghi lại URL (vd: `http://192.168.1.15:8080`)
4. Cập nhật `camera.url` trong `config.yaml`
5. Test: `python src/app.py --source http://192.168.1.15:8080/video --model transfer`

> ⚠️ **Mẹo giảm độ trễ:** Giảm độ phân giải điện thoại xuống 640×480 trong app IP camera.

### 4.3. Cấu trúc báo cáo `report.pdf` (4–6 trang)
1. **Mô tả dữ liệu** – FER-2013, phân bố lớp, biện pháp xử lý mất cân bằng
2. **Kiến trúc M1** – sơ đồ layers, lý do thiết kế
3. **Kiến trúc M2** – MobileNetV2, chiến lược fine-tuning 2 giai đoạn
4. **Bảng so sánh M3** – accuracy, F1, params, FPS
5. **Confusion matrix & phân tích lỗi M4**
6. **Ảnh chụp màn hình demo** với camera điện thoại

### 4.4. Đóng gói
```bash
Compress-Archive -Path MSSV_HoTen_Lab1 -DestinationPath MSSV_HoTen_Lab1.zip
```

---

## 🎤 Chuẩn bị Phỏng vấn (60% điểm)

| Câu hỏi | Gợi ý trả lời |
|---------|--------------|
| Tại sao chọn MobileNetV2? | Nhẹ, chạy CPU tốt, đủ accuracy, thiết kế cho edge devices với depthwise separable convolution |
| Batch Normalization làm gì? | Chuẩn hóa activation giữa các layer → train ổn định, nhanh hội tụ, giảm dependency vào learning rate |
| Tại sao dùng class weights? | FER-2013 mất cân bằng (disgust 436 vs happy 7215 ảnh), class weights phạt nặng hơn khi sai lớp thiểu số |
| Transfer Learning là gì? | Dùng lại features đã học từ ImageNet (edges, textures, shapes), chỉ fine-tune cho task mới |
| Dropout có tác dụng gì? | Regularization — tắt ngẫu nhiên neurons khi train, giúp model không overfit vào training data |
| Tại sao fear/sad hay bị nhầm? | Biểu cảm cơ mặt tương đồng (mắt cụp, miệng xuống), dataset ít đặc trưng phân biệt rõ |
| Progressive fine-tuning là gì? | Train classifier trước (backbone đóng băng), sau đó unfreeze toàn bộ với LR nhỏ hơn để tránh "catastrophic forgetting" |

---

## ✅ Checklist cuối cùng trước buổi chấm

- [ ] `weights/model_cnn.pt` tồn tại
- [ ] `weights/model_mobilenet.pt` tồn tại
- [ ] `config.yaml` — URL camera **không** hard-code trong code
- [ ] `requirements.txt` đầy đủ, `pip install -r requirements.txt` chạy được
- [ ] `notebooks/train.ipynb` đã chạy xong, còn nguyên output cells
- [ ] `python src/app.py --source test.jpg --model transfer` → OK
- [ ] `python src/app.py --source http://<ip>:8080/video --model cnn` → OK
- [ ] FPS ≥ 10 khi demo realtime trên CPU
- [ ] Xử lý graceful khi mất kết nối camera (không crash)
- [ ] `report.pdf` đủ 4–6 trang, có ảnh chụp màn hình demo
- [ ] Đọc lại và hiểu từng dòng code để trả lời phỏng vấn
