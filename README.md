# Lab 1 – Nhận diện cảm xúc khuôn mặt bằng Deep Learning

> **Môn:** IMP302 | **Giảng viên:** TS. Nguyễn Đình Vinh  
> **Chiến lược:** Train trên Kaggle GPU → Deploy local CPU

---

## ⚙️ Cài đặt

### 1. Clone / Giải nén project
```bash
# Nếu dùng ZIP:
# Giải nén MSSV_HoTen_Lab1.zip vào thư mục lab1/
```

### 2. Tạo virtual environment và cài thư viện
```bash
python -m venv venv
venv\Scripts\activate        # Windows

# Cài PyTorch CPU-only (nhẹ hơn, đủ để inference)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài các thư viện còn lại
pip install -r requirements.txt
```

### 3. Đặt file weights
Sau khi train xong trên Kaggle, download 2 file và đặt vào:
```
weights/
├── model_cnn.pt          ← M1: CNN tự thiết kế
└── model_mobilenet.pt    ← M2: MobileNetV2
```

### 4. Cấu hình IP Camera
Chỉnh URL trong `config.yaml`:
```yaml
camera:
  url: "http://<IP_DIEN_THOAI>:8080/video"
```
*(Thay `<IP_DIEN_THOAI>` bằng IP thực tế hiển thị trong app IP Webcam)*

---

## 🚀 Cách chạy

### Realtime từ IP Camera điện thoại
```bash
python src/app.py --source http://192.168.x.x:8080/video --model transfer
python src/app.py --source http://192.168.x.x:8080/video --model cnn
```

### Ảnh tĩnh
```bash
python src/app.py --source test.jpg --model transfer
python src/app.py --source test.jpg --model cnn
```

### File video
```bash
python src/app.py --source video.mp4 --model transfer
```

**Phím tắt khi đang chạy:**
- `q` – thoát
- `s` – lưu screenshot

---

## 📊 Kết quả chính

| Metric | M1 – CNN tự thiết kế | M2 – MobileNetV2 |
|--------|----------------------|------------------|
| Test Accuracy | XX% | XX% |
| Macro F1 | X.XX | X.XX |
| Số tham số | ~1.5M | ~2.2M |
| FPS (CPU) | ~XX | ~XX |

*(Cập nhật sau khi train trên Kaggle)*

---

## 📁 Cấu trúc thư mục

```
lab1/
├── README.md
├── requirements.txt
├── config.yaml
├── DEVELOPMENT_GUIDE.md
├── notebooks/
│   └── train.ipynb        ← train trên Kaggle
├── src/
│   ├── model.py           ← kiến trúc M1, M2
│   ├── detector.py        ← phát hiện khuôn mặt
│   ├── predict.py         ← inference
│   └── app.py             ← ứng dụng chính
└── weights/
    ├── model_cnn.pt
    └── model_mobilenet.pt
```

---

## 📱 Kết nối IP Camera điện thoại

1. Cài **IP Webcam** (Android) từ CH Play
2. Kết nối điện thoại và laptop **cùng mạng Wi-Fi**
3. Mở app → **Start Server**
4. Ghi lại URL hiển thị (vd: `http://192.168.1.15:8080`)
5. Cập nhật vào `config.yaml`
6. Chạy: `python src/app.py --source http://192.168.1.15:8080/video --model transfer`
