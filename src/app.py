"""
src/app.py
==========
Entry point của ứng dụng nhận diện cảm xúc thời gian thực.

Lệnh chạy:
    # Realtime từ IP camera điện thoại
    python src/app.py --source http://192.168.x.x:8080/video --model transfer

    # Ảnh tĩnh
    python src/app.py --source test.jpg --model cnn

    # File video
    python src/app.py --source video.mp4 --model transfer

    # Chỉ định file config khác (mặc định: config.yaml)
    python src/app.py --source test.jpg --model cnn --config my_config.yaml

Phím tắt khi đang chạy:
    q - thoát
    s - lưu screenshot (chỉ ở chế độ video/camera)
"""

import argparse
import sys
import time
from pathlib import Path

# Đảm bảo in tiếng Việt không bị lỗi charmap trên Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import torch
import yaml
import numpy as np

# Thêm thư mục src vào sys.path để import được khi chạy từ gốc dự án
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import FaceDetector
from model import load_model
from predict import predict_emotion

# ─────────────────────────────────────────────
# Màu sắc cho từng cảm xúc (BGR)
# ─────────────────────────────────────────────
EMOTION_COLORS = {
    "Angry":    (0,   0,   220),   # đỏ
    "Disgust":  (0,   128, 0  ),   # xanh lá
    "Fear":     (128, 0,   128),   # tím
    "Happy":    (0,   220, 220),   # vàng
    "Neutral":  (180, 180, 180),   # xám
    "Sad":      (220, 100, 0  ),   # xanh dương
    "Surprise": (0,   165, 255),   # cam
}
DEFAULT_COLOR = (0, 255, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nhan dien cam xuc khuon mat thoi gian thuc (Realtime FER-2013)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        required=False,
        help=(
            "Nguon dau vao: '0'/'usb' (Webcam/USB Type-C), 'adb' (Mobile cam Type-C 127.0.0.1:8080), "
            "URL IP Camera (http://...:8080/video), anh (.jpg/.png), hoac video (.mp4). "
            "Mac dinh lay tu config.yaml"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="transfer",
        choices=["cnn", "transfer"],
        help="Chon model: 'cnn' (M1 Custom CNN) hoac 'transfer' (M2 MobileNetV2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Duong dan file config (mac dinh: config.yaml)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"[ERROR] Không tìm thấy file config: {config_path}")
        sys.exit(1)
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_face_result(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    label: str | None,
    conf: float,
) -> np.ndarray:
    """Vẽ bounding box + nhãn cảm xúc + confidence lên frame."""
    color = EMOTION_COLORS.get(label, DEFAULT_COLOR) if label else (100, 100, 100)

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Nhãn + confidence
    if label:
        text = f"{label}: {conf:.0%}"
    else:
        text = "Unknown"

    # Background cho text
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
    cv2.putText(
        frame, text, (x + 3, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
    )
    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    model_type: str,
    num_faces: int,
) -> np.ndarray:
    """Vẽ HUD info (FPS, model, số khuôn mặt) góc trên trái."""
    model_label = "CNN (M1)" if model_type == "cnn" else "MobileNetV2 (M2)"
    lines = [
        f"FPS  : {fps:5.1f}",
        f"Model: {model_label}",
        f"Faces: {num_faces}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (10, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2
        )
    return frame


# ─────────────────────────────────────────────
# Chế độ 1: Ảnh tĩnh
# ─────────────────────────────────────────────
def run_image(source: str, model, model_type: str,
              detector: FaceDetector, cfg: dict, device):
    frame = cv2.imread(source)
    if frame is None:
        print(f"[ERROR] Không đọc được ảnh: {source}")
        sys.exit(1)

    class_names = cfg["model"]["class_names"]
    conf_thresh  = cfg["model"]["confidence_threshold"]
    faces = detector.detect(frame)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        result = predict_emotion(
            face_crop, model, model_type, device, class_names, conf_thresh
        )
        label, conf = result if result else (None, 0.0)
        draw_face_result(frame, x, y, w, h, label, conf)

    model_label = "CNN" if model_type == "cnn" else "MobileNetV2"
    cv2.putText(
        frame, f"Model: {model_label} | Faces: {len(faces)}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2
    )
    cv2.imshow("Emotion Detection – Ảnh tĩnh", frame)
    print(f"[INFO] Phát hiện {len(faces)} khuôn mặt. Nhấn phím bất kỳ để đóng.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Stream Capture Đa Luồng (Chống trễ / Zero-Latency Buffer)
# ─────────────────────────────────────────────
import threading

class ThreadedStream:
    """
    Stream capture đa luồng loại bỏ hoàn toàn hiện tượng dồn ứ bộ đệm (buffer pile-up)
    trong OpenCV khi stream qua IP camera, Wi-Fi hoặc USB. Luôn đảm bảo model inference
    nhận được khung hình mới nhất tại thời gian thực.
    """
    def __init__(self, src, buf_size=1):
        self.src = src
        self.buf_size = buf_size
        self.cap = self._open_capture(src)
        self.lock = threading.Lock()
        self.ret = False
        self.frame = None
        self.stopped = False

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buf_size)
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def _open_capture(self, src):
        # Trên Windows, dùng cv2.CAP_DSHOW cho camera cục bộ để mở nhanh và tránh lỗi backend MSMF
        if isinstance(src, int) and sys.platform.startswith("win"):
            return cv2.VideoCapture(src, cv2.CAP_DSHOW)
        return cv2.VideoCapture(src)

    def _capture_loop(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
            if not ret:
                time.sleep(0.02)

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()


# ─────────────────────────────────────────────
# Chế độ 2: Realtime (camera / video)
# ─────────────────────────────────────────────
def run_stream(source: str, model, model_type: str,
               detector: FaceDetector, cfg: dict, device):
    class_names = cfg["model"]["class_names"]
    conf_thresh  = cfg["model"]["confidence_threshold"]
    buf_size     = cfg["camera"]["buffer_size"]

    # Phân tích linh hoạt nguồn đầu vào (Webcam, USB type-C, ADB, IP camera, video file)
    src_lower = str(source).lower().strip()
    if src_lower in ["usb", "webcam", "0"]:
        src_val = 0
        is_live_stream = True
        print("[INFO] Khởi động Webcam máy tính / USB Camera (Index 0, DirectShow)...")
    elif src_lower in ["usb1", "1"]:
        src_val = 1
        is_live_stream = True
        print("[INFO] Khởi động USB Camera ngoài (Index 1, DirectShow)...")
    elif src_lower == "adb":
        src_val = "http://127.0.0.1:8080/video"
        is_live_stream = True
        print("[INFO] Chế độ kết nối Mobile Cam qua cáp Type-C (ADB Port Forwarding: http://127.0.0.1:8080/video)...")
    elif str(source).isdigit():
        src_val = int(source)
        is_live_stream = True
    elif src_lower.startswith("http://") or src_lower.startswith("https://") or src_lower.startswith("rtsp://"):
        src_val = source
        is_live_stream = True
    else:
        src_val = source
        is_live_stream = False  # File video tĩnh (mp4, avi...)

    target_w     = cfg["camera"].get("width", 640)
    target_h     = cfg["camera"].get("height", 480)

    # Khởi tạo capture tương ứng
    if is_live_stream:
        cap = ThreadedStream(src_val, buf_size=buf_size)
    else:
        cap = cv2.VideoCapture(src_val)

    if not cap.isOpened():
        print(f"[ERROR] Không mở được nguồn: {source}")
        if isinstance(src_val, int):
            print("        [Gợi ý] Hãy kiểm tra quyền truy cập Camera của Windows hoặc thử cắm lại cổng USB.")
        sys.exit(1)

    prev_time = time.time()
    fps = 0.0
    screenshot_count = 0

    print("[INFO] Đang chạy. Nhấn 'q' để thoát, 's' để chụp màn hình.")

    while True:
        ok, frame = cap.read()

        # ── Xử lý mất kết nối camera ──────────────────
        if not ok or frame is None:
            source_str = str(source)
            if source_str.startswith("http"):
                print("[WARN] Đang đợi tín hiệu từ camera stream...")
                time.sleep(0.5)
                continue
            else:
                # File video đã hết hoặc camera bị ngắt kết nối
                print("[INFO] Kết thúc luồng video.")
                break

        # ── Tự động resize về chuẩn 640x480 (Tăng tốc Face Detection gấp 8 lần) ──
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # ── Face detection + emotion prediction ────────
        faces = detector.detect(frame)
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            result = predict_emotion(
                face_crop, model, model_type, device, class_names, conf_thresh
            )
            label, conf = result if result else (None, 0.0)
            draw_face_result(frame, x, y, w, h, label, conf)

        # ── FPS tính toán ──────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── HUD ───────────────────────────────────────
        draw_hud(frame, fps, model_type, len(faces))

        cv2.imshow("Emotion Detection - Realtime", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Thoát.")
            break
        elif key == ord("s"):
            fname = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Đã lưu: {fname}")
            screenshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cpu")

    # Chọn đường dẫn weights
    weights_path = (
        cfg["model"]["cnn_path"]
        if args.model == "cnn"
        else cfg["model"]["mobilenet_path"]
    )

    # Kiểm tra file weights tồn tại
    if not Path(weights_path).exists():
        print(
            f"[ERROR] Không tìm thấy file weights: '{weights_path}'\n"
            f"        Hãy download model_cnn.pt / model_mobilenet.pt từ Kaggle\n"
            f"        và đặt vào thư mục 'weights/'"
        )
        sys.exit(1)

    print(f"[INFO] Loading model '{args.model}' từ: {weights_path}")
    model    = load_model(args.model, weights_path, device)
    detector = FaceDetector(method=cfg["inference"]["face_detector"])

    source = args.source if args.source is not None else cfg["camera"]["url"]
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    if str(source).lower().endswith(image_exts):
        run_image(source, model, args.model, detector, cfg, device)
    else:
        run_stream(source, model, args.model, detector, cfg, device)


if __name__ == "__main__":
    main()
