"""
src/detector.py
===============
Face detection module – tương thích OpenCV 4.x và 5.x.

Backends hỗ trợ:
  - 'yn'  : FaceDetectorYN (YuNet ONNX) – OpenCV 5.x, chính xác (mặc định)
            Model file: weights/face_detection_yunet_2023mar.onnx (~227KB)
  - 'haar': Haar Cascade – OpenCV 4.x
  - 'dnn' : OpenCV DNN SSD face detector – cần file model riêng

OpenCV 5.x đã loại bỏ CascadeClassifier khỏi namespace chính.
"""

import cv2
import numpy as np


def _get_best_method() -> str:
    """Tự động chọn method tốt nhất theo phiên bản OpenCV."""
    if hasattr(cv2, 'FaceDetectorYN'):
        return 'yn'       # OpenCV 5.x+ (YuNet built-in)
    elif hasattr(cv2, 'CascadeClassifier'):
        return 'haar'     # OpenCV 4.x
    else:
        return 'dnn'


class FaceDetector:
    """
    Phát hiện khuôn mặt trong một frame BGR.
    Tự động chọn backend phù hợp với phiên bản OpenCV.

    Sử dụng:
        detector = FaceDetector()
        faces = detector.detect(frame)   # [(x, y, w, h), ...]
    """

    def __init__(self, method: str = 'auto'):
        """
        Args:
            method: 'auto' (tự chọn), 'yn' (YuNet/OpenCV5),
                    'haar' (OpenCV4), 'dnn' (SSD caffe)
        """
        if method == 'auto':
            method = _get_best_method()

        self.method = method
        print(f"[FaceDetector] Using backend: '{method}' (OpenCV {cv2.__version__})")

        if method == 'yn':
            self._init_yunet()
        elif method == 'haar':
            self._init_haar()
        elif method == 'dnn':
            self._init_dnn()
        else:
            raise ValueError(f"method phải là 'auto', 'yn', 'haar', hoặc 'dnn'. Nhận: '{method}'")

    # Đường dẫn tới YuNet ONNX model
    YUNET_MODEL = "weights/face_detection_yunet_2023mar.onnx"

    # ── YuNet (OpenCV 5.x với ONNX model) ────────────────────────────────
    def _init_yunet(self):
        """
        FaceDetectorYN dùng YuNet ONNX model.
        Download: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
        """
        import os
        if not os.path.exists(self.YUNET_MODEL):
            raise FileNotFoundError(
                f"Không tìm thấy YuNet model: {self.YUNET_MODEL}\n"
                "Chạy lệnh sau để tải về:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', "
                "'weights/face_detection_yunet_2023mar.onnx')\""
            )
        self._yn_input_size = (640, 480)
        self._detector_yn = cv2.FaceDetectorYN.create(
            model=self.YUNET_MODEL,
            config="",
            input_size=self._yn_input_size,
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

    # ── Haar Cascade (OpenCV 4.x) ─────────────────────────────────────────
    def _init_haar(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector_haar = cv2.CascadeClassifier(cascade_path)
        if self._detector_haar.empty():
            raise RuntimeError(f"Không tải được Haar Cascade từ: {cascade_path}")

    # ── OpenCV DNN SSD ────────────────────────────────────────────────────
    def _init_dnn(self):
        prototxt = "weights/deploy.prototxt"
        caffemodel = "weights/res10_300x300_ssd_iter_140000.caffemodel"
        import os
        if not os.path.exists(prototxt):
            raise FileNotFoundError(
                f"Không tìm thấy: {prototxt}\n"
                "Tải về từ: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
            )
        self._detector_dnn = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self._dnn_conf_threshold = 0.5

    # ── Public API ────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list:
        """
        Phát hiện khuôn mặt trong frame BGR.

        Args:
            frame: ảnh BGR (H×W×3 numpy array)

        Returns:
            list of (x, y, w, h) – bounding boxes. [] nếu không có mặt.
        """
        if self.method == 'yn':
            return self._detect_yn(frame)
        elif self.method == 'haar':
            return self._detect_haar(frame)
        elif self.method == 'dnn':
            return self._detect_dnn(frame)
        return []

    def _detect_yn(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        # Cập nhật input size nếu frame thay đổi kích thước
        if (w, h) != self._yn_input_size:
            self._yn_input_size = (w, h)
            self._detector_yn.setInputSize((w, h))

        _, faces = self._detector_yn.detect(frame)
        if faces is None:
            return []

        results = []
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # Clamp về trong frame
            x  = max(0, x);  y  = max(0, y)
            fw = min(fw, w - x); fh = min(fh, h - y)
            if fw > 0 and fh > 0:
                results.append((x, y, fw, fh))
        return results

    def _detect_haar(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector_haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def _detect_dnn(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self._detector_dnn.setInput(blob)
        detections = self._detector_dnn.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self._dnn_conf_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces


if __name__ == "__main__":
    detector = FaceDetector()  # auto-detect best backend
    print(f"Backend selected: {detector.method}")

    # Test với frame ngẫu nhiên
    dummy = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    faces = detector.detect(dummy)
    print(f"Faces detected on random frame: {len(faces)}")
