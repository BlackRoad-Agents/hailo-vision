#!/usr/bin/env python3
"""Hailo-8 Vision Pipeline — object detection with YOLOv8s on Hailo-8 accelerator.

Falls back to CPU inference via OpenCV DNN if no Hailo device is available.
"""

import json
import sys
import os
import time
import numpy as np
from pathlib import Path

# Detection config
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class HailoDetector:
    """Object detection using Hailo-8 NPU."""

    def __init__(self, model_path=None):
        self.backend = None
        self.runner = None
        self.model_path = model_path or os.environ.get(
            "HAILO_MODEL", "/usr/share/hailo-models/yolov8s.hef"
        )
        self._init_backend()

    def _init_backend(self):
        """Try Hailo first, fall back to OpenCV DNN."""
        try:
            from hailo_platform import HEF, VDevice, ConfigureParams, FormatType
            self.hef = HEF(self.model_path)
            self.vdevice = VDevice()
            configure_params = ConfigureParams.create_from_hef(
                hef=self.hef, interface=HailoConfigureParams.default_interface()
            )
            self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
            self.input_vstreams_params = self.network_group.make_input_vstream_params(
                quantized=False, format_type=FormatType.FLOAT32
            )
            self.output_vstreams_params = self.network_group.make_output_vstream_params(
                quantized=False, format_type=FormatType.FLOAT32
            )
            self.backend = "hailo"
            print("[vision] Hailo-8 backend initialized", file=sys.stderr)
        except Exception as e:
            print(f"[vision] Hailo unavailable ({e}), falling back to CPU", file=sys.stderr)
            self._init_cpu_backend()

    def _init_cpu_backend(self):
        """Initialize OpenCV DNN backend for CPU inference."""
        try:
            import cv2
            onnx_path = os.environ.get("YOLO_ONNX", "yolov8s.onnx")
            if os.path.exists(onnx_path):
                self.net = cv2.dnn.readNetFromONNX(onnx_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "cpu"
                print("[vision] CPU backend initialized (OpenCV DNN)", file=sys.stderr)
            else:
                print(f"[vision] No model found at {onnx_path}", file=sys.stderr)
                self.backend = "stub"
        except ImportError:
            print("[vision] OpenCV not available, using stub backend", file=sys.stderr)
            self.backend = "stub"

    def detect(self, image_path: str) -> dict:
        """Run object detection on an image. Returns JSON-serializable results."""
        start = time.time()

        if self.backend == "hailo":
            detections = self._detect_hailo(image_path)
        elif self.backend == "cpu":
            detections = self._detect_cpu(image_path)
        else:
            detections = self._detect_stub(image_path)

        elapsed = time.time() - start
        return {
            "image": image_path,
            "backend": self.backend,
            "inference_ms": round(elapsed * 1000, 1),
            "num_detections": len(detections),
            "detections": detections,
        }

    def _detect_hailo(self, image_path: str) -> list:
        """Run detection on Hailo-8 NPU."""
        import cv2
        from hailo_platform import InferVStreams

        img = cv2.imread(image_path)
        if img is None:
            return []
        h, w = img.shape[:2]
        blob = cv2.resize(img, INPUT_SIZE).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        with InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params,
        ) as pipeline:
            input_data = {self.hef.get_input_vstream_infos()[0].name: blob}
            raw = pipeline.infer(input_data)

        output_name = list(raw.keys())[0]
        return self._parse_yolo_output(raw[output_name], w, h)

    def _detect_cpu(self, image_path: str) -> list:
        """Run detection on CPU via OpenCV DNN."""
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return []
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self._parse_yolo_output(outputs[0], w, h)

    def _detect_stub(self, image_path: str) -> list:
        """Stub detector when no backend is available."""
        return [
            {
                "class": "person",
                "confidence": 0.0,
                "bbox": [0, 0, 0, 0],
                "note": "stub — install hailo_platform or opencv-python + yolov8s.onnx"
            }
        ]

    def _parse_yolo_output(self, output, orig_w, orig_h) -> list:
        """Parse YOLOv8 output tensor into detection list."""
        detections = []
        if output is None:
            return detections

        output = np.squeeze(output)
        if output.ndim == 2 and output.shape[0] == 84:
            output = output.T

        for row in output:
            if len(row) < 84:
                continue
            scores = row[4:84]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            cx, cy, bw, bh = row[0:4]
            scale_x, scale_y = orig_w / INPUT_SIZE[0], orig_h / INPUT_SIZE[1]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)

            detections.append({
                "class": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}",
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
            })

        return detections


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vision.py <image_path> [model_path]")
        print("  env HAILO_MODEL=/path/to/yolov8s.hef for Hailo")
        print("  env YOLO_ONNX=/path/to/yolov8s.onnx for CPU fallback")
        sys.exit(1)

    model = sys.argv[2] if len(sys.argv) > 2 else None
    detector = HailoDetector(model_path=model)
    result = detector.detect(sys.argv[1])
    print(json.dumps(result, indent=2))
