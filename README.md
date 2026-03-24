# hailo-vision

Hailo-8 accelerated object detection pipeline for BlackRoad agents. Uses YOLOv8s with automatic CPU fallback.

## Hardware

- 2x Hailo-8 NPUs across the fleet (52 TOPS combined)
- Cecilia (192.168.4.96) — Hailo-8 at /dev/hailo0
- Octavia (192.168.4.101) — Hailo-8 at /dev/hailo0

## Usage

```python
from vision import HailoDetector

detector = HailoDetector()
result = detector.detect("photo.jpg")
print(result)
# {
#   "image": "photo.jpg",
#   "backend": "hailo",
#   "inference_ms": 12.3,
#   "num_detections": 3,
#   "detections": [
#     {"class": "person", "confidence": 0.92, "bbox": [100, 50, 300, 400]},
#     ...
#   ]
# }
```

## CLI

```bash
python vision.py <image_path> [model_path]

# Environment variables:
HAILO_MODEL=/usr/share/hailo-models/yolov8s.hef  # Hailo model
YOLO_ONNX=yolov8s.onnx                           # CPU fallback model
```

## Backends

1. **Hailo-8** — uses hailort Python bindings, requires HEF model file
2. **CPU** — uses OpenCV DNN with ONNX model, works anywhere
3. **Stub** — returns empty detections when no backend available

## Part of BlackRoad-Agents

Remember the Road. Pave Tomorrow.

BlackRoad OS, Inc. — Incorporated 2025.
