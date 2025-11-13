import cv2
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


class MotionDetector:
    def __init__(self, video_source=0):
        if os.name == "nt":
            cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
            self.cap = cap if cap.isOpened() else cv2.VideoCapture(video_source)
        else:
            self.cap = cv2.VideoCapture(video_source)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40)
        self.previous_centroids = []
        self.frame_counter = 0
        self.alert_active_until = 0.0

    def _process(self, frame):
        self.frame_counter += 1
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        human_like = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150 or area > 7000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w + 0.001)
            if 1.2 < aspect_ratio < 3.5:
                cx, cy = x + w // 2, y + h // 2
                self.previous_centroids.append((cx, cy))
                if len(self.previous_centroids) > 10:
                    self.previous_centroids.pop(0)
                if len(self.previous_centroids) >= 2:
                    dx = self.previous_centroids[-1][0] - self.previous_centroids[0][0]
                    dy = self.previous_centroids[-1][1] - self.previous_centroids[0][1]
                    distance = np.sqrt(dx**2 + dy**2)
                    avg_speed = distance / len(self.previous_centroids)
                else:
                    avg_speed = 0
                if 2 < avg_speed < 50:
                    human_like = True
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Human~like Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if human_like:
            self.alert_active_until = time.time() + 2.0
        if human_like:
            cv2.putText(frame, "ALERT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return frame, fgmask, human_like

    def next_frame(self):
        if not self.cap or not self.cap.isOpened():
            return self._placeholder_jpeg(), None, False
        ret, frame = self.cap.read()
        if not ret:
            return self._placeholder_jpeg(), None, False
        processed, _, human_like = self._process(frame)
        ok, buf = cv2.imencode('.jpg', processed)
        if not ok:
            return self._placeholder_jpeg(), None, False
        return buf.tobytes(), processed, human_like

    def release(self):
        self.cap.release()

    def _placeholder_jpeg(self):
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, "NO VIDEO SOURCE", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        ok, buf = cv2.imencode('.jpg', img)
        return buf.tobytes() if ok else None

    def is_alert_active(self):
        return time.time() < self.alert_active_until


# ------------------------------
# Deep Learning Human Detector
# ------------------------------

@dataclass
class DetectionConfig:
    """Configuration for human detection using a deep learning model.

    Attributes:
        model_path: Path to YOLOv8 weights (e.g., 'yolov8n.pt').
        confidence_threshold: Minimum score to keep detections.
        iou_threshold: IoU threshold used by NMS in the model.
        min_box_area: Minimum bbox area in pixels to keep.
        aspect_ratio_range: Allowed (w/h) ratio range for person bboxes.
        frame_resize_width: Optional width to resize before inference.
        frame_resize_height: Optional height to resize before inference.
        device: 'cpu' or a CUDA device string like 'cuda:0'.
        visualize: Whether to draw detections on output frame.
    """

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    min_box_area: int = 200
    aspect_ratio_range: Tuple[float, float] = (0.3, 0.8)
    frame_resize_width: Optional[int] = None
    frame_resize_height: Optional[int] = None
    device: str = "cpu"
    visualize: bool = True


def load_detector(config: DetectionConfig):
    """Load YOLOv8 detector based on the provided configuration.

    Lazily imports Ultralytics to avoid import errors for users not using DL.
    Returns the loaded model or raises a RuntimeError with a helpful message.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ultralytics YOLO is not available. Install with: pip install ultralytics"
        ) from e

    try:
        model = YOLO(config.model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{config.model_path}': {e}") from e

    return model


def _preprocess_frame(frame: np.ndarray, config: DetectionConfig) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Preprocess frame (resize if requested). Returns processed frame and scale factors.

    If resizing is applied, returns (fx, fy) scale to map boxes back to original size.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Invalid frame provided to detector")

    h, w = frame.shape[:2]
    target_w = config.frame_resize_width
    target_h = config.frame_resize_height
    if target_w and target_h and target_w > 0 and target_h > 0:
        processed = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        fx = w / float(target_w)
        fy = h / float(target_h)
        return processed, (fx, fy)
    return frame, (1.0, 1.0)


def run_detection(frame: np.ndarray, model: Any, config: DetectionConfig) -> List[Dict[str, Any]]:
    """Run human detection on a single BGR frame.

    Returns list of detections: {"bbox": (x1,y1,x2,y2), "confidence": float, "class": "person"}
    Only the person class is kept.
    """
    processed, scale = _preprocess_frame(frame, config)

    try:
        results = model.predict(
            processed,
            conf=config.confidence_threshold,
            iou=config.iou_threshold,
            classes=[0],  # person class
            device=config.device,
            imgsz=(config.frame_resize_height or processed.shape[0], config.frame_resize_width or processed.shape[1]),
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}") from e

    detections: List[Dict[str, Any]] = []
    if not results:
        return detections

    res = results[0]
    if not hasattr(res, "boxes") or res.boxes is None:
        return detections

    fx, fy = scale
    for b in res.boxes:
        try:
            xyxy = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
        except Exception:
            continue
        if cls_id != 0:
            continue
        x1, y1, x2, y2 = xyxy
        x1, y1, x2, y2 = x1 * fx, y1 * fy, x2 * fx, y2 * fy
        detections.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": conf,
            "class": "person",
        })

    return detections


def filter_detections(detections: List[Dict[str, Any]], config: DetectionConfig) -> List[Dict[str, Any]]:
    """Apply additional filters: bbox area and aspect-ratio (w/h) range."""
    filtered: List[Dict[str, Any]] = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        if area < config.min_box_area:
            continue
        ratio = w / float(h + 1e-6)
        rmin, rmax = config.aspect_ratio_range
        if not (rmin <= ratio <= rmax):
            continue
        filtered.append(d)
    return filtered


def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]):
    """Draw bounding boxes and confidence labels on the frame."""
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        conf = d.get("confidence", 0.0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {conf:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Human detection with YOLOv8")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--min_area", type=int, default=200)
    parser.add_argument("--ratio_min", type=float, default=0.3)
    parser.add_argument("--ratio_max", type=float, default=0.8)
    parser.add_argument("--resize_w", type=int, default=0)
    parser.add_argument("--resize_h", type=int, default=0)
    args = parser.parse_args()

    cfg = DetectionConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        min_box_area=args.min_area,
        aspect_ratio_range=(args.ratio_min, args.ratio_max),
        frame_resize_width=(args.resize_w or None),
        frame_resize_height=(args.resize_h or None),
        device=args.device,
        visualize=True,
    )

    try:
        model = load_detector(cfg)
    except Exception as e:
        print("Detector load error:", e)
        raise SystemExit(1)

    src = args.source
    cap: Optional[cv2.VideoCapture]
    if src.isdigit():
        index = int(src)
        if os.name == "nt":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(src)

    if not cap or not cap.isOpened():
        print("Failed to open source:", src)
        raise SystemExit(1)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        try:
            dets = run_detection(frame, model, cfg)
            dets = filter_detections(dets, cfg)
        except Exception as e:
            print("Inference error:", e)
            break
        if cfg.visualize:
            draw_detections(frame, dets)
        cv2.imshow("Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()