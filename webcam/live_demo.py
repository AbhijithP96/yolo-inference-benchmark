# webcam/live_demo.py

"""
Live Webcam demo: switch backends in real time.
Shows bounding boxes, FPS, latency, and GPU usage on screen.
"""

import argparse
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pynvml

from inference.base_inferencer import InferenceConfig
from inference.pytorch_inferencer import PyTorchInferencer
from inference.onnx_inferencer import ONNXInferencer
from inference.tensorrt_inferencer import TensorRTInferencer

# Constants

BACKENDS = {
    "1": "PyTorch",
    "2": "ONNX",
    "3": "TensorRT",
}

MODEL_PATHS = {
    "PyTorch": "models/yolov8n_head_detector.pt",
    "ONNX": "models/yolov8n_head_detector.onnx",
    "TensorRT": "models/yolov8n_head_detector.engine",
}

INFERENCER_MAP = {
    "PyTorch": PyTorchInferencer,
    "ONNX": ONNXInferencer,
    "TensorRT": TensorRTInferencer,
}

# Color Map (BGR)
COLORS = {
    "PyTorch": (255, 255, 0),  # cyan
    "ONNX": (255, 0, 220),  # purple
    "TensorRT": (0, 170, 255),  # orange
    "box": (0, 255, 0),  # green
    "text_bg": (0, 0, 0),  # black
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}

SCREENSHOT_DIR = Path("results/screenshots")

# GPU Monitor


class GPUMonitor:

    def __init__(self, gpu_index: int = 0):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_util = 0.0
        self.gpu_mem_mb = 0.0
        self.gpu_mem_total = (
            pynvml.nvmlDeviceGetMemoryInfo(self._handle).total / 1024 / 1024
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.gpu_util = float(util.gpu)
                self.gpu_mem_mb = mem.used / 1024 / 1024
            except Exception:
                pass
            time.sleep(0.2)

    def stop(self):
        self._stop.set()


# Drawing


def draw_metric(
    frame: np.ndarray,
    backend_name: str,
    fps: float,
    latency_ms: float,
    num_dets: int,
    fps_history: deque,
    gpu_util: float,
    gpu_mem_mb: float,
    gpu_mem_total: float,
) -> np.ndarray:
    """Draw the frame with all informations"""

    h, w = frame.shape[:2]
    color = COLORS[backend_name]

    # top-left
    panel_lines = [
        (f"Backend : {backend_name}", color),
        (f"FPS     : {fps:.1f}", COLORS["yellow"]),
        (f"Latency : {latency_ms:.1f} ms", COLORS["yellow"]),
        (f"Objects : {num_dets}", COLORS["white"]),
    ]

    panel_x, panel_y = 12, 12
    line_h = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    panel_w = 250
    panel_h = len(panel_lines) * line_h + 12
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x - 6, panel_y - 6),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (text, col) in enumerate(panel_lines):
        y = panel_y + i * line_h + line_h
        cv2.putText(frame, text, (panel_x, y), font, font_scale, col, thickness)

    # top right
    gpu_lines = [
        (f"GPU Util : {gpu_util:.0f}%", COLORS["white"]),
        (f"GPU Mem  : {gpu_mem_mb:.0f}/{gpu_mem_total:.0f} MB", COLORS["white"]),
    ]

    gx = w - 250
    gy = 12
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (gx - 6, gy - 6),
        (gx + 244, gy + len(gpu_lines) * line_h + 6),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0.0, frame)

    for i, (text, col) in enumerate(gpu_lines):
        y = gy + i * line_h + line_h
        cv2.putText(frame, text, (gx, y), font, font_scale - 0.12, col, thickness)

    # bottom left
    if len(fps_history) > 1:
        graph_w, graph_h = 200, 50
        gx2, gy2 = 12, h - graph_h - 40
        max_fps = max(max(fps_history), 1)

        overlay3 = frame.copy()
        cv2.rectangle(
            overlay3,
            (gx2 - 4, gy2 - 4),
            (gx2 + graph_w + 4, gy2 + graph_h + 4),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay3, 0.6, frame, 0.4, 0, frame)

        pts = []
        for i, f in enumerate(fps_history):
            x = gx2 + int(i * graph_w / len(fps_history))
            y = gy2 + graph_h - int(f / max_fps * graph_h)
            pts.append((x, y))

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)

        cv2.putText(
            frame,
            f"FPS history (last {len(fps_history)}f)",
            (gx2, gy2 + graph_h + 20),
            font,
            0.45,
            COLORS["white"],
            1,
        )

    # bottom right
    controls = ["[1] PyTorch  [2] ONNX  [3] TensorRT  [S] Save  [Q] Quit"]
    for i, text in enumerate(controls):
        x = w - 400 + i * 20
        cv2.putText(frame, text, (x, h - 12), font, 0.42, COLORS["white"], 1)

    # frame background color
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

    return frame


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and confidence scores."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        conf = det.confidence

        # box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["box"], 2)

        # label background
        label = f"{conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), COLORS["text_bg"], -1
        )

        # label text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLORS["white"],
            1,
        )

    return frame


def main(args):
    # get the logger
    from loguru import logger

    logger.info("Loading all backends. This will take some time...")
    inferencer = {}

    try:
        for name, infer_cls in INFERENCER_MAP.items():
            config = InferenceConfig(
                model_path=MODEL_PATHS[name],
                imgsz=640,
                conf_thresh=args.conf,
                device="cuda",
                warmup_runs=5,
            )

            inferer = infer_cls(config)
            inferer.load()
            inferencer[name] = inferer

        logger.info("All Backends Loaded.")
    except Exception as e:
        logger.error("Error Occured while loading backends.")
        raise RuntimeError(str(e))

    # GPU monitor
    gpu_monitor = GPUMonitor(gpu_index=args.gpu)
    logger.info("GPU Montior Initialized.")

    # check source and set cap size
    source = 0 if args.source == "cam" else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot find source: {args.source}")
        raise RuntimeError

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # main running loop
    current_backend = args.backend
    fps_history = deque(maxlen=60)
    frame_count = 0
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Active Backend: {current_backend}")

    while True:
        ret, frame = cap.read()

        # loop the video
        if not ret:
            if args.source != "cam":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        inferer = inferencer[current_backend]
        result = inferer.run(frame)

        fps_history.append(result.fps)
        frame_count += 1

        # draw results on frame
        frame = draw_detections(frame, result.detections)
        frame = draw_metric(
            frame=frame,
            backend_name=current_backend,
            fps=result.fps,
            latency_ms=result.total_ms,
            num_dets=len(result.detections),
            fps_history=fps_history,
            gpu_util=gpu_monitor.gpu_util,
            gpu_mem_mb=gpu_monitor.gpu_mem_mb,
            gpu_mem_total=gpu_monitor.gpu_mem_total,
        )

        cv2.imshow("YOLO Inference", frame)

        c = cv2.waitKey(1) & 0xFF

        if c == ord("q"):
            logger.info("Quitting Live Demo")
            break

        if c == ord("s"):
            path = SCREENSHOT_DIR / f"{current_backend}_{frame_count:.0f}.png"
            cv2.imwrite(str(path), frame)
            logger.info(f"Screenshot saved: {str(path)}")

        if chr(c) in BACKENDS:
            new_backend = BACKENDS[chr(c)]
            if new_backend != current_backend:
                current_backend = new_backend
                fps_history.clear()
                logger.info(f"Backend Switched to {current_backend}")

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    gpu_monitor.stop()
    logger.info("Demo Ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Live Inference Demo")
    parser.add_argument(
        "--source",
        type=str,
        default="cam",
        help="'cam' for live feed or path to video file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="PyTorch",
        choices=["PyTorch", "ONNX", "TensorRT"],
        help="Starting backend",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    main(args)
