# inference/base_inferencer.py

"""
Abstract base class for all inference backends.
Uses Pydantic for result modeling.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import time
import torch
import cv2

import numpy as np
from pydantic import BaseModel, Field, computed_field, model_validator
from loguru import logger


# ---------------------------------------------------------------
#                           Base Models
# ---------------------------------------------------------------


class Detection(BaseModel):
    """Single Detection Result from one forward pass."""

    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] in pixel coordinates",
    )

    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int = 0

    @model_validator(mode="after")
    def validate_bbox(self) -> "Detection":
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            logger.error("Invalid bbox: x2 must > x1 and y2 must > y1")
            raise ValueError
        return self

    @computed_field
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class InferenceResult(BaseModel):
    """Full output from one forward pass including process times"""

    detections: list[Detection] = Field(default_factory=list)
    preprocess_ms: float = Field(..., ge=0.0, description="Resize + normalize time")
    inference_ms: float = Field(..., ge=0.0, description="Pure model forward pass time")
    postprocess_ms: float = Field(..., ge=0.0, description="NMS time")
    backend: str = Field(default="", description="Backend name tag")

    @computed_field
    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    @computed_field
    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0

    @computed_field
    @property
    def num_detections(self) -> int:
        return len(self.detections)


class InferenceConfig(BaseModel):
    """Config for any backend"""

    model_path: str = Field(..., description="Path to model file")
    imgsz: int = Field(640)
    conf_thresh: float = Field(0.25, ge=0.0, le=1.0)
    iou_thresh: float = Field(0.45, ge=0.0, le=1.0)
    device: str = Field("cuda", description="cuda or cpu")
    warmup_runs: int = Field(3, ge=0, description="Warmup Passes before benchmarking.")

    @model_validator(mode="after")
    def validate_model_path(self) -> "InferenceConfig":
        if not Path(self.model_path).exists():
            logger.error("Model Not Found")
            raise FileNotFoundError
        return self


# ---------------------------------------------------------------
#                         Abstract Class
# ---------------------------------------------------------------


class BaseInferencer(ABC):
    """
    All backend inherit from this class.
    Enforces a consistent preprocess -> infer -> postprocess pipeline
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.is_loaded = False

    # Abstract Methods

    @abstractmethod
    def load(self) -> None:
        """Load Model Weigths into memory"""
        ...

    def preprocess(self, frame: np.ndarray) -> tuple:
        """
        Resize, normalize, convert to tensor/array.
        Returns (preprocessed_input, scale_factor)
        """
        orig_h, orig_w = frame.shape[:2]
        target = self.config.imgsz

        # compute scale
        scale = min(target / orig_w, target / orig_h)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # pad to reahc target size
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2

        padded = cv2.copyMakeBorder(
            resized,
            pad_y,
            target - new_h - pad_y,
            pad_x,
            target - new_w - pad_x,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # bgr to rgb
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # normalized tensor
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        # add batch and move to correct device
        tensor = tensor.unsqueeze(0).to(self.config.device)

        return tensor, (scale, pad_x, pad_y)

    @abstractmethod
    def infer(self, preprocessed_input) -> np.ndarray:
        """
        Run the forward pass.
        Returns raw model output as numpy array.
        """
        ...

    def postprocess(
        self, raw_output: np.ndarray, scale_factor: tuple
    ) -> list[Detection]:
        """
        Decode raw output -> Detection list.
        Applies NMS.
        """
        scale, pad_x, pad_y = scale_factor

        # squeeze batch dim
        pred = raw_output[0].T

        # filter
        scores = pred[:, 4]
        mask = scores >= self.config.conf_thresh
        pred = pred[mask]
        scores = scores[mask]

        if len(pred) == 0:
            return []

        # cxcywh -> xyxy
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = self._nms(boxes, scores, self.config.iou_thresh)
        boxes = boxes[keep]
        scores = scores[keep]

        # remap to original frame coords
        detections = []
        for i, box in enumerate(boxes):
            bx1 = (box[0] - pad_x) / scale
            by1 = (box[1] - pad_y) / scale
            bx2 = (box[2] - pad_x) / scale
            by2 = (box[3] - pad_y) / scale

            if bx2 <= bx1 or by2 <= by1:
                continue

            detections.append(
                Detection(
                    bbox=list(map(float, [bx1, by1, bx2, by2])),
                    confidence=float(scores[i]),
                    class_id=0,
                )
            )

        return detections

    @abstractmethod
    def warmup(self) -> None:
        """
        Run config.warmup_runs dummy forward passes.
        Always called inside load(); never benchmark a cold model.
        """
        ...

    # nms
    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
        """
        Pure numpy Non-Maximum Suppression.
        Removes duplicates detections of the same object.
        """

        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[1:][iou <= iou_thresh]

        return keep

    # identical methods for all backends
    def run(self, frame: np.ndarray) -> InferenceResult:
        """
        Pipeline: preprocess -> infer -> postprocess.
        Each stage is timed independently.
        """
        if not self.is_loaded:
            logger.error(f"{self.__class__.__name__} not loaded")
            raise RuntimeError

        # preprocess
        t0 = time.perf_counter()
        preprocessed, scale_factor = self.preprocess(frame)
        t1 = time.perf_counter()

        # infer
        raw_output = self.infer(preprocessed)
        t2 = time.perf_counter()

        # postprocess
        detections = self.postprocess(raw_output, scale_factor)
        t3 = time.perf_counter()

        return InferenceResult(
            detections=detections,
            preprocess_ms=(t1 - t0) * 1000,
            inference_ms=(t2 - t1) * 1000,
            postprocess_ms=(t3 - t2) * 1000,
            backend=self.__class__.__name__,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={Path(self.config.model_path).name}, "
            f"imgsz={self.config.imgsz}, "
            f"conf={self.config.conf_thresh}, "
            f"loaded={self.is_loaded})"
        )
