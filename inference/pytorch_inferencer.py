# inference/pytorch_inferencer.py

"""
PyTorch inference backend using Ultralytics YOLOv8.
This is the baseline backend.
"""

import numpy as np
import torch
from ultralytics import YOLO

from inference.base_inferencer import (
    BaseInferencer,
    InferenceConfig,
)

from loguru import logger


class PyTorchInferencer(BaseInferencer):
    """
    YOLOV8 inference using native Pytorch(.pt weights)
    """

    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self._model: torch.nn.Module | None = None

    def load(self) -> None:
        """Load .pt weights onto device and run warmup process."""

        yolo = YOLO(self.config.model_path)

        # get undelying pytorch model (nn.Module)
        self._model = yolo.model

        # eval mode and move model to correct device
        self._model.eval()
        self._model.to(self.config.device)

        self.is_loaded = True
        logger.info(
            f"[PyTorch] Raw Model Extracted and Moved to {self.config.device.upper()}"
        )
        logger.info(f"[PyTorch] Model Type: {type(self._model).__name__}")

        # warmup the model
        self.warmup()

    def warmup(self) -> None:
        """Run N dummy forward passes to reach steady state."""

        if not self.is_loaded:
            return

        logger.info(f"[PyTorch] Warming up ({self.config.warmup_runs} passes)")
        dummy = torch.zeros(
            1,
            3,
            self.config.imgsz,
            self.config.imgsz,
            dtype=torch.float32,
            device=self.config.device,
        )

        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                _ = self._model(dummy)

        if self.config.device == "cuda":
            torch.cuda.synchronize()

        logger.info("[PyTorch] Warmup Complete.")

    def infer(self, preprocessed_input):
        """
        Run YOLOv8 forward pass.
        """

        with torch.no_grad():
            raw = self._model(preprocessed_input)

        if self.config.device == "cuda":
            torch.cuda.synchronize()

        output = raw[0] if isinstance(raw, (list, tuple)) else raw

        return output.cpu().numpy()


# test
if __name__ == "__main__":
    from inference.base_inferencer import InferenceConfig

    config = InferenceConfig(
        model_path="models/yolov8n_head_detector.pt",
        imgsz=640,
        conf_thresh=0.25,
        device="cuda",
        warmup_runs=3,
    )

    inferencer = PyTorchInferencer(config)
    inferencer.load()

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = inferencer.run(dummy_frame)

    print("\nPyTorch Raw Model Test")
    print(f"  FPS            : {result.fps:.1f}")
    print(f"  Total ms       : {result.total_ms:.2f}")
    print(f"  Preprocess ms  : {result.preprocess_ms:.2f}")
    print(f"  Inference ms   : {result.inference_ms:.2f}")
    print(f"  Postprocess ms : {result.postprocess_ms:.2f}")
    print(f"  Detections     : {result.num_detections}")
    print("\n  JSON output:")
    print(result.model_dump_json(indent=2))
