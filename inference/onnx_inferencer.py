# inference/onnx_inferencer.py


"""
ONNX Runtime inference backend.
"""

import numpy as np
import torch
import onnxruntime as ort

from inference.base_inferencer import (
    BaseInferencer,
    InferenceConfig,
)

from loguru import logger


class ONNXInferencer(BaseInferencer):
    """
    YOLOv8 inference using ONNX RUntime with CUDA.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._output_name: str = ""

    def load(self) -> None:
        """
        Create an ONNX Runtime with inference session with CUDA.
        Falls back to CPU if CUDA is not available.
        """

        logger.info(f"[ONNX] Loading {self.config.model_path} ")

        # provider priority
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.config.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        # session options
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3

        self._session = ort.InferenceSession(
            self.config.model_path, sess_options=opts, providers=providers
        )

        # cache names
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        active = self._session.get_providers()[0]

        self.is_loaded = True

        logger.info(f"[ONNX] Session Created with provider: {active}")

        self.warmup()

    def warmup(self) -> None:
        """Run dummy passes to warm up the CUDA Provider"""

        if not self.is_loaded:
            return

        logger.info(f"[ONNX] Warming Up ({self.config.warmup_runs}) passes")

        dummy = np.zeros((1, 3, self.config.imgsz, self.config.imgsz), dtype=np.float32)

        for _ in range(self.config.warmup_runs):
            self._session.run([self._output_name], {self._input_name: dummy})

        logger.info("[ONNX] Warmup Complete.")

    def infer(self, preprocessed_input) -> np.ndarray:
        """
        Run ONNX Runtime Forward Pass
        """

        if isinstance(preprocessed_input, torch.Tensor):
            preprocessed_input = preprocessed_input.cpu().numpy()

        outputs = self._session.run(
            [self._output_name], {self._input_name: preprocessed_input}
        )

        return outputs[0]


# test
if __name__ == "__main__":
    from inference.base_inferencer import InferenceConfig

    config = InferenceConfig(
        model_path="models/yolov8n_head_detector.onnx",
        imgsz=640,
        conf_thresh=0.25,
        device="cuda",
        warmup_runs=3,
    )

    inferencer = ONNXInferencer(config)
    inferencer.load()

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = inferencer.run(dummy_frame)

    print("\n ONNX Test")
    print(f"  FPS            : {result.fps:.1f}")
    print(f"  Total ms       : {result.total_ms:.2f}")
    print(f"  Preprocess ms  : {result.preprocess_ms:.2f}")
    print(f"  Inference ms   : {result.inference_ms:.2f}")
    print(f"  Postprocess ms : {result.postprocess_ms:.2f}")
    print(f"  Detections     : {result.num_detections}")
    print("\n  JSON output:")
    print(result.model_dump_json(indent=2))
