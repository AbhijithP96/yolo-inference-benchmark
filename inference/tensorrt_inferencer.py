# inference/tensorrt_inferencer.py

"""
TensorRT inference backend.
Loads a pre-built .engine file and runs inference using TensorRT + PyCUDA.
"""

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch

from inference.base_inferencer import BaseInferencer, InferenceConfig

from loguru import logger

# suppress TensorRT info logs — only show warnings and errors
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTInferencer(BaseInferencer):
    """
    YOLOv8 inference using a pre-built TensorRT engine file.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__(config)

        self._engine: trt.ICudaEngine | None = None
        self._context: trt.IExecutionContext | None = None

        self._d_input: cuda.DeviceAllocation | None = None
        self._d_output: cuda.DeviceAllocation | None = None
        self._h_output: np.ndarray | None = None

        self._stream: cuda.Stream | None = None

        self._input_shape: tuple = ()
        self._output_shape: tuple = ()
        self._input_dtype = None
        self._output_dtype = None

    def load(self):
        """
        Deserialize the .engine file and allocate GPU memory buffers.
        """

        logger.info(f"[TensorRT] Loading {self.config.model_path}")

        # deserialize engine
        runtime = trt.Runtime(TRT_LOGGER)

        with open(self.config.model_path, "rb") as f:
            engine_data = f.read()

        self._engine = runtime.deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError(
                f"[TensorRT] Failed to deserialize engine: {self.config.model_path}\n"
                "The engine may have been built with a different TensorRT version. "
                "Rebuild the engine with the current TRT version."
            )
        self._context = self._engine.create_execution_context()

        logger.info("[TensorRT] Engine deserialized")

        # tensor shapes from the engine
        input_name = self._engine.get_tensor_name(0)
        output_name = self._engine.get_tensor_name(1)

        self._input_shape = tuple(self._engine.get_tensor_shape(input_name))
        self._output_shape = tuple(self._engine.get_tensor_shape(output_name))

        self._input_dtype = trt.nptype(self._engine.get_tensor_dtype(input_name))
        self._output_dtype = trt.nptype(self._engine.get_tensor_dtype(output_name))
        _input_dtype = self._input_dtype
        _output_dtype = self._output_dtype

        logger.info(f"[TensorRT] Input  tensor : {input_name} {self._input_shape}")
        logger.info(f"[TensorRT] Output tensor : {output_name} {self._output_shape}")

        # allocartion
        input_bytes = int(np.prod(self._input_shape) * np.dtype(_input_dtype).itemsize)
        output_bytes = int(
            np.prod(self._output_shape) * np.dtype(_output_dtype).itemsize
        )

        self._d_input = cuda.mem_alloc(input_bytes)
        self._d_output = cuda.mem_alloc(output_bytes)

        self._h_output = np.empty(self._output_shape, dtype=self._output_dtype)

        self._context.set_tensor_address(input_name, int(self._d_input))
        self._context.set_tensor_address(output_name, int(self._d_output))

        self._stream = cuda.Stream()

        self.is_loaded = True

        logger.info(
            f"[TensorRT] Input  buffer : {input_bytes  / 1024:.1f} KB  dtype={_input_dtype}"
        )
        logger.info(
            f"[TensorRT] Output buffer : {output_bytes / 1024:.1f} KB  dtype={_output_dtype}"
        )

        self.warmup()

    def warmup(self) -> None:
        if not self.is_loaded:
            return

        logger.info(f"[TensorRT] Warming up ({self.config.warmup_runs} passes)")

        dummy = np.zeros(self._input_shape, dtype=self._input_dtype)  # ← correct dtype

        for _ in range(self.config.warmup_runs):
            self.infer(dummy)

        logger.info("[TensorRT] Warmup complete")

    def infer(self, preprocessed_input):
        """
        Run TensorRT forward pass via PyCUDA.
        """
        if isinstance(preprocessed_input, torch.Tensor):
            preprocessed_input = preprocessed_input.cpu().numpy()

        input_array = np.ascontiguousarray(preprocessed_input, dtype=np.float32)

        # 1. H2D: host (CPU) to device (GPU)
        cuda.memcpy_htod_async(self._d_input, input_array, self._stream)

        # 2. RUN: execute engine on stream
        self._context.execute_async_v3(stream_handle=self._stream.handle)

        # 3. D2H: device (GPU) to host (CPU)
        cuda.memcpy_dtoh_async(self._h_output, self._d_output, self._stream)

        # 4. SYNC: block CPU until all stream operations are complete
        self._stream.synchronize()

        return self._h_output.copy()

    def __del__(self):
        """Free GPU memory when object is destroyed."""
        try:
            if self._d_input:
                self._d_input.free()
            if self._d_output:
                self._d_output.free()
        except Exception:
            pass


if __name__ == "__main__":
    from inference.base_inferencer import InferenceConfig

    config = InferenceConfig(
        model_path="models/yolov8n_head_detector.engine",
        imgsz=640,
        conf_thresh=0.25,
        device="cuda",
        warmup_runs=3,
    )

    inferencer = TensorRTInferencer(config)
    inferencer.load()

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = inferencer.run(dummy_frame)

    print("TensorRT Test")
    print(f"  FPS            : {result.fps:.1f}")
    print(f"  Total ms       : {result.total_ms:.2f}")
    print(f"  Preprocess ms  : {result.preprocess_ms:.2f}")
    print(f"  Inference ms   : {result.inference_ms:.2f}")
    print(f"  Postprocess ms : {result.postprocess_ms:.2f}")
    print(f"  Detections     : {result.num_detections}")
    print("JSON output:")
    print(result.model_dump_json(indent=2))
