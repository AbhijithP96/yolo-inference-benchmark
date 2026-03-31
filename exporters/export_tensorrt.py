# exporters/export_tensorrt.py

"""
Builds a raw TensorRT engine directly from ONNX using the TRT Python API.

Usage: uv run exporters/export_tensorrt.py --onnx models/best.onnx
"""

import argparse
from pathlib import Path
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path: str, engine_path: str, fp16: bool, workspace_gb: int = 2):

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    precision = "FP16" if fp16 else "FP32"
    print(f"\n{'─'*50}")
    print(f"  Building TensorRT engine ({precision})")
    print(f"{'─'*50}")
    print(f"  ONNX source : {onnx_path}")
    print(f"  Engine dest : {engine_path}")
    print(f"  Workspace   : {workspace_gb} GB")
    print(f"  TRT version : {trt.__version__}")
    print(f"  NOTE        : First build takes 2-10 mins")
    print(f"{'─'*50}\n")

    # builder + config
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()

    # workspace memory TRT can use during engine build
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_gb * (1 << 30),  # convert GB to bytes
    )

    # FP16 mode — uses Tensor Cores on RTX GPUs for 2x throughput
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 Tensor Cores: enabled")
    elif fp16:
        print("  FP16 not supported on this GPU — falling back to FP32")

    # parse ONNX
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX")
    with open(onnx_path, "rb") as f:
        success = parser.parse(f.read())

    if not success:
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed:\n" + "\n".join(str(e) for e in errors))

    # log input / output info
    print(f"  Network inputs  : {network.num_inputs}")
    print(f"  Network outputs : {network.num_outputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    input  [{i}]: {inp.name} {inp.shape} {inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    output [{i}]: {out.name} {out.shape} {out.dtype}")

    # build and serialize engine
    print(f"\n  Building engine — this takes a few minutes ...")
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError("Engine build failed — serialized network is None")

    # save raw engine bytes — no wrapper, no metadata
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = engine_path.stat().st_size / 1024 / 1024
    print(f"\n✅ Raw TensorRT engine saved: {engine_path} ({size_mb:.1f} MB)")

    # verify it loads back cleanly
    print(f"  Verifying engine loads ...")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("Verification failed — engine could not be deserialized")

    print(f"✅ Engine verified — {engine.num_io_tensors} I/O tensors")
    return str(engine_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="models/yolov8n_head_detector.onnx")
    parser.add_argument(
        "--engine", type=str, default="models/yolov8n_head_detector.engine"
    )
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--workspace", type=int, default=2, help="Workspace size in GB")
    args = parser.parse_args()

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=args.fp16,
        workspace_gb=args.workspace,
    )
