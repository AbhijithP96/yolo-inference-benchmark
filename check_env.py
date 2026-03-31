"""
check_env.py
Run this before starting the project to verify your full stack.
Usage: python check_env.py
"""

import sys
import importlib
import subprocess

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "


def section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def check(label: str, fn):
    try:
        result = fn()
        print(f"{PASS} {label}: {result}")
        return True
    except Exception as e:
        print(f"{FAIL} {label}: {e}")
        return False


# ── 1. Python ─────────────────────────────────────────────────────────────────

section("1. Python")
check("Version (need 3.10+)", lambda: sys.version.split()[0])


# ── 2. PyTorch + CUDA ─────────────────────────────────────────────────────────

section("2. PyTorch + CUDA")
try:
    import torch

    check("PyTorch version", lambda: torch.__version__)
    check("CUDA available", lambda: str(torch.cuda.is_available()))
    check("CUDA version", lambda: torch.version.cuda)
    check("cuDNN version", lambda: str(torch.backends.cudnn.version()))
    check("GPU name", lambda: torch.cuda.get_device_name(0))
    check(
        "GPU memory (GB)",
        lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}",
    )
    check("GPU compute capability", lambda: str(torch.cuda.get_device_capability(0)))
    check("CUDA tensor op", lambda: str(torch.tensor([1.0]).cuda().sum().item()))
except ImportError as e:
    print(f"{FAIL} PyTorch not installed: {e}")


# ── 3. Ultralytics ────────────────────────────────────────────────────────────

section("3. Ultralytics")
try:
    import ultralytics
    from ultralytics import YOLO

    check("Ultralytics version", lambda: ultralytics.__version__)
    check("YOLO class import", lambda: str(YOLO))
except ImportError as e:
    print(f"{FAIL} Ultralytics not installed: {e}")


# ── 4. ONNX + ONNX Runtime ────────────────────────────────────────────────────

section("4. ONNX + ONNX Runtime GPU")
try:
    import onnx

    check("ONNX version", lambda: onnx.__version__)
except ImportError as e:
    print(f"{FAIL} onnx not installed: {e}")

try:
    import torch  # preload torch DLLs before ort
    import onnxruntime as ort

    check("ONNXRuntime version", lambda: ort.__version__)
    check("Available providers", lambda: ort.get_available_providers())

    # Confirm CUDAExecutionProvider is present
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        print(f"{PASS} CUDAExecutionProvider is available")
    else:
        print(f"{WARN} CUDAExecutionProvider NOT found — ORT will run on CPU only")

except ImportError as e:
    print(f"{FAIL} onnxruntime-gpu not installed: {e}")


# ── 5. TensorRT ───────────────────────────────────────────────────────────────

section("5. TensorRT")
try:
    import tensorrt as trt

    check("TensorRT version", lambda: trt.__version__)
    check("TRT Logger init", lambda: str(trt.Logger(trt.Logger.WARNING)))

    # Check CUDA + TRT can see each other
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    check("TRT Builder init", lambda: str(builder))
except ImportError as e:
    print(f"{FAIL} TensorRT not installed: {e}")
except Exception as e:
    print(f"{WARN} TensorRT installed but error during init: {e}")


# ── 6. PyCUDA ─────────────────────────────────────────────────────────────────

section("6. PyCUDA")
try:
    import pycuda.driver as cuda
    import pycuda.autoinit

    check("PyCUDA device name", lambda: cuda.Device(0).name())
    check("PyCUDA compute capability", lambda: str(cuda.Device(0).compute_capability()))
    check(
        "PyCUDA total memory (GB)", lambda: f"{cuda.Device(0).total_memory() / 1e9:.1f}"
    )
except ImportError as e:
    print(f"{FAIL} pycuda not installed: {e}")
except Exception as e:
    print(f"{WARN} pycuda error: {e}")


# ── 7. Profiling Tools ────────────────────────────────────────────────────────

section("7. Profiling Tools")
try:
    import psutil

    check("psutil version", lambda: psutil.__version__)
    check("CPU count", lambda: str(psutil.cpu_count()))
    check("RAM total (GB)", lambda: f"{psutil.virtual_memory().total / 1e9:.1f}")
except ImportError as e:
    print(f"{FAIL} psutil not installed: {e}")

try:
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    check("pynvml version", lambda: pynvml.nvmlSystemGetNVMLVersion())
    check("pynvml GPU name", lambda: pynvml.nvmlDeviceGetName(handle))
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    check(
        "pynvml GPU mem (GB)",
        lambda: f"total={mem.total/1e9:.1f} free={mem.free/1e9:.1f}",
    )
except ImportError as e:
    print(f"{FAIL} pynvml not installed: {e}")
except Exception as e:
    print(f"{WARN} pynvml error: {e}")


# ── 8. OpenCV ─────────────────────────────────────────────────────────────────

section("8. OpenCV")
try:
    import cv2

    check("OpenCV version", lambda: cv2.__version__)
    check(
        "OpenCV build info",
        lambda: (
            "CUDA" if "CUDA" in cv2.getBuildInformation() else "No CUDA (OK for now)"
        ),
    )

    # Quick webcam check — non-fatal
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print(f"{PASS} Webcam (index 0): accessible")
        cap.release()
    else:
        print(
            f"{WARN} Webcam (index 0): not accessible — connect one before running live_demo.py"
        )
except ImportError as e:
    print(f"{FAIL} opencv-python not installed: {e}")


# ── 9. Dashboard / Reporting ──────────────────────────────────────────────────

section("9. Dashboard & Reporting")
for pkg in ["matplotlib", "pandas", "rich", "tabulate", "seaborn", "numpy"]:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "ok")
        print(f"{PASS} {pkg}: {ver}")
    except ImportError:
        print(f"{FAIL} {pkg}: not installed")


# ── 10. Summary ───────────────────────────────────────────────────────────────

section("Summary")
print(
    """
  Stack targets:
    Python        ≥ 3.12
    PyTorch       2.11 + cu128
    Ultralytics   ≥ 8.4.0
    ONNX Runtime  ≥ 1.19  (CUDA 12.x default)
    TensorRT      ≥ 10.x  (cu12 variant)
    PyCUDA        latest

  If you see any ❌ above, re-run the corresponding install step.
  If you see ⚠️  for CUDAExecutionProvider, torch may not have been
  imported before onnxruntime — check your install order.
"""
)
