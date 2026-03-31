# benchmark/benchmark_runner.py

"""
Runs all three backends against the same frames and collects ProfilerStats.
Saves results to results/benchmark_report.json and prints comparison table.

Usage:
    uv run benchmark/benchmark_runner.py --frames 200 --source webcam
    uv run benchmark/benchmark_runner.py --frames 200 --source video.mp4
    uv run benchmark/benchmark_runner.py --frames 200 --source image.jpg
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from benchmark.profiler import Profiler
from inference.base_inferencer import InferenceConfig
from inference.pytorch_inferencer import PyTorchInferencer
from inference.onnx_inferencer import ONNXInferencer
from inference.tensorrt_inferencer import TensorRTInferencer


# Config

MODELS = {
    "PyTorch": "models/yolov8n_head_detector.pt",
    "ONNX": "models/yolov8n_head_detector.onnx",
    "TensorRT": "models/yolov8n_head_detector.engine",
}

INFERENCER_MAP = {
    "PyTorch": PyTorchInferencer,
    "ONNX": ONNXInferencer,
    "TensorRT": TensorRTInferencer,
}


# Frame source


def get_frames(source: str, n: int) -> list[np.ndarray]:
    """
    Collect N frames from source.
    Source can be: 'webcam', a video file path, or an image path.
    All backends run on the exact same frames -> fair comparison.
    """
    frames = []
    console = Console()

    if source == "webcam":
        console.print(f"  [cyan]Capturing {n} frames from webcam...[/cyan]")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open webcam — connect one or use --source video.mp4"
            )
        while len(frames) < n:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    elif Path(source).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        console.print(f"  [cyan]Capturing {n} frames from {source}...[/cyan]")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        while len(frames) < n:
            ret, frame = cap.read()
            if not ret:
                # loop video if it runs out of frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            frames.append(frame)
        cap.release()

    elif Path(source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        console.print(f"  [cyan]Using image {source} × {n} frames...[/cyan]")
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {source}")
        frames = [frame.copy() for _ in range(n)]

    else:
        raise ValueError(
            f"Unknown source: {source}. Use 'webcam', a video, or an image path."
        )

    console.print(f"  [green] {len(frames)} frames ready[/green]\n")
    return frames


# Single backend benchmark


def run_backend(
    backend_name: str,
    inferencer_cls,
    model_path: str,
    frames: list[np.ndarray],
    imgsz: int,
    conf_thresh: float,
) -> dict:
    """
    Run one backend against all frames.
    Returns ProfilerStats as a dict for JSON serialization.
    """
    console = Console()
    console.rule(f"[bold cyan]{backend_name}")

    config = InferenceConfig(
        model_path=model_path,
        imgsz=imgsz,
        conf_thresh=conf_thresh,
        device="cuda",
        warmup_runs=5,  # more warmup for accurate benchmarking
    )

    inferencer = inferencer_cls(config)
    profiler = Profiler(gpu_index=0, sample_interval_ms=100)

    # load + warmup
    inferencer.load()

    # benchmark loop
    console.print(f"  [yellow]Running {len(frames)} frames...[/yellow]")
    profiler.start(backend_name)

    for i, frame in enumerate(frames):
        result = inferencer.run(frame)
        profiler.record_frame(result)

        # progress every 50 frames
        if (i + 1) % 50 == 0:
            stats = profiler.get_stats()
            console.print(
                f"  frame {i+1:>4}/{len(frames)} | "
                f"fps={stats.fps_mean:.1f} | "
                f"lat={stats.latency_mean_ms:.1f}ms | "
                f"gpu={stats.gpu_util_mean:.0f}%"
            )

    profiler.stop()
    profiler.print_summary()

    # save individual backend result
    out_path = Path("results") / f"{backend_name.lower()}_stats.json"
    profiler.save(str(out_path))

    return profiler.get_stats().model_dump()


# Comparison table


def print_comparison(all_stats: dict[str, dict]) -> None:
    """Print side-by-side comparison of all backends using rich."""
    console = Console()
    console.rule("[bold white]BENCHMARK COMPARISON")

    table = Table(show_lines=True, title="All Backends — Side by Side")
    table.add_column("Metric", style="cyan", no_wrap=True)

    # add one column per backend
    colors = {"PyTorch": "green", "ONNX": "yellow", "TensorRT": "magenta"}
    for name in all_stats:
        table.add_column(name, style=colors.get(name, "white"))

    def row(label, key, fmt=".1f", suffix=""):
        vals = []
        for stats in all_stats.values():
            v = stats.get(key, 0)
            vals.append(f"{v:{fmt}}{suffix}")
        table.add_row(label, *vals)

    def divider(label):
        table.add_row(f"[bold]── {label}", *[""] * len(all_stats))

    divider("Throughput")
    row("FPS (mean)", "fps_mean", ".1f")
    row("FPS (max)", "fps_max", ".1f")
    divider("Latency")
    row("Latency mean  (ms)", "latency_mean_ms", ".2f")
    row("Latency P50   (ms)", "latency_p50_ms", ".2f")
    row("Latency P95   (ms)", "latency_p95_ms", ".2f")
    row("Latency P99   (ms)", "latency_p99_ms", ".2f")
    row("Latency min   (ms)", "latency_min_ms", ".2f")
    row("Latency max   (ms)", "latency_max_ms", ".2f")
    divider("GPU")
    row("GPU util mean  (%)", "gpu_util_mean", ".1f")
    row("GPU util max   (%)", "gpu_util_max", ".1f")
    row("GPU mem mean  (MB)", "gpu_mem_mean_mb", ".1f")
    row("GPU mem max   (MB)", "gpu_mem_max_mb", ".1f")
    divider("CPU / RAM")
    row("CPU util mean  (%)", "cpu_util_mean", ".1f")
    row("RAM mean       (MB)", "ram_mean_mb", ".1f")
    divider("Frames")
    row("Total frames", "total_frames", "d")

    console.print(table)


# Main


def main(args):
    console = Console()
    console.rule("[bold green]YOLO Inference Benchmark")
    console.print(f"  Frames     : {args.frames}")
    console.print(f"  Source     : {args.source}")
    console.print(f"  Image size : {args.imgsz}")
    console.print(f"  Conf thresh: {args.conf}")
    console.print(f"  Backends   : {', '.join(args.backends)}\n")

    # collect frames once — all backends use identical input
    frames = get_frames(args.source, args.frames)

    # run each backend
    all_stats = {}
    for backend_name in args.backends:
        if backend_name not in INFERENCER_MAP:
            console.print(f"[red]Unknown backend: {backend_name} — skipping[/red]")
            continue

        all_stats[backend_name] = run_backend(
            backend_name=backend_name,
            inferencer_cls=INFERENCER_MAP[backend_name],
            model_path=MODELS[backend_name],
            frames=frames,
            imgsz=args.imgsz,
            conf_thresh=args.conf,
        )

        # cooldown between backends — let GPU settle
        console.print("  [dim]Cooling down 3s...[/dim]")
        time.sleep(3)

    # save combined report
    Path("results").mkdir(exist_ok=True)
    report_path = Path("results/benchmark_report.json")
    report = {
        "meta": {
            "frames": args.frames,
            "source": args.source,
            "imgsz": args.imgsz,
            "conf_thresh": args.conf,
            "backends": args.backends,
        },
        "results": all_stats,
    }
    report_path.write_text(json.dumps(report, indent=2))
    console.print(f"\n  [green]✅ Full report saved: {report_path}[/green]")

    # comparison table
    print_comparison(all_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Inference Benchmark Runner")
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Number of frames to benchmark per backend",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Frame source: 'webcam', video path, or image path",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["PyTorch", "ONNX", "TensorRT"],
        choices=["PyTorch", "ONNX", "TensorRT"],
        help="Which backends to benchmark",
    )
    args = parser.parse_args()
    main(args)
