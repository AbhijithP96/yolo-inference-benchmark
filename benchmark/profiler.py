# benchmark/profiler.py

"""
Background profiler — samples GPU/CPU/memory every 100ms on a separate thread.
Uses Pydantic for stats modeling.
"""

import time
import threading
from typing import Optional

import numpy as np
import psutil
import pynvml
from pydantic import BaseModel, Field, computed_field

from loguru import logger


# ---------------------------------------------------------------
#                           Base Models
# ---------------------------------------------------------------


class FrameSample(BaseModel):
    """Stats captured for a single inference frame."""

    frame_idx: int
    latency_ms: float = Field(..., ge=0.0)
    fps: float = Field(..., ge=0.0)
    preprocess_ms: float = Field(..., ge=0.0)
    inference_ms: float = Field(..., ge=0.0)
    postprocess_ms: float = Field(..., ge=0.0)
    num_detections: int = Field(..., ge=0)


class SystemSample(BaseModel):
    """One background thread hardware snapshot."""

    timestamp_s: float = Field(
        ..., description="Time since profiler.start() was called"
    )
    gpu_util_pct: float = Field(..., ge=0.0, le=100.0)
    gpu_mem_mb: float = Field(..., ge=0.0)
    cpu_util_pct: float = Field(..., ge=0.0)
    ram_mb: float = Field(..., ge=0.0)


class ProfilerStats(BaseModel):
    """
    Full stats for one backend benchmark run.
    Serializes directly to JSON via .model_dump_json().
    """

    backend: str = Field(default="")
    gpu_name: str = Field(default="")
    frame_samples: list[FrameSample] = Field(default_factory=list)
    system_samples: list[SystemSample] = Field(default_factory=list)

    # Per-frame computed fields

    @computed_field
    @property
    def fps_mean(self) -> float:
        vals = [f.fps for f in self.frame_samples]
        return float(np.mean(vals)) if vals else 0.0

    @computed_field
    @property
    def fps_max(self) -> float:
        vals = [f.fps for f in self.frame_samples]
        return float(np.max(vals)) if vals else 0.0

    @computed_field
    @property
    def latency_mean_ms(self) -> float:
        vals = [f.latency_ms for f in self.frame_samples]
        return float(np.mean(vals)) if vals else 0.0

    @computed_field
    @property
    def latency_p50_ms(self) -> float:
        vals = [f.latency_ms for f in self.frame_samples]
        return float(np.percentile(vals, 50)) if vals else 0.0

    @computed_field
    @property
    def latency_p95_ms(self) -> float:
        vals = [f.latency_ms for f in self.frame_samples]
        return float(np.percentile(vals, 95)) if vals else 0.0

    @computed_field
    @property
    def latency_p99_ms(self) -> float:
        vals = [f.latency_ms for f in self.frame_samples]
        return float(np.percentile(vals, 99)) if vals else 0.0

    @computed_field
    @property
    def latency_min_ms(self) -> float:
        vals = [f.latency_ms for f in self.frame_samples]
        return float(np.min(vals)) if vals else 0.0

    @computed_field
    @property
    def latency_max_ms(self) -> float:
        vals = [f.latency_ms for f in self.frame_samples]
        return float(np.max(vals)) if vals else 0.0

    # System computed fields

    @computed_field
    @property
    def gpu_util_mean(self) -> float:
        vals = [s.gpu_util_pct for s in self.system_samples]
        return float(np.mean(vals)) if vals else 0.0

    @computed_field
    @property
    def gpu_util_max(self) -> float:
        vals = [s.gpu_util_pct for s in self.system_samples]
        return float(np.max(vals)) if vals else 0.0

    @computed_field
    @property
    def gpu_mem_mean_mb(self) -> float:
        vals = [s.gpu_mem_mb for s in self.system_samples]
        return float(np.mean(vals)) if vals else 0.0

    @computed_field
    @property
    def gpu_mem_max_mb(self) -> float:
        vals = [s.gpu_mem_mb for s in self.system_samples]
        return float(np.max(vals)) if vals else 0.0

    @computed_field
    @property
    def cpu_util_mean(self) -> float:
        vals = [s.cpu_util_pct for s in self.system_samples]
        return float(np.mean(vals)) if vals else 0.0

    @computed_field
    @property
    def ram_mean_mb(self) -> float:
        vals = [s.ram_mb for s in self.system_samples]
        return float(np.mean(vals)) if vals else 0.0

    @computed_field
    @property
    def total_frames(self) -> int:
        return len(self.frame_samples)


# ---------------------------------------------------------------
#                           Profiler
# ---------------------------------------------------------------


class Profiler:
    """
    Two-layer profiler:
      Layer 1 -> per frame : call record_frame(inference_result) after each run()
      Layer 2 -> background: thread samples GPU/CPU/RAM every 100ms automatically.
    """

    def __init__(self, gpu_index: int = 0, sample_interval_ms: float = 100.0):
        self.gpu_index = gpu_index
        self.sample_interval_s = sample_interval_ms / 1000.0

        self._stats: Optional[ProfilerStats] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0

        # init pynvml once
        pynvml.nvmlInit()
        self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._gpu_name = pynvml.nvmlDeviceGetName(self._gpu_handle)

    def start(self, backend_name: str = "") -> None:
        """Start a new profiling session — resets all stats."""
        self._stats = ProfilerStats(
            backend=backend_name,
            gpu_name=self._gpu_name,
        )
        self._start_time = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info(f"[Profiler] started -> backend : {backend_name or 'unknown'}")
        logger.info(f"[Profiler] GPU -> {self._gpu_name}")

    def stop(self) -> None:
        """Stop the background sampling thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        n = len(self._stats.frame_samples) if self._stats else 0
        logger.info(f"[Profiler] stopped -> {n} frames recorded")

    def record_frame(self, inference_result) -> None:
        """
        Call once per frame after inferencer.run().
        Captures per-frame latency, FPS, and timing breakdown.
        """
        if self._stats is None:
            logger.error("Profiler not started. Call start() first.")
            raise RuntimeError

        idx = len(self._stats.frame_samples)

        sample = FrameSample(
            frame_idx=idx,
            latency_ms=inference_result.total_ms,
            fps=inference_result.fps,
            preprocess_ms=inference_result.preprocess_ms,
            inference_ms=inference_result.inference_ms,
            postprocess_ms=inference_result.postprocess_ms,
            num_detections=inference_result.num_detections,
        )
        self._stats.frame_samples.append(sample)

    def get_stats(self) -> ProfilerStats:
        if self._stats is None:
            logger.error("Profiler not started. Call start() first.")
            raise RuntimeError
        return self._stats

    def save(self, path: str) -> None:
        """Save full ProfilerStats to a JSON file."""
        # import json
        from pathlib import Path

        if self._stats is None:
            logger.error("No stats to save. Run a benchmark first.")
            raise RuntimeError

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self._stats.model_dump_json(indent=2))
        logger.info(f"[Profiler] saved -> {out}")

    def print_summary(self) -> None:
        """Pretty-prints the summary table using rich."""
        from rich.table import Table
        from rich.console import Console

        if self._stats is None:
            print("No stats yet.")
            return

        s = self._stats
        table = Table(
            title=f"Benchmark Summary — {s.backend}  |  {s.gpu_name}",
            show_lines=True,
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        rows = [
            ("Total frames", str(s.total_frames)),
            ("─── Throughput", ""),
            ("FPS (mean)", f"{s.fps_mean:.1f}"),
            ("FPS (max)", f"{s.fps_max:.1f}"),
            ("─── Latency", ""),
            ("Latency mean (ms)", f"{s.latency_mean_ms:.2f}"),
            ("Latency min  (ms)", f"{s.latency_min_ms:.2f}"),
            ("Latency P50  (ms)", f"{s.latency_p50_ms:.2f}"),
            ("Latency P95  (ms)", f"{s.latency_p95_ms:.2f}"),
            ("Latency P99  (ms)", f"{s.latency_p99_ms:.2f}"),
            ("Latency max  (ms)", f"{s.latency_max_ms:.2f}"),
            ("─── GPU", ""),
            ("GPU util mean (%)", f"{s.gpu_util_mean:.1f}"),
            ("GPU util max  (%)", f"{s.gpu_util_max:.1f}"),
            ("GPU mem  mean (MB)", f"{s.gpu_mem_mean_mb:.1f}"),
            ("GPU mem  max  (MB)", f"{s.gpu_mem_max_mb:.1f}"),
            ("─── CPU / RAM", ""),
            ("CPU util mean (%)", f"{s.cpu_util_mean:.1f}"),
            ("RAM       mean (MB)", f"{s.ram_mean_mb:.1f}"),
        ]

        for label, val in rows:
            table.add_row(label, val)

        Console().print(table)

    def _sample_loop(self) -> None:
        """Runs on a daemon thread — samples hardware every sample_interval_s."""
        while not self._stop_event.is_set():
            try:
                now = time.perf_counter() - self._start_time
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().used / 1024 / 1024

                sample = SystemSample(
                    timestamp_s=now,
                    gpu_util_pct=float(util.gpu),
                    gpu_mem_mb=mem.used / 1024 / 1024,
                    cpu_util_pct=float(cpu),
                    ram_mb=float(ram),
                )
                if self._stats is not None:
                    self._stats.system_samples.append(sample)

            except Exception:
                pass  # never crash the benchmark from a sampling error

            time.sleep(self.sample_interval_s)

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
