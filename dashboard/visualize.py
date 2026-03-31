# dashboard/visualize.py

"""
Reads the benchmark_results.json and generates comparison charts.

Usage: uv run -m dashboard.visualize --report results/benchmark_report.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from loguru import logger

# Styling

BACKEND_COLORS = {
    "PyTorch": "#00ffff",  # cyan
    "ONNX": "#dc00ff",  # purple
    "TensorRT": "#ffaa00",  # orange
}

plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "text.color": "#c9d1d9",
        "font.family": "monospace",
        "figure.dpi": 150,
    }
)

CHART_DIR = Path("results/charts")

# Helper Functions


def load_report(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    return data


def bar_chart(
    ax,
    backends: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    highlight: str = "max",  # "max" or "min" — which bar to mark as winner
    fmt: str = ".1f",
):
    """Reusable bar chart with value labels and winner highlight."""
    colors = [BACKEND_COLORS[b] for b in backends]
    bars = ax.bar(backends, values, color=colors, width=0.5, zorder=3)
    ax.set_title(title, pad=10, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#c9d1d9",
        )

    # highlight winner
    winner_idx = (
        values.index(max(values)) if highlight == "max" else values.index(min(values))
    )
    bars[winner_idx].set_edgecolor("#FFD700")
    bars[winner_idx].set_linewidth(2.5)

    return ax


# Chart 1 : FPS Comparison


def chart_fps(results: dict, backends: list[str]):
    logger.info("FPS Comparison..")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Throughput Comparison", fontsize=14, fontweight="bold", y=1.02)

    # mean FPS
    bar_chart(
        axes[0],
        backends,
        [results[b]["fps_mean"] for b in backends],
        title="FPS (Mean)",
        ylabel="Frames per Second",
        highlight="max",
        fmt=".1f",
    )

    # max FPS
    bar_chart(
        axes[1],
        backends,
        [results[b]["fps_max"] for b in backends],
        title="FPS (Max)",
        ylabel="Frames per Second",
        highlight="max",
        fmt=".1f",
    )

    plt.tight_layout()
    out = CHART_DIR / "01_fps_comparison.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved : {out}")


# Chart 2: Latency Percentiles


def chart_latency(results: dict, backends: list[str]):
    logger.info("Latency Comparison..")
    percentiles = [
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    ]
    labels = ["Mean", "P50", "P95", "P99"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Latency Percentiles (ms) — lower is better", fontsize=14, fontweight="bold"
    )

    x = np.arange(len(labels))
    width = 0.25
    offset = np.linspace(-width, width, len(backends))

    for i, backend in enumerate(backends):
        vals = [results[backend][p] for p in percentiles]
        bars = ax.bar(
            x + offset[i],
            vals,
            width,
            label=backend,
            color=BACKEND_COLORS[backend],
            zorder=3,
            alpha=0.9,
        )

        # value labels
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#c9d1d9",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = CHART_DIR / "02_latency_percentiles.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved : {out}")


# Chart 3: Per frame Latency


def chart_latency_timeline(report: dict, backends: list[str]):
    """
    Plots per-frame latency over time for each backend.
    Shows spikes and consistency — not visible from averages alone.
    """
    logger.info("Latency Timeline..")
    fig, axes = plt.subplots(
        len(backends), 1, figsize=(14, 4 * len(backends)), sharex=False
    )
    if len(backends) == 1:
        axes = [axes]

    fig.suptitle(
        "Per-Frame Latency Timeline — spikes visible here",
        fontsize=14,
        fontweight="bold",
    )

    for ax, backend in zip(axes, backends):
        frames = report["results"][backend].get("frame_samples", [])
        if not frames:
            ax.set_title(f"{backend} — no frame data")
            continue

        latencies = [f["latency_ms"] for f in frames]
        x = list(range(len(latencies)))
        color = BACKEND_COLORS[backend]

        ax.plot(x, latencies, color=color, linewidth=0.8, alpha=0.8)
        ax.axhline(
            np.mean(latencies),
            color="#FFD700",
            linewidth=1.2,
            linestyle="--",
            label=f"mean {np.mean(latencies):.1f}ms",
        )
        ax.axhline(
            np.percentile(latencies, 99),
            color="#FF5252",
            linewidth=1.0,
            linestyle=":",
            label=f"P99 {np.percentile(latencies, 99):.1f}ms",
        )

        ax.fill_between(x, latencies, alpha=0.15, color=color)
        ax.set_title(f"{backend}", fontsize=11, fontweight="bold", color=color)
        ax.set_ylabel("Latency (ms)", fontsize=9)
        ax.set_xlabel("Frame index", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(zorder=0)

    plt.tight_layout()
    out = CHART_DIR / "03_latency_timeline.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved : {out}")


# Chart 4: GPU Utilization and Memory


def chart_gpu(results: dict, backends: list[str]):
    logger.info("GPU Utilization..")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GPU Resource Usage", fontsize=14, fontweight="bold", y=1.02)

    # GPU utilization
    bar_chart(
        axes[0],
        backends,
        [results[b]["gpu_util_mean"] for b in backends],
        title="GPU Utilization Mean (%)",
        ylabel="Utilization (%)",
        highlight="max",
        fmt=".1f",
    )

    # GPU memory
    bar_chart(
        axes[1],
        backends,
        [results[b]["gpu_mem_mean_mb"] for b in backends],
        title="GPU Memory Mean (MB)",
        ylabel="Memory (MB)",
        highlight="min",
        fmt=".0f",  # lower memory = better = highlight min
    )

    plt.tight_layout()
    out = CHART_DIR / "04_gpu_resources.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved : {out}")


# Chart 5 : Backend Process breakdown


def chart_timing_breakdown(report: dict, backends: list[str]):
    """
    Stacked bar chart showing preprocess / inference / postprocess time.
    Shows where time is actually spent per backend.
    """
    logger.info("Backend Process Timeline..")
    pre_vals = []
    inf_vals = []
    post_vals = []

    for backend in backends:
        frames = report["results"][backend].get("frame_samples", [])
        if frames:
            pre_vals.append(np.mean([f["preprocess_ms"] for f in frames]))
            inf_vals.append(np.mean([f["inference_ms"] for f in frames]))
            post_vals.append(np.mean([f["postprocess_ms"] for f in frames]))
        else:
            pre_vals.append(0)
            inf_vals.append(0)
            post_vals.append(0)

    x = np.arange(len(backends))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Latency Breakdown — Preprocess / Inference / Postprocess",
        fontsize=14,
        fontweight="bold",
    )

    b1 = ax.bar(x, pre_vals, width, label="Preprocess", color="#42A5F5", zorder=3)
    b2 = ax.bar(
        x,
        inf_vals,
        width,
        label="Inference",
        color="#EF5350",
        bottom=pre_vals,
        zorder=3,
    )
    b3 = ax.bar(
        x,
        post_vals,
        width,
        label="Postprocess",
        color="#FFCA28",
        bottom=[p + i for p, i in zip(pre_vals, inf_vals)],
        zorder=3,
    )

    # value labels on each segment
    for i, (p, inf, po) in enumerate(zip(pre_vals, inf_vals, post_vals)):
        if p > 0.3:
            ax.text(x[i], p / 2, f"{p:.1f}", ha="center", fontsize=8, color="white")
        if inf > 0.3:
            ax.text(
                x[i], p + inf / 2, f"{inf:.1f}", ha="center", fontsize=8, color="white"
            )
        if po > 0.3:
            ax.text(
                x[i],
                p + inf + po / 2,
                f"{po:.1f}",
                ha="center",
                fontsize=8,
                color="white",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(backends, fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = CHART_DIR / "05_timing_breakdown.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved : {out}")


# Chart 6 : Summary


def chart_summary_dashboard(results: dict, backends: list[str]):
    """
    Single-image summary with 4 key metrics.
    """
    logger.info("Summary Dashbord...")
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "YOLOv8 Inference Benchmark — PyTorch vs ONNX vs TensorRT",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    bar_chart(
        axes[0],
        backends,
        [results[b]["fps_mean"] for b in backends],
        "FPS (Mean)",
        "FPS",
        "max",
    )

    bar_chart(
        axes[1],
        backends,
        [results[b]["latency_p50_ms"] for b in backends],
        "Latency P50 (ms)",
        "ms",
        "min",
    )

    bar_chart(
        axes[2],
        backends,
        [results[b]["latency_p99_ms"] for b in backends],
        "Latency P99 (ms)",
        "ms",
        "min",
    )

    bar_chart(
        axes[3],
        backends,
        [results[b]["gpu_util_mean"] for b in backends],
        "GPU Util Mean (%)",
        "%",
        "max",
    )

    bar_chart(
        axes[4],
        backends,
        [results[b]["gpu_mem_mean_mb"] for b in backends],
        "GPU Memory Mean (MB)",
        "MB",
        "min",
    )

    bar_chart(
        axes[5],
        backends,
        [results[b]["cpu_util_mean"] for b in backends],
        "CPU Util Mean (%)",
        "%",
        "max",
    )

    # legend
    patches = [mpatches.Patch(color=BACKEND_COLORS[b], label=b) for b in backends]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=3,
        fontsize=11,
        framealpha=0.2,
        bbox_to_anchor=(0.5, 0.01),
    )

    out = CHART_DIR / "00_summary_dashboard.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved : {out}")


# Chart 7 : GPU Timeline


def chart_gpu_timeline(report: dict, backends: list[str]):
    """GPU utilization over time sampled by background thread."""
    logger.info("GPU Util Timeline...")
    fig, axes = plt.subplots(
        len(backends), 1, figsize=(14, 3.5 * len(backends)), sharex=False
    )
    if len(backends) == 1:
        axes = [axes]

    fig.suptitle("GPU Utilization Timeline", fontsize=14, fontweight="bold")

    for ax, backend in zip(axes, backends):
        samples = report["results"][backend].get("system_samples", [])
        if not samples:
            continue

        timestamps = [s["timestamp_s"] for s in samples]
        gpu_utils = [s["gpu_util_pct"] for s in samples]
        gpu_mems = [s["gpu_mem_mb"] for s in samples]
        color = BACKEND_COLORS[backend]

        ax.plot(timestamps, gpu_utils, color=color, linewidth=1.2, label="GPU util %")
        ax.fill_between(timestamps, gpu_utils, alpha=0.2, color=color)
        ax.set_title(f"{backend}", fontsize=11, fontweight="bold", color=color)
        ax.set_ylabel("GPU util (%)", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(zorder=0)

    plt.tight_layout()
    out = CHART_DIR / "06_gpu_timeline.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved : {out}")


def main(args):
    logger.info(f"Chart Generator from {args.report}")

    report = load_report(args.report)
    results = report["results"]
    backends = list(results.keys())

    logger.debug(f"Backends found: {backends}")

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Charts")
    chart_summary_dashboard(results, backends)
    chart_fps(results, backends)
    chart_latency(results, backends)
    chart_latency_timeline(report, backends)
    chart_gpu(results, backends)
    chart_timing_breakdown(report, backends)
    chart_gpu_timeline(report, backends)

    logger.info(f"All Charts Generated and Saved to {str(CHART_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=str,
        default="results/benchmark_report.json",
    )
    args = parser.parse_args()
    main(args)
