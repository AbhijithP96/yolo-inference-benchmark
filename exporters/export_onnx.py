# exporters/export_onnx.py

"""
Export Yolo V8 .pt model to ONNX format.
Usage: python exporters/export_onnx.py --weights models/yolov8n_head_detector.pt --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

from loguru import logger


def export_onnx(
    weights: str, imgsz: int, dynamic: bool, simplify: bool, opset: int
) -> str:
    weights_path = Path(weights)
    if not weights_path.exists():
        logger.error(f"Weights not found at {weights}")
        raise FileNotFoundError

    logger.info("Exporting to ONNX")
    logger.info(
        f"Weigts : {weights}, Image Size: {imgsz}, Dynamic: {dynamic}, Simplify: {simplify}, Opset: {opset}"
    )

    model = YOLO(weights_path.as_posix())

    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
        half=False,
    )

    logger.info(f"Model exported to : {export_path}")
    return export_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dynamic", action="store_true", default=False)
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    export_onnx(
        weights=args.weights,
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )
