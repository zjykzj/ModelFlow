# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/6 20:38
@File    : main.py
@Author  : zj
@Description:
"""
import os.path

import cv2
import sys
import time
import argparse

from pathlib import Path
from typing import Union, Dict, Any
from tqdm import tqdm

# Setup logging
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Resolve paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

CURRENT_DIR = Path.cwd()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import local modules
from core.utils.general import CLASSES_NAME, draw_results, IMAGE_EXTS, VIDEO_EXTS


def predict_image(
        model: Any,
        img_path: Union[Path, str],
        output_dir: Union[Path, str] = "output",
        suffix: str = "",
        save: bool = False,
        conf: float = 0.25,
        iou: float = 0.45,
):
    """
    Predict on a single image and log end-to-end latency.
    """
    img_path = Path(img_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")

    logging.info(f"Processing image: {img_path.name}")

    # End-to-end time (load + inference + draw)
    start_time = time.time()
    boxes, confs, cls_ids = model.detect(image, conf, iou)
    overlay = draw_results(image, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
    total_time_ms = (time.time() - start_time) * 1000
    fps = 1000 / total_time_ms if total_time_ms > 0 else 0

    logging.info(f"Detected {len(boxes)} objects | Latency: {total_time_ms:.2f}ms | FPS: {fps:.1f}")

    if save:
        if suffix != "":
            suffix = '_' + suffix
        save_path = output_dir / f"{img_path.stem}{suffix}.jpg"
        cv2.imwrite(str(save_path), overlay)
        logging.info(f"Result saved to: {save_path}")


def predict_video(
        model: Any,
        video_file: Union[Path, str],
        output_dir: Union[Path, str] = "output",
        suffix: str = "",
        save: bool = False,
        conf: float = 0.25,
        iou: float = 0.45,
):
    """
    Process a video file frame by frame and compute average FPS.
    """
    video_path = Path(video_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(
        f"Video info: {video_path.name} | {fps:.1f} FPS | {total_frames} frames | {frame_width}x{frame_height}")

    writer = None
    if save:
        if suffix != "":
            suffix = '_' + suffix
        save_path = output_dir / f"{video_path.stem}{suffix}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (frame_width, frame_height))

    total_inference_time = 0.0
    processed_frames = 0

    try:
        pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            boxes, confs, cls_ids = model.detect(frame, conf, iou)
            overlay = draw_results(frame, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
            frame_time_ms = (time.time() - start_time) * 1000

            total_inference_time += frame_time_ms
            processed_frames += 1
            pbar.update(1)

            if save and writer:
                writer.write(overlay)

        pbar.close()
    except Exception as e:
        logging.error(f"Error during video processing: {e}")
    finally:
        cap.release()
        if writer:
            writer.release()

    # Compute average performance
    avg_time = total_inference_time / processed_frames if processed_frames > 0 else 0
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    logging.info(f"Average latency: {avg_time:.2f}ms/frame | Average FPS: {avg_fps:.1f}")

    if save:
        logging.info(f"Video saved to: {save_path}")


def parse_opt() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="YOLOv5 Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "model",
        type=str,
        help="Path to model file (e.g., yolov5s.onnx)"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input image or video"
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="onnxruntime",
        choices=["onnxruntime", "tensorrt", "triton"],
        help="Inference backend to use"
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Pre/Post-processing backend: use NumPy or PyTorch"
    )

    # Add confidence and IOU (NMS) threshold arguments
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for object detection"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for Non-Max Suppression (NMS)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file does not exist: {args.input}")

    # Detect mode (image or video)
    ext = input_path.suffix.lower()
    if ext in IMAGE_EXTS:
        args.mode = "image"
    elif ext in VIDEO_EXTS:
        args.mode = "video"
    else:
        parser.error(f"Unsupported file extension: {ext}. Supported: {IMAGE_EXTS | VIDEO_EXTS}")

    logging.info(f"Parsed arguments: {args}")
    return args


def main():
    args = parse_opt()

    if args.backend == "onnxruntime":
        if args.processor == "torch":
            try:
                from yolov5_runtime_w_torch import YOLOv5Runtime
                logging.info("Using YOLOv5Runtime with PyTorch")
            except ImportError:
                raise ImportError(f"PyTorch processor selected, but YOLOv5Runtime is not available.")
        else:
            try:
                from yolov5_runtime_w_numpy import YOLOv5Runtime
                logging.info("Using YOLOv5Runtime with Numpy")
            except ImportError:
                raise ImportError(f"Numpy processor selected, but YOLOv5Runtime is not available.")

        ModelClass = YOLOv5Runtime
    elif args.backend == "tensorrt":
        raise ValueError(f"Backend '{args.backend}' is not supported yet.")
    else:
        raise ValueError(f"Unsupported backend type: {args.backend}")

    # Load model
    model = ModelClass(args.model)
    logging.info(f"Model loaded: {args.model} | Processor: {args.processor} | Backend: {args.backend}")

    model_name = os.path.basename(args.model)
    output_dir = os.path.join(args.output, model_name)

    # Run inference
    predict = predict_image if args.mode == "image" else predict_video
    predict(model, args.input, output_dir, save=True, conf=args.conf, iou=args.iou)


if __name__ == "__main__":
    main()
