# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/7 17:55
@File    : infer.py
@Author  : zj
@Description: 
"""

import os.path

import cv2
import sys
import argparse

from pathlib import Path
from typing import Union, Dict, Any, List

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
from core.utils.general import yaml_load
from core.utils.results import save_txt
from core.utils.dataloaders import LoadImages
from core.utils.ops import masks2segments, scale_coords
from core.utils.v8.plots import Annotator, colors


def predict_source(
        model: Any,
        source: str,
        save_dir: str = "output",
        save: bool = False,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        vid_stride: int = 1,  # video frame-rate stride
        line_thickness: int = 3,  # bounding box thickness (pixels)
):
    """
    Predict on a single image and log end-to-end latency.
    """
    dataset = LoadImages(source, vid_stride)
    vid_path, vid_writer = None, None

    for path, im0, vid_cap, s in dataset:
        im0_shape = im0.shape[:2]
        boxes, confs, cls_ids, masks, dt = model.detect(im0, conf_thresh, iou_thresh)
        # Print time (inference-only)
        logging.info(f"{s}{'' if len(boxes) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # --- Timing statistics ---
        pre_time = dt[0].t * 1000  # ms
        inf_time = dt[1].t * 1000
        post_time = dt[2].t * 1000
        total_time = sum([t.t for t in dt]) * 1000

        logging.info(
            f"Detect time - Pre: {pre_time:.2f}ms | "
            f"Infer: {inf_time:.2f}ms | "
            f"Post: {post_time:.2f}ms | "
            f"Total: {total_time:.2f}ms"
        )

        annotator = Annotator(im0, line_width=line_thickness)
        if len(masks):
            idx = reversed(range(len(masks)))
            annotator.masks(masks, colors=[colors(x, True) for x in idx])
        if len(boxes):
            for i in reversed(range(len(boxes))):
                xyxy = boxes[i]
                conf = float(confs[i][0])
                cls_id = int(cls_ids[i][0])

                label = f'{model.classes[cls_id]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(cls_id, True))
        im0 = annotator.result()

        if save:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            p = Path(path)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg

            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                logging.info(f"Result saved to: {save_path}")

                segments = None
                if len(masks) > 0:
                    segments = masks2segments(masks)
                    segments = scale_coords((model.net_h, model.net_w), segments[0], im0_shape, normalize=True)

                label_path = save_path.split('.')[0] + '.txt'
                save_txt(boxes, confs, cls_ids, im0_shape, label_path, save_conf=True, segments=segments)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)


def parse_opt() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="YOLOv8-seg Inference",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'weight',
        type=str,
        help='Path to model file (e.g., yolov8s-seg.onnx)'
    )
    parser.add_argument(
        'source',
        type=str,
        help='Path to input image or video'
    )
    parser.add_argument(
        'data',
        type=str,
        help='Path to dataset.yaml'
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
        default=0.25,  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
        help="Confidence threshold for object detection"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,  # (float) intersection over union (IoU) threshold for NMS
        help="IOU threshold for Non-Max Suppression (NMS)"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="output",
        help="Output directory for results"
    )

    args = parser.parse_args()

    logging.info(f"Parsed arguments: {args}")
    return args


def main():
    args = parse_opt()

    if args.backend == "onnxruntime":
        if args.processor == "torch":
            try:
                from yolov8_seg_runtime_w_torch import YOLOv8RuntimeTorch
                logging.info("Using YOLOv8RuntimeTorch")
                ModelClass = YOLOv8RuntimeTorch
            except ImportError:
                raise ImportError(f"PyTorch processor selected, but YOLOv8RuntimeTorch is not available.")
        else:
            pass
            # try:
            #     from yolov8_runtime_w_numpy import YOLOv8RuntimeNumpy
            #     logging.info("Using YOLOv8RuntimeNumpy")
            #     ModelClass = YOLOv8RuntimeNumpy
            # except ImportError:
            #     raise ImportError(f"Numpy processor selected, but YOLOv8RuntimeNumpy is not available.")

    elif args.backend == "tensorrt":
        pass
    else:
        raise ValueError(f"Unsupported backend type: {args.backend}")

    # Load names
    names = yaml_load(args.data)['names'] if args.data else {i: f'class{i}' for i in range(999)}

    # Load model
    model = ModelClass(names, weight=args.weight)
    logging.info(f"Model loaded: {args.weight} | Processor: {args.processor} | Backend: {args.backend}")

    model_name = os.path.basename(args.weight).split('.')[0]
    save_dir = os.path.join(str(args.save_dir), str(model_name))

    # Run inference
    predict_source(model, args.source,
                   save_dir=save_dir, save=True, conf_thresh=args.conf, iou_thresh=args.iou)


if __name__ == "__main__":
    main()
