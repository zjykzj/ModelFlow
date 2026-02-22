# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/227 14:16
@File    : segment_detect.py
@Author  : zj
@Description: 
"""

import os
import os.path as osp
import cv2
import json
import glob

import numpy as np
from PIL import Image
from pathlib import Path

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])

from core.utils.logger import get_logger, LOGGER_NAME

logger = get_logger(LOGGER_NAME)


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def get_imgs_labels(img_dirs):
    if not isinstance(img_dirs, list):
        img_dirs = [img_dirs]

    img_paths = []
    for img_dir in img_dirs:
        assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"
        img_paths.extend(glob.glob(osp.join(img_dir, "**/*"), recursive=True))

    img_paths = sorted([
        p for p in img_paths
        if osp.isfile(p) and p.split('.')[-1].lower() in IMG_FORMATS
    ])
    assert img_paths, f"No images found in {img_dirs}."

    label_paths = img2label_paths(img_paths)
    labels = []  # Each element is a list of [cls_id, x1, y1, x2, y2, ...], variable length per object
    for lp in label_paths:
        if osp.exists(lp):
            try:
                with open(lp, 'r') as f:
                    lines = f.read().strip().splitlines()
                lb = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 3 or (len(parts) - 1) % 2 != 0:
                        raise ValueError(f"Malformed label at {lp}: {line}")
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    lb.append([cls] + coords)  # Keep as Python list
                labels.append(lb)  # Append list of objects (each object = [cls, x1, y1, ..., xn, yn])
            except Exception as e:
                raise RuntimeError(f"Error processing file {lp}: {e}")
        else:
            labels.append([])  # Empty list for no labels
    return img_paths, labels


class SegmentDataset:

    def __init__(self, img_dir, class_list, img_size=640):
        self.img_size = img_size
        self.class_list = class_list
        self.img_paths, self.labels = get_imgs_labels(img_dir)

        logger.info(f"data_root is {img_dir}")
        logger.info(f"data_list length is {len(self.img_paths)}")

    def __len__(self):
        return len(self.img_paths)

    def __call__(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        # Load image
        path = self.img_paths[index]
        img = cv2.imread(path)  # BGR image
        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {path}")

        h, w = img.shape[:2]

        # Load label: labels_per_img is a list of [cls_id, x1, y1, x2, y2, ..., xn, yn]
        labels_per_img = self.labels[index]

        # Initialize mask with the same size as the image but with a single channel
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(labels_per_img) > 0:
            for label in labels_per_img:
                cls_id = int(label[0])
                coords_norm = np.array(label[1:], dtype=np.float32)  # Convert to numpy array
                if len(coords_norm) % 2 != 0:
                    raise ValueError(f"Odd number of coordinates in label from {path}: {label}")
                coords_norm = coords_norm.reshape(-1, 2)  # (N, 2)
                coords_abs = (coords_norm * np.array([w, h])).astype(np.int32)

                # Fill the polygon on the mask
                cv2.fillPoly(mask, [coords_abs], color=cls_id)

        return img, path, mask

    def get_anno_json(self, anno_json_path):
        """
        Generate COCO-style annotation JSON with RLE-encoded masks.
        Saves to `anno_json_path` and returns the dict.
        """
        from datetime import datetime
        from pycocotools import mask as mask_utils

        coco_dict = {
            "info": {
                "description": "YOLO-format Segmentation Dataset converted to COCO (RLE masks)",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"id": 1, "name": "MIT", "url": ""}],
            "categories": [],
            "images": [],
            "annotations": []
        }

        # Collect all class IDs
        all_cls_ids = set()
        for labels_per_img in self.labels:
            if len(labels_per_img) > 0:
                # Extract class IDs using pure Python (each label is [cls, x1, y1, ...])
                cls_ids = [int(label[0]) for label in labels_per_img]
                all_cls_ids.update(cls_ids)
        all_cls_ids = sorted(all_cls_ids)
        print(f"all_cls_ids = {all_cls_ids}")
        coco_dict["categories"] = [
            {"id": cid + 1, "name": self.class_list[cid], "supercategory": "object"} for cid in all_cls_ids
        ]

        ann_id = 1
        for img_id, (img_path, labels) in enumerate(zip(self.img_paths, self.labels), start=1):
            # Image info
            img_pil = Image.open(img_path)
            width, height = img_pil.size
            file_name = osp.basename(img_path)
            coco_dict["images"].append({
                "id": int(Path(img_path).stem),
                "width": width,
                "height": height,
                "file_name": file_name,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })

            if len(labels) == 0:
                continue

            for label in labels:
                cls_id = int(label[0])
                coords_norm = np.array(label[1:], dtype=np.float32)
                if len(coords_norm) % 2 != 0:
                    raise ValueError(f"Odd number of coordinates in label from {img_path}: {label}")
                coords_norm = coords_norm.reshape(-1, 2)  # (N, 2)
                coords_abs = coords_norm * np.array([[width, height]])  # to absolute pixel coords

                # Step 1: Render binary mask (H, W)
                mask = np.zeros((height, width), dtype=np.uint8)
                poly = np.round(coords_abs).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)

                # Step 2: Encode to RLE
                rle = mask_utils.encode(np.asfortranarray(mask))
                # pycocotools returns counts as bytes; COCO format expects string in Python 3
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode('utf-8')

                # Step 3: Compute area as sum of mask pixels
                area = int(np.sum(mask))

                # Optional: compute bbox from mask (more accurate than polygon bbox)
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue  # skip empty
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

                coco_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": int(Path(img_path).stem),
                    "category_id": cls_id + 1,
                    "segmentation": rle,  # RLE dict
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1

        # Save
        os.makedirs(osp.dirname(anno_json_path), exist_ok=True)
        with open(anno_json_path, 'w') as f:
            json.dump(coco_dict, f, indent=2)

        return coco_dict
