# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 11:48
@File    : detect_dataset.py
@Author  : zj
@Description: 
"""

import os
import os.path as osp
import cv2
import json
import glob

from PIL import Image
from pathlib import Path

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])

from core.utils.logger import get_logger, LOGGER_NAME

logger = get_logger(LOGGER_NAME)


def img2label_paths(img_paths):
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
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
    labels = []  # Each element: list of [cls_id, cx, cy, w, h] (normalized)
    for lp in label_paths:
        if osp.exists(lp):
            try:
                with open(lp, 'r') as f:
                    lines = f.read().strip().splitlines()
                lb = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        raise ValueError(f"Malformed detection label at {lp}: {line}")
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    lb.append([cls, cx, cy, w, h])
                labels.append(lb)
            except Exception as e:
                raise RuntimeError(f"Error processing file {lp}: {e}")
        else:
            labels.append([])  # No objects
    return img_paths, labels


class DetectDataset:

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
        path = self.img_paths[index]
        img = cv2.imread(path)  # BGR
        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {path}")

        img_h, img_w = img.shape[:2]

        # Labels: list of [cls, cx, cy, w, h] (normalized)
        # labels_per_img = self.labels[index]
        #
        # Convert normalized bbox to absolute pixel coordinates [x1, y1, x2, y2]
        # bboxes_abs = []
        # classes = []
        # for label in labels_per_img:
        #     cls_id, cx, cy, nw, nh = label
        #     cx_abs = cx * w
        #     cy_abs = cy * h
        #     w_abs = nw * w
        #     h_abs = nh * h
        #
        #     x1 = cx_abs - w_abs / 2
        #     y1 = cy_abs - h_abs / 2
        #     x2 = cx_abs + w_abs / 2
        #     y2 = cy_abs + h_abs / 2
        #
        #     bboxes_abs.append([x1, y1, x2, y2])
        #     classes.append(int(cls_id))

        # Return image, path, bounding boxes (xyxy), and class IDs
        return img, path, img_h, img_w

    def get_anno_json(self, anno_json_path):
        """
        Generate COCO-style annotation JSON for detection (no masks, only bbox).
        """
        from datetime import datetime

        coco_dict = {
            "info": {
                "description": "YOLO-format Detection Dataset converted to COCO",
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
            if labels_per_img:
                cls_ids = [int(label[0]) for label in labels_per_img]
                all_cls_ids.update(cls_ids)
        all_cls_ids = sorted(all_cls_ids)
        print(f"all_cls_ids = {all_cls_ids}")
        coco_dict["categories"] = [
            {"id": cid + 1, "name": self.class_list[cid], "supercategory": "object"} for cid in all_cls_ids
        ]

        ann_id = 1
        for img_id, (img_path, labels) in enumerate(zip(self.img_paths, self.labels), start=1):
            img_pil = Image.open(img_path)
            width, height = img_pil.size
            file_name = osp.basename(img_path)
            image_id = int(Path(img_path).stem) if Path(img_path).stem.isdigit() else img_id

            coco_dict["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": file_name,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })

            if not labels:
                continue

            for label in labels:
                cls_id, cx, cy, nw, nh = label
                # Convert normalized to absolute
                w_abs = nw * width
                h_abs = nh * height
                x_min = (cx - nw / 2) * width
                y_min = (cy - nh / 2) * height

                # COCO bbox format: [x, y, width, height]
                bbox = [round(x_min), round(y_min), round(w_abs), round(h_abs)]
                area = w_abs * h_abs

                coco_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []  # Empty for detection
                })
                ann_id += 1

        os.makedirs(osp.dirname(anno_json_path), exist_ok=True)
        with open(anno_json_path, 'w') as f:
            json.dump(coco_dict, f, indent=2)

        return coco_dict
