# Data Module

> YAML-driven dataset loading with a unified factory — zero dependency on the rest of ModelFlow.

<br>

## 1. Overview

The `data/` module provides dataset loading and configuration for model evaluation. It is a **standalone module** — no imports from `modelflow/` or `export/`. External dependencies are limited to `numpy`, `cv2` (OpenCV), and `pyyaml`.

**Design principles:**

| Principle | Implementation |
|---|---|
| **YAML-driven** | Datasets are defined in config files (Ultralytics-style), not hardcoded |
| **Factory pattern** | `build_dataset()` resolves configs and returns the right dataset type |
| **ABC contract** | `BaseDataset` enforces `__len__`, `__getitem__`, `get_gt_json` |
| **Zero coupling** | No reference to `modelflow/` or `export/` modules |

<br>

## 2. Quick Start

### 2.1 Load by Config Name

```python
from data import build_dataset, get_class_names

# COCO detection
ds = build_dataset("coco", path="/data/coco/val2017")
class_names = get_class_names("coco")   # ["person", "bicycle", ...]

# ImageNet classification
ds = build_dataset("imagenet", path="/data/imagenet", val="val")
class_names = get_class_names("imagenet")

# COCO instance segmentation
ds = build_dataset("coco-seg", path="/data/coco", val="val2017")
```

### 2.2 Load from Custom YAML

```python
ds = build_dataset("./my_custom_dataset.yaml", path="/data/mydata")

# Override fields at call time
ds = build_dataset("coco", path="/data/coco128", val="images/train2017")
```

### 2.3 Direct Instantiation

```python
from data import ClassifyDataset, COCODataset

# Classification
ds = ClassifyDataset(root="val/", class_list=["cat", "dog", "bird"])

# Detection / segmentation
ds = COCODataset(
    img_dir="val2017/",
    class_list=["person", "car", ...],
    task="detect",          # or "segment"
    anno_json="annotations/instances_val2017.json",
)
```

### 2.4 Iterate

```python
image, gt = ds[0]
# image: np.ndarray, shape=(H, W, 3), dtype=uint8, BGR
# gt:    dict with dataset-specific keys
```

<br>

## 3. YAML Configuration

### 3.1 Config Format

```yaml
# my_dataset.yaml
name: my_dataset          # config name
task: detect              # classify | detect | segment
path: datasets/mydata     # root directory

# Sub-directories (relative to path)
val: val2017              # validation image directory
anno: annotations/instances_val2017.json   # ground truth JSON (evaluation)

# Class names (dict or list)
names:
  0: person
  1: car
  2: dog
```

### 3.2 Built-in Configs

| Config Name | Task | Classes | Path Default |
|---|---|---|---|
| `coco` | detect | COCO 80 | `datasets/coco` |
| `coco-seg` | segment | COCO 80 | `datasets/coco` |
| `imagenet` | classify | ImageNet-1K (20 sample) | `datasets/imagenet` |

> **Note:** `imagenet.yaml` includes only 20 representative classes. For the full 1000-class list, create a custom YAML or load names from `torchvision.datasets`.

### 3.3 Loading & Overriding

```python
from data import load_config

# Load without building
cfg = load_config("coco")
print(cfg["task"])   # "detect"

# build_dataset merges **overrides into the config
ds = build_dataset("coco", path="/alt/path", val="images/train")
```

Overrides are merged **after** YAML loading — call-time values take precedence.

<br>

## 4. Dataset Classes

### 4.1 `BaseDataset` (ABC)

All datasets implement three methods:

| Method | Returns | Purpose |
|---|---|---|
| `__len__()` | `int` | Number of samples |
| `__getitem__(idx)` | `(image: ndarray, gt: dict)` | Image in HWC BGR `uint8` + ground truth |
| `get_gt_json()` | `str` | Path to COCO JSON annotation, `""` if none |

```python
from data import BaseDataset

class MyDataset(BaseDataset):
    def __len__(self): ...
    def __getitem__(self, idx): ...
    def get_gt_json(self): return ""
```

### 4.2 `ClassifyDataset`

Classification dataset organized by directory structure:

```
root/
├── cat/
│   ├── img1.jpg
│   └── img2.jpg
├── dog/
│   └── img3.jpg
└── bird/
    └── img4.jpg
```

Each subdirectory name maps to a class index via `class_list`.

```python
ds = ClassifyDataset(root="imagenet/val/", class_list=["cat", "dog", "bird"])
image, gt = ds[0]
# gt = {"class_id": 0, "class_name": "cat", "image_path": "imagenet/val/cat/img1.jpg"}
```

### 4.3 `COCODataset`

Detection and instance segmentation on COCO-format image directories.

```python
ds = COCODataset(
    img_dir="coco/val2017/",
    class_list=["person", "bicycle", ...],   # COCO 80 classes
    task="detect",                            # or "segment"
    anno_json="coco/annotations/instances_val2017.json",
)

image, gt = ds[0]
# gt = {"image_path": "...", "image_id": "000000000139", "task": "detect"}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `img_dir` | `str` | required | Directory containing images |
| `class_list` | `List[str]` | required | Class name list indexed by class ID |
| `task` | `str` | `"detect"` | `"detect"` or `"segment"` |
| `anno_json` | `Optional[str]` | `None` | COCO JSON annotation path for evaluation |
| `img_size` | `int` | `640` | Target image size (reserved field) |

<br>

## 5. Factory API

### `build_dataset(config, task=None, **overrides) -> BaseDataset`

Build a dataset from a YAML config name or file path.

```python
from data import build_dataset

# By built-in name
ds = build_dataset("coco", path="/data/coco")

# From a file
ds = build_dataset("/path/to/custom.yaml")

# Override task type
ds = build_dataset("coco-seg", task="segment")
```

**Resolution order:** config name → `<config>.yaml` in `data/configs/` → user-provided YAML file path.

### `get_class_names(config) -> List[str]`

Extract class name list from a YAML config.

```python
from data import get_class_names

class_list = get_class_names("coco")
# ["person", "bicycle", "car", "motorcycle", ...]
```

Supports both `dict` (`{0: "name", 1: "name2"}`) and `list` (`["name", "name2"]`) formats for the `names` field.

### `load_config(config) -> dict`

Load and parse a YAML config without building a dataset instance.

```python
from data import load_config

cfg = load_config("coco")
# {"name": "coco", "task": "detect", "path": "datasets/coco", "val": "val2017", ...}
```

<br>

## 6. Package Structure

```
data/
├── __init__.py          # Public API exports
├── base.py              # BaseDataset ABC
├── classify.py          # ClassifyDataset
├── coco.py              # COCODataset (detect + segment)
├── build.py             # Factory: build_dataset, get_class_names, load_config
└── configs/             # Built-in YAML configs
    ├── coco.yaml        #   COCO detection (80 classes)
    ├── coco-seg.yaml    #   COCO segmentation (80 classes)
    └── imagenet.yaml    #   ImageNet-1K classification (20 sample classes)
```

<br>

## 7. Custom Dataset

Extend `BaseDataset` to add a new dataset type, then register it in `build.py`:

```python
# 1. Implement the ABC
from data import BaseDataset
import cv2, numpy as np

class VOCDataset(BaseDataset):
    def __init__(self, img_dir, class_list, anno_dir=None):
        self.img_dir = img_dir
        self.class_list = class_list
        self.images = sorted(os.listdir(img_dir))
        ...

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        gt = {"image_path": self.images[idx], ...}
        return img, gt

    def get_gt_json(self):
        return ""   # VOC has no COCO JSON

# 2. Add a branch in build_dataset (in build.py)
# elif task == "voc":
#     return VOCDataset(val_dir, class_list, ...)
```

