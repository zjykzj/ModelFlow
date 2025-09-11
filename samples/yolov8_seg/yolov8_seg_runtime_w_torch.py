# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/30 15:57
@File    : yolov8_runtime_w_torch.py
@Author  : zj
@Description: 
"""
from typing import Union, Tuple, Optional, Any, List

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from core.backends.backend_runtime import BackendRuntime
from core.utils.general import Profile
from core.utils.v8.preprocessor import LetterBox
from samples.yolov8.torch_util import non_max_suppression, scale_boxes
from samples.yolov8_seg.torch_util import process_mask, scale_image


def pre_transform(im, imgsz: Union[int, Tuple] = 640, stride: int = 32, auto: bool = False):
    """
    Apply preprocessing transformation (e.g., letterbox resizing) to a list of input images before model inference.

    This function ensures all images are resized to the target size while preserving aspect ratio,
    padding with gray values if necessary (letterboxing). It's typically used when inputs are raw images.

    Args:
        im (List[np.ndarray]): List of input images. Each image is in HWC (height, width, channels) format, BGR color space.
                               Shape: [(H, W, 3)] * N, where N is the number of images.
        imgsz (Union[int, Tuple]): Target input size for the model. If int, a square image (imgsz x imgsz) is assumed.
                                   Can be a tuple (h, w) for rectangular input.
        stride (int): Model's stride (e.g., 32 for YOLOv8). Output shape must be divisible by stride.
                      Used to adjust padding in letterbox.
        auto (bool): If True, enables dynamic padding based on image shape. When combined with `same_shapes=True`,
                     disables padding entirely (i.e., uses simple resize).

    Returns:
        (list): A list of preprocessed NumPy arrays (in HWC format), resized and padded as needed.
                Each element has shape (resized_h, resized_w, 3).
    """
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(imgsz, auto=same_shapes and auto, stride=stride)
    return [letterbox(image=x) for x in im]


def preprocess(im: ndarray, imgsz: Union[int, Tuple] = 640, stride: int = 32, auto: bool = False,
               fp16: bool = False) -> Any:
    """
      Sourced from https://github.com/ultralytics/ultralytics/blob/25307552100e4c03c8fec7b0f7286b4244018e15/ultralytics/engine/predictor.py#L115

      Preprocess input image(s) for inference with the YOLOv8 model.

      This function handles both NumPy arrays and PyTorch tensors. For NumPy inputs, it performs:
      - Batch dimension expansion (if missing)
      - Letterbox resizing
      - Color space conversion (BGR -> RGB)
      - Channel ordering (HWC -> CHW)
      - Contiguity enforcement
      - Normalization (0-255 -> 0.0-1.0)
      - Data type conversion (uint8 -> float32 or float16)

      For tensor inputs, only normalization and type conversion are applied.

      Args:
          im (torch.Tensor | List[np.ndarray]): Input image(s).
              - If tensor: Expected in BCHW format (batch, channels, height, width), uint8.
              - If list of arrays: List of HWC images (BGR), no batch dim; will be stacked.
          imgsz (Union[int, Tuple]): Target size for resizing (used only if input is list/array).
          stride (int): Downsample stride of the model (used in letterbox padding alignment).
          auto (bool): Whether to enable auto-padding based on image shape (see `LetterBox`).
          fp16 (bool): If True, convert image tensor to half precision (float16). Otherwise, float32.

      Returns:
          (np.ndarray): Preprocessed image as a NumPy array in BCHW format, normalized to [0, 1],
                        with dtype float16 (if fp16=True) or float32.
                        Shape: (N, 3, resized_h, resized_w)
      """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im = np.stack(pre_transform(im, imgsz, stride=stride, auto=auto))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im
    im = im.half() if fp16 else im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im.numpy()


def postprocess(
        pred: Union[Tensor, List[Tensor]],
        im_shape: Tuple,  # (h, w) of input to model
        im0_shape: Tuple,  # (h, w) of original image
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[list] = None,
        agnostic: bool = False,
        max_det: int = 300,
        nc: int = 0,  # number of classes (optional)
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
       Post-process model predictions for segmentation models (e.g., YOLOv8-seg).

       Applies Non-Max Suppression (NMS) to detections, scales bounding boxes and segmentation masks
       to the original image dimensions, and separates outputs into boxes, confidences, class IDs, and masks.

       Args:
           pred (Union[Tensor, List[Tensor]]): Raw model output, typically a tuple of (detections, proto).
               - detections: tensor of shape (batch, num_boxes, 4 + 1 + num_classes + mask_coefficients)
               - proto: tensor of shape (batch, num_masks, mask_h, mask_w) for mask prototype generation.
           im_shape (Tuple): Shape of the input image to the model (after resizing/padding), as (height, width).
           im0_shape (Tuple): Original shape of the image (before any preprocessing), as (height, width).
           conf (float): Confidence threshold for filtering detections before NMS.
           iou (float): IoU threshold for NMS to suppress overlapping boxes.
           classes (Optional[list]): List of class indices to keep. If None, all classes are kept.
           agnostic (bool): Whether to perform class-agnostic NMS (merge boxes across classes).
           max_det (int): Maximum number of detections to keep per image.
           nc (int, optional): Number of classes in the model. Used to determine where mask coefficients start.

       Returns:
           Tuple[ndarray, ndarray, ndarray, ndarray]: A tuple containing:
               - boxes (ndarray): Scaled bounding boxes in xyxy format, shape (N, 4), relative to original image.
                 Values are rounded integers. Empty array of shape (0, 4) if no detections.
               - confs (ndarray): Confidence scores for each detection, shape (N, 1). Empty array of shape (0, 1) if no detections.
               - cls_ids (ndarray): Predicted class IDs (integer), shape (N, 1). Empty array of shape (0, 1) if no detections.
               - masks (ndarray): Segmentation masks for each detection, shape (N, H, W), where H and W match im0_shape.
                 Masks are binary (0.0 or 1.0) or soft masks in [0,1], resized to original image size.
                 Empty array of shape (0, 1, 1) if no detections (maintains 3D structure).
       """
    proto = pred[1][-1] if isinstance(pred[1], tuple) else pred[1]  # tuple if PyTorch model or array if exported
    proto = proto[0]  # [1, 32, 160, 160] -> [32, 160, 160]

    pred = non_max_suppression(
        pred[0],
        conf,
        iou,
        classes=classes,
        agnostic=agnostic,
        max_det=max_det,
        nc=nc,
    )
    pred = pred[0]  # [1, 300, 6] -> [300, 6]

    if len(pred) > 0:
        masks = process_mask(proto, pred[:, 6:], pred[:, :4], im_shape, upsample=True)  # HWC

        masks = scale_image(masks.permute(1, 2, 0).numpy(), im0_shape)
        masks = np.transpose(masks, (2, 0, 1))

        boxes = scale_boxes(im_shape, pred[:, :4], im0_shape).round().cpu().numpy()
        confs = pred[:, 4:5].cpu().numpy()
        cls_ids = pred[:, 5:6].cpu().numpy()
    else:
        # ✅ 返回二维空数组，保持 shape 一致性
        boxes = np.zeros((0, 4), dtype=np.float32)
        confs = np.zeros((0, 1), dtype=np.float32)
        cls_ids = np.zeros((0, 1), dtype=np.float32)
        masks = np.zeros((0, 1, 1), dtype=np.float32)
    return boxes, confs, cls_ids, masks


class YOLOv8RuntimeTorch:

    def __init__(self, classes: List[str], weight: str = 'yolov8s-seg.onnx', providers: List[str] = None):
        super().__init__()
        self.classes = classes
        self.nc = len(classes)

        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = BackendRuntime(weight, providers=providers)
        self.session.load()

        self.input_name = self.session.get_input_names()[0]
        self.net_h, self.net_w = self.session.get_input_shapes()[self.input_name][2:]
        self.output_names = self.session.output_names

        self.device = torch.device('cpu')

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def infer(self, im: ndarray) -> List[Any]:
        output_dict = self.session.infer({self.input_name: im})

        pred = []
        for output_name in self.output_names:
            pred.append(self.from_numpy(output_dict[output_name]))
        return pred

    def detect(self, im0: ndarray, conf: float = 0.25, iou: float = 0.45) -> Tuple[
        ndarray, ndarray, ndarray, ndarray, Tuple]:
        """
        Detect objects in the image and measure time consumption for each stage.
        Returns:
            boxes, confs, cls_ids, masks, dt
        """
        # Record start time
        dt = (Profile(), Profile(), Profile())

        # --- Preprocessing ---
        with dt[0]:
            im = preprocess(im0, (self.net_h, self.net_w))
            im_shape = im.shape[2:]  # Model input shape (h, w)
            im0_shape = im0.shape[:2]  # Original image shape (h, w)

        # --- Inference ---
        with dt[1]:
            pred = self.infer(im)

        # --- Postprocessing ---
        with dt[2]:
            boxes, confs, cls_ids, masks = postprocess(
                pred,
                im_shape,
                im0_shape,
                conf=conf,
                iou=iou,
                nc=self.nc,
            )
        return boxes, confs, cls_ids, masks, dt
