# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/30 15:57
@File    : yolov8_runtime_w_torch.py
@Author  : zj
@Description: 
"""

from typing import Union, Tuple, Optional, Any, List

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

from core.backends.backend_runtime import BackendRuntime
from core.utils.general import Profile
from core.utils.v8.preprocessor import LetterBox
from samples.yolov8.torch_util import non_max_suppression, scale_boxes
from samples.yolov8_seg.torch_util import process_mask


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


import cv2


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (np.ndarray): The masks that are being returned with shape [h, w, num].
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def plot_masks(masks, colors, im_gpu, im0_shape, alpha=0.5, retina_masks=False):
    """
    Plot masks on image.

    Args:
        masks (tensor): Predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
        im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
        alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
        retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
    """
    if len(masks) == 0:
        im = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
    if im_gpu.device != masks.device:
        im_gpu = im_gpu.to(masks.device)
    colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
    colors = colors[:, None, None]  # shape(n,1,1,3)
    masks = masks.unsqueeze(3)  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

    im_gpu = im_gpu.flip(dims=[0])  # flip channel
    im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
    im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
    im_mask = im_gpu * 255
    im_mask_np = im_mask.byte().cpu().numpy()
    im = im_mask_np if retina_masks else scale_image(im_mask_np, im0_shape)
    return im


def postprocess(
        im0,
        pred: Union[Tensor, List[Tensor]],
        im_shape: Tuple,  # (h, w) of input to model
        im0_shape: Tuple,  # (h, w) of original image
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[list] = None,
        agnostic: bool = False,
        max_det: int = 300,
        nc: int = 0,  # number of classes (optional)
) -> Tuple:
    """
    Post-process model predictions (detections) after inference.

    Applies Non-Max Suppression (NMS) to filter overlapping bounding boxes,
    scales detection boxes back to original image coordinates, and separates outputs.

    Args:
        pred (Union[Tensor, List[Tensor]]): Raw model output detections. Shape: (batch, num_boxes, 4 + 1 + num_classes)
        im_shape (Tuple): Shape of the image fed into the model (after preprocessing), as (height, width).
        im0_shape (Tuple): Original shape of the input image (before any preprocessing), as (height, width).
        conf (float): Confidence threshold for filtering detections.
        iou (float): IoU threshold for NMS.
        classes (Optional[list]): List of class indices to keep. If None, keep all classes.
        agnostic (bool): If True, perform NMS across all classes (class-agnostic).
        max_det (int): Maximum number of detections to keep per image.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.

    Returns:
        (Tuple): A tuple containing:
            - boxes (np.ndarray or []): Scaled bounding boxes in xyxy format, shape (N, 4), relative to original image.
            - confs (np.ndarray or []): Confidence scores for each kept detection, shape (N, 1).
            - cls_ids (np.ndarray or []): Predicted class IDs, shape (N, 1).
            If no detections, returns empty lists.
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

    masks = process_mask(proto, pred[:, 6:], pred[:, :4], im_shape, upsample=True)  # HWC

    img = LetterBox(im_shape)(image=im0)
    im_gpu = (
            torch.as_tensor(img, dtype=torch.float16, device=torch.device("cpu"))
            .permute(2, 0, 1)
            .flip(0)
            .contiguous()
            / 255
    )

    idx = reversed(range(len(masks)))
    from annotator import colors
    im = plot_masks(masks, colors=[colors(x, True) for x in idx], im_gpu=im_gpu, im0_shape=im0_shape)
    print(f"im shape: {im.shape} - im type: {type(im)}")
    cv2.imwrite("im_seg.jpg", im)

    if len(pred) > 0:
        boxes = scale_boxes(im_shape, pred[:, :4], im0_shape)
        confs = pred[:, 4:5]
        cls_ids = pred[:, 5:6]
    else:
        boxes, confs, cls_ids = [], [], []
    return boxes, confs, cls_ids


class YOLOv8RuntimeTorch:

    def __init__(self, classes: List[str], weight: str = 'yolov8s.onnx', providers: List[str] = None):
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

    def detect(self, im0: ndarray, conf: float = 0.25, iou: float = 0.45) -> Tuple:
        """
        Detect objects in the image and measure time consumption for each stage.
        Returns:
            boxes, confs, cls_ids, dt
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
            boxes, confs, cls_ids = postprocess(
                im0,
                pred,
                im_shape,
                im0_shape,
                conf=conf,
                iou=iou,
                nc=self.nc
            )

        return boxes, confs, cls_ids, dt
