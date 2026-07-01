# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : profile.py
@Author  : zj
@Description: 推理延迟测量 — 纯backend / 完整管线 / 真实数据

用法:
    from utils.profile import measure_inference_latency, measure_pipeline_latency

    # 仅 backend 推理延迟（dummy input，跳过 pre/post）
    stats = measure_inference_latency(pipeline, input_shape=(1, 3, 640, 640))

    # 完整管线延迟（preprocessor + backend + postprocessor）
    stats = measure_pipeline_latency(pipeline, image)

    # 从数据集随机采样测量
    stats = measure_pipeline_latency_from_dataset(pipeline, dataset, iters=50)
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def _to_stats(times_ms: np.ndarray) -> dict:
    """将原始时间列表转换为统计量（保留 mean 和 p95 两个核心指标）。"""
    return {
        "mean_ms": round(float(times_ms.mean()), 3),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 3),
    }


def measure_inference_latency(
    pipeline,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    dtype: np.dtype = np.float32,
    warmup: int = 10,
    iters: int = 100,
) -> dict:
    """测量纯 backend 推理延迟（跳过 preprocessor 和 postprocessor）。

    用固定大小的 dummy 输入直接调用 pipeline.infer()，
    适合比较不同后端的纯推理吞吐。

    Args:
        pipeline: InferencePipeline 实例
        input_shape: 输入张量 shape (N, C, H, W)
        dtype: 输入数据类型
        warmup: 预热迭代次数
        iters: 测量迭代次数

    Returns:
        dict: {
            mean_ms, std_ms, min_ms, max_ms, p50_ms, p95_ms,
            fps, iters, warmup, input_shape, mode: "backend_only"
        }
    """
    dummy = np.random.randn(*input_shape).astype(dtype)

    # warmup
    for _ in range(warmup):
        pipeline.infer(dummy)

    # measure
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        pipeline.infer(dummy)
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000

    return {
        **_to_stats(times_ms),
        "fps": round(float(1000 / times_ms.mean()), 1),
        "iters": iters,
        "warmup": warmup,
        "input_shape": list(input_shape),
        "mode": "backend_only",
    }


def measure_pipeline_latency(
    pipeline,
    image: np.ndarray,
    warmup: int = 10,
    iters: int = 100,
    **postproc_kwargs,
) -> dict:
    """测量完整管线延迟：preprocess → inference → postprocess 三段。

    使用真实图片，计时包含每阶段的 perf_counter() 开销。

    Args:
        pipeline: InferencePipeline 实例
        image: HWC BGR 图像 (uint8 ndarray)
        warmup: 预热迭代次数
        iters: 测量迭代次数
        **postproc_kwargs: 传给后处理的参数（conf_thres, iou_thres 等）

    Returns:
        dict: {
            preprocess:   {mean_ms, std_ms, min_ms, max_ms},
            inference:    {mean_ms, std_ms, min_ms, max_ms},
            postprocess:  {mean_ms, std_ms, min_ms, max_ms},
            total:        {mean_ms, std_ms, min_ms, max_ms},
            fps, iters, warmup, image_shape, mode: "full_pipeline"
        }
    """
    original_shape = image.shape[:2]

    # warmup
    for _ in range(warmup):
        pipeline(image, **postproc_kwargs)

    # measure
    pre_times, inf_times, post_times, tot_times = [], [], [], []

    for _ in range(iters):
        # preprocess
        t0 = time.perf_counter()
        tensor = pipeline.preprocessor(image)
        t1 = time.perf_counter()

        # inference
        raw = pipeline.backend(tensor)
        t2 = time.perf_counter()

        # postprocess
        kwargs = dict(postproc_kwargs)
        kwargs.setdefault("original_shape", original_shape)
        pipeline.postprocessor(raw, **kwargs)
        t3 = time.perf_counter()

        pre_times.append(t1 - t0)
        inf_times.append(t2 - t1)
        post_times.append(t3 - t2)
        tot_times.append(t3 - t0)

    def _ms(arr):
        return [x * 1000 for x in arr]

    return {
        "preprocess": _to_stats(np.array(_ms(pre_times))),
        "inference": _to_stats(np.array(_ms(inf_times))),
        "postprocess": _to_stats(np.array(_ms(post_times))),
        "total": _to_stats(np.array(_ms(tot_times))),
        "fps": round(float(1000 / (np.array(_ms(tot_times)).mean())), 1),
        "iters": iters,
        "warmup": warmup,
        "image_shape": list(image.shape),
        "mode": "full_pipeline",
    }


def measure_pipeline_latency_from_dataset(
    pipeline,
    dataset,
    warmup: int = 10,
    iters: int = 100,
    **postproc_kwargs,
) -> dict:
    """从数据集中随机采样图片测量完整管线延迟。

    每次迭代随机取一张图，适合测量真实数据分布下的推理速度。

    Args:
        pipeline: InferencePipeline 实例
        dataset: BaseDataset 实例（支持 __len__ 和 __getitem__）
        warmup: 预热迭代次数
        iters: 测量迭代次数
        **postproc_kwargs: 传给后处理的参数

    Returns:
        dict: 同 measure_pipeline_latency，额外含 dataset_size 字段
    """
    ds_size = len(dataset)
    actual_iters = min(iters, ds_size)

    # 随机采样索引
    indices = np.random.default_rng().choice(ds_size, size=actual_iters, replace=False)

    pre_times, inf_times, post_times, tot_times = [], [], [], []

    for idx in indices.tolist():
        image, _ = dataset[idx]
        original_shape = image.shape[:2]

        # 预热只在第一张图上执行（避免多次 warmup 浪费）
        if len(pre_times) < warmup:
            for _ in range(warmup - len(pre_times)):
                _ = pipeline(image, **postproc_kwargs)
            # warmup 后不在 timing 结果中记录
            continue

        t0 = time.perf_counter()
        tensor = pipeline.preprocessor(image)
        t1 = time.perf_counter()

        raw = pipeline.backend(tensor)
        t2 = time.perf_counter()

        kwargs = dict(postproc_kwargs)
        kwargs.setdefault("original_shape", original_shape)
        pipeline.postprocessor(raw, **kwargs)
        t3 = time.perf_counter()

        pre_times.append(t1 - t0)
        inf_times.append(t2 - t1)
        post_times.append(t3 - t2)
        tot_times.append(t3 - t0)

    if not pre_times:
        # dataset 太小，连 warmup 都不够
        return {"error": f"Dataset too small (size={ds_size}) for warmup={warmup}"}

    def _ms(arr):
        return [x * 1000 for x in arr]

    return {
        "preprocess": _to_stats(np.array(_ms(pre_times))),
        "inference": _to_stats(np.array(_ms(inf_times))),
        "postprocess": _to_stats(np.array(_ms(post_times))),
        "total": _to_stats(np.array(_ms(tot_times))),
        "fps": round(float(1000 / (np.array(_ms(tot_times)).mean())), 1),
        "iters": actual_iters - warmup,
        "warmup": warmup,
        "dataset_size": ds_size,
        "mode": "dataset_sample",
    }


# ==================== Formatted Output ====================


def _fmt_val(v, precision=2):
    """Format a numeric value for display."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    if isinstance(v, int) and v > 1_000_000:
        return f"{v:,}"
    return str(v)


def _fmt_size(n: Optional[int]) -> str:
    """Format parameter count in human readable form."""
    if n is None:
        return "N/A"
    if n >= 1e9:
        return f"{n/1e9:.2f} G"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{n/1e3:.1f} K"
    return str(n)


def _fmt_flops(n: Optional[int]) -> str:
    """Format FLOPs in human readable form."""
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"{n/1e12:.2f} T"
    if n >= 1e9:
        return f"{n/1e9:.2f} G"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    return str(n)


def _print_section(title: str):
    """Print a section header."""
    width = 58
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _print_kv(key: str, value: str, indent: int = 2):
    """Print a key-value pair with aligned values."""
    prefix = " " * indent
    print(f"{prefix}{key:<20} {value}")


def print_summary(model_info: dict, latency: dict, accuracy: dict,
                  task: str, dataset_name: str = "",
                  eval_time_s: float = 0.0, dataset_size: int = 0) -> None:
    """打印格式化的评估汇总输出。

    Args:
        model_info: get_model_info() 的返回值
        latency: measure_*_latency() 的返回值
        accuracy: evaluator.run() 的返回值
        task: 任务类型
        dataset_name: 数据集名称
        eval_time_s: 评估总耗时（秒）
        dataset_size: 数据集大小
    """
    # --- Model Info ---
    _print_section("Model Info")
    _print_kv("Model path", model_info.get("model_path", "N/A"))
    _print_kv("Backend", model_info.get("backend", "N/A"))
    _print_kv("File size", f"{model_info.get('file_size_mb', 'N/A')} MB"
              if model_info.get("file_size_mb") else "N/A")
    _print_kv("Parameters", _fmt_size(model_info.get("param_count")))
    _print_kv("FLOPs", _fmt_flops(model_info.get("flops")))
    shapes = model_info.get("input_shapes", [])
    _print_kv("Input shape", str(shapes[0]) if shapes else "N/A")
    shapes = model_info.get("output_shapes", [])
    _print_kv("Output shape", str(shapes[0]) if shapes else "N/A")

    # --- Latency ---
    _print_section("Latency")
    mode = latency.get("mode", "")
    _print_kv("Mode", mode.replace("_", " "))

    if mode in ("full_pipeline", "dataset_sample"):
        for stage in ("preprocess", "inference", "postprocess", "total"):
            s = latency.get(stage, {})
            if s:
                _print_kv(
                    f"{stage.capitalize()} (ms)",
                    f"mean={s['mean_ms']:.2f}  p95={s['p95_ms']:.2f}"
                )
    elif mode == "backend_only":
        _print_kv("Inference (ms)",
                  f"mean={latency.get('mean_ms', 0):.2f}  "
                  f"p95={latency.get('p95_ms', 0):.2f}")

    _print_kv("Throughput", f"{latency.get('fps', 'N/A')} FPS")
    _print_kv("Iterations",
              f"{latency.get('iters', 'N/A')} "
              f"(warmup={latency.get('warmup', 'N/A')})")

    # --- Accuracy ---
    _print_section(f"Accuracy{f' ({dataset_name})' if dataset_name else ''}")
    if accuracy:
        # 按 key 排序，mAP 类指标优先
        def _sort_key(k):
            k_lower = k.lower()
            if k_lower.startswith("m"):
                return (0, k_lower)
            if k_lower.startswith("ap"):
                return (1, k_lower)
            if k_lower.startswith("ar"):
                return (2, k_lower)
            if k_lower in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
                return (3, k_lower)
            return (4, k_lower)

        for k in sorted(accuracy.keys(), key=_sort_key):
            v = accuracy[k]
            if isinstance(v, float):
                _print_kv(k, f"{v:.4f}")
            else:
                _print_kv(k, str(v))
    else:
        _print_kv("(no metrics)", "")

    # --- Footer ---
    footer_parts = []
    if eval_time_s > 0:
        footer_parts.append(f"Eval time: {eval_time_s:.1f}s")
    if dataset_size > 0:
        footer_parts.append(f"{dataset_size} images")
    if footer_parts:
        print(f"{'─' * 58}")
        print(f"  {' | '.join(footer_parts)}")
    print()
