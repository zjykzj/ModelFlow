# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/26 21:33
@File    : safe_int8_build_by_pycuda.py
@Author  : zj
@Description:

### 📜 脚本说明：PyCUDA 版 INT8 量化构建器

**适用场景**：NVIDIA Jetson (Orin/Xavier/Nano)、嵌入式设备、Docker 精简镜像、纯推理环境。
**核心优势**：极致轻量，无重型框架依赖，直接绑定系统 CUDA 驱动，稳定性最高。

#### ✨ 功能特性
- **🛡️ 数据自愈**：自动检测并修复校准数据中的 `NaN`/`Inf`，跳过尺寸不匹配文件，防止构建崩溃。
- **⚡ 零拷贝优化**：使用 `pagelocked_empty` (锁页内存) 加速 CPU 到 GPU 的数据传输。
- **🎯 混合精度**：默认开启 `INT8` + `FP16`，确保最佳性能与精度平衡。
- **🔧 全参数化**：支持命令行动态配置模型路径、校准目录、输入形状及显存限制。

#### ⚖️ 与 PyTorch 版对比
| 特性 | 本版本 (PyCUDA) | PyTorch 版 |
| :--- | :--- | :--- |
| **依赖** | `pycuda`, `tensorrt` (轻量) | `torch`, `tensorrt` (重型) |
| **内存占用** | **极低** (<100MB 额外开销) | 高 (>2GB 额外开销) |
| **推荐设备** | **Jetson**, 边缘盒子 | RTX 服务器, 开发机 |

#### 💡 使用示例
# 基础用法 (YOLOv5/v8 检测)
python3 safe_int8_build.py \
    --onnx yolov5s.onnx \
    --calib_dir ./calib_data \
    --output yolov5s_int8.engine \
    --input_shape 1 3 640 640

# 分割模型 (YOLOv8-seg)
python3 safe_int8_build.py \
    --onnx yolov8n-seg.onnx \
    --calib_dir ./calib_seg \
    --output yolov8n_seg_int8.engine \
    --input_shape 1 3 640 640 \
    --workspace 4


# ==========================================
# 🧠 分类模型专用示例 (ImageNet 风格)
# ==========================================

# 1. 标准分类模型 (ResNet50, EfficientNet 等)
# 输入通常为 224x224 或 256x256
python3 safe_int8_build.py \
    --onnx resnet50.onnx \
    --calib_dir ./calib_imagenet_224 \
    --output resnet50_int8.engine \
    --input_shape 1 3 224 224

# 2. 轻量级分类模型 (MobileNetV3, ShuffleNet)
python3 safe_int8_build.py \
    --onnx mobilenet_v3.onnx \
    --calib_dir ./calib_mobilenet \
    --output mobilenet_v3_int8.engine \
    --input_shape 1 3 224 224

# 3. 高分辨率分类模型 (ViT, ConvNeXt 等)
# 输入可能为 384x384 或更大
python3 safe_int8_build.py \
    --onnx vit_base.onnx \
    --calib_dir ./calib_vit_384 \
    --output vit_base_int8.engine \
    --input_shape 1 3 384 384 \
    --workspace 8

"""

import os
import sys
import argparse
import numpy as np

import tensorrt as trt

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("❌ 错误：未找到 pycuda 库。")
    print("💡 安装命令：pip install pycuda")
    print("💡 若环境已有 PyTorch，请改用 safe_int8_build_v2.py")
    sys.exit(1)

logger = trt.Logger(trt.Logger.INFO)


class SafeCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_data_dir, input_shape):
        super().__init__()
        self.calib_data_dir = calib_data_dir
        self.input_shape = input_shape

        if not os.path.isdir(calib_data_dir):
            raise FileNotFoundError(f"校准目录不存在：{calib_data_dir}")

        self.files = sorted([
            os.path.join(calib_data_dir, f)
            for f in os.listdir(calib_data_dir)
            if f.endswith('.bin')
        ])

        if not self.files:
            raise FileNotFoundError(f"未在 {calib_data_dir} 中找到任何 .bin 文件")

        print(f"📂 找到 {len(self.files)} 个校准文件。")
        self.idx = 0

        if len(input_shape) != 4:
            raise ValueError("Input shape 必须是 4 维: (N, C, H, W)")
        self.n, self.c, self.h, self.w = input_shape
        self.single_vol = self.c * self.h * self.w

        # 分配锁页内存
        self.host_input = cuda.pagelocked_empty(self.single_vol, dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)

        print(f"⚙️  预期单图体积：{self.single_vol} floats ({self.single_vol * 4 / 1024 / 1024:.2f} MB)")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        while self.idx < len(self.files):
            file_path = self.files[self.idx]
            self.idx += 1
            try:
                data = np.fromfile(file_path, dtype=np.float32)
                if data.size != self.single_vol:
                    print(f"⚠️  跳过 {os.path.basename(file_path)}: 大小不匹配")
                    continue
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    print(f"⚠️  修复 {os.path.basename(file_path)}: 发现 NaN/Inf")
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                np.copyto(self.host_input, data)
                cuda.memcpy_htod(self.device_input, self.host_input)

                if self.idx % 20 == 0:
                    print(f"   🔄 进度：{self.idx} / {len(self.files)}")
                return [int(self.device_input)]
            except Exception as e:
                print(f"❌ 读取失败：{e}")
                continue
        return None

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        with open("yolo_safe_calib.cache", "wb") as f:
            f.write(cache)
        print("✅ 校准缓存已保存。")


def build_engine(args):
    if not os.path.exists(args.onnx):
        print(f"❌ 找不到模型：{args.onnx}")
        return False

    print("=" * 70)
    print(f"🚀 [{args.device_name}] 开始 INT8 量化 (PyCUDA 安全模式)")
    print("=" * 70)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False
    print("✅ ONNX 解析成功。")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 30)

    try:
        calibrator = SafeCalibrator(args.calib_dir, tuple(args.input_shape))
        config.int8_calibrator = calibrator
    except Exception as e:
        print(f"❌ 校准器初始化失败：{e}")
        return False

    print("⏳ 正在构建引擎...")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"❌ 构建崩溃：{e}")
        return False

    if serialized_engine is None:
        print("❌ 构建失败。")
        return False

    with open(args.output, 'wb') as f:
        f.write(serialized_engine)

    print(f"🎉 成功！引擎已保存至：{args.output} ({os.path.getsize(args.output) / 1024 / 1024:.2f} MB)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe INT8 Builder (PyCUDA)")
    parser.add_argument("--onnx", type=str, required=True, help="输入 ONNX 路径")
    parser.add_argument("--calib_dir", type=str, required=True, help="校准数据目录 (.bin)")
    parser.add_argument("--output", type=str, default="model_int8.engine", help="输出引擎文件名")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 640, 640], help="N C H W")
    parser.add_argument("--workspace", type=int, default=4, help="工作空间 (GB)")
    parser.add_argument("--device_name", type=str, default="GPU", help="设备描述")

    args = parser.parse_args()
    sys.exit(0 if build_engine(args) else 1)
