# C++ 推理模块规格

> **Version:** 0.1
> **Status:** Draft
> **Dependencies:** `spec_architecture.md`（Pipeline 模式、独立模块设计原则）

## 1. 模块定位

`cpp/` 是一个**独立模块**，不依赖 `modelflow/` 或 `export/` 的代码。提供基于 OpenCV + ONNX Runtime / TensorRT 的 C++ 推理 Pipeline，与 Python 实现共享相同的 Pipeline 语义。

**关键设计原则**：

- **每个后端独立可提取**：`backends/onnx/` 和 `backends/tensorrt/` 各有独立 CMakeLists.txt
- **核心库共享**：`core/` 提供 OpenCV 图像处理 + 通用后处理，被两后端共用
- **CPU 部署场景**：只需复制 `core/` + `backends/onnx/` + 自写顶层 CMakeLists.txt，零 TensorRT 依赖
- **GPU 部署场景**：只需复制 `core/` + `backends/tensorrt/` + 自写顶层 CMakeLists.txt
- **Pipeline 语义与 Python 一致**：同一张图、同一个模型，输出必须精度对齐

## 2. 目录结构

```
cpp/
├── CMakeLists.txt                  # 顶层构建（选项式包含各子目录）
├── README.md
│
├── core/                           # 核心库（可独立使用）
│   ├── CMakeLists.txt              # 独立子项目
│   ├── types.h                     # 数据类型
│   ├── preprocess.h                # OpenCV 预处理函数
│   └── postprocess.h               # 通用后处理函数
│
├── backends/
│   ├── onnx/                       # ONNX Runtime 后端（可独立提取）
│   │   ├── CMakeLists.txt          # 独立子项目
│   │   ├── onnx_backend.h          # OnnxBackend 类
│   │   └── onnx_backend.cpp
│   │
│   └── tensorrt/                   # TensorRT 后端（可独立提取）
│       ├── CMakeLists.txt          # 独立子项目
│       ├── tensorrt_backend.h      # TensorrtBackend 类
│       └── tensorrt_backend.cpp
│
├── pipelines/                      # Pipeline 组合
│   ├── include/                    # 公共头文件
│   └── src/                        # 各 Pipeline 实现
│
├── examples/                       # 可执行示例
│   └── .../*.cpp
│
└── tests/                          # 测试
    └── .../*.cpp
```

## 3. 核心接口

### 3.1 数据类型 (`core/types.h`)

```cpp
// 模型输入输出信息
struct ModelInfo {
    std::string name;
    std::vector<int64_t> shape;
    int dtype;  // 对应 nvinfer1::DataType 或 ONNX TensorType
};

// 检测结果
struct Detection {
    cv::Rect2f bbox;      // xyxy 格式
    float confidence;
    int class_id;
};

// 分割结果
struct SegmentResult : Detection {
    cv::Mat mask;          // 二值掩码
};
```

### 3.2 预处理 (`core/preprocess.h`)

```cpp
namespace preprocess {

// letterbox：保持长宽比的 resize + padding
cv::Mat letterbox(const cv::Mat& src, int target_w, int target_h,
                  int stride = 32, const cv::Scalar& color = {114, 114, 114});

// 直接 resize
cv::Mat resize(const cv::Mat& src, int w, int h);

// 中心裁剪
cv::Mat center_crop(const cv::Mat& src, int size);

// BGR → RGB + HWC → CHW + float32 + normalize
std::vector<float> hwc_to_chw(const cv::Mat& src, bool bgr_to_rgb = true,
                              float scale = 1.0f / 255.0f);

// ImageNet normalize
void imagenet_normalize(float* data, int n, const float* mean, const float* std);

}  // namespace preprocess
```

### 3.3 后处理 (`core/postprocess.h`)

```cpp
namespace postprocess {

// softmax
void softmax(const float* input, float* output, int num_classes);

// NMS（非极大值抑制）
std::vector<int> nms(const std::vector<cv::Rect2f>& boxes,
                     const std::vector<float>& scores,
                     float iou_threshold);

// scale_boxes：将模型输出的框缩放到原始图像尺寸
void scale_boxes(const std::vector<cv::Rect2f>& boxes,
                 std::vector<cv::Rect2f>& output,
                 const cv::Size& model_size, const cv::Size& image_size);

}  // namespace postprocess
```

### 3.4 推理后端

#### ONNX Runtime (`backends/onnx/onnx_backend.h`)

```cpp
class OnnxBackend {
public:
    OnnxBackend(const std::string& model_path);
    ~OnnxBackend();

    // 同步推理，输入输出均为 float 向量
    std::vector<std::vector<float>> infer(const std::vector<float>& input,
                                           const std::vector<int64_t>& input_shape);

    ModelInfo get_input_info() const;
    std::vector<ModelInfo> get_output_info() const;
    void warmup();

private:
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};
```

#### TensorRT (`backends/tensorrt/tensorrt_backend.h`)

```cpp
class TensorrtBackend {
public:
    TensorrtBackend(const std::string& engine_path, int device_id = 0);
    ~TensorrtBackend();

    std::vector<std::vector<float>> infer(const std::vector<float>& input,
                                           const std::vector<int64_t>& input_shape);

    ModelInfo get_input_info() const;
    std::vector<ModelInfo> get_output_info() const;
    void warmup();

private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    // CUDA buffer 管理
};
```

### 3.5 Pipeline

```cpp
// pipelines/include/detect_pipeline.h
class DetectPipeline {
public:
    DetectPipeline(std::unique_ptr<OnnxBackend> backend,
                   int input_size = 640, float conf_thres = 0.25f, float iou_thres = 0.45f);

    std::vector<Detection> run(const cv::Mat& image);

private:
    std::unique_ptr<OnnxBackend> backend_;
    int input_size_;
    float conf_thres_;
    float iou_thres_;
};
```

## 4. 提取性设计

### CPU 场景：只复制 ONNX 后端

```
my_cpu_project/
├── CMakeLists.txt                  # find_package(OpenCV) + find_package(onnxruntime)
├── core/
│   ├── types.h                     # 从 cpp/core/ 复制
│   ├── preprocess.h                # 从 cpp/core/ 复制
│   └── postprocess.h               # 从 cpp/core/ 复制
├── backends/onnx/
│   ├── onnx_backend.h              # 从 cpp/backends/onnx/ 复制
│   └── onnx_backend.cpp            # 从 cpp/backends/onnx/ 复制
└── main.cpp
```

**零 TensorRT 头文件/库/链接配置**。

### GPU 场景：只复制 TensorRT 后端

```
my_gpu_project/
├── CMakeLists.txt                  # find_package(OpenCV) + find_package(TensorRT) + find_package(CUDA)
├── core/                           # 同上
├── backends/tensorrt/
│   ├── tensorrt_backend.h          # 从 cpp/backends/tensorrt/ 复制
│   └── tensorrt_backend.cpp        # 从 cpp/backends/tensorrt/ 复制
└── main.cpp
```

## 5. Pipeline 示例

### 分类

```cpp
#include "core/preprocess.h"
#include "core/postprocess.h"
#include "backends/onnx/onnx_backend.h"

int main(int argc, char** argv) {
    // 1. 加载模型
    OnnxBackend backend(argv[1]);

    // 2. 读取图像
    cv::Mat image = cv::imread(argv[2]);

    // 3. 预处理
    cv::Mat resized = preprocess::resize(image, 224, 224);
    std::vector<float> tensor = preprocess::hwc_to_chw(resized, true, 1.0f/255.0f);
    preprocess::imagenet_normalize(tensor.data(), 224*224,
        new float[]{0.485f, 0.456f, 0.406f},
        new float[]{0.229f, 0.224f, 0.225f});

    // 4. 推理
    auto outputs = backend.infer(tensor, {1, 3, 224, 224});

    // 5. 后处理
    std::vector<float> probs(1000);
    postprocess::softmax(outputs[0].data(), probs.data(), 1000);
    int top1 = std::max_element(probs.begin(), probs.end()) - probs.begin();

    std::cout << "Top-1: " << top1 << " (" << probs[top1] << ")" << std::endl;
    return 0;
}
```

### 检测

```cpp
#include "pipelines/include/detect_pipeline.h"

int main(int argc, char** argv) {
    auto backend = std::make_unique<OnnxBackend>(argv[1]);
    DetectPipeline pipeline(std::move(backend), 640, 0.25f, 0.45f);

    cv::Mat image = cv::imread(argv[2]);
    auto detections = pipeline.run(image);

    for (const auto& det : detections) {
        cv::rectangle(image, det.bbox, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("output.jpg", image);
    return 0;
}
```

## 6. Python ↔ C++ 精度对齐

```python
# tests/test_cpp_alignment.py
class TestCppAlignment:
    def test_preprocess_alignment(self):
        """同一张图片，Python preprocessor vs C++ preprocessor 输出一致"""
        image = cv2.imread("assets/bus.jpg")
        py_tensor = python_preprocessor(image)
        cpp_tensor = run_cpp_preprocessor(image)  # 调用编译好的 C++ 可执行文件
        np.testing.assert_allclose(py_tensor, cpp_tensor, rtol=1e-5)

    def test_pipeline_alignment(self):
        """同一模型 + 同一图片，Python Pipeline vs C++ Pipeline 输出一致"""
        py_result = python_pipeline(image)
        cpp_result = run_cpp_pipeline(image)
        np.testing.assert_allclose(py_result.boxes, cpp_result.boxes, rtol=1e-5)
        np.testing.assert_allclose(py_result.scores, cpp_result.scores, rtol=1e-5)
```
