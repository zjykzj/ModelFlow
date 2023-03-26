<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

 <div align="center"><a title="" href="git@github.com:zjykzj/onnx.git"><img align="center" src="./imgs/onnx.png"></a></div>

<p align="center">
  «onnx» implements the model deployment phase.
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

As a computer vision engineer, how to better apply image algorithms to landing scenes is crucial. In practice, C++ can provide a faster reasoning speed and a more practical deployment platform; In addition, Python can provide more convenient simulation and processing.

## Requirements

* [2.4.0 NNAPI后端/CUDA后端支持量化模型](https://github.com/alibaba/MNN/releases/tag/2.4.0)
* [ONNX Runtime v1.14.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.14.1)
* [OpenCV 4.7.0](https://github.com/opencv/opencv/releases/tag/4.7.0)
* [ultralytics/yolov5 v7.0 - YOLOv5 SOTA Realtime Instance Segmentation](https://github.com/ultralytics/yolov5/releases/tag/v7.0)

## Troubleshooting

```text
[ERROR:0@2.663] global onnx_importer.cpp:1051 handleNode DNN/ONNX: ERROR during processing node with 2 inputs and 3 outputs: [Split]:(onnx_node!/model.24/Split) from domain='ai.onnx'
```

* [global onnx_importer.cpp:1051 handleNode DNN/ONNX](https://github.com/opencv/opencv/issues/23227)
* [OPENCV部署ONNX模型报错 ERROR during processing node with 1 inputs and 1 outputs](https://ask.csdn.net/questions/7795689)

>In short, OpenCV 4.7.0 only supports ONNX models with fixed input sizes, and this issue will be resolved after the 5. X. X series

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [Open Neural Network Exchange](https://onnx.ai/)
* [pytorch/pytorch](https://github.com/pytorch/pytorch)
* [pytorch/vision](https://github.com/pytorch/vision)
* [alibaba/MNN](https://github.com/alibaba/MNN)
* [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
* [rockchip-linux/rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
* [libjpeg-turbo/libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo)
* [opencv/opencv](https://github.com/search?q=opencv)
* [opencv/opencv-python](https://github.com/opencv/opencv-python)
* [ermig1979/Simd](https://github.com/ermig1979/Simd)
* [nothings/stb](https://github.com/nothings/stb)
* [gabime/spdlog](https://github.com/gabime/spdlog)
* [facebookresearch/faiss](https://github.com/facebookresearch/faiss)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/onnx/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2021 zjykzj