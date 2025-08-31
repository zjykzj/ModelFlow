# -*- coding: utf-8 -*-

"""
@Time    : 2025/5/13 16:04
@File    : segment_predict.py
@Author  : zj
@Description:

$ python segment_predict.py --model /data/zj/renjie_train/rj_train/tmp/5132451234123_eval_1747124485/model/model.pt
                            --source /data/zj/renjie_train/rj_train/tmp/TinyXfyPlate6323_train_1747124303/dst_data/images
                            --device 1 --imgsz 640 --save_txt true --save_conf true
"""

import argparse


def parse_device(device_str):
    """
    自定义类型函数，用于解析 --device 参数。
    支持输入格式如：'cpu', '0', '0,1,2,3'。
    """
    if device_str.lower() == 'cpu':
        return 'cpu'
    else:
        try:
            # 尝试将输入解析为整数列表
            ids = [int(id) for id in device_str.split(',')]
            return ids
        except ValueError:
            raise argparse.ArgumentTypeError("Device must be 'cpu' or a comma-separated list of integers.")


def is_float(s):
    try:
        float(s)  # 直接尝试转换成float，因为int的字符串也可以成功转换
        return True
    except ValueError:
        return False


def parse_args(folder_pict=False):
    # 创建解析器
    parser = argparse.ArgumentParser(description="YOLO11Face Script")

    # 添加参数
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the model file')
    parser.add_argument('--source', type=str, default=None,
                        help='Path to the data configuration')

    parser.add_argument('--device', type=parse_device, default="cpu",
                        help='Device ID for CUDA execution, cpu for CPU (default: cpu)')

    # 解析已知和未知的参数
    args, unknown = parser.parse_known_args()
    print(f"args: {args} - unknown: {unknown}")

    # 构建 overrides 字典
    overrides = {
        "model": args.model,
        "source": args.source,
        "device": args.device,
    }

    # 处理额外的参数
    extra_args = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].replace('--', '')  # 去除 '--' 前缀
        value = unknown[i + 1]
        try:
            if value.isdecimal():
                extra_args[key] = int(value)
            elif is_float(value):
                extra_args[key] = float(value)
            elif value.lower() == 'true':
                extra_args[key] = True
            elif value.lower() == 'false':
                extra_args[key] = False
            else:
                extra_args[key] = value
        except ValueError:
            extra_args[key] = value

    # 将额外的参数合并到 overrides
    overrides.update(extra_args)

    return overrides


def main():
    overrides = parse_args()
    assert overrides['model'] is not None, 'model must be specified'
    assert overrides['source'] is not None, 'source must be specified'
    overrides['mode'] = 'predict'
    print(f"overrides: {overrides}")

    # 初始化训练器并开始训练
    from ultralytics.models.yolo.segment import SegmentationPredictor
    predictor = SegmentationPredictor(overrides=overrides)
    predictor.predict_cli()


if __name__ == "__main__":
    main()
