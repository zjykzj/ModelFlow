# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 17:18
@File    : helpers.py
@Author  : zj
@Description: 
"""

import re
import os
import time
import yaml
import shutil
import hashlib
import requests
import subprocess

from typing import List, Dict, Optional
from contextlib import ContextDecorator
from urllib.parse import urlparse

from core.utils.logger import get_logger

logger = get_logger()


def validate_return_name(return_name):
    """
    校验 return_name 是否符合格式 <训练类型>_<业务场景>_<模型架构>_<模型版本>，
    其中 <模型架构> 可以包含多个下划线。

    :param return_name: str, 要校验的 return_name
    :return: True if valid, else raises ValueError
    """
    # 正则表达式解释：
    # ^([^_]+)_([^_]+)_([^_].*?)_([^_]+)$
    # 分别对应：<训练类型>_<业务场景>_<模型架构>_<模型版本>
    pattern = r'^([^_]+)_([^_]+)_([^_][^_]*)_([^_]+)$'
    match = re.fullmatch(pattern, return_name)

    if not match:
        raise ValueError(
            f"'return_name' '{return_name}' is invalid. "
            f"It must follow the format '<训练类型>_<业务场景>_<模型架构>_<模型版本>', where only <模型架构> can contain multiple underscores."
        )

    return True


def parse_model_architecture_regex(model_code):
    """
    使用正则表达式校验并提取模型架构字段。
    格式要求：5个由下划线连接的部分，整体格式为 <训练类型>_<业务场景>_<模型架构>_<导出格式>_<模型版本>
    其中 <模型架构> 可以包含任意字符（包括多个下划线），其余部分仍限制为字母、数字和下划线。
    """
    # 分别匹配前两段、中间一段（含任意字符）、最后两段
    pattern = r'^([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)_([^_].*?)_([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)$'
    match = re.fullmatch(pattern, model_code)
    if not match:
        return False, None
    return True, match.group(3)


def generate_new_model_name(model_name, target_format):
    """
    根据输入的 model_name 和目标格式生成新的模型名称。

    :param model_name: str, 模型名称，格式为 <训练类型>_<业务场景>_<模型架构>_<模型版本>。
                       其中 <模型架构> 可以包含多个下划线。
    :param target_format: str, 目标模型格式 ("ONNX" 或 "TensorRT")。
    :return: str, 新的模型名称，格式为 <训练类型>_<业务场景>_<模型架构>_<导出格式>_<模型版本>。
    """
    # 使用正则表达式匹配四段结构：<1>_<2>_<3>_<5>，其中 <3> 可以含多个下划线
    pattern = r'^([^_]+)_([^_]+)_([^_].*?)_([^_]+)$'
    match = re.fullmatch(pattern, model_name)

    if not match:
        raise ValueError(
            f"Invalid model_name '{model_name}'. "
            f"It must follow the format: <训练类型>_<业务场景>_<模型架构>_<模型版本>, "
            f"其中 <模型架构> 可以包含多个下划线。"
        )

    train_type, business_scene, model_architecture, model_version = match.groups()

    # 插入 target_format 作为第四部分
    new_model_name = f"{train_type}_{business_scene}_{model_architecture}_{target_format}_{model_version}"

    return new_model_name


def extract_filename_from_url(url):
    """
    从给定的URL中提取文件名。

    参数:
        url (str): 完整的URL字符串。

    返回:
        str: 提取到的文件名。如果无法提取，则返回None。
    """
    try:
        # 使用urlparse解析URL
        parsed_url = urlparse(url)

        # 获取路径部分
        path = parsed_url.path

        # 分割路径并获取最后一部分作为文件名
        filename = path.split('/')[-1]

        # 检查文件名是否为空（例如路径以斜杠结尾）
        if not filename:
            return None

        return filename
    except Exception as e:
        print(f"解析URL时出错: {e}")
        return None


## TODO 文件校验
# 海草链接示例 http://10.218.223.107/inspection-gateway/sphFile/getStream/4quality?fileId=689,a0c4e882fc8a&fileName=quality_1663853274_20220922212754.jpg
def sign_url(url):
    if not url:
        logger.error(f"Url:{url} is empty!")
        return url
    if "fileId" not in url or "fileName" not in url:
        logger.error(f"Url:{url} is error! Url must have fileId string and fileName string !")
        return url

    info = url.split("?")
    if len(info) < 2:
        logger.error(f"Url:{url} is error! Url must have user param !")
        return url

    url_param = info[1]
    param = url_param.split("&")
    fileId = param[0].replace("fileId=", "")
    fileName = param[1].replace("fileName=", "")
    date = time.strftime("%Y%m%d")
    sign = fileId + "_" + fileName + "_" + date
    sign = md5_encode(sign).upper()

    return url + "&sign=" + sign


def md5_encode(data):
    # 创建一个md5对象
    m = hashlib.md5()
    # 使用md5对象的update方法对数据进行MD5加密
    m.update(data.encode('utf-8'))
    # 使用hexdigest函数获取MD5加密后的十六进制输出
    return m.hexdigest()


# 创建全局 Session（关键：复用 TCP 连接）
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,  # 连接池大小
    pool_maxsize=50,  # 最大连接数
    max_retries=3  # 自动重试
)
session.mount('http://', adapter)
session.mount('https://', adapter)


def download_file(url: str, save_path: str) -> bool:
    """
    下载指定 URL 的文件并保存到本地（带重试和连接复用）。
    """
    final_url = str(url)
    max_retries = 3
    backoff_factor = 1.0

    for attempt in range(max_retries + 1):
        try:
            if 'volces.com' in final_url:
                from core.utils.tosutil import download_from_volcengine, parse_tos_url
                from core.utils.globals import BUCKET_NAME
                object_key = parse_tos_url(final_url)
                return download_from_volcengine(BUCKET_NAME, object_key, save_path)

            if 'inspection-gateway' in final_url:
                final_url = sign_url(final_url)

            # 使用 session 复用连接，stream=True 避免内存溢出
            response = session.get(final_url, stream=True, timeout=30)

            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                return True
            else:
                logger.error(f"Failed to download {final_url}. Status: {response.status_code}")

        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Error downloading {final_url} after {max_retries} retries: {str(e)}")
            else:
                sleep_time = backoff_factor * (2 ** attempt)
                logger.warning(
                    f"Download failed: {final_url}, retry {attempt + 1}/{max_retries} in {sleep_time}s. Error: {str(e)}")
                time.sleep(sleep_time)  # 指数退避
                continue  # 重试

        break  # 成功或不再重试

    return False


# 回调函数：用于发送状态更新
def send_callback(callback_url, task_id, current_stage, status_code, message=None, data=None, log_url=None):
    payload = {
        "current_stage": current_stage,
        "message": message,
        "task_id": task_id,
        "status_code": status_code,
    }
    if data:
        payload["data"] = data
    if log_url:
        payload['log'] = log_url
    response = requests.post(callback_url, json=payload)
    if response.status_code != 200:
        logger.error(f"Failed to send callback for stage {current_stage}: {response.text}")
        raise ValueError(f"Callback failed with status code {response.status_code}: {response.text}")


# 辅助函数，用于简化日志记录与状态更新
def log_and_send_status(callback_url, task_id, stage, status_code, info_msg):
    """
    记录信息到日志，并发送回调通知。

    :param callback_url: 回调URL。
    :param task_id: 任务ID。
    :param stage: 当前处理阶段。
    :param status_code: 状态码（例如2000表示成功，5000表示错误）。
    :param info_msg: 要记录的信息或错误消息。
    """
    logger.info(info_msg)
    if callback_url is not None:
        send_callback(callback_url, task_id, stage, status_code, info_msg)


def get_gpu_name():
    try:
        # 检查 nvidia-smi 是否存在
        if not shutil.which("nvidia-smi"):
            print("当前环境没有 nvidia-smi，可能不是桌面端 NVIDIA GPU 环境（如 Jetson 设备）。")
            return None

        # 使用 nvidia-smi 命令获取 GPU 名称
        result = subprocess.run(['nvidia-smi', '-q'], stdout=subprocess.PIPE)
        output = result.stdout.decode()

        # 解析输出以找到 Product Name 行
        for line in output.split('\n'):
            if 'Product Name' in line:
                return line.split(':')[-1].strip()
        return None
    except Exception as e:
        print(f"无法获取 GPU 名称: {e}")
        return None


def parse_onnx_model(onnx_path):
    # 加载ONNX模型
    import onnx

    model = onnx.load(onnx_path)

    # 初始化输入和输出信息
    input_names = []
    output_names = []
    input_shapes = []
    output_shapes = []
    input_dtypes = []
    output_dtypes = []

    # 提取输入信息
    for input in model.graph.input:
        input_name = input.name
        input_shape = [dim.dim_value if dim.dim_value != 0 else 'dynamic' for dim in input.type.tensor_type.shape.dim]
        input_dtype = input.type.tensor_type.elem_type
        input_names.append(input_name)
        input_shapes.append(input_shape)
        input_dtypes.append(input_dtype)

    # 提取输出信息
    for output in model.graph.output:
        output_name = output.name
        output_shape = [dim.dim_value if dim.dim_value != 0 else 'dynamic' for dim in output.type.tensor_type.shape.dim]
        output_dtype = output.type.tensor_type.elem_type
        output_names.append(output_name)
        output_shapes.append(output_shape)
        output_dtypes.append(output_dtype)

    # 将ONNX的数据类型转换为更易读的字符串形式
    dtype_map = {1: 'float32', 2: 'uint8', 3: 'int8', 4: 'uint16', 5: 'int16', 6: 'int32', 7: 'int64', 9: 'bool',
                 10: 'float16', 11: 'double'}

    input_dtypes = [dtype_map.get(dtype, dtype) for dtype in input_dtypes]
    output_dtypes = [dtype_map.get(dtype, dtype) for dtype in output_dtypes]

    # 返回导出模型的相关元信息
    return {
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "input_dtypes": input_dtypes,
        "output_dtypes": output_dtypes
    }


class Profile(ContextDecorator):
    """
    A class to measure the execution time of a block of code or function.
    Can be used as a decorator or as a context manager.

    Example usage:
        @Profile()
        def my_function():
            # Your code here

        with Profile() as prof:
            # Your code here
        print(f"Execution time: {prof.dt:.2f} seconds")
    """

    def __init__(self, name=None):
        self.name = name  # Optional name for the profiled section
        self.start_time = None
        self.end_time = None
        self.dt = 0.0  # Delta time (execution duration)

    def __enter__(self):
        self.start_time = time.time()  # Record start time
        return self

    def __exit__(self, *exc):
        self.end_time = time.time()  # Record end time
        self.dt = self.end_time - self.start_time  # Calculate delta time
        if self.name:
            print(f"[{self.name}] Execution time: {self.dt:.2f} seconds")
        return False


def validate_gpu_id(gpu_id, single_gpu_only=False):
    """
    校验指定的 GPU ID 是否有效。
    :param gpu_id: 用户指定的 GPU ID（字符串）。
    :param single_gpu_only: 是否仅允许单个 GPU。默认为 False。
    :raises ValueError: 如果校验失败，抛出异常。
    """
    try:
        # 确保 gpu_id 是一个非空字符串
        if not isinstance(gpu_id, str) or gpu_id.strip() == "":
            error_msg = (
                "GPU 配置错误：未指定有效的 GPU ID。\n"
                "请确保输入的是单个 GPU ID（例如 '0'）或逗号分隔的多个 GPU ID（例如 '0,1,2'）。"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 解析用户指定的 GPU ID
        try:
            gpu_ids = [int(gpu.strip()) for gpu in gpu_id.split(",") if gpu.strip().isdigit()]
            if len(gpu_ids) != len(gpu_id.split(",")):
                raise ValueError("包含非法字符或空值")
        except ValueError:
            error_msg = (
                f"GPU 配置错误：无法解析指定的 GPU ID。\n"
                f"您输入的 GPU ID 为 '{gpu_id}'。\n"
                f"请确保输入的是单个非负整数（例如 '0'）或逗号分隔的多个非负整数（例如 '0,1,2'）。"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 检查是否仅允许单个 GPU
        if single_gpu_only and len(gpu_ids) > 1:
            error_msg = (
                "GPU 配置错误：当前设置仅允许指定单个 GPU。\n"
                "请确保输入的是单个 GPU ID，例如 '0'。"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 获取可用的 GPU 数量
        import torch

        available_gpus = torch.cuda.device_count()

        # 检查指定的 GPU ID 是否超出范围
        for gpu_id_int in gpu_ids:
            if gpu_id_int >= available_gpus:
                error_msg = (
                    f"GPU 配置错误：指定的 GPU ID 超出了机器的可用 GPU 数量。\n"
                    f"当前机器可用的 GPU 数量为 {available_gpus}，最大可用 GPU ID 为 {available_gpus - 1}。\n"
                    f"但您指定了 GPU ID {gpu_id_int}。\n"
                    f"请确保指定的 GPU ID 在范围 [0, {available_gpus - 1}] 内。"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # 如果通过检查，记录日志
        logger.info(f"GPU 配置检查通过。指定的 GPU ID: {gpu_ids}")
    except Exception as e:
        # 捕获所有异常并记录日志
        error_msg = f"GPU 配置检查失败：{str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def modify_dataset_config(input_file, output_file, new_path, new_train, new_val, new_names, is_coco=False):
    """
    修改数据集配置文件并保存到指定路径。

    :param input_file: 原始配置文件路径
    :param output_file: 修改后的配置文件保存路径
    :param new_path: 新的 dataset root dir
    :param new_train: 新的 train images 路径（相对路径）
    :param new_val: 新的 val images 路径（相对路径）
    :param new_names: 新的类别名称列表（例如 ["person", "bicycle", "car"]）
    """
    # 读取原始配置文件
    with open(input_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 修改配置内容
    if 'path' not in config.keys():
        # 适配YOLOv6
        config['train'] = os.path.join(new_path, new_train)
        config['val'] = os.path.join(new_path, new_val)
        config['nc'] = len(new_names)
        config['is_coco'] = "true" if is_coco else "false"
    else:
        # 适配ultralytics
        config['path'] = new_path
        config['train'] = new_train
        config['val'] = new_val

    # 处理 names 字段：将列表转换为字典形式
    config['names'] = {i: name for i, name in enumerate(new_names)}

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # # 将修改后的配置写入目标文件
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    # 将修改后的配置写入目标文件
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    return config


def run_task(
        command: List[str],
        env: Dict[str, str] = None,
        work_dir: Optional[str] = None,
        log_real_time: bool = True,
):
    """
    通用任务执行器，用于运行任意命令并实时捕获其输出日志。
    可用于训练、评估、预处理等各类脚本的执行。

    :param command: 完整的命令行参数列表，如 ["python", "train.py", "--arg", "value"]
    :param env: 要设置的环境变量字典，如 {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "..."}
    :param work_dir: 子进程的工作目录，默认使用当前项目根目录
    :param log_real_time: 是否逐行实时记录日志（默认为 True）
    """
    # 默认工作目录为项目根目录
    if not work_dir:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        work_dir = project_root

    env = env.copy() if env is not None else os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    logger.info(f"正在启动任务。执行命令: {' '.join(command)}")
    logger.debug(f"工作目录: {work_dir}")
    logger.debug(f"环境变量: {env}")

    try:
        if log_real_time:
            # 实时读取日志输出
            process = subprocess.Popen(
                command,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )

            for line in process.stdout:
                logger.info(line.strip())

            return_code = process.wait()
            if return_code != 0:
                error_message = f"任务执行失败，退出码: {return_code}"
                logger.error(error_message)
                raise RuntimeError(error_message)

        else:
            # 非实时输出方式（一次性获取结果）
            result = subprocess.run(
                command,
                check=True,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            logger.info(f"任务执行成功。标准输出内容: {result.stdout}")
            if result.stderr:
                logger.warning(f"任务执行过程中有警告信息: {result.stderr}")

        logger.info("任务执行成功。")

    except subprocess.CalledProcessError as e:
        error_message = f"任务执行失败: {e.stderr.strip()}"
        logger.error(error_message)
        raise RuntimeError(error_message) from e
