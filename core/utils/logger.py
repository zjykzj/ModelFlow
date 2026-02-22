# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 17:07
@File    : logger_config.py
@Author  : zj
@Description:
"""

import os
import logging
from concurrent_log import ConcurrentTimedRotatingFileHandler

LOGGER_NAME = "MODEL_FLOW"


def get_logger(name=None):
    return logging.getLogger(LOGGER_NAME if name is None else name)


def setup_task_logger(logger: logging.Logger, task_id: str, working_dir: str) -> (logging.FileHandler, str):
    """
    为指定 logger 添加任务专属的日志 handler

    Args:
        logger: 要添加 handler 的 logger 对象
        task_id: 当前任务 ID
        working_dir: 任务工作目录

    Returns:
        file_handler: 创建的 FileHandler 实例
        log_path: 日志文件路径
    """
    log_path = os.path.join(working_dir, f"task_{task_id}.log")
    file_handler = logging.FileHandler(log_path)

    formatter = logging.Formatter(
        '[%(asctime)s] {%(pathname)s:%(lineno)d} [PID %(process)d] %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return file_handler, log_path


def cleanup_task_logger(logger: logging.Logger, file_handler: logging.Handler):
    """
    安全地从 logger 中移除指定 handler

    Args:
        logger: 要清理的 logger
        file_handler: 要移除的 handler
    """
    if file_handler:
        try:
            file_handler.flush()
            file_handler.close()
            logger.removeHandler(file_handler)
        except Exception as e:
            logger.error(f"[cleanup_task_logger] Error during cleanup: {e}")


class EnhancedLogger:
    loggers = {}

    def __init__(self, name=None, log_dir='logs', log_file='app.log', level=logging.DEBUG, backup_count=30,
                 use_file_handler=True, use_stream_handler=True):
        self.name = LOGGER_NAME if name is None else name
        self.log_dir = log_dir
        self.log_file = log_file
        self.level = level
        self.backup_count = backup_count
        self.use_file_handler = use_file_handler
        self.use_stream_handler = use_stream_handler
        self.logger = self.setup_logger(name, log_dir, log_file, level, backup_count,
                                        use_file_handler, use_stream_handler)

    @classmethod
    def setup_logger(cls, name, log_dir='logs', log_file='app.log', level=logging.DEBUG, backup_count=30,
                     use_file_handler=True, use_stream_handler=True):
        if name in cls.loggers:
            return cls.loggers[name]

        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger = logging.getLogger(name)
        # 清除所有现有的handlers（如果需要）
        if logger.handlers:
            logger.handlers = []
        logger.setLevel(level)

        # 创建日志格式
        detailed_formatter = logging.Formatter(
            '[%(asctime)s] {%(pathname)s:%(lineno)d} [PID %(process)d] %(levelname)s - %(message)s'
        )

        if use_file_handler:
            # 创建ConcurrentTimedRotatingFileHandler并将格式化器与之关联
            log_path = os.path.join(log_dir, log_file)
            file_handler = ConcurrentTimedRotatingFileHandler(
                log_path, when='midnight', interval=1, backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.suffix = "_%Y-%m-%d.log"
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

        if use_stream_handler:
            # 创建StreamHandler以便输出到控制台
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(detailed_formatter)
            logger.addHandler(stream_handler)

        cls.loggers[name] = logger
        return logger
