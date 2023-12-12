# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/12 14:33
@File    : general.py
@Author  : zj
@Description: 
"""

import os
import logging.config
import platform

LOGGING_NAME = 'ultralytics'
VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'  # global verbose mode
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans


def emojis(string=''):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string


class EmojiFilter(logging.Filter):
    """
    A custom logging filter class for removing emojis in log messages.

    This filter is particularly useful for ensuring compatibility with Windows terminals that may not support the
    display of emojis in log messages.
    """

    def filter(self, record):
        """Filter logs by emoji unicode characters on windows."""
        record.msg = emojis(record.msg)
        return super().filter(record)


def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})


# Set logger
set_logging(LOGGING_NAME, verbose=VERBOSE)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if WINDOWS:  # emoji-safe logging
    LOGGER.addFilter(EmojiFilter())
