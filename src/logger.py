# -*- coding: utf-8 -*-
"""统一日志系统

日志自动写入 log/YYYYMMDD-HHMMSS.log，屏幕仅输出 ERROR 级别简短提示。
全局单例，禁止各脚本自建。
"""

import logging
import os
import sys
import shlex
from datetime import datetime
from pathlib import Path
from typing import Optional

_logger: Optional[logging.Logger] = None
_log_file_path: Optional[Path] = None


def _init_logger() -> logging.Logger:
    """初始化日志系统"""
    global _log_file_path

    # 确保日志目录存在
    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)

    # 生成日志文件名，加入 PID 避免并发子进程（如 dispatcher 批量启动时）写入同一文件
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _log_file_path = log_dir / f"{timestamp}-{os.getpid()}.log"

    # 创建 logger
    logger = logging.getLogger("memoRaxis")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 文件处理器：记录所有级别
    file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台处理器：仅输出 ERROR 级别
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter("[错误] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    cmd = " ".join([shlex.quote(sys.executable)] + [shlex.quote(arg) for arg in sys.argv])
    logger.info("启动命令: %s", cmd)
    logger.info("日志系统初始化完成，日志文件: %s", _log_file_path)
    return logger


def get_logger() -> logging.Logger:
    """获取全局日志实例"""
    global _logger
    if _logger is None:
        _logger = _init_logger()
    return _logger


def get_log_file_path() -> Optional[Path]:
    """获取当前日志文件路径"""
    return _log_file_path
