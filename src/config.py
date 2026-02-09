# -*- coding: utf-8 -*-
"""统一配置管理

加载 config/prompts.yaml，提供全局配置访问接口。
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .logger import get_logger

_config: Optional["Config"] = None


class Config:
    """配置管理类"""

    def __init__(self, config_dir: Optional[Path] = None):
        self._logger = get_logger()
        self._prompts: Dict[str, Any] = {}
        self._app_config: Dict[str, Any] = {}

        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"

        self._load_prompts(config_dir / "prompts.yaml")
        self._load_app_config(config_dir / "config.yaml")

    def _load_prompts(self, config_path: Path) -> None:
        """加载 Prompt 配置"""
        if not config_path.exists():
            self._logger.error("配置文件不存在: %s", config_path)
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._prompts = yaml.safe_load(f) or {}

        self._logger.info("Prompt 配置加载完成: %s", config_path)

    def _load_app_config(self, config_path: Path) -> None:
        """加载应用配置 (LLM, Embedding, DB)"""
        if not config_path.exists():
            self._logger.warning("应用配置文件不存在: %s，将使用默认值", config_path)
            return

        with open(config_path, "r", encoding="utf-8") as f:
            self._app_config = yaml.safe_load(f) or {}

        self._logger.info("应用配置加载完成: %s", config_path)

    def get_prompt(self, adaptor: str, template_name: str) -> str:
        """获取指定适配器的 Prompt 模板

        Args:
            adaptor: 适配器名称 (single_turn, iterative, plan_and_act)
            template_name: 模板名称 (synthesis, decision, planning 等)

        Returns:
            Prompt 模板字符串
        """
        if adaptor not in self._prompts:
            self._logger.error("未知的适配器: %s", adaptor)
            raise KeyError(f"未知的适配器: {adaptor}")

        if template_name not in self._prompts[adaptor]:
            self._logger.error("未知的模板: %s.%s", adaptor, template_name)
            raise KeyError(f"未知的模板: {adaptor}.{template_name}")

        return self._prompts[adaptor][template_name]

    @property
    def prompts(self) -> Dict[str, Any]:
        """获取所有 Prompt 配置"""
        return self._prompts

    @property
    def llm(self) -> Dict[str, Any]:
        """获取 LLM 配置"""
        return self._app_config.get("llm", {})

    @property
    def embedding(self) -> Dict[str, Any]:
        """获取 Embedding 配置"""
        return self._app_config.get("embedding", {})

    @property
    def database(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return self._app_config.get("database", {})



def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config()
    return _config
