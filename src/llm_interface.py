# -*- coding: utf-8 -*-
"""LLM 抽象接口

定义抽象 LLM 客户端接口，用户需自行实现具体的 LLM 调用。
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .logger import get_logger


class BaseLLMClient(ABC):
    """抽象 LLM 客户端接口"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本响应

        Args:
            prompt: 输入提示
            **kwargs: 额外参数

        Returns:
            生成的文本
        """
        pass

    @abstractmethod
    def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成结构化 JSON 响应

        Args:
            prompt: 输入提示
            **kwargs: 额外参数

        Returns:
            解析后的 JSON 字典
        """
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI 兼容接口客户端"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        if OpenAI is None:
            raise ImportError("请先安装 openai 库: pip install openai")

        self._logger = get_logger()
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        """返回 Token 消耗 (估算或从 API 返回获取)"""
        return self._total_tokens

    def reset_stats(self) -> None:
        """重置统计"""
        self._total_tokens = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """调用 OpenAI 生成文本"""
        self._logger.debug("OpenAI generate 调用")
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
            )
            
            content = response.choices[0].message.content
            
            if response.usage:
                self._total_tokens += response.usage.total_tokens
                
            return content
        except Exception as e:
            self._logger.error("OpenAI 调用失败: %s", e)
            raise

    def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """调用 OpenAI 生成 JSON"""
        self._logger.debug("OpenAI generate_json 调用")
        # 强制添加 JSON 指令
        json_prompt = prompt + "\n\n请确保输出为纯 JSON 格式，不要包含 Markdown 代码块标记。"
        
        content = self.generate(json_prompt, temperature=0.1, **kwargs) # 低温以保证结构稳定
        
        return self._parse_json(content)

    def _parse_json(self, content: str) -> Dict[str, Any]:
        """解析 JSON 字符串，处理 Markdown 代码块"""
        try:
            # 尝试直接解析
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试去除 Markdown 代码块 ```json ... ```
            pattern = r"```(?:json)?\s*(.*?)\s*```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            self._logger.error("无法解析 JSON: %s", content)
            # 返回空字典或抛出异常，这里选择返回空字典避免崩溃，但在 Adaptor 层可能需要处理
            return {}



class MockLLMClient(BaseLLMClient):
    """模拟 LLM 客户端，用于演示和测试"""

    def __init__(self):
        self._logger = get_logger()
        self._call_count = 0
        self._total_tokens = 0

    @property
    def call_count(self) -> int:
        """返回调用次数"""
        return self._call_count

    @property
    def total_tokens(self) -> int:
        """返回模拟的 Token 消耗"""
        return self._total_tokens

    def reset_stats(self) -> None:
        """重置统计"""
        self._call_count = 0
        self._total_tokens = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """模拟生成文本"""
        self._call_count += 1
        # 模拟 Token 消耗：输入 + 输出
        self._total_tokens += len(prompt.split()) + 50
        self._logger.debug("MockLLM generate 调用 #%d", self._call_count)

        # 根据 prompt 内容返回不同的模拟响应
        if "任务:" in prompt and "记忆上下文:" in prompt:
            # synthesis prompt
            return self._generate_synthesis_response(prompt)
        else:
            return "这是模拟的 LLM 响应。"

    def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """模拟生成 JSON 响应"""
        self._call_count += 1
        self._total_tokens += len(prompt.split()) + 30
        self._logger.debug("MockLLM generate_json 调用 #%d", self._call_count)

        # 根据 prompt 内容返回不同的模拟 JSON
        if "判断: 信息是否足够回答任务" in prompt:
            return self._generate_decision_response(prompt)
        elif "生成一个有序执行计划" in prompt:
            return self._generate_plan_response(prompt)
        elif "判断: 检索结果是否满足当前步骤需求" in prompt:
            return self._generate_replan_check_response()
        else:
            return {"action": "ANSWER"}

    def _generate_synthesis_response(self, prompt: str) -> str:
        """生成综合响应"""
        # 从 prompt 中提取任务
        task_start = prompt.find("任务:") + 3
        task_end = prompt.find("\n", task_start)
        task = prompt[task_start:task_end].strip() if task_end > task_start else "未知任务"

        return f"基于检索到的上下文信息，关于「{task}」的回答如下：这是一个模拟的综合响应，实际应用中 LLM 会根据检索到的证据生成准确的答案。"

    def _generate_decision_response(self, prompt: str) -> Dict[str, Any]:
        """生成决策响应"""
        # 模拟迭代逻辑：前两次返回 SEARCH，之后返回 ANSWER
        # 通过检查已有上下文的长度来判断迭代次数
        if "暂无" in prompt or prompt.count("- ") < 3:
            return {"action": "SEARCH", "query": "深度学习 Transformer 模型"}
        else:
            return {"action": "ANSWER"}

    def _generate_plan_response(self, prompt: str) -> Dict[str, Any]:
        """生成计划响应"""
        return {
            "plan": [
                {
                    "step_id": 1,
                    "description": "了解基础概念",
                    "query": "机器学习 基础概念",
                },
                {
                    "step_id": 2,
                    "description": "了解深度学习",
                    "query": "深度学习 神经网络",
                },
                {
                    "step_id": 3,
                    "description": "了解 Transformer 架构",
                    "query": "Transformer 注意力机制",
                },
            ]
        }

    def _generate_replan_check_response(self) -> Dict[str, Any]:
        """生成重规划检查响应"""
        # 模拟：大多数情况继续，偶尔重规划
        return {"action": "CONTINUE"}
