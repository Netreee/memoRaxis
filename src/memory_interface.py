# -*- coding: utf-8 -*-
"""记忆体接口

定义抽象记忆系统接口和用于测试的 MockMemory 实现。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel

from .logger import get_logger


class Evidence(BaseModel):
    """检索到的证据"""

    content: str
    metadata: Dict[str, Any] = {}


class BaseMemorySystem(ABC):
    """抽象记忆系统接口"""

    @abstractmethod
    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        """添加记忆

        Args:
            data: 记忆内容
            metadata: 元数据
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """检索记忆

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            检索到的证据列表
        """
        pass

    def reset(self) -> None:
        """重置记忆系统"""
        pass


class MockMemory(BaseMemorySystem):
    """模拟记忆系统，用于测试和演示"""

    def __init__(self):
        self._logger = get_logger()
        self._memories: List[Dict[str, Any]] = []
        self._init_default_memories()

    def _init_default_memories(self) -> None:
        """初始化默认测试数据"""
        default_data = [
            {
                "content": "Python 是一种解释型、高级编程语言，由 Guido van Rossum 于 1991 年创建。",
                "metadata": {"source": "编程语言百科", "topic": "Python"},
            },
            {
                "content": "Python 的设计哲学强调代码可读性，使用缩进来表示代码块。",
                "metadata": {"source": "编程语言百科", "topic": "Python"},
            },
            {
                "content": "机器学习是人工智能的一个子领域，通过数据训练模型来做出预测或决策。",
                "metadata": {"source": "AI 基础教程", "topic": "机器学习"},
            },
            {
                "content": "深度学习是机器学习的一种方法，使用多层神经网络来学习数据表示。",
                "metadata": {"source": "AI 基础教程", "topic": "深度学习"},
            },
            {
                "content": "Transformer 架构是一种基于注意力机制的神经网络架构，广泛用于自然语言处理。",
                "metadata": {"source": "深度学习进阶", "topic": "Transformer"},
            },
            {
                "content": "BERT 是 Google 提出的预训练语言模型，使用双向 Transformer 编码器。",
                "metadata": {"source": "NLP 模型介绍", "topic": "BERT"},
            },
            {
                "content": "GPT 系列模型由 OpenAI 开发，使用自回归方式生成文本。",
                "metadata": {"source": "NLP 模型介绍", "topic": "GPT"},
            },
            {
                "content": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术，用于增强 LLM 的知识能力。",
                "metadata": {"source": "LLM 应用技术", "topic": "RAG"},
            },
        ]

        for item in default_data:
            self._memories.append(item)

        self._logger.info("MockMemory 初始化完成，共 %d 条记忆", len(self._memories))

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        """添加记忆"""
        self._memories.append({"content": data, "metadata": metadata})
        self._logger.debug("添加记忆: %s", data[:50])

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """检索记忆（简单关键词匹配）"""
        self._logger.debug("检索查询: %s, top_k=%d", query, top_k)

        # 简单的关键词匹配评分
        scored_memories = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for memory in self._memories:
            content_lower = memory["content"].lower()
            # 计算匹配分数
            score = 0
            for word in query_words:
                if word in content_lower:
                    score += 1
            if score > 0:
                scored_memories.append((score, memory))

        # 按分数排序，取 top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        results = [
            Evidence(content=m["content"], metadata=m["metadata"])
            for _, m in scored_memories[:top_k]
        ]

        self._logger.debug("检索结果: %d 条", len(results))
        return results

    def reset(self) -> None:
        """重置记忆"""
        self._memories.clear()
        self._init_default_memories()
        self._logger.info("MockMemory 已重置")
