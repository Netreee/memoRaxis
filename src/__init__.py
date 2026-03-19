# -*- coding: utf-8 -*-
"""Agent 推理范式适配器模块"""

from .logger import get_logger
from .config import Config, get_config
from .memory_interface import Evidence, BaseMemorySystem, MockMemory

# amem_memory must be imported BEFORE llm_interface (which loads Fornax SDK) and
# simple_memory (which loads psycopg2), to avoid a native-library conflict:
#   fornax gRPC → psycopg2/libpq → chromadb/hnswlib crashes in that order.
try:
    from .amem_memory import AMemMemory
except ImportError:
    AMemMemory = None

from .llm_interface import BaseLLMClient, MockLLMClient, OpenAIClient
from .adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor
from .simple_memory import SimpleRAGMemory

__all__ = [
    "get_logger",
    "Config",
    "get_config",
    "Evidence",
    "BaseMemorySystem",
    "MockMemory",
    "SimpleRAGMemory",
    "AMemMemory",
    "BaseLLMClient",
    "MockLLMClient",
    "OpenAIClient",
    "SingleTurnAdaptor",
    "IterativeAdaptor",
    "PlanAndActAdaptor",
]
