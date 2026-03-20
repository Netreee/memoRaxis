# -*- coding: utf-8 -*-
"""HippoRAG 适配器（接入 memoRaxis 的统一 Memory 接口）

封装 HippoRAG 的知识图谱构建（OpenIE + PPR）与检索，
让现有 R1/R2/R3 推理范式可以无缝调用 HippoRAG 作为记忆后端。
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List

from .config import get_config
from .logger import get_logger
from .memory_interface import BaseMemorySystem, Evidence

logger = get_logger()

# Embedding proxy 配置
EMBEDDING_PROXY_URL = "http://127.0.0.1:8284/v1"
EMBEDDING_MODEL_ALIAS = "text-embedding-3-small"  # triggers OpenAIEmbeddingModel in HippoRAG


def _add_hipporag_to_syspath(project_root: Path) -> None:
    """将 external/hipporag_repo/src 加入 sys.path"""
    candidate = project_root / "external" / "hipporag_repo" / "src"
    if candidate.exists() and (candidate / "hipporag").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            logger.info("[HippoRAGMemory] Added HippoRAG src to sys.path: %s", candidate)
        return

    raise FileNotFoundError(
        f"找不到 HippoRAG 源码: {candidate}\n"
        "请确认 external/hipporag_repo/ 已 clone 且其中包含 src/hipporag/ 目录。"
    )


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path, onerror=lambda func, p, _: (os.chmod(p, 0o777), func(p)))


class HippoRAGMemory(BaseMemorySystem):
    """HippoRAG 记忆体封装"""

    def __init__(
        self,
        index_dir: str,
        openie_mode: str = "online",
        force_rebuild: bool = False,
        top_k_default: int = 5,
        seed: int = 42,
    ):
        self._logger = get_logger()
        self._config = get_config()

        self.index_dir = Path(index_dir).resolve()
        self.openie_mode = openie_mode
        self.force_rebuild = force_rebuild
        self.top_k_default = top_k_default
        self.seed = seed

        if self.force_rebuild:
            self._logger.warning("[HippoRAGMemory] force_rebuild=True, removing: %s", self.index_dir)
            _safe_rmtree(self.index_dir)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 设置 OpenAI 环境变量（HippoRAG 内部需要）
        llm_conf = self._config.llm
        api_key = llm_conf.get("api_key") or os.getenv("OPENAI_API_KEY")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # 导入 HippoRAG
        project_root = Path(__file__).resolve().parents[1]
        _add_hipporag_to_syspath(project_root)

        from hipporag import HippoRAG  # type: ignore
        from hipporag.utils.config_utils import BaseConfig  # type: ignore

        # HippoRAG 配置：LLM 用 Ark，Embedding 走 proxy
        global_config = BaseConfig(
            llm_name=llm_conf.get("model", "gpt-4o-mini"),
            llm_base_url=llm_conf.get("base_url"),
            embedding_model_name=EMBEDDING_MODEL_ALIAS,
            embedding_base_url=EMBEDDING_PROXY_URL,
            openie_mode=self.openie_mode,
            save_dir=str(self.index_dir),
            force_index_from_scratch=self.force_rebuild,
            force_openie_from_scratch=self.force_rebuild,
            seed=self.seed,
        )

        self._hippo = HippoRAG(global_config=global_config)
        self._buffer: List[str] = []

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        if isinstance(data, str) and data.strip():
            self._buffer.append(data)

    def build_index(self) -> None:
        """构建 HippoRAG 索引（ingest 阶段调用一次）"""
        if not self._buffer:
            self._logger.warning("[HippoRAGMemory] No chunks to index.")
            return
        self._logger.info("[HippoRAGMemory] Building index with %d chunks", len(self._buffer))
        self._hippo.index(self._buffer)
        self._logger.info("[HippoRAGMemory] Index build done.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        k = top_k or self.top_k_default
        if not query:
            return []

        try:
            out = self._hippo.retrieve([query], num_to_retrieve=k)
        except TypeError:
            out = self._hippo.retrieve([query], k)

        if isinstance(out, tuple) and len(out) == 2:
            solutions = out[0]
        else:
            solutions = out

        if not solutions:
            return []

        sol = solutions[0]
        docs = getattr(sol, "docs", None)
        if docs is None:
            docs = []
        scores = getattr(sol, "doc_scores", None)

        evidences: List[Evidence] = []
        for i, doc in enumerate(docs[:k]):
            score_val = 0.0
            if scores is not None:
                try:
                    score_val = float(scores[i])
                except Exception:
                    pass
            evidences.append(
                Evidence(
                    content=doc,
                    metadata={
                        "source": "HippoRAG",
                        "rank": i + 1,
                        "score": score_val,
                        "index_dir": str(self.index_dir),
                    },
                )
            )
        return evidences

    def reset(self) -> None:
        self._buffer = []
