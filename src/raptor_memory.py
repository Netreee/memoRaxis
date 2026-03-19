# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests
import threading

from src.memory_interface import BaseMemorySystem, Evidence
from src.config import get_config
from scipy.spatial import distance
from external.raptor_repo.raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from external.raptor_repo.raptor.EmbeddingModels import BaseEmbeddingModel
from external.raptor_repo.raptor.SummarizationModels import BaseSummarizationModel
from external.raptor_repo.raptor.QAModels import BaseQAModel
class _NoQAModel(BaseQAModel):
    def answer_question(self, *args, **kwargs):
        return ""



class _CompatEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        conf = get_config().embedding
        self.provider = conf.get('provider', 'openai_compat')
        self.base_url = conf.get('base_url')
        self.api_key = conf.get('api_key')
        self.model = conf.get('model', 'text-embedding-3-small')

        if self.provider != 'ark_multimodal':
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self._client = None

    def create_embedding(self, text: str):
        import time as _time
        import logging as _logging

        text = text.replace('\n', ' ')
        max_retries = 8
        wait = 30
        for attempt in range(max_retries):
            try:
                if self.provider == 'ark_multimodal':
                    url = (self.base_url or '').rstrip('/') + '/embeddings/multimodal'
                    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + str(self.api_key)}
                    payload = {'model': self.model, 'input': [{'type': 'text', 'text': text}]}
                    r = requests.post(url, headers=headers, json=payload, timeout=60)
                    r.raise_for_status()
                    return r.json()['data']['embedding']
                resp = self._client.embeddings.create(input=text, model=self.model)
                return resp.data[0].embedding
            except Exception as e:
                err_str = str(e)
                retryable = (
                    "429" in err_str or "RateLimit" in err_str or "rate_limit" in err_str.lower()
                    or "ConnectionError" in type(e).__name__ or "BrokenPipe" in err_str
                    or "ConnectionReset" in err_str or "Timeout" in type(e).__name__
                )
                if retryable and attempt < max_retries - 1:
                    _logging.warning("RAPTOR embedding error (%s), waiting %ds (attempt %d/%d)", type(e).__name__, wait, attempt + 1, max_retries)
                    _time.sleep(wait)
                    wait = min(wait * 2, 300)
                    continue
                raise
        raise RuntimeError("RAPTOR embedding failed after max retries")


class _CompatSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        conf = get_config().llm
        from openai import OpenAI
        self._client = OpenAI(api_key=conf.get('api_key'), base_url=conf.get('base_url'))
        self.model = conf.get('model')
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._stats_lock = threading.Lock()

    def summarize(self, context, max_tokens=180):
        import time as _time
        import logging as _logging

        # 1) 防止 max_tokens 意外为 0/None
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = 180
        if max_tokens < 32:
            max_tokens = 32

        # 指数退避重试（与 OpenAIClient.generate 对齐）
        max_retries = 8
        wait = 30
        resp = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': 'You are a careful summarizer.'},
                        {'role': 'user', 'content': f'Summarize the following. Keep key entities, dates, decisions:\n{context}'},
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RateLimit" in err_str or "rate_limit" in err_str.lower():
                    if attempt < max_retries - 1:
                        _logging.warning("RAPTOR summarize 429, waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                        _time.sleep(wait)
                        wait = min(wait * 2, 300)
                        continue
                raise
        if resp is None:
            return (context or "")[:2000]

        # 2) 主路径：标准 OpenAI-compat
        content = ""
        try:
            content = resp.choices[0].message.content or ""
        except Exception:
            content = ""

        # 3) 兼容路径：某些网关会把内容放在“分段 content”里
        #    这里用 model_dump/dict 把原始结构拿出来再捞一遍
        if not content.strip():
            try:
                raw = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
                msg = raw.get("choices", [{}])[0].get("message", {})
                c = msg.get("content", "")

                if isinstance(c, str):
                    content = c
                elif isinstance(c, list):
                    # 常见形态：[{type: "...", text: "..."}] 或类似字段
                    buf = []
                    for part in c:
                        if isinstance(part, dict):
                            buf.append(part.get("text") or part.get("content") or "")
                        elif isinstance(part, str):
                            buf.append(part)
                    content = "".join(buf)
            except Exception:
                pass

        # 4) 最终兜底：如果仍为空，用原文截断代替，确保树节点不为空
        if not content.strip():
            content = (context or "")[:2000]

        # 5) 审计：累积 token 消耗（线程安全）
        with self._stats_lock:
            self.llm_calls += 1
            try:
                usage = resp.usage
                self.prompt_tokens += getattr(usage, "prompt_tokens", 0)
                self.completion_tokens += getattr(usage, "completion_tokens", 0)
            except Exception:
                pass

        return content


class RaptorTreeMemory(BaseMemorySystem):
    def __init__(self, tree_path: Optional[str] = None, tb_num_layers: int = 3):
        self._buffer: List[str] = []
        emb = _CompatEmbeddingModel()
        summ = _CompatSummarizationModel()

        self._summ = summ
        self._config = RetrievalAugmentationConfig(
            embedding_model=emb,
            summarization_model=summ,
            qa_model=_NoQAModel(),
            tb_num_layers=tb_num_layers,
            tb_max_tokens=200,
            tb_summarization_length=120,
            tr_threshold=0.5,
            tr_top_k=5,
        )
        self._ra = RetrievalAugmentation(config=self._config, tree=tree_path)

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        self._buffer.append(data)

    def build_tree(self) -> None:
        if self._ra.tree is not None:
            return
        text = '\n\n'.join(self._buffer)
        self._ra.add_documents(text)

    def save_tree(self, path: str) -> None:
        self.build_tree()
        self._ra.save(path)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        self.build_tree()
        context, layer_info = self._ra.retrieve(
            question=query,
            top_k=top_k,
            collapse_tree=True,
            return_layer_information=True,
        )

        # 用 RAPTOR 同一套 embedding 给 query 做向量（保证一致）
        q_emb = self._ra.retriever.create_embedding(query)
        emb_key = getattr(self._ra.retriever, "context_embedding_model", None) or "EMB"

        evidences: List[Evidence] = []
        for item in layer_info:
            idx = int(item["node_index"])
            layer = int(item["layer_number"])
            node = self._ra.tree.all_nodes[idx]
            node_text = node.text

        # 取节点 embedding
            node_emb = None
            try:
                if isinstance(node.embeddings, dict):
                    node_emb = node.embeddings.get(emb_key)
                    if node_emb is None and node.embeddings:
                        node_emb = next(iter(node.embeddings.values()))
                else:
                    node_emb = node.embeddings
            except Exception:
                node_emb = None

        # 计算 cosine distance，再映射到 [0,1] 的相似度 score
        # scipy cosine distance: d = 1 - cos_sim, d∈[0,2]
        # 映射：score = 1 - d/2 ∈ [0,1]
            if node_emb is not None:
                d = float(distance.cosine(q_emb, node_emb))
                score = float(1.0 - d / 2.0)
            else:
                d = None
                score = 0.0

            evidences.append(
                Evidence(
                    content=node_text,
                    metadata={
                        "source": "RAPTOR",
                        "node_index": idx,
                        "layer": layer,
                        "score": score,
                        "distance": d,
                        "score_source": "cosine_score=1-d/2",
                        "emb_key": emb_key,
                    },
                )
            )

        if not evidences and isinstance(context, str) and context.strip():
            evidences.append(
                Evidence(
                    content=context,
                    metadata={"source": "RAPTOR", "node_index": -1, "layer": -1, "score": 0.0, "score_source": "fallback_context"},
                )
            )

        return evidences

    def get_llm_stats(self) -> dict:
        """返回 ingest 阶段 LLM 调用统计（仅 summarization）。"""
        summ = self._summ
        return {
            "llm_calls": getattr(summ, "llm_calls", 0),
            "prompt_tokens": getattr(summ, "prompt_tokens", 0),
            "completion_tokens": getattr(summ, "completion_tokens", 0),
            "total_tokens": getattr(summ, "prompt_tokens", 0) + getattr(summ, "completion_tokens", 0),
        }

    def reset(self) -> None:
        self._buffer.clear()
        self._ra = RetrievalAugmentation(config=self._config, tree=None)


