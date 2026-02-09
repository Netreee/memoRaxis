# -*- coding: utf-8 -*-
"""基于 PostgreSQL + pgvector 的 RAG 记忆系统"""

import json
import logging
import uuid
import requests
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import psycopg2
from psycopg2.extras import Json
from src.memory_interface import BaseMemorySystem, Evidence
from src.config import get_config

# 尝试导入 openai，如果不存在则报错
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("simple_memory requires 'openai' package.")


class SimpleRAGMemory(BaseMemorySystem):
    """
    基于 PostgreSQL + pgvector 的简单 RAG 记忆系统。
    继承自 memoRaxis 的 BaseMemorySystem。
    """

    def __init__(self, table_name: str = "memory_records"):
        self._logger = logging.getLogger(__name__)
        self._config = get_config()
        self._table_name = table_name

        # 初始化 Embedding 配置
        emb_conf = self._config.embedding
        self._emb_provider = emb_conf.get("provider", "openai_compat")
        self._emb_base_url = emb_conf.get("base_url")
        self._emb_api_key = emb_conf.get("api_key")
        self._emb_model = emb_conf.get("model", "text-embedding-v4")
        self._emb_dim = emb_conf.get("dim", 1536)

        # 仅当非 ark_multimodal 时才初始化 OpenAI 客户端
        if self._emb_provider != "ark_multimodal":
            self._emb_client = OpenAI(
                api_key=self._emb_api_key,
                base_url=self._emb_base_url
            )
        else:
            self._emb_client = None

        # 初始化数据库连接
        db_conf = self._config.database
        self._db_url = db_conf.get("url")
        self._init_db()

    def _init_db(self):
        """初始化数据库表和扩展"""
        conn = psycopg2.connect(self._db_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            # 启用 pgvector 扩展
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # 创建表
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id UUID PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR({self._emb_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # 创建向量索引 (IVFFlat 或 HNSW，这里用简单的 IVFFlat)
            # 注意：数据量少时创建索引可能会失败或不生效，但在生产中是必须的
            # 这里为了简单暂不自动创建索引，依赖 pgvector 的默认扫描
        conn.close()
        self._logger.info(f"Database initialized: table '{self._table_name}' ready.")

    def _get_embedding(self, text: str) -> List[float]:
        """获取 Embedding，支持 OpenAI 兼容接口和 Ark 多模态接口"""
        text = text.replace("\n", " ")
        
        if self._emb_provider == "ark_multimodal":
            return self._get_embedding_ark_multimodal(text)
        
        try:
            resp = self._emb_client.embeddings.create(
                input=[text],
                model=self._emb_model
            )
            return resp.data[0].embedding
        except Exception as e:
            self._logger.error(f"Embedding failed: {e}")
            # 返回零向量作为 fallback，或者直接抛出
            return [0.0] * self._emb_dim

    def _get_embedding_ark_multimodal(self, text: str) -> List[float]:
        """调用 Ark 多模态 Embedding 接口"""
        url = f"{self._emb_base_url}/embeddings/multimodal" if not self._emb_base_url.endswith("/embeddings/multimodal") else self._emb_base_url
        # 如果 base_url 只是 host，需要补全路径
        if "/embeddings/multimodal" not in url:
             url = f"{self._emb_base_url.rstrip('/')}/embeddings/multimodal"

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self._emb_api_key
        }
        payload = {
            "model": self._emb_model,
            "input": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['data']['embedding']
        except Exception as e:
            self._logger.error(f"Ark Multimodal Embedding failed: {e}")
            if 'response' in locals():
                self._logger.error(f"Response content: {response.text}")
            import traceback
            self._logger.error(traceback.format_exc())
            return [0.0] * self._emb_dim

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        """实现 BaseMemorySystem 的 add_memory 接口"""
        vector = self._get_embedding(data)
        record_id = uuid.uuid4()
        
        conn = psycopg2.connect(self._db_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._table_name} (id, content, metadata, embedding) VALUES (%s, %s, %s, %s)",
                (str(record_id), data, Json(metadata), vector)
            )
        conn.close()
        self._logger.debug("添加记忆: %s... (ID: %s)", data[:30], record_id)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """实现 BaseMemorySystem 的 retrieve 接口"""
        query_vector = self._get_embedding(query)
        
        conn = psycopg2.connect(self._db_url)
        with conn.cursor() as cur:
            # 使用 pgvector 的 <-> (欧氏距离) 或 <=> (余弦距离) 操作符
            # 这里使用余弦距离 (cosine distance)，注意 pgvector 的 <=> 是 cosine distance
            # 相似度 = 1 - distance
            cur.execute(
                f"""
                SELECT content, metadata, 1 - (embedding <=> %s::vector) as score
                FROM {self._table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vector, query_vector, top_k)
            )
            rows = cur.fetchall()
            
        conn.close()
        
        evidence_list = []
        for row in rows:
            content, metadata, score = row
            # 可以选择把 score 放进 metadata
            if metadata is None:
                metadata = {}
            metadata["score"] = float(score)
            
            evidence_list.append(Evidence(content=content, metadata=metadata))
            
        self._logger.debug("检索完成: query='%s', 结果数=%d", query[:50], len(evidence_list))
        return evidence_list

    def reset(self) -> None:
        """重置记忆系统：删除并重新创建表（以处理维度变化）"""
        conn = psycopg2.connect(self._db_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self._table_name};")
        conn.close()
        self._init_db()
        self._logger.info(f"记忆已重置 (表 {self._table_name} 已重建)")