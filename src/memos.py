from typing import List, Dict, Any, Optional
import uuid
import requests
import json
import time
from src.memory_interface import BaseMemorySystem, Evidence
from src.logger import get_logger
from neo4j import GraphDatabase
from src.config import get_config

_logger = get_logger()

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

BASE_URL = "http://127.0.0.1:8000/product"
HEADERS = {"Content-Type": "application/json"}

class MemOS(BaseMemorySystem):
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id

    def add_memory(self, data: str, metadata: Dict[str, Any] = None) -> None:
        metadata = metadata or {}
        payload = {
            "user_id": self.user_id,
            "vector_sync": "success",
            "writable_cube_ids": [self.user_id],
            "messages": [{"role": "user", "content": data}],
        }
        
        try:
            resp = requests.post(
                f"{BASE_URL}/add", headers=HEADERS, data=json.dumps(payload), timeout=60
            )
            # _logger.info(f"Response: {resp.status_code} - {resp.text}")
        except Exception as e:
            _logger.error(f"Request failed with exception: {e!r}")
            return

    def retrieve(self, query: str, top_k: int = 5, conversation_id: str = None) -> List[Evidence]:
        # 截断超长 query，避免 embedding 超时（LRU 等任务的问题可长达 1000+ 字）
        query = query[:500] if len(query) > 500 else query
        payload = json.dumps(
            {
                "query": query,
                "user_id": self.user_id,
                "readable_cube_ids": [self.user_id],
                # "conversation_id": conversation_id,
                "top_k": top_k,
                "include_preference": True,
                "mode": "fast",
                "relativity": 0,
                # "pref_top_k": 6,
            },
            ensure_ascii=False,
        )
        for attempt in range(4):
            try:
                response = requests.request("POST", f"{BASE_URL}/search", data=payload, headers=HEADERS, timeout=120)
                response.raise_for_status()
                result = response.json()
                break
            except Exception as e:
                if attempt < 3:
                    _logger.warning(f"Failed to retrieve (attempt {attempt+1}/4), retrying in 8s: {e}")
                    time.sleep(8)
                else:
                    _logger.error(f"Failed to retrieve after 4 attempts: {e}")
                    return []

        _logger.debug(f"[MemOS] search_memory result: {str(result)[:300]}...")

        data = result.get("data", {})
        evidences = []

        # 只获取 text_mem 类型的记忆
        cubes_data = data.get("text_mem", [])
        if isinstance(cubes_data, list):
            for cube in cubes_data:
                memories = cube.get("memories", [])
                for mem in memories:
                    # 获取记忆内容
                    content = mem.get("memory", "")
                    mem_metadata = mem.get("metadata", {})
                    
                    # 提取关键元数据
                    score = mem_metadata.get("relativity")
                    confidence = mem_metadata.get("confidence")
                    tags = mem_metadata.get("tags")
                    
                    meta: Dict[str, Any] = {
                        "source": "memos",
                        "score": score,
                        "id": mem.get("id"),
                        "confidence": confidence,
                        "conversation_id": conversation_id, 
                        "tags": tags,
                    }
                    evidences.append(Evidence(content=content, metadata=meta))

        # 按相关性分数降序排序
        evidences.sort(key=lambda x: (x.metadata.get("score") or 0.0), reverse=True)

        return evidences[:top_k]

    def reset(self) -> None:
        """
        连接到 system 库，强制删除当前的数据库 (self.user_id)。
        
        注意：该操作会彻底删除该用户的记忆数据。
        """
        _logger.info(f"[MemOS] 正在删除数据库: {self.user_id} ...")

        # 获取 Neo4j 配置
        try:
            conf = get_config()
            db_conf = conf.database
            neo4j_uri = db_conf.get("neo4j_url") or "bolt://localhost:7687"
            neo4j_auth = (
                db_conf.get("neo4j_user") or "neo4j",
                db_conf.get("neo4j_password") or "password"
            )
        except Exception as e:
            _logger.warning(f"[MemOS] 无法加载数据库配置: {e}，跳过重置操作。")
            return

        # 连接 Neo4j 并执行删除
        try:
            # 关键点：必须连接到 "system" 数据库来管理其他数据库
            with GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) as driver:
                with driver.session(database="system") as session:
                    # 1. 先停止数据库 (可选，防止有连接卡住)
                    stop_query = f"STOP DATABASE `{self.user_id}` IF EXISTS WAIT"
                    session.run(stop_query)
                    
                    # 2. 彻底删除
                    drop_query = f"DROP DATABASE `{self.user_id}` IF EXISTS DESTROY DATA WAIT"
                    session.run(drop_query)
                    
                    _logger.info(f"[MemOS] 数据库 {self.user_id} 已成功删除。")
        except Exception as e:
            _logger.error(f"[MemOS] 重置失败: {e}")
