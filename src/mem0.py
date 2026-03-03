from typing import List, Dict, Any, Optional
from src.memory_interface import BaseMemorySystem, Evidence
from src.logger import get_logger

_logger = get_logger()


class Mem0(BaseMemorySystem):
    def __init__(self, mem0_instance):
        self.mem0 = mem0_instance
        self.user_id = "ingest_user" # Match ingest user

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """
        Adapt Mem0 search to return list of Evidence objects.
        """
        results = self.mem0.search(query, user_id=self.user_id, limit=top_k)
        
        search_hits = results.get("results") if isinstance(results, dict) else results
        
        evidences = []
        for hit in search_hits:
            # 构建 Evidence 对象
            content = hit.get("memory", "")
            metadata = {
                "score": hit.get("score", 0.0),
                "id": hit.get("id", ""),
                "source": "mem0"
            }
            # 如果 hit 中有其他元数据，也添加进去
            if "metadata" in hit and hit["metadata"] is not None:
                metadata.update(hit["metadata"])
            
            evidences.append(Evidence(content=content, metadata=metadata))
        
        return evidences

    def add_memory(self, text: str, metadata: dict = None):
        self.mem0.add(text, user_id=self.user_id, metadata=metadata)
        
    def reset(self):
        self.mem0.reset()


class Mem0G(BaseMemorySystem):
    def __init__(self, mem0_instance, user_id: str = "ingest_user", filters: dict = None):
        self.mem0 = mem0_instance
        self.user_id = user_id
        self.filters = filters

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """
        Retrieve evidence from both:
        - vector memory hits ("results")
        - graph relations ("relations")

        Return unified Evidence list for benchmark.
        """

        # ---- Call Mem0 search ----
        results = self.mem0.search(
            query,
            user_id=self.user_id,
            limit=top_k,
            filters=self.filters
        )

        # ---- Debug: 记录原始返回，验证图检索是否生效 ----
        _logger.debug("[Mem0G.retrieve] raw search result keys: %s", list(results.keys()) if isinstance(results, dict) else type(results))
        if isinstance(results, dict):
            vector_hits = results.get("results", [])
            graph_hits  = results.get("relations", [])
            _logger.info(
                "[Mem0G.retrieve] query=%r | vector_hits=%d | graph_relations=%d",
                query[:60],
                len(vector_hits),
                len(graph_hits),
            )
            if graph_hits:
                for i, rel in enumerate(graph_hits[:5]):
                    _logger.info("  [graph #%d] %s", i, rel)
            else:
                _logger.warning("[Mem0G.retrieve] graph_relations 为空 — Neo4j 未返回任何关系！")

        evidences: List[Evidence] = []

        # ---- Defensive parse ----
        if not isinstance(results, dict):
            raise ValueError(
                f"Mem0G expects dict return, got {type(results)}"
            )

        # ============================
        # 1. Vector-like chunk results
        # ============================
        memory_hits = results.get("results", [])

        for hit in memory_hits:
            content = hit.get("memory", "")

            metadata = {
                "backend": "mem0g",
                "type": "chunk",
                "score": hit.get("score", 0.0),
                "id": hit.get("id", None),
            }

            # Merge extra metadata
            if hit.get("metadata"):
                metadata.update(hit["metadata"])

            evidences.append(
                Evidence(content=content, metadata=metadata)
            )

        # ============================
        # 2. Graph relations evidence
        # ============================
        relations = results.get("relations", [])

        for rel in relations:
            """
            Graph entity format usually like:

            {
                "source": "...",
                "relation": "...",
                "target": "...",
                ...
            }
            """

            # Convert relation into readable text evidence
            source = rel.get("source", "")
            # relation key might be 'relationship' or 'relation'
            relation = rel.get("relation") or rel.get("relationship", "")
            # target key might be 'destination' or 'target'
            target = rel.get("target") or rel.get("destination", "")

            relation_text = f"{source} --[{relation}]--> {target}"

            metadata = {
                "backend": "mem0g",
                "type": "graph_relation",
            }

            # Keep full raw relation for debugging
            metadata.update(rel)

            evidences.append(
                Evidence(content=relation_text, metadata=metadata)
            )

        return evidences

    def add_memory(self, text: str, metadata: Optional[dict] = None):
        """
        Add memory into Mem0 graph system.

        Mem0 will automatically:
        - embed chunk
        - extract entities
        - insert relations into Memgraph
        """

        return self.mem0.add(
            text,
            user_id=self.user_id,
            metadata=metadata
        )

    def reset(self):
        """
        Reset graph + vector memory.

        Mem0 itself may not provide unified reset,
        so here are two strategies:

        1. If mem0 has reset/delete API → call it
        2. Otherwise raise NotImplementedError
        """

        if hasattr(self.mem0, "reset"):
            self.mem0.reset(user_id=self.user_id)

        elif hasattr(self.mem0, "delete_all"):
            self.mem0.delete_all(user_id=self.user_id)

        else:
            raise NotImplementedError(
                "Mem0G reset not supported: no reset/delete_all API found."
            )
