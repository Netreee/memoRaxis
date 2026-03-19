# -*- coding: utf-8 -*-
"""A-MEM (Agentic Memory) 适配器

将 agiresearch/A-mem 的 AgenticMemorySystem 包装为 BaseMemorySystem 接口，
支持 ChromaDB 持久化、Ark LLM 端点和 response_format 降级。
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.memory_interface import BaseMemorySystem, Evidence
from src.config import get_config

# A-MEM imports (via external/amem_repo)
from external.amem_repo.agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
from external.amem_repo.agentic_memory.llm_controller import LLMController
from external.amem_repo.agentic_memory.retrievers import ChromaRetriever

logger = logging.getLogger(__name__)


class AMemMemory(BaseMemorySystem):
    """A-MEM 记忆系统适配器

    核心特性:
    - ChromaDB PersistentClient 持久化 (每个 instance 一个目录)
    - 本地 SentenceTransformer embedding (all-MiniLM-L6-v2, 384-dim)
    - 记忆演化 (evolution): 每条新记忆触发 LLM 分析 + 邻居更新
    - Fornax / Ark / OpenAI 兼容 (provider: fornax|openai in config)
    - response_format 降级 (json_schema → json_object, Ark 用)
    - 限速自动重试 (429 / TPM 错误指数退避)
    """

    def __init__(
        self,
        chroma_dir: str,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        enable_evolution: bool = True,
        evo_threshold: int = 100,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        conf = get_config()
        self._model = llm_model or conf.llm.get("model") or "placeholder"
        self._base_url = llm_base_url or conf.llm.get("base_url") or ""
        # When using Fornax provider, there's no api_key in config; use a placeholder
        # since we'll replace the OpenAI client with FornaxOpenAI in __init__ anyway.
        self._api_key = llm_api_key or conf.llm.get("api_key") or "placeholder"
        self._chroma_dir = str(chroma_dir)
        self._model_name = model_name
        self._enable_evolution = enable_evolution

        # LLM stats
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._stats_lock = threading.Lock()

        # --- Init AgenticMemorySystem (ephemeral ChromaDB, will be replaced) ---
        self._amem = AgenticMemorySystem(
            model_name=model_name,
            llm_backend="openai",
            llm_model=self._model,
            api_key=self._api_key,
            evo_threshold=evo_threshold,
        )

        # Replace LLM controller with one that has base_url
        self._amem.llm_controller = LLMController(
            backend="openai",
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
        )

        # If provider is fornax, swap out the OpenAI client with FornaxOpenAI
        provider = conf.llm.get("provider", "openai")
        self._is_fornax = provider == "fornax"
        if self._is_fornax:
            from fornax.fornax_openai import FornaxOpenAI
            fornax_client = FornaxOpenAI(
                ak=conf.llm["fornax_ak"],
                sk=conf.llm["fornax_sk"],
                prompt_key=conf.llm.get("fornax_prompt_key", "lark.memo.main"),
                env={"SERVICE_ENV": "boe", "RUNTIME_IDC_NAME": "boe", "FORNAX_CUSTOM_REGION": "CN"},
            )
            # FornaxOpenAI has the same .chat.completions.create() interface as openai.OpenAI
            self._amem.llm_controller.llm.client = fornax_client
            logger.info("A-MEM LLM backend: Fornax (prompt_key=%s)", conf.llm.get("fornax_prompt_key"))
        else:
            logger.info("A-MEM LLM backend: OpenAI-compatible (model=%s)", self._model)

        # Wrap get_completion for response_format fallback + retry + stats
        self._patch_llm()

        # Replace ephemeral retriever with persistent ChromaDB
        self._setup_persistent_retriever()

        # Load existing memories dict from pickle (if resuming)
        self._load_memories()

        # Patch consolidate_memories to use persistent retriever
        self._patch_consolidate()

        # Patch find_related_memories to truncate neighbor content (prevents Fornax output truncation)
        self._patch_find_related_memories()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup_persistent_retriever(self):
        """Replace ephemeral ChromaDB with PersistentClient."""
        Path(self._chroma_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=self._chroma_dir)
        ef = SentenceTransformerEmbeddingFunction(model_name=self._model_name)
        collection = client.get_or_create_collection(
            name="memories", embedding_function=ef
        )

        # Build a ChromaRetriever-compatible object with persistent backend
        retriever = ChromaRetriever.__new__(ChromaRetriever)
        retriever.client = client
        retriever.embedding_function = ef
        retriever.collection = collection
        self._amem.retriever = retriever

    def _patch_llm(self):
        """Wrap get_completion for: retry on rate limits, json_schema→json_object fallback, stats."""
        original_fn = self._amem.llm_controller.llm.get_completion
        adapter = self

        def wrapped_get_completion(prompt, response_format=None, temperature=0.7):
            max_retries = 8
            wait = 30
            # Fornax ignores response_format (silently via **kwargs), and doesn't need it;
            # pass None to avoid confusion and potential future errors.
            rf = None if adapter._is_fornax else response_format
            for attempt in range(max_retries):
                try:
                    result = original_fn(prompt, response_format=rf, temperature=temperature)
                    with adapter._stats_lock:
                        adapter._llm_calls += 1
                    # Best-effort JSON repair for Fornax (no structured output enforcement)
                    if adapter._is_fornax and isinstance(result, str):
                        result = AMemMemory._try_extract_json(result)
                    return result
                except Exception as e:
                    err_str = str(e)
                    # Rate limit → exponential backoff
                    # Fornax uses FornaxAPIError with "TPM" keyword; Ark uses HTTP 429
                    is_rate_limit = (
                        "429" in err_str
                        or "RateLimit" in err_str
                        or "rate_limit" in err_str.lower()
                        or "TPM" in err_str
                        or "tpm" in err_str.lower()
                    )
                    if is_rate_limit:
                        if attempt < max_retries - 1:
                            logger.warning(
                                "A-MEM LLM rate limit, waiting %ds (attempt %d/%d): %s",
                                wait, attempt + 1, max_retries, err_str[:120],
                            )
                            _time.sleep(wait)
                            wait = min(wait * 2, 300)
                            continue
                    # json_schema not supported → degrade to json_object (Ark only)
                    if rf and rf.get("type") == "json_schema":
                        logger.warning(
                            "json_schema not supported, falling back to json_object: %s",
                            err_str[:200],
                        )
                        rf = {"type": "json_object"}
                        continue
                    raise
            raise RuntimeError("A-MEM LLM call failed after max retries")

        self._amem.llm_controller.llm.get_completion = wrapped_get_completion

    def _patch_consolidate(self):
        """Override consolidate_memories to use persistent retriever."""
        amem = self._amem
        adapter = self

        def patched_consolidate():
            # Recreate persistent collection (clear + re-add)
            client = chromadb.PersistentClient(path=adapter._chroma_dir)
            try:
                client.delete_collection("memories")
            except Exception:
                pass
            ef = SentenceTransformerEmbeddingFunction(
                model_name=adapter._model_name
            )
            collection = client.get_or_create_collection(
                name="memories", embedding_function=ef
            )
            retriever = ChromaRetriever.__new__(ChromaRetriever)
            retriever.client = client
            retriever.embedding_function = ef
            retriever.collection = collection
            amem.retriever = retriever

            for memory in amem.memories.values():
                metadata = _note_to_metadata(memory)
                amem.retriever.add_document(memory.content, metadata, memory.id)

        amem.consolidate_memories = patched_consolidate

    def _patch_find_related_memories(self, max_content_chars: int = 400):
        """Truncate neighbor content in evolution prompt.

        Root cause: find_related_memories embeds full chunk content (up to 24K chars)
        for each of 5 neighbors → prompt input easily 100K+ chars → Fornax output
        gets truncated mid-JSON.  Tags/context/keywords are sufficient for the
        evolution decision; the full content is not needed.
        """
        amem = self._amem

        def patched_find_related_memories(query, k=5):
            try:
                results = amem.retriever.search(query, k)
            except Exception as e:
                logger.error("find_related_memories search error: %s", e)
                return "", []

            memory_str = ""
            indices = []
            if "ids" not in results or not results["ids"] or not results["ids"][0]:
                return "", []

            for i, doc_id in enumerate(results["ids"][0]):
                if i >= len(results["metadatas"][0]):
                    continue
                metadata = results["metadatas"][0][i]
                content = metadata.get("content", "")
                if len(content) > max_content_chars:
                    content = content[:max_content_chars] + "..."
                memory_str += (
                    f"memory index:{i}\t"
                    f"talk start time:{metadata.get('timestamp', '')}\t"
                    f"memory content: {content}\t"
                    f"memory context: {metadata.get('context', '')}\t"
                    f"memory keywords: {str(metadata.get('keywords', []))}\t"
                    f"memory tags: {str(metadata.get('tags', []))}\n"
                )
                indices.append(i)

            return memory_str, indices

        amem.find_related_memories = patched_find_related_memories

    @staticmethod
    def _try_extract_json(text: str) -> str:
        """Best-effort JSON extraction from a possibly-truncated LLM response.

        Handles three common Fornax failure modes:
        1. Extra data after JSON  → take the first {...} block
        2. Truncated mid-string   → find last complete key-value pair and close brackets
        3. Minor formatting issues → return as-is and let caller handle
        """
        text = text.strip()

        # Fast path: already valid
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Case 1: extra trailing content — extract first balanced {...} block
        depth = 0
        start = text.find("{")
        if start != -1:
            for idx in range(start, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : idx + 1]
                        try:
                            json.loads(candidate)
                            logger.debug("JSON repaired: extracted first balanced block")
                            return candidate
                        except json.JSONDecodeError:
                            break

        # Case 2: response truncated — strip incomplete tail and close open brackets
        # Remove trailing partial token (unfinished string / bare word)
        cleaned = re.sub(r'[^}\]"]*$', "", text.rstrip())
        # Count unclosed brackets
        opens = cleaned.count("{") - cleaned.count("}")
        arr_opens = cleaned.count("[") - cleaned.count("]")
        if opens > 0 or arr_opens > 0:
            # Strip trailing comma before closing
            cleaned = re.sub(r",\s*$", "", cleaned)
            cleaned += "]" * max(arr_opens, 0) + "}" * max(opens, 0)
            try:
                json.loads(cleaned)
                logger.debug("JSON repaired: closed truncated brackets")
                return cleaned
            except json.JSONDecodeError:
                pass

        # Give up — return original and let caller's JSONDecodeError propagate
        return text

    @property
    def _memories_pkl_path(self) -> Path:
        return Path(self._chroma_dir) / "memories.pkl"

    def _load_memories(self):
        """Load pickled memories dict (for resume)."""
        if self._memories_pkl_path.exists():
            with open(self._memories_pkl_path, "rb") as f:
                self._amem.memories = pickle.load(f)
            logger.info(
                "Loaded %d memories from %s",
                len(self._amem.memories), self._memories_pkl_path,
            )

    # ------------------------------------------------------------------
    # Public API (BaseMemorySystem)
    # ------------------------------------------------------------------

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        time_val = metadata.get("time")
        if self._enable_evolution:
            # 1) LLM call: analyze content → keywords / context / tags
            analysis = self._amem.analyze_content(data)
            if not isinstance(analysis, dict):
                analysis = {"keywords": [], "context": "General", "tags": []}
            # 2) add_note → process_memory (evolution LLM call if neighbors exist)
            self._amem.add_note(
                content=data,
                time=time_val,
                keywords=analysis.get("keywords", []),
                context=analysis.get("context", "General"),
                tags=analysis.get("tags", []),
            )
        else:
            # No LLM calls — raw storage only
            note = MemoryNote(content=data, timestamp=time_val)
            self._amem.memories[note.id] = note
            md = _note_to_metadata(note)
            self._amem.retriever.add_document(note.content, md, note.id)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        results = self._amem.search_agentic(query, k=top_k)
        evidences = []
        for r in results:
            evidences.append(
                Evidence(
                    content=r.get("content", ""),
                    metadata={
                        "source": "A-MEM",
                        "id": r.get("id", ""),
                        "context": r.get("context", ""),
                        "keywords": r.get("keywords", []),
                        "tags": r.get("tags", []),
                        "score": r.get("score", 0.0),
                        "is_neighbor": r.get("is_neighbor", False),
                    },
                )
            )
        return evidences

    def reset(self) -> None:
        self._amem.memories.clear()
        self._setup_persistent_retriever()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self):
        """Persist memories dict to pickle (ChromaDB auto-persists)."""
        with open(self._memories_pkl_path, "wb") as f:
            pickle.dump(self._amem.memories, f)
        logger.info(
            "Saved %d memories to %s",
            len(self._amem.memories), self._memories_pkl_path,
        )

    def get_llm_stats(self) -> dict:
        return {
            "llm_calls": self._llm_calls,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
        }

    @property
    def memory_count(self) -> int:
        return len(self._amem.memories)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _note_to_metadata(note: MemoryNote) -> dict:
    return {
        "id": note.id,
        "content": note.content,
        "keywords": note.keywords,
        "links": note.links,
        "retrieval_count": note.retrieval_count,
        "timestamp": note.timestamp,
        "last_accessed": note.last_accessed,
        "context": note.context,
        "evolution_history": note.evolution_history,
        "category": note.category,
        "tags": note.tags,
    }
