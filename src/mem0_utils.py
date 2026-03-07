from typing import Optional, Dict, Any
import os
import re
from urllib.parse import urlparse

from .config import get_config
from .logger import get_logger

logger = get_logger()


# --- Monkey-patch Mem0 OpenAIEmbedding to support Ark Multimodal ---
from mem0.embeddings.openai import OpenAIEmbedding
import requests

original_init = OpenAIEmbedding.__init__

def patched_init(self, config=None):
    original_init(self, config)
    # Ensure config has api_key and base_url
    # config is a pydantic model in newer mem0ai versions
    self.api_key = getattr(config, "api_key", None) if config else None
    
    # Store model name
    self.model = getattr(config, "model", None) if config else None
    if not self.model and hasattr(self, "config") and hasattr(self.config, "model"):
        self.model = self.config.model
        
    # Some configs might store base_url differently or in extra kwargs
    if config and hasattr(config, "openai_base_url"):
        self.base_url = config.openai_base_url
    elif config and hasattr(config, "model_extra") and config.model_extra:
        self.base_url = config.model_extra.get("openai_base_url")
    else:
        self.base_url = None

def patched_embed(self, text, memory_action=None):
    """Custom embed method that formats the payload for Ark Multimodal."""
    # Check if it's the multimodal provider setup
    conf = get_config()
    if conf.embedding.get("provider") == "ark_multimodal":
        url = self.base_url
        if url and not url.endswith("/embeddings/multimodal"):
            url = f"{url.rstrip('/')}/embeddings/multimodal"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Handle list of texts or single text
        if isinstance(text, str):
            inputs = [{"type": "text", "text": text}]
        else:
            inputs = [{"type": "text", "text": t} for t in text]
            
        payload = {
            "model": self.model,
            "input": inputs
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            # Ark returns data as a dict with "embedding" key directly, not a list of elements
            # Wait, for multiple inputs, how does it return? 
            # In our previous probe, data["data"]["embedding"] was the array. 
            # If text is a single string, it returns a single embedding list.
            if isinstance(text, str):
                return data['data']['embedding']
            else:
                # If mem0 expects a list of embeddings for list of texts
                # This might need adjustment if Ark returns multiple embeddings differently
                return [data['data']['embedding']] # Simplified assumption
        except Exception as e:
            logger.error(f"Monkey-patched Ark Multimodal Embedding failed: {e}")
            if 'response' in locals():
                logger.error(f"Response: {response.text}")
            # Fallback zero vector
            dim = conf.embedding.get("dim", 2048)
            if isinstance(text, str):
                return [0.0] * dim
            return [[0.0] * dim for _ in text]
    else:
        # Fallback to original litellm embedding if not ark_multimodal
        from litellm import embedding
        response = embedding(
            model=self.model,
            input=text,
            api_key=self.api_key,
            api_base=self.base_url,
        )
        return [r["embedding"] for r in response.data]

OpenAIEmbedding.__init__ = patched_init
OpenAIEmbedding.embed = patched_embed
# -------------------------------------------------------------------


def _sanitize_neo4j_database_name(name: str) -> str:
    """Neo4j DB names allow lowercase ascii letters, digits, dots and dashes."""
    candidate = (name or "neo4j").strip().lower().replace("_", "-")
    candidate = re.sub(r"[^a-z0-9.-]", "-", candidate)
    candidate = candidate.strip(".-")
    return candidate or "neo4j"


def get_mem0_config(
    collection_name: str,
    include_graph: bool = False,
) -> Dict[str, Any]:
    """Build the Mem0 configuration and optionally attach graph storage."""
    conf = get_config()

    # Per requirement: forbid changing Qdrant port/url via CLI/config file
    # Use Docker configuration (environment variables)
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

    llm_provider = conf.llm.get("provider", "openai")
    embed_provider = conf.embedding.get("provider", "openai")
    if embed_provider == "ark_multimodal":
        embed_provider = "openai"  # Force openai provider so mem0 accepts it (we monkey-patched it)

    config: Dict[str, Any] = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": conf.embedding.get("dim", 384),
                "host": qdrant_host,
                "port": qdrant_port,
            },
        },
        "embedder": {
            "provider": embed_provider,
            "config": {
                "model": conf.embedding.get("model"),
                "api_key": conf.embedding.get("api_key"),
                "openai_base_url": conf.embedding.get("base_url"),
            },
        },
        "llm": {
            "provider": llm_provider,
            "config": {
                "model": conf.llm.get("model"),
                "api_key": conf.llm.get("api_key"),
                "openai_base_url": conf.llm.get("base_url"),
                "max_tokens": 4096,
                "temperature": 0.1,
            },
        },
    }

    if include_graph:
        db_conf = conf.database
        # Read graph config from config file; NEO4J_URL env var takes priority.
        # This allows parallel ingest processes to each point at a different
        # Docker Neo4j CE instance without touching the shared config file.
        # Usage: NEO4J_URL=bolt://localhost:7688 python mem0g_MAB/ingest.py ...
        neo4j_url = (
            os.getenv("NEO4J_URL")
            or db_conf.get("neo4j_url", "bolt://localhost:7687")
        )
        neo4j_username = db_conf.get("neo4j_username", "neo4j")
        neo4j_password = db_conf.get("neo4j_password", "password")
        # Neo4j specific: use collection_name as target DB name.
        # Community Edition only supports 'neo4j' and 'system'.
        target_db = _sanitize_neo4j_database_name(collection_name if collection_name else "neo4j")
        
        # Check and create database if needed
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))
            with driver.session(database="system") as session:
                # Check if database exists
                result = session.run("SHOW DATABASES")
                existing_dbs = [record["name"] for record in result]
                
                if target_db not in existing_dbs:
                    logger.info(f"Database '{target_db}' not found. Attempting to create it...")
                    try:
                        session.run(f"CREATE DATABASE `{target_db}`")
                        logger.info(f"Successfully created database '{target_db}'")
                    except Exception as e:
                        logger.warning(f"Failed to create database '{target_db}'. It might be a Community Edition restriction. Falling back to 'neo4j'. Error: {e}")
                        target_db = "neo4j"
            driver.close()
        except ImportError:
            logger.warning("neo4j python package not installed. Skipping database check/creation.")
        except Exception as e:
            logger.warning(f"Could not verify/create Neo4j database: {e}. Defaulting to 'neo4j'.")
            target_db = "neo4j"

        config["graph_store"] = {
            "provider": "neo4j",
            "config": {
                "url": neo4j_url,
                "username": neo4j_username,
                "password": neo4j_password,
                # NOTE: mem0ai==1.0.3 与部分 langchain-neo4j 版本组合下，
                # 显式 database 可能被下游错误映射到 bearer token 参数。
                # 这里省略 database，使用 Neo4j 默认数据库（通常为 neo4j）。
            },
        }

        # mem0ai==1.0.3 does not reliably pass graph_store.config.database through
        # with newer langchain-neo4j; use env var for selected DB.
        os.environ["NEO4J_DATABASE"] = target_db

        # Explicit dims help when graph reasoning expects them.
        config["embedder"]["config"]["embedding_dims"] = conf.embedding.get("dim", 384)

    return config
