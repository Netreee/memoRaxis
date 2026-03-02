from typing import Optional, Dict, Any
import os
import re
from urllib.parse import urlparse

from .config import get_config
from .logger import get_logger

logger = get_logger()


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
            },
        },
        "llm": {
            "provider": llm_provider,
            "config": {
                "model": conf.llm.get("model"),
                "api_key": conf.llm.get("api_key"),
                "max_tokens": 2000,
                "temperature": 0.1,
            },
        },
    }

    if include_graph:
        db_conf = conf.database
        # Read graph config strictly from config file (supports memgraph or neo4j keys)
        neo4j_url = db_conf.get("neo4j_url", "bolt://localhost:7687")
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
