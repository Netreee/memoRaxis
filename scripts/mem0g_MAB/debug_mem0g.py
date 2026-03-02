#!/usr/bin/env python3
"""Mem0G debug utility to validate vector and graph paths."""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None 

# Allow imports from the project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from mem0 import Memory  # type: ignore

from src.mem0 import Mem0G
from src.mem0_utils import get_mem0_config

LOGGER = logging.getLogger("mem0g.debug")


def configure_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mem0G graph and vector sanity checks")
    parser.add_argument("--collection", default="mem0g_debug_collection", help="Qdrant collection name")
    parser.add_argument("--user", default="mem0g_debug_user", help="User identifier used for the run")
    parser.add_argument("--skip-add", action="store_true", help="Skip adding sample memories")
    parser.add_argument("--skip-search", action="store_true", help="Skip search validation")
    parser.add_argument("--reset", action="store_true", help="Reset the user namespace before running")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to wait between insert and search")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def print_header(title: str) -> None:
    border = "=" * 60
    print(f"\n{border}\n{title}\n{border}")


def summarize_records(label: str, raw: Any) -> None:
    """Render a concise view of both vector hits and relations."""
    if raw is None:
        print(f"{label}: no data returned")
        return

    if isinstance(raw, dict):
        results = raw.get("results", [])
        relations = raw.get("relations", [])
    elif isinstance(raw, list):
        results = raw
        relations = []
    else:
        print(f"{label}: unexpected payload type {type(raw)}")
        return

    print(f"{label}: {len(results)} vector hits, {len(relations)} relations")

    for idx, item in enumerate(results[:5], start=1):
        memory_text = item.get("memory") or item.get("value") or ""
        score = item.get("score")
        identifier = item.get("id")
        print(f"  [{idx}] score={score} id={identifier}\n      {memory_text}")

    for idx, relation in enumerate(relations[:5], start=1):
        source = relation.get("source")
        rel = relation.get("relation")
        target = relation.get("target")
        print(f"  (relation {idx}) {source} --[{rel}]--> {target}")


def sanitize_db_name(name: str) -> str:
    # Neo4j DB names: lowercase ascii letters, digits, dots and dashes.
    candidate = (name or "neo4j").strip().lower().replace("_", "-")
    candidate = re.sub(r"[^a-z0-9.-]", "-", candidate)
    candidate = candidate.strip(".-")
    return candidate or "neo4j"


def ensure_graph_database(uri: str, user: str, password: str, database: str) -> str:
    """Ensure target database exists and return the database to use."""
    if GraphDatabase is None:
        print("⚠ neo4j driver unavailable; fallback to default database 'neo4j'.")
        return "neo4j"

    target_db = sanitize_db_name(database)
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database="system") as system_session:
                dbs = [record["name"] for record in system_session.run("SHOW DATABASES")]
                if target_db not in dbs:
                    try:
                        system_session.run(f"CREATE DATABASE `{target_db}`")
                        print(f"✓ Created database: {target_db}")
                    except Exception as exc:
                        print(f"⚠ Could not create database '{target_db}', fallback to 'neo4j': {exc}")
                        target_db = "neo4j"
                else:
                    print(f"✓ Database exists: {target_db}")
    except Exception as exc:
        print(f"⚠ Could not verify/create database '{target_db}', fallback to 'neo4j': {exc}")
        target_db = "neo4j"

    return target_db


def inspect_graph_connection(config: Dict[str, Any], target_db: str) -> None:
    """Validate Neo4j connection and inspect graph content."""
    print_header("Step 3.5: Inspect Graph Connection")

    if GraphDatabase is None:
        print("⚠ Neo4j driver not available (pip install neo4j).")
        return

    graph_conf = config.get("graph_store", {}).get("config", {})
    uri = graph_conf.get("url")
    user = graph_conf.get("username")
    password = graph_conf.get("password")

    if not (uri and user and password):
        print("⚠ Graph store configuration missing or incomplete.")
        return

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            print(f"✓ Connected to Neo4j at {uri}")

            with driver.session() as session:
                # 1. List databases
                try:
                    # 'SHOW DATABASES' is standard in Neo4j 4.x/5.x
                    result = session.run("SHOW DATABASES")
                    dbs = [record["name"] for record in result]
                    print(f"  Found {len(dbs)} databases: {', '.join(dbs[:5])}{'...' if len(dbs) > 5 else ''}")
                except Exception as e:
                    print(f"  ⚠ Could not list databases: {e}")

                # Check if target_db actually exists in the list
                if "dbs" in locals() and target_db not in dbs and "system" not in dbs:
                    print(f"  ⚠ Target database '{target_db}' not found in server.")
                
                print(f"  Inspect nodes in database: {target_db}")
                # We need a new session for a specific DB if it's not the default
                # But let's try generic query first.
                
                # For safety, use a separate session for the targeted DB query
                try:
                    with driver.session(database=target_db) as db_session:
                        result = db_session.run("MATCH (n) RETURN n LIMIT 20")
                        nodes = list(result)
                        print(f"  Found {len(nodes)} sample nodes in '{target_db}':")
                        for record in nodes:
                            node = record["n"]
                            labels = list(node.labels)
                            props = dict(node)
                            print(f"    - Labels: {labels}, Props: {props}")
                            
                        # Count total
                        count_res = db_session.run("MATCH (n) RETURN count(n) as count")
                        total = count_res.single()["count"]
                        print(f"  Total nodes in '{target_db}': {total}")
                except Exception as e:
                    print(f"  ⚠ Failed to query database '{target_db}': {e}")

    except Exception as exc:
        print(f"✗ Graph connection failed: {exc}")
        LOGGER.exception("Graph connection error")


def add_sample_memories(mem0g: Mem0G) -> None:
    """Insert a fixed mini knowledge graph for validation."""
    samples = [
        {
            "text": "Alice Johnson leads Project Atlas at Contoso Labs.",
            "metadata": {"topic": "team", "entity": "Alice Johnson"},
        },
        {
            "text": "Project Atlas is a research effort located in Berlin, Germany.",
            "metadata": {"topic": "project", "location": "Berlin"},
        },
        {
            "text": "Bob Smith collaborates with Alice Johnson on Project Atlas.",
            "metadata": {"topic": "team", "entity": "Bob Smith"},
        },
        {
            "text": "Contoso Labs is a subsidiary of Wide World Importers.",
            "metadata": {"topic": "org", "entity": "Contoso Labs"},
        },
    ]

    print_header("Step 4: Insert sample memories")
    for entry in samples:
        LOGGER.info("Adding memory: %s", entry["text"])
        try:
            mem0g.add_memory(entry["text"], metadata=entry.get("metadata"))
            print(f"✓ Added: {entry['text']}")
        except Exception as exc:
            print(f"✗ Failed to add: {entry['text']}")
            LOGGER.exception("Add failed: %s", exc)


def validate_search(mem0g: Mem0G, user_id: str, queries: List[str], delay: float) -> None:
    """Run a set of searches and display both chunks and relations."""
    print_header("Step 6: Search checks")
    time.sleep(max(delay, 0.0))

    for query in queries:
        print(f"\nQuery: {query}")
        try:
            evidences = mem0g.retrieve(query, top_k=5)
            print(f"  Evidence objects: {len(evidences)}")
            for idx, evidence in enumerate(evidences, start=1):
                print(f"    [{idx}] {evidence.content}")
            raw = mem0g.mem0.search(query, user_id=user_id, limit=5)
            summarize_records("  Raw search", raw)
        except Exception as exc:
            print(f"✗ Search failed: {exc}")
            LOGGER.exception("Search exception")


def reset_namespace(mem0g: Mem0G, user_id: str) -> None:
    try:
        mem0g.reset()
        print(f"✓ Reset memory for user {user_id}")
    except NotImplementedError:
        print("⚠ Reset not supported by backend")
    except Exception as exc:
        print(f"✗ Reset failed: {exc}")
        LOGGER.exception("Reset failed")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    print_header("Step 1: Build configuration")
    # Use a specific debug collection name that will be used as the database name
    debug_collection = args.collection if args.collection else "mem0g_debug_db"
    
    try:
        mem0_config = get_mem0_config(
            collection_name=debug_collection,
            include_graph=True,
        )
    except Exception as exc:
        print(f"✗ Failed to load configuration: {exc}")
        LOGGER.exception("Configuration error")
        sys.exit(1)

    print("Configuration preview:")
    print(mem0_config)

    graph_conf = mem0_config.get("graph_store", {}).get("config", {})
    neo4j_uri = graph_conf.get("url")
    neo4j_user = graph_conf.get("username")
    neo4j_password = graph_conf.get("password")
    requested_db = sanitize_db_name(debug_collection)
    selected_db = "neo4j"
    if neo4j_uri and neo4j_user and neo4j_password:
        selected_db = ensure_graph_database(neo4j_uri, neo4j_user, neo4j_password, requested_db)
    os.environ["NEO4J_DATABASE"] = selected_db
    print(f"Using Qdrant collection: {debug_collection}")
    print(f"Using Neo4j database (derived from collection): {selected_db}")

    # # Move inspection BEFORE initialization to debug connection issues
    # inspect_graph_connection(mem0_config, target_db=selected_db)

    print_header("Step 2: Initialize Mem0G")
    try:
        mem0_instance = Memory.from_config(mem0_config)
        mem0g = Mem0G(mem0_instance)
        mem0g.user_id = args.user  # override default ingest user for isolation
        print("✓ Mem0G initialized")
    except Exception as exc:
        print(f"✗ Failed to initialize Mem0G: {exc}")
        LOGGER.exception("Initialization error")
        sys.exit(1)

    if args.reset:
        print_header("Step 3: Reset namespace")
        reset_namespace(mem0g, args.user)

    print_header("Step 3: Inspect existing data")
    try:
        existing = mem0g.mem0.get_all(user_id=args.user, limit=20)
        summarize_records("Existing", existing)
    except Exception as exc:
        print(f"✗ Failed to inspect existing data: {exc}")
        LOGGER.exception("Inspection error")

    if not args.skip_add:
        add_sample_memories(mem0g)
    else:
        print("Skipping inserts as requested")

    print_header("Step 5: Inspect data after inserts")
    try:
        updated = mem0g.mem0.get_all(user_id=args.user, limit=20)
        summarize_records("Updated", updated)
    except Exception as exc:
        print(f"✗ Failed to fetch updated data: {exc}")
        LOGGER.exception("Post-insert inspection error")

    if not args.skip_search:
        validate_search(
            mem0g,
            args.user,
            queries=[
                "Project Atlas",
                "Where is Project Atlas located?",
                "Alice Johnson",
            ],
            delay=args.sleep,
        )
    else:
        print("Skipping search validation as requested")

    print_header("Step 7: Debug completed")


if __name__ == "__main__":
    main()
