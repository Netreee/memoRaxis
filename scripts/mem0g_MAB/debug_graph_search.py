#!/usr/bin/env python3
"""
专项调试：验证 Mem0G 图检索不返回 relations 的根因。

步骤：
  1. 直接查 Neo4j，确认节点存在、user_id 是什么
  2. 模拟 LLM entity extraction，打印原始返回
  3. 用提取到的 entity 直接跑 Cypher，绕过 threshold 限制
  4. 对比 ingest user_id vs search user_id

用法：
  python scripts/mem0g_MAB/debug_graph_search.py \
      --collection mem0g_long_range_0 \
      --query "What happened to Alice?"
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import get_config
from src.mem0_utils import get_mem0_config

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: pip install neo4j")
    sys.exit(1)


SEP = "=" * 60


def step1_inspect_neo4j(uri: str, user: str, pwd: str, db: str) -> None:
    """直接查 Neo4j，打印节点分布和 user_id 列表。"""
    print(f"\n{SEP}")
    print(f"STEP 1: inspect Neo4j  db={db}")
    print(SEP)
    with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
        with driver.session(database=db) as s:
            # 节点总数
            total = s.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            print(f"  Total nodes: {total}")

            # 有 embedding 的节点数
            with_emb = s.run(
                "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) AS cnt"
            ).single()["cnt"]
            print(f"  Nodes with embedding: {with_emb}")

            # 所有 user_id 分布
            uid_rows = s.run(
                "MATCH (n) RETURN n.user_id AS uid, count(n) AS cnt ORDER BY cnt DESC LIMIT 10"
            )
            print("  user_id distribution:")
            for r in uid_rows:
                print(f"    user_id={r['uid']!r}  count={r['cnt']}")

            # 前 5 条节点
            sample = s.run("MATCH (n) RETURN n.name AS name, n.user_id AS uid LIMIT 5")
            print("  Sample nodes:")
            for r in sample:
                print(f"    name={r['name']!r}  user_id={r['uid']!r}")

            # 前 5 条关系
            rels = s.run(
                "MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 5"
            )
            print("  Sample relations:")
            for r in rels:
                print(f"    ({r['a.name']}) -[{r['type(r)']}]-> ({r['b.name']})")


def step2_llm_entity_extraction(query: str) -> list:
    """调用 LLM tool call 提取 query 中的实体，打印原始响应。"""
    print(f"\n{SEP}")
    print("STEP 2: LLM entity extraction")
    print(SEP)

    from mem0.utils.factory import LlmFactory  # type: ignore
    from mem0.graphs.tools import EXTRACT_ENTITIES_TOOL  # type: ignore

    conf = get_config()
    llm_provider = conf.llm.get("provider", "openai")

    # 使用 dict 格式，与 get_mem0_config() 中传给 mem0 的 llm config 完全一致
    llm_conf_dict = {
        "model": conf.llm.get("model"),
        "api_key": conf.llm.get("api_key"),
        "max_tokens": 2000,
        "temperature": 0.1,
    }
    llm = LlmFactory.create(llm_provider, llm_conf_dict)
    print(f"  LLM provider: {llm_provider},  model: {conf.llm.get('model')}")

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a smart assistant who understands entities and their types in a given text. "
                f"If user message contains self reference such as 'I', 'me', 'my' etc. then use ingest_user as the source entity. "
                f"Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question."
            ),
        },
        {"role": "user", "content": query},
    ]

    try:
        resp = llm.generate_response(messages=messages, tools=[EXTRACT_ENTITIES_TOOL])
        print(f"  Raw LLM response keys: {list(resp.keys()) if isinstance(resp, dict) else type(resp)}")
        tool_calls = resp.get("tool_calls", []) if isinstance(resp, dict) else []
        print(f"  tool_calls count: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"    name={tc.get('name')}  args={tc.get('arguments')}")

        entities = []
        for tc in tool_calls:
            if tc.get("name") == "extract_entities":
                for item in tc.get("arguments", {}).get("entities", []):
                    entities.append(item["entity"].lower().replace(" ", "_"))
        print(f"  Extracted entities: {entities}")
        return entities
    except Exception as e:
        print(f"  ERROR during LLM call: {e}")
        import traceback; traceback.print_exc()
        return []


def step3_cypher_bypass(uri: str, user: str, pwd: str, db: str, entities: list, neo4j_user_id: str) -> None:
    """用提取到的 entity embedding 直接跑 Cypher，以不同 threshold 测试。"""
    print(f"\n{SEP}")
    print("STEP 3: direct Cypher embedding search (bypass threshold)")
    print(SEP)

    from mem0.utils.factory import EmbedderFactory  # type: ignore
    conf = get_config()
    embed_provider = conf.embedding.get("provider", "huggingface")

    from mem0.configs.embeddings.base import BaseEmbedderConfig  # type: ignore
    emb_conf = BaseEmbedderConfig(model=conf.embedding.get("model"))
    emb_model = EmbedderFactory.create(embed_provider, emb_conf)

    if not entities:
        print("  No entities to search for — skipping.")
        return

    with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
        with driver.session(database=db) as s:
            for entity in entities[:3]:
                print(f"\n  Entity: {entity!r}")
                n_emb = emb_model.embed(entity)
                cypher = """
                MATCH (n)
                WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
                WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS sim
                RETURN n.name AS name, n.user_id AS uid, sim
                ORDER BY sim DESC LIMIT 10
                """
                rows = s.run(cypher, n_embedding=n_emb, user_id=neo4j_user_id)
                results = list(rows)
                if results:
                    print(f"  Top matches (user_id={neo4j_user_id!r}):")
                    for r in results:
                        print(f"    sim={r['sim']:.4f}  name={r['name']!r}")
                else:
                    print(f"  No matches with user_id={neo4j_user_id!r}")

                    # 再不过滤 user_id 试试
                    cypher2 = """
                    MATCH (n)
                    WHERE n.embedding IS NOT NULL
                    WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS sim
                    RETURN n.name AS name, n.user_id AS uid, sim
                    ORDER BY sim DESC LIMIT 10
                    """
                    rows2 = s.run(cypher2, n_embedding=n_emb)
                    results2 = list(rows2)
                    if results2:
                        print(f"  Top matches WITHOUT user_id filter:")
                        for r in results2:
                            print(f"    sim={r['sim']:.4f}  name={r['name']!r}  user_id={r['uid']!r}")
                    else:
                        print("  Still no matches even without user_id filter — embedding issue or empty DB.")


def main():
    parser = argparse.ArgumentParser(description="Debug mem0g graph search")
    parser.add_argument("--collection", required=True, help="Qdrant/Neo4j collection name, e.g. mem0g_long_range_0")
    parser.add_argument("--query", default="What is the main topic?", help="Query string to test")
    parser.add_argument("--user_id", default="ingest_user", help="user_id used during ingest (default: ingest_user)")
    parser.add_argument("--db", default=None, help="Neo4j database override (default: derived from --collection)")
    args = parser.parse_args()

    conf = get_config()
    neo4j_uri = conf.database.get("neo4j_url", "bolt://localhost:7687")
    neo4j_user = conf.database.get("neo4j_username", "neo4j")
    neo4j_pwd = conf.database.get("neo4j_password", "password")

    import re
    def sanitize(name):
        c = (name or "neo4j").strip().lower().replace("_", "-")
        c = re.sub(r"[^a-z0-9.-]", "-", c)
        return c.strip(".-") or "neo4j"

    neo4j_db = args.db if args.db else sanitize(args.collection)
    print(f"Neo4j: {neo4j_uri}  db={neo4j_db}  user_id={args.user_id!r}")
    print(f"Query: {args.query!r}")

    step1_inspect_neo4j(neo4j_uri, neo4j_user, neo4j_pwd, neo4j_db)
    entities = step2_llm_entity_extraction(args.query)
    step3_cypher_bypass(neo4j_uri, neo4j_user, neo4j_pwd, neo4j_db, entities, args.user_id)

    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
