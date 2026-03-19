import argparse
import json
import os
import sys
import re
import time
from pathlib import Path
from typing import List, Optional
from mem0 import Memory
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.mem0 import Mem0G
from src.logger import get_logger
from src.config import get_config
from src.mem0_utils import get_mem0_config
from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor, AdaptorResult
from MemoryAgentBench.utils.templates import get_template

logger = get_logger()

# --- Shared Utilities ---

def evaluate_adaptor(name: str, adaptor, questions: list, limit: int, llm, template_name: str = None, template_type: str = 'query', agent_type: str = 'rag_agent') -> list:
    results = []
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)

    query_template = None
    if template_name:
        query_template = get_template(template_name, template_type, agent_type)

    for i, q in enumerate(target_questions):
        logger.info(f"[{name}] Running Q{i+1}/{total}: {q}")
        formatted_q = query_template.format(question=q) if query_template else q
        try:
            llm.reset_stats()
            t0 = time.time()
            res: AdaptorResult = adaptor.run(formatted_q)
            latency = round(time.time() - t0, 2)
            tokens = res.token_consumption
            logger.info(f"[{name}] Q{i+1} done | latency={latency}s | tokens={tokens} | steps={res.steps_taken}")
            results.append({
                "question": q,
                "answer": res.answer,
                "steps": res.steps_taken,
                "tokens": tokens,
                "latency_s": latency,
                "replan": res.replan_count
            })
        except Exception as e:
            logger.error(f"[{name}] Failed on Q{i+1}: {e}", exc_info=True)
            results.append({"question": q, "error": str(e)})
    return results

def sanitize_neo4j_db_name(name: str) -> str:
    """Neo4j DB names: lowercase letters/digits/dot/dash; no underscore."""
    candidate = (name or "neo4j").strip().lower().replace("_", "-")
    candidate = re.sub(r"[^a-z0-9.-]", "-", candidate)
    candidate = candidate.strip(".-")
    return candidate or "neo4j"

def ensure_neo4j_database(uri: str, username: str, password: str, db_name: str) -> str:
    """Ensure target Neo4j DB exists, fallback to neo4j on failure."""
    target_db = sanitize_neo4j_db_name(db_name)
    if GraphDatabase is None:
        logger.warning("neo4j driver not installed, fallback to default database 'neo4j'.")
        return "neo4j"

    try:
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            with driver.session(database="system") as session:
                existing_dbs = [record["name"] for record in session.run("SHOW DATABASES")]
                if target_db not in existing_dbs:
                    session.run(f"CREATE DATABASE `{target_db}`")
                    logger.info(f"Created Neo4j database: {target_db}")
                else:
                    logger.info(f"Neo4j database already exists: {target_db}")
        return target_db
    except Exception as e:
        logger.warning(f"Failed to ensure Neo4j database '{target_db}', fallback to 'neo4j'. Error: {e}")
        return "neo4j"

def setup_mem0g_and_llm(collection_name: str, instance_idx: int, dataset_type: str, neo4j_db_override: str = None):
    logger.info(f"Using collection: {collection_name} for evaluation")
    
    mem0_config = get_mem0_config(
        collection_name=collection_name, 
        include_graph=True
    )
    
    # Neo4j database selection matching ingest
    graph_conf = mem0_config.get("graph_store", {}).get("config", {})
    neo4j_uri = graph_conf.get("url")
    neo4j_user = graph_conf.get("username")
    neo4j_password = graph_conf.get("password")

    desired_db = neo4j_db_override if neo4j_db_override else collection_name
    desired_db = sanitize_neo4j_db_name(desired_db)
    selected_db = desired_db
    if neo4j_uri and neo4j_user and neo4j_password:
        selected_db = ensure_neo4j_database(neo4j_uri, neo4j_user, neo4j_password, desired_db)
    os.environ["NEO4J_DATABASE"] = selected_db
    logger.info(f"Using Qdrant collection: {collection_name}")
    logger.info(f"Using Neo4j database: {selected_db}")

    try:
        mem0_inst = Memory.from_config(mem0_config)
    except Exception as e:
        logger.error(f"Failed to initialize Mem0G: {e}")
        return None, None
        
    filters = {
        "dataset_type": dataset_type,
        "instance_idx": instance_idx
    }
    # Using instance-specific user_id to match ingest.py for graph isolation
    user_id = f"instance_{instance_idx}"
    memory = Mem0G(mem0_inst, user_id=user_id)
    
    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm.get("api_key"),
        base_url=conf.llm.get("base_url"),
        model=conf.llm.get("model")
    )
    return memory, llm

def save_results(final_report: dict, task_name: str, instance_idx: int, output_suffix: str):
    output_dir = Path("out/mem0g")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"mem0g_{task_name}_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    output_file = output_dir / filename
    
    try:
        # 合并已有结果，避免覆盖其他 adaptor 的数据
        if output_file.exists():
            existing = json.load(open(output_file, encoding="utf-8"))
            for a, v in existing.get("results", {}).items():
                if a not in final_report["results"]:
                    final_report["results"][a] = v
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Instance {instance_idx} Finished. Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write results: {e}")


# --- Main Evaluation Logic ---

def evaluate_one_instance(task: str, instance_idx: int, adaptors_to_run: List[str], limit: int, output_suffix: str = "", neo4j_db_override: str = None):
    logger.info(f"=== Evaluating {task} Instance {instance_idx} (Mem0G) ===")
    
    # Task specific configuration
    questions = []
    collection_name = ""
    template_name = "ruler_qa" # Default
    dataset_type = ""
    
    if task == "Accurate_Retrieval":
        data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
        try:
            data = load_benchmark_data(data_path, instance_idx)
            questions = list(data["questions"])
        except Exception as e:
            logger.error(f"Error loading instance {instance_idx}: {e}")
            return
        collection_name = f"mem0g_acc_ret_{instance_idx}"
        dataset_type = "acc_ret"
        template_name = "ruler_qa"

    elif task == "Conflict_Resolution":
        data_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"
        if not Path(data_path).exists():
            logger.error(f"Data file not found: {data_path}")
            return
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions = list(data.get("questions", []))
        except Exception as e:
            logger.error(f"Error loading instance {instance_idx}: {e}")
            return
        collection_name = f"mem0g_conflict_{instance_idx}"
        dataset_type = "conflict_res"
        template_name = "factconsolidation_"

    elif task == "Long_Range_Understanding":
        data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"
        if not Path(data_path).exists():
            logger.error(f"Data file not found: {data_path}")
            return
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions = list(data.get("questions", []))
        except Exception as e:
            logger.error(f"Error loading instance {instance_idx}: {e}")
            return
        collection_name = f"mem0g_long_range_{instance_idx}"
        dataset_type = "long_range"
        if instance_idx >= 100:
            template_name = "detective_qa"
        else:
            template_name = None

    elif task == "Test_Time_Learning":
        data_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"
        if not Path(data_path).exists():
            logger.error(f"Data file not found: {data_path}")
            return
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions = list(data.get("questions", []))
        except Exception as e:
            logger.error(f"Error loading instance {instance_idx}: {e}")
            return
        collection_name = f"mem0g_ttl_{instance_idx}"
        dataset_type = "ttl"
        if instance_idx == 0:
            template_name = "recsys_redial"
        else:
            template_name = "icl_"
    
    else:
        logger.error(f"Unknown task: {task}")
        return

    # Setup Memory and LLM
    memory, llm = setup_mem0g_and_llm(collection_name, instance_idx, dataset_type, neo4j_db_override)
    if not memory or not llm:
        return

    # Run Adaptors
    results = {}
    
    task_short_map = {
        "Accurate_Retrieval": "acc_ret",
        "Conflict_Resolution": "conflict",
        "Long_Range_Understanding": "long_range",
        "Test_Time_Learning": "ttl"
    }
    task_short_name = task_short_map.get(task, task.lower())

    if "all" in adaptors_to_run or "R1" in adaptors_to_run:
        res = evaluate_adaptor("R1", SingleTurnAdaptor(llm, memory), questions, limit, llm, template_name)
        results["R1"] = res

    if "all" in adaptors_to_run or "R2" in adaptors_to_run:
        res = evaluate_adaptor("R2", IterativeAdaptor(llm, memory), questions, limit, llm, template_name)
        results["R2"] = res

    if "all" in adaptors_to_run or "R3" in adaptors_to_run:
        res = evaluate_adaptor("R3", PlanAndActAdaptor(llm, memory), questions, limit, llm, template_name)
        results["R3"] = res

    final_report = {
        "dataset": task,
        "backend": "mem0g",
        "instance_idx": instance_idx,
        "results": results
    }
    
    save_results(final_report, task_short_name, instance_idx, output_suffix)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptors on MemoryAgentBench with Mem0G")
    
    parser.add_argument("--task", type=str, required=True, 
                        choices=["Accurate_Retrieval", "Conflict_Resolution", "Long_Range_Understanding", "Test_Time_Learning"],
                        help="The task/dataset to evaluate")
                        
    parser.add_argument("--adaptor", nargs='+', default=["all"], choices=["R1", "R2", "R3", "all"], help="Adaptors to run")
    parser.add_argument("--limit", type=int, default=-1, help="Number of questions to run (-1 for all)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0-5', '1,3')")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filename")
    parser.add_argument(
        "--neo4j_db",
        type=str,
        default=None,
        help="Neo4j database name override. Default follows collection name per instance."
    )
    
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Task: {args.task}")
    logger.info(f"Target instances: {indices}")
    logger.info(f"Target adaptors: {args.adaptor}")

    for idx in indices:
        evaluate_one_instance(args.task, idx, args.adaptor, args.limit, args.output_suffix, args.neo4j_db)

if __name__ == "__main__":
    main()
