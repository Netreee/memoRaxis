"""RAPTOR infer — 统一 4 任务推理脚本

用法:
    python scripts/RAPTOR/infer.py --task Conflict_Resolution --instance_idx 0-7 --adaptor R1 R2 R3
    python scripts/RAPTOR/infer.py --task Accurate_Retrieval --instance_idx 7-14 --adaptor R1
    python scripts/RAPTOR/infer.py --task Test_Time_Learning --instance_idx 1-5 --adaptor R1 R2 R3
    python scripts/RAPTOR/infer.py --task Long_Range_Understanding --instance_idx 0-39

产物: out/raptor/raptor_{task_short}_{idx}.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.config import get_config
from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor, AdaptorResult
from src.raptor_memory import RaptorTreeMemory
from MemoryAgentBench.utils.templates import get_template

logger = get_logger()

TREE_DIR = Path("out/raptor/trees")
OUTPUT_DIR = Path("out/raptor")

TASK_CONFIG = {
    "Accurate_Retrieval": {
        "tree_prefix": "raptor_acc_ret",
        "task_short": "acc_ret",
        "template_name": "ruler_qa",
        "load_fn": "parquet",
    },
    "Conflict_Resolution": {
        "tree_prefix": "raptor_conflict",
        "task_short": "conflict",
        "template_name": "factconsolidation_",
        "load_fn": "json",
        "folder": "Conflict_Resolution",
    },
    "Long_Range_Understanding": {
        "tree_prefix": "raptor_long_range",
        "task_short": "long_range",
        "template_name": None,  # inst < 100: None, inst >= 100: detective_qa
        "load_fn": "json",
        "folder": "Long_Range_Understanding",
    },
    "Test_Time_Learning": {
        "tree_prefix": "raptor_ttl",
        "task_short": "ttl",
        "template_name": None,  # inst 0: recsys_redial, inst 1-5: icl_
        "load_fn": "json",
        "folder": "Test_Time_Learning",
    },
}


def get_template_name(task: str, instance_idx: int) -> Optional[str]:
    if task == "Long_Range_Understanding":
        return "detective_qa" if instance_idx >= 100 else None
    elif task == "Test_Time_Learning":
        return "recsys_redial" if instance_idx == 0 else "icl_"
    return TASK_CONFIG[task]["template_name"]


def load_questions(task: str, instance_idx: int) -> list:
    cfg = TASK_CONFIG[task]
    if cfg["load_fn"] == "parquet":
        data = load_benchmark_data(
            "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet",
            instance_idx,
        )
        return list(data["questions"])
    else:
        data_path = Path(f"MemoryAgentBench/preview_samples/{cfg['folder']}/instance_{instance_idx}.json")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return list(data.get("questions", []))


def evaluate_adaptor(
    name: str,
    adaptor,
    questions: list,
    limit: int,
    llm,
    template_name: Optional[str] = None,
) -> list:
    results = []
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)

    query_template = None
    if template_name:
        query_template = get_template(template_name, "query", "rag_agent")

    for i, q in enumerate(target_questions):
        logger.info(f"[{name}] Running Q{i + 1}/{total}: {q[:80]}...")
        formatted_q = query_template.format(question=q) if query_template else q
        try:
            llm.reset_stats()
            t0 = time.time()
            res: AdaptorResult = adaptor.run(formatted_q)
            latency = round(time.time() - t0, 2)
            tokens = res.token_consumption
            logger.info(f"[{name}] Q{i + 1} done | latency={latency}s | tokens={tokens} | steps={res.steps_taken}")
            results.append({
                "question": q,
                "answer": res.answer,
                "steps": res.steps_taken,
                "tokens": tokens,
                "latency_s": latency,
                "replan": res.replan_count,
            })
        except Exception as e:
            logger.error(f"[{name}] Failed on Q{i + 1}: {e}")
            results.append({"question": q, "error": str(e)})
    return results


def evaluate_one_instance(
    task: str,
    instance_idx: int,
    adaptors_to_run: List[str],
    limit: int,
    output_suffix: str = "",
    tree_dir: str = None,
):
    cfg = TASK_CONFIG[task]
    _tree_dir = Path(tree_dir) if tree_dir else TREE_DIR
    tree_path = _tree_dir / f"{cfg['tree_prefix']}_{instance_idx}.pkl"

    if not tree_path.exists():
        logger.error(f"RAPTOR tree not found: {tree_path} (run ingest first)")
        return

    logger.info(f"=== Evaluating {task} Instance {instance_idx} (RAPTOR) ===")
    logger.info(f"Using tree: {tree_path}")

    try:
        questions = load_questions(task, instance_idx)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    memory = RaptorTreeMemory(tree_path=str(tree_path))

    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm["api_key"],
        base_url=conf.llm["base_url"],
        model=conf.llm["model"],
    )

    template_name = get_template_name(task, instance_idx)
    logger.info(f"Template: {template_name or '(none, raw question)'}")

    results = {}

    if "all" in adaptors_to_run or "R1" in adaptors_to_run:
        results["R1"] = evaluate_adaptor("R1", SingleTurnAdaptor(llm, memory), questions, limit, llm, template_name)
    if "all" in adaptors_to_run or "R2" in adaptors_to_run:
        results["R2"] = evaluate_adaptor("R2", IterativeAdaptor(llm, memory), questions, limit, llm, template_name)
    if "all" in adaptors_to_run or "R3" in adaptors_to_run:
        results["R3"] = evaluate_adaptor("R3", PlanAndActAdaptor(llm, memory), questions, limit, llm, template_name)

    final_report = {
        "dataset": task,
        "backend": "raptor",
        "instance_idx": instance_idx,
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"raptor_{cfg['task_short']}_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    output_file = OUTPUT_DIR / filename

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Instance {instance_idx} Finished. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptors on MemoryAgentBench with RAPTOR")
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_CONFIG.keys()),
                        help="The task/dataset to evaluate")
    parser.add_argument("--adaptor", nargs="+", default=["all"],
                        choices=["R1", "R2", "R3", "all"],
                        help="Adaptors to run")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Number of questions to run (-1 for all)")
    parser.add_argument("--instance_idx", type=str, default="0",
                        help="Index range (e.g., '0-5', '1,3')")
    parser.add_argument("--tree_dir", type=str, default=None,
                        help=f"Directory containing RAPTOR pkl trees (default: {TREE_DIR})")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output filename")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Task: {args.task}")
    logger.info(f"Target instances: {indices}")
    logger.info(f"Target adaptors: {args.adaptor}")

    for idx in indices:
        evaluate_one_instance(args.task, idx, args.adaptor, args.limit, args.output_suffix, args.tree_dir)


if __name__ == "__main__":
    main()
