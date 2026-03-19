"""A-MEM infer — 统一 4 任务推理脚本 (带 per-question checkpoint)

用法:
    python scripts/amem_MAB/infer.py --task Accurate_Retrieval --instance_idx 0-21
    python scripts/amem_MAB/infer.py --task Conflict_Resolution --instance_idx 0-7
    python scripts/amem_MAB/infer.py --task Long_Range_Understanding --instance_idx 0-39
    python scripts/amem_MAB/infer.py --task Test_Time_Learning --instance_idx 0-5

产物: out/amem/amem_{task_short}_{idx}.json
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
from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor, AdaptorResult
from src.amem_memory import AMemMemory
from MemoryAgentBench.utils.templates import get_template

logger = get_logger()

CHROMA_BASE = Path("out/amem/chroma")
OUTPUT_DIR = Path("out/amem")

TASK_CONFIG = {
    "Accurate_Retrieval": {
        "task_short": "acc_ret",
        "template_name": "ruler_qa",
        "load_fn": "parquet",
    },
    "Conflict_Resolution": {
        "task_short": "conflict",
        "template_name": "factconsolidation_",
        "load_fn": "json",
        "folder": "Conflict_Resolution",
    },
    "Long_Range_Understanding": {
        "task_short": "long_range",
        "template_name": None,
        "load_fn": "json",
        "folder": "Long_Range_Understanding",
    },
    "Test_Time_Learning": {
        "task_short": "ttl",
        "template_name": None,
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


def _get_output_path(task: str, instance_idx: int, output_suffix: str = "") -> Path:
    cfg = TASK_CONFIG[task]
    filename = f"amem_{cfg['task_short']}_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    return OUTPUT_DIR / filename


def _load_checkpoint(output_path: Path) -> dict:
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("results", {})
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def _save_checkpoint(output_path: Path, task: str, instance_idx: int, results: dict):
    report = {
        "dataset": task,
        "backend": "amem",
        "instance_idx": instance_idx,
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix('.json.tmp')
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    tmp.rename(output_path)


def evaluate_adaptor(
    name: str,
    adaptor,
    questions: list,
    limit: int,
    llm,
    template_name: Optional[str] = None,
    existing_results: Optional[list] = None,
    checkpoint_fn=None,
) -> list:
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)

    results = list(existing_results) if existing_results else []
    start_idx = len(results)
    if start_idx > 0:
        logger.info(f"[{name}] Resuming from Q{start_idx + 1}/{total} ({start_idx} already done)")
    if start_idx >= total:
        logger.info(f"[{name}] Already complete ({total}/{total})")
        return results

    query_template = None
    if template_name:
        query_template = get_template(template_name, "query", "rag_agent")

    for i in range(start_idx, total):
        q = target_questions[i]
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

        if checkpoint_fn:
            checkpoint_fn()

    return results


def evaluate_one_instance(
    task: str,
    instance_idx: int,
    adaptors_to_run: List[str],
    limit: int,
    output_suffix: str = "",
):
    cfg = TASK_CONFIG[task]
    task_short = cfg["task_short"]
    chroma_dir = CHROMA_BASE / f"{task_short}_{instance_idx}"

    if not chroma_dir.exists():
        logger.error(f"A-MEM data not found: {chroma_dir} (run ingest first)")
        return

    output_path = _get_output_path(task, instance_idx, output_suffix)

    logger.info(f"=== Evaluating {task} Instance {instance_idx} (A-MEM) ===")
    logger.info(f"ChromaDB dir: {chroma_dir}")
    logger.info(f"Output: {output_path}")

    try:
        questions = load_questions(task, instance_idx)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    memory = AMemMemory(
        chroma_dir=str(chroma_dir),
        enable_evolution=False,
    )
    logger.info(f"Loaded {memory.memory_count} memories")

    # OpenAIClient() 无参调用自动从 config.yaml 读取 provider/model/key
    # 支持 Fornax (provider: fornax) 和 OpenAI-compatible (provider: openai)
    llm = OpenAIClient()

    template_name = get_template_name(task, instance_idx)
    logger.info(f"Template: {template_name or '(none, raw question)'}")

    checkpoint = _load_checkpoint(output_path)
    results = {}
    for key in ("R1", "R2", "R3"):
        if key in checkpoint:
            results[key] = checkpoint[key]

    adaptor_map = {
        "R1": lambda: SingleTurnAdaptor(llm, memory),
        "R2": lambda: IterativeAdaptor(llm, memory),
        "R3": lambda: PlanAndActAdaptor(llm, memory),
    }

    run_list = (
        ["R1", "R2", "R3"]
        if "all" in adaptors_to_run
        else [a for a in ["R1", "R2", "R3"] if a in adaptors_to_run]
    )

    for adaptor_name in run_list:
        existing = results.get(adaptor_name, [])

        def make_checkpoint_fn(res_ref=results, path=output_path):
            def fn():
                _save_checkpoint(path, task, instance_idx, res_ref)
            return fn

        results[adaptor_name] = evaluate_adaptor(
            adaptor_name,
            adaptor_map[adaptor_name](),
            questions,
            limit,
            llm,
            template_name,
            existing_results=existing,
            checkpoint_fn=make_checkpoint_fn(),
        )

    _save_checkpoint(output_path, task, instance_idx, results)
    logger.info(f"Instance {instance_idx} Finished. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptors on MemoryAgentBench with A-MEM")
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
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output filename")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Task: {args.task}")
    logger.info(f"Target instances: {indices}")
    logger.info(f"Target adaptors: {args.adaptor}")

    for idx in indices:
        evaluate_one_instance(args.task, idx, args.adaptor, args.limit, args.output_suffix)


if __name__ == "__main__":
    main()
