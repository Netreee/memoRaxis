import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.logger import get_logger
from src.config import get_config
from src.benchmark_utils import parse_instance_indices
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor, AdaptorResult
from src.simple_memory import SimpleRAGMemory
from MemoryAgentBench.utils.templates import get_template

logger = get_logger()

TEMPLATE_NAME = "factconsolidation_"


def evaluate_adaptor(name: str, adaptor, questions: list, answers: list, limit: int, llm, template_name: str = None) -> list:
    results = []
    target_questions = questions if limit == -1 else questions[:limit]
    target_answers = answers if limit == -1 else answers[:limit]
    total = len(target_questions)

    query_template = None
    if template_name:
        query_template = get_template(template_name, 'query', 'rag_agent')

    for i, (q, a) in enumerate(zip(target_questions, target_answers)):
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
                "ground_truth": a,
                "steps": res.steps_taken,
                "tokens": tokens,
                "latency_s": latency,
                "replan": res.replan_count
            })
        except Exception as e:
            logger.error(f"[{name}] Failed on Q{i+1}: {e}")
            results.append({"question": q, "error": str(e)})
    return results


def evaluate_instance(instance_idx: int, adaptors_to_run: List[str], limit: int = -1, output_suffix: str = ""):
    logger.info(f"=== Evaluating Conflict_Resolution Instance {instance_idx} ===")

    data_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data["questions"]
    answers = data["answers"]

    table_name = f"bench_conflict_{instance_idx}"
    logger.info(f"Using table: {table_name}")
    memory = SimpleRAGMemory(table_name=table_name)

    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm["api_key"],
        base_url=conf.llm["base_url"],
        model=conf.llm["model"]
    )

    results = {}

    if "all" in adaptors_to_run or "R1" in adaptors_to_run:
        results["R1"] = evaluate_adaptor("R1", SingleTurnAdaptor(llm, memory), questions, answers, limit, llm, TEMPLATE_NAME)
    if "all" in adaptors_to_run or "R2" in adaptors_to_run:
        results["R2"] = evaluate_adaptor("R2", IterativeAdaptor(llm, memory), questions, answers, limit, llm, TEMPLATE_NAME)
    if "all" in adaptors_to_run or "R3" in adaptors_to_run:
        results["R3"] = evaluate_adaptor("R3", PlanAndActAdaptor(llm, memory), questions, answers, limit, llm, TEMPLATE_NAME)

    final_report = {
        "dataset": "Conflict_Resolution",
        "instance_idx": instance_idx,
        "results": results
    }

    output_dir = Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"conflict_res_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    out_file = output_dir / filename

    # 合并已有结果，避免覆盖其他 adaptor 数据
    if out_file.exists():
        try:
            existing = json.load(open(out_file, encoding="utf-8"))
            for a, v in existing.get("results", {}).items():
                if a not in final_report["results"]:
                    final_report["results"][a] = v
        except Exception:
            pass
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conflict_Resolution")
    parser.add_argument("--instance_idx", type=str, default="0-7", help="e.g., '0-7'")
    parser.add_argument("--adaptor", nargs="+", default=["R1", "R2"], choices=["R1", "R2", "R3", "all"], help="R1, R2, R3")
    parser.add_argument("--limit", type=int, default=-1, help="Limit questions per instance (-1 for all)")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filename")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    logger.info(f"Target adaptors: {args.adaptor}")

    for idx in indices:
        evaluate_instance(idx, args.adaptor, args.limit, args.output_suffix)


if __name__ == "__main__":
    main()
