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


def get_template_name(instance_idx: int) -> str:
    """idx==0 为电影推荐任务，其余为意图分类（ICL）任务。"""
    return "recsys_redial" if instance_idx == 0 else "icl_"


def evaluate_instance(instance_idx: int, adaptors_to_run: List[str], limit: int = -1, output_suffix: str = ""):
    logger.info(f"=== Evaluating Test_Time_Learning Instance {instance_idx} ===")

    data_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Output File Setup
    output_dir = Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ttl_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    out_file = output_dir / filename

    # Load Existing Results (Checkpointing)
    results = {
        "dataset": "Test_Time_Learning",
        "instance_idx": instance_idx,
        "results": {}
    }

    if out_file.exists():
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if existing_data.get("instance_idx") == instance_idx:
                    results = existing_data
                    logger.info(f"Loaded checkpoint from {out_file}. Resuming...")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    # Initialize Memory and LLM
    table_name = f"bench_ttl_{instance_idx}"
    logger.info(f"Using table: {table_name}")
    memory = SimpleRAGMemory(table_name=table_name)

    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm["api_key"],
        base_url=conf.llm["base_url"],
        model=conf.llm["model"]
    )

    template_name = get_template_name(instance_idx)
    query_template = get_template(template_name, 'query', 'rag_agent')
    logger.info(f"Template: {template_name}")

    questions = data["questions"]
    answers = data["answers"]

    if limit > 0:
        questions = questions[:limit]
        answers = answers[:limit]

    adaptor_map = {
        "R1": SingleTurnAdaptor(llm, memory),
        "R2": IterativeAdaptor(llm, memory),
        "R3": PlanAndActAdaptor(llm, memory),
    }

    for adaptor_name in adaptors_to_run:
        if adaptor_name not in adaptor_map:
            logger.warning(f"Unknown adaptor: {adaptor_name}, skipping.")
            continue

        logger.info(f"Running Adaptor: {adaptor_name}")
        adaptor = adaptor_map[adaptor_name]

        # Ensure adaptor list exists
        if adaptor_name not in results["results"]:
            results["results"][adaptor_name] = []

        adaptor_results = results["results"][adaptor_name]

        # Determine start index based on existing results
        start_idx = len(adaptor_results)
        if start_idx >= len(questions):
            logger.info(f"Adaptor {adaptor_name} already completed ({start_idx}/{len(questions)}). Skipping.")
            continue

        logger.info(f"Resuming {adaptor_name} from Q{start_idx+1}...")

        for i in range(start_idx, len(questions)):
            q = questions[i]
            a = answers[i]

            logger.info(f"[{adaptor_name}] Q{i+1}/{len(questions)}")
            formatted_q = query_template.format(question=q)

            try:
                llm.reset_stats()
                t0 = time.time()
                res: AdaptorResult = adaptor.run(formatted_q)
                latency = round(time.time() - t0, 2)
                tokens = res.token_consumption
                logger.info(f"[{adaptor_name}] Q{i+1} done | latency={latency}s | tokens={tokens} | steps={res.steps_taken}")

                new_entry = {
                    "question": q,
                    "answer": res.answer,
                    "ground_truth": a,
                    "steps": res.steps_taken,
                    "tokens": tokens,
                    "latency_s": latency,
                    "replan": res.replan_count
                }
            except Exception as e:
                logger.error(f"[{adaptor_name}] Error on Q{i+1}: {e}")
                new_entry = {"question": q, "error": str(e)}

            # Append and save immediately (checkpoint)
            adaptor_results.append(new_entry)
            results["results"][adaptor_name] = adaptor_results
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Instance {instance_idx} Finished. Results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Test_Time_Learning")
    parser.add_argument("--instance_idx", type=str, default="0-5", help="e.g., '0-5'")
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
