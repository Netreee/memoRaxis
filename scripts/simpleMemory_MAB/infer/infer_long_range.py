
import argparse
import sys
import logging
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices
from src.simple_memory import SimpleRAGMemory
from src.adaptors import run_r1_single_turn, run_r2_iterative, run_r3_plan_act

logger = get_logger()

def evaluate_instance(instance_idx: int, adaptors: list, limit: int = -1, output_suffix: str = ""):
    logger.info(f"=== Evaluating Long_Range_Understanding Instance {instance_idx} ===")
    
    data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    table_name = f"bench_long_range_{instance_idx}"
    logger.info(f"Using table: {table_name}")
    memory = SimpleRAGMemory(table_name=table_name)

    questions = data["questions"]
    answers = data["answers"]
    
    if limit > 0:
        questions = questions[:limit]
        answers = answers[:limit]

    results = {
        "dataset": "Long_Range_Understanding",
        "instance_idx": instance_idx,
        "results": {}
    }

    for adaptor_name in adaptors:
        logger.info(f"Running Adaptor: {adaptor_name}")
        adaptor_results = []
        
        for i, (q, a) in enumerate(zip(questions, answers)):
            logger.info(f"[{adaptor_name}] Q{i+1}/{len(questions)}")
            try:
                if adaptor_name == "R1":
                    pred, meta = run_r1_single_turn(q, memory)
                elif adaptor_name == "R2":
                    pred, meta = run_r2_iterative(q, memory)
                elif adaptor_name == "R3":
                    pred, meta = run_r3_plan_act(q, memory)
                else:
                    continue
                
                adaptor_results.append({
                    "question": q,
                    "answer": pred,
                    "ground_truth": a,
                    "steps": meta.get("steps", 0),
                    "tokens": meta.get("total_tokens", 0)
                })
            except Exception as e:
                logger.error(f"Error on Q{i}: {e}")
                adaptor_results.append({"question": q, "error": str(e)})

        results["results"][adaptor_name] = adaptor_results

    output_dir = Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"long_range_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    out_file = output_dir / filename
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Long_Range_Understanding")
    # 默认 Top 40
    parser.add_argument("--instance_idx", type=str, default="0-39", help="e.g., '0-39'")
    parser.add_argument("--adaptor", nargs="+", default=["R1", "R2"], help="R1, R2, R3")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filename")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    for idx in indices:
        evaluate_instance(idx, args.adaptor, args.limit, args.output_suffix)

if __name__ == "__main__":
    main()
