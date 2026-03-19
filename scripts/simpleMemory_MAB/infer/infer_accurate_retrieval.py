import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

# Add project root to sys.path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.logger import get_logger
from src.config import get_config
from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor, AdaptorResult
from src.simple_memory import SimpleRAGMemory
from MemoryAgentBench.utils.templates import get_template

logger = get_logger()

def evaluate_adaptor(name: str, adaptor, questions: list, limit: int, llm, template_name: str = None) -> list:
    results = []
    # limit -1 表示跑所有
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)

    query_template = None
    if template_name:
        query_template = get_template(template_name, 'query', 'rag_agent')

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
            logger.error(f"[{name}] Failed on Q{i+1}: {e}")
            results.append({"question": q, "error": str(e)})
    return results

def evaluate_one_instance(instance_idx: int, adaptors_to_run: List[str], limit: int, output_suffix: str = ""):
    logger.info(f"=== Evaluating Instance {instance_idx} ===")
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
    
    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return
        
    questions = list(data["questions"])

    table_name = f"bench_acc_ret_{instance_idx}"
    logger.info(f"Using table: {table_name} for evaluation")
    
    memory = SimpleRAGMemory(table_name=table_name)
    
    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm["api_key"],
        base_url=conf.llm["base_url"],
        model=conf.llm["model"]
    )

    results = {}
    
    if "all" in adaptors_to_run or "R1" in adaptors_to_run:
        results["R1"] = evaluate_adaptor("R1", SingleTurnAdaptor(llm, memory), questions, limit, llm, "ruler_qa")
    if "all" in adaptors_to_run or "R2" in adaptors_to_run:
        results["R2"] = evaluate_adaptor("R2", IterativeAdaptor(llm, memory), questions, limit, llm, "ruler_qa")
    if "all" in adaptors_to_run or "R3" in adaptors_to_run:
        results["R3"] = evaluate_adaptor("R3", PlanAndActAdaptor(llm, memory), questions, limit, llm, "ruler_qa")

    final_report = {
        "dataset": "Accurate_Retrieval",
        "instance_idx": instance_idx,
        "results": results
    }
    
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)
    
    filename = f"acc_ret_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    output_file = output_dir / filename
    
    # 合并已有结果，避免覆盖其他 adaptor 数据
    if output_file.exists():
        try:
            existing = json.load(open(output_file, encoding="utf-8"))
            for a, v in existing.get("results", {}).items():
                if a not in final_report["results"]:
                    final_report["results"][a] = v
        except Exception:
            pass
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Instance {instance_idx} Finished. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptors on MemoryAgentBench")
    parser.add_argument("--adaptor", nargs='+', default=["all"], choices=["R1", "R2", "R3", "all"], help="Adaptors to run (e.g., R1 R2)")
    parser.add_argument("--limit", type=int, default=-1, help="Number of questions to run (-1 for all)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0-5', '1,3')")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filename (e.g., 'new_r3')")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    logger.info(f"Target adaptors: {args.adaptor}")

    for idx in indices:
        evaluate_one_instance(idx, args.adaptor, args.limit, args.output_suffix)

if __name__ == "__main__":
    main()
