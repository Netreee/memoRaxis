
import argparse
import sys
import json
from pathlib import Path

# 确保能找到 src 目录
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.logger import get_logger
from src.config import get_config
from src.simple_memory import SimpleRAGMemory
from src.adaptors import run_r1_single_turn, run_r2_iterative, run_r3_plan_act

logger = get_logger()

def load_q2_dataset(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser(description="Infer Q2 Benchmark using SimpleMemory")
    parser.add_argument("--dataset_path", type=str, default="q2DataBase/q2_benchmark_release/dataset_clean.jsonl")
    parser.add_argument("--adaptor", nargs="+", default=["R1", "R2"], help="R1, R2, R3")
    parser.add_argument("--limit", type=int, default=-1, help="Number of questions to run (-1 for all)")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filename")
    args = parser.parse_args()

    logger.info("=== Starting Q2 Benchmark Inference ===")
    
    # 1. Load Data
    try:
        dataset = load_q2_dataset(args.dataset_path)
        logger.info(f"Loaded {len(dataset)} questions from {args.dataset_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    if args.limit > 0:
        dataset = dataset[:args.limit]

    # 2. Setup Memory
    table_name = "bench_q2_md"
    logger.info(f"Connecting to memory table: {table_name}")
    memory = SimpleRAGMemory(table_name=table_name)
    
    # 3. Setup LLM
    conf = get_config()
    from src.llm_interface import OpenAIClient
    llm = OpenAIClient(
        api_key=conf.llm["api_key"],
        base_url=conf.llm["base_url"],
        model=conf.llm["model"]
    )

    # 4. Output File Setup
    output_dir = Path("out/q2_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "q2_infer_results"
    if args.output_suffix:
        filename += f"_{args.output_suffix}"
    filename += ".json"
    out_file = output_dir / filename

    # Load existing for Checkpointing
    results = {
        "dataset": "Q2_Benchmark",
        "results": {}
    }
    
    if out_file.exists():
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                logger.info(f"Loaded checkpoint from {out_file}.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    # 5. Run Inference
    for adaptor_name in args.adaptor:
        logger.info(f"Running Adaptor: {adaptor_name}")
        
        if adaptor_name not in results["results"]:
            results["results"][adaptor_name] = []
        
        adaptor_results = results["results"][adaptor_name]
        start_idx = len(adaptor_results)
        
        if start_idx >= len(dataset):
            logger.info(f"Adaptor {adaptor_name} already completed. Skipping.")
            continue
            
        logger.info(f"Resuming {adaptor_name} from Q{start_idx+1}...")
        
        for i in range(start_idx, len(dataset)):
            item = dataset[i]
            original_q = item["question"]
            gt_answer = item["answer"]
            cluster_id = item.get("cluster_id", "")
            
            logger.info(f"[{adaptor_name}] Q{i+1}/{len(dataset)} (Cluster: {cluster_id})")
            
            # --- Prompt Engineering for Q2 ---
            # 强化“拒答”指令，匹配 GT 的“无法回答”格式
            instruction = "重要指令】：请仔细阅读提供的所有上下文。如果上下文中完全没有足够的信息来回答这个问题，请务必直接输出“无法回答”四个字，不要进行任何猜测或输出其他解释。"
            task_query = original_q + instruction
            
            try:
                if adaptor_name == "R1":
                    pred, meta = run_r1_single_turn(task_query, memory)
                elif adaptor_name == "R2":
                    pred, meta = run_r2_iterative(task_query, memory)
                elif adaptor_name == "R3":
                    pred, meta = run_r3_plan_act(task_query, memory)
                else:
                    continue
                
                new_entry = {
                    "question": original_q,
                    "cluster_id": cluster_id,
                    "answer": pred,
                    "ground_truth": gt_answer,
                    "steps": meta.get("steps", 0),
                    "tokens": meta.get("total_tokens", 0)
                }
                
                adaptor_results.append(new_entry)
                results["results"][adaptor_name] = adaptor_results
                
                # Real-time Save
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                logger.error(f"Error on Q{i}: {e}")
                error_entry = {"question": original_q, "error": str(e)}
                adaptor_results.append(error_entry)
                results["results"][adaptor_name] = adaptor_results
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Inference Complete. Results saved to {out_file}")

if __name__ == "__main__":
    main()
