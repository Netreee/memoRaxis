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
                # Merge existing results
                if existing_data.get("instance_idx") == instance_idx:
                    results = existing_data
                    logger.info(f"Loaded checkpoint from {out_file}. Resuming...")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    # Initialize Memory
    table_name = f"bench_ttl_{instance_idx}"
    logger.info(f"Using table: {table_name}")
    memory = SimpleRAGMemory(table_name=table_name)

    questions = data["questions"]
    answers = data["answers"]
    
    if limit > 0:
        questions = questions[:limit]
        answers = answers[:limit]

    for adaptor_name in adaptors:
        logger.info(f"Running Adaptor: {adaptor_name}")
        
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
            
            # --- Instruction Injection ---
            if instance_idx == 0:
                task_query = q + "\n\nInstruction: Based on the dialogue history, recommend 5 movies. Output their titles or IDs."
            else:
                task_query = q + "\n\nInstruction: Classify the intent of the user query based on the labeled examples in memory. Output ONLY the numeric Label ID."
            
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
                    "question": q,
                    "answer": pred,
                    "ground_truth": a,
                    "steps": meta.get("steps", 0),
                    "tokens": meta.get("total_tokens", 0)
                }
                
                # Append and Save Immediately
                adaptor_results.append(new_entry)
                
                # Update main dict
                results["results"][adaptor_name] = adaptor_results
                
                # Real-time Save
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                logger.error(f"Error on Q{i}: {e}")
                # Optional: save error state so we don't get stuck? 
                # For now, just skip saving and retry next time or crash.
                # Ideally we append an error entry to move forward.
                error_entry = {"question": q, "error": str(e)}
                adaptor_results.append(error_entry)
                results["results"][adaptor_name] = adaptor_results
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Instance {instance_idx} Finished. Results saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Test_Time_Learning")
    parser.add_argument("--instance_idx", type=str, default="0-5", help="e.g., '0-5'")
    parser.add_argument("--adaptor", nargs="+", default=["R1", "R2"], help="R1, R2, R3")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filename")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    for idx in indices:
        evaluate_instance(idx, args.adaptor, args.limit, args.output_suffix)

if __name__ == "__main__":
    main()