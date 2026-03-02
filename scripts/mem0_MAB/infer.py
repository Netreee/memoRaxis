import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
from mem0 import Memory

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.mem0 import Mem0
from src.logger import get_logger
from src.config import get_config
from src.mem0_utils import get_mem0_config
from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor, AdaptorResult
from MemoryAgentBench.utils.templates import get_template

logger = get_logger()

# --- Shared Utilities ---

def evaluate_adaptor(name: str, adaptor, questions: list, limit: int, template_name: str, template_type: str = 'query', agent_type: str = 'rag_agent') -> list:
    results = []
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)
    
    if template_name:
        query_template = get_template(template_name, template_type, agent_type)
    
    for i, q in enumerate(target_questions):
        logger.info(f"[{name}] Running Q{i+1}/{total}: {q}")
        if template_name:
            formatted_q = query_template.format(question=q)
        else:
            formatted_q = q
        try:
            res: AdaptorResult = adaptor.run(formatted_q)
            results.append({
                "question": q,
                "answer": res.answer,
                "steps": res.steps_taken,
                "tokens": res.token_consumption,
                "replan": res.replan_count
            })
        except Exception as e:
            logger.error(f"[{name}] Failed on Q{i+1}: {e}")
            results.append({"question": q, "error": str(e)})
    return results

def setup_mem0_and_llm(collection_name: str):
    logger.info(f"Using collection: {collection_name} for evaluation")
    mem0_config = get_mem0_config(collection_name)
    try:
        mem0_inst = Memory.from_config(mem0_config)
    except Exception as e:
        logger.error(f"Failed to initialize Mem0: {e}")
        return None, None
        
    memory = Mem0(mem0_inst)
    
    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm.get("api_key"),
        base_url=conf.llm.get("base_url"),
        model=conf.llm.get("model")
    )
    return memory, llm

def save_results(final_report: dict, task_name: str, instance_idx: int, output_suffix: str):
    output_dir = Path("out/mem0")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"mem0_{task_name}_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    output_file = output_dir / filename
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Instance {instance_idx} Finished. Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write results: {e}")

# --- Task Specific Helpers ---

# (Helper functions removed as requested)

# --- Main Evaluation Logic ---

def evaluate_one_instance(task: str, instance_idx: int, adaptors_to_run: List[str], limit: int, output_suffix: str = ""):
    logger.info(f"=== Evaluating {task} Instance {instance_idx} (Mem0) ===")
    
    # Task specific configuration
    questions = []
    collection_name = ""
    template_name = "ruler_qa" # Default
    
    if task == "Accurate_Retrieval":
        data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
        try:
            data = load_benchmark_data(data_path, instance_idx)
            questions = list(data["questions"])
        except Exception as e:
            logger.error(f"Error loading instance {instance_idx}: {e}")
            return
        collection_name = f"mem0_acc_ret_{instance_idx}"
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
        collection_name = f"mem0_conflict_{instance_idx}"
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
        collection_name = f"mem0_long_range_{instance_idx}"
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
        collection_name = f"mem0_ttl_{instance_idx}"
        if instance_idx == 0:
            template_name = "recsys_"
        else:
            template_name = "icl_"
    
    else:
        logger.error(f"Unknown task: {task}")
        return

    # Setup Memory and LLM
    memory, llm = setup_mem0_and_llm(collection_name)
    if not memory or not llm:
        return

    # Run Adaptors
    results = {}
    
    # For file naming purposes, map task to short name
    task_short_map = {
        "Accurate_Retrieval": "acc_ret",
        "Conflict_Resolution": "conflict",
        "Long_Range_Understanding": "long_range",
        "Test_Time_Learning": "ttl"
    }
    task_short_name = task_short_map.get(task, task.lower())

    if "all" in adaptors_to_run or "R1" in adaptors_to_run:
        res = evaluate_adaptor("R1", SingleTurnAdaptor(llm, memory), questions, limit, template_name)
        results["R1"] = res
        
    if "all" in adaptors_to_run or "R2" in adaptors_to_run:
        res = evaluate_adaptor("R2", IterativeAdaptor(llm, memory), questions, limit, template_name)
        results["R2"] = res
        
    if "all" in adaptors_to_run or "R3" in adaptors_to_run:
        res = evaluate_adaptor("R3", PlanAndActAdaptor(llm, memory), questions, limit, template_name)
        results["R3"] = res

    final_report = {
        "dataset": task,
        "backend": "mem0",
        "instance_idx": instance_idx,
        "results": results
    }
    
    save_results(final_report, task_short_name, instance_idx, output_suffix)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptors on MemoryAgentBench with Mem0")
    
    parser.add_argument("--task", type=str, required=True, 
                        choices=["Accurate_Retrieval", "Conflict_Resolution", "Long_Range_Understanding", "Test_Time_Learning"],
                        help="The task/dataset to evaluate")
                        
    parser.add_argument("--adaptor", nargs='+', default=["all"], choices=["R1", "R2", "R3", "all"], help="Adaptors to run")
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to run (-1 for all)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0-5', '1,3')")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filename")
    
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Task: {args.task}")
    logger.info(f"Target instances: {indices}")
    logger.info(f"Target adaptors: {args.adaptor}")

    for idx in indices:
        evaluate_one_instance(args.task, idx, args.adaptor, args.limit, args.output_suffix)

if __name__ == "__main__":
    main()
