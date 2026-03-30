import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.memos import MemOS
from src.logger import get_logger
from src.config import get_config
from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.llm_interface import OpenAIClient
from MemoryAgentBench.utils.templates import get_template as get_mab_template

# --- Adaptor Imports (Assuming similar structure or placeholders) ---
# Since imports in mem0_MAB/infer.py are from src.adaptors, we use them too.
from src.adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor

logger = get_logger()

# --- Shared Utilities ---

def evaluate_adaptor(name: str, adaptor, questions: List[str], limit: int, 
                     template_name: Optional[str] = None, 
                     template_type: str = 'query', 
                     agent_type: str = 'rag_agent') -> List[dict]:
    """
    Runs the adaptor on a list of questions.
    """
    results = []
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)
    
    query_template = None
    if template_name:
        try:
            # Attempt to retrieve template from MemoryAgentBench utils
            query_template = get_mab_template(template_name, template_type, agent_type)
        except Exception as e:
            logger.warning(f"Could not load template '{template_name}': {e}. Using raw question.")

    for i, q in enumerate(target_questions):
        logger.info(f"[{name}] Running Q{i+1}/{total}")

        formatted_q = q
        if query_template:
            try:
                formatted_q = query_template.format(question=q)
            except KeyError:
                formatted_q = q

        # 记录本题开始前的 token 累计值，用于计算本题的 token 增量
        llm_client = getattr(adaptor, '_llm', None)
        tokens_before = getattr(llm_client, 'total_tokens', 0)
        t0 = time.perf_counter()

        def _timeout_handler(signum, frame):
            raise TimeoutError("question timeout (600s)")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(600)  # 10 min per question max
        try:
            res = adaptor.run(formatted_q)
            signal.alarm(0)
            elapsed = time.perf_counter() - t0
            tokens_delta = getattr(llm_client, 'total_tokens', 0) - tokens_before

            results.append({
                "question": q,
                "answer": res.answer,
                "steps": getattr(res, "steps_taken", []),
                "tokens": tokens_delta,
                "replan": getattr(res, "replan_count", 0),
                "elapsed_sec": round(elapsed, 2),
            })
            logger.info(f"[{name}] Q{i+1} done: {elapsed:.1f}s, tokens={tokens_delta}")
        except Exception as e:
            signal.alarm(0)
            elapsed = time.perf_counter() - t0
            logger.error(f"[{name}] Failed on Q{i+1} ({elapsed:.1f}s): {e}")
            results.append({"question": q, "error": str(e), "elapsed_sec": round(elapsed, 2)})
            
    return results

def setup_memos_and_llm(user_id: str):
    """
    Initialize MemOS and LLM client.
    """
    logger.info(f"Using MemOS user_id: {user_id}")
    
    try:
        # Initialize MemOS with the specific user_id (which acts as the collection/cube ID context)
        memory = MemOS(user_id=user_id)
    except Exception as e:
        logger.error(f"Failed to initialize MemOS: {e}")
        return None, None
    
    # Initialize LLM — MemOS infer 固定走 Ark（不读全局 config，避免影响其他系统）
    try:
        llm = OpenAIClient(
            provider="openai",
            api_key=os.environ.get("ARK_API_KEY", ""),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model="ep-20251113195357-4gftp",
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return None, None
        
    return memory, llm

def save_results(final_report: dict, task_name: str, instance_idx: int, output_suffix: str):
    """
    Save evaluation results to JSON.
    """
    output_dir = Path("out/memOS")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"memOS_{task_name}_{instance_idx}"
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


def evaluate_one_instance(task: str, instance_idx: int, adaptors_to_run: List[str], limit: int, output_suffix: str):
    logger.info(f"=== Evaluating {task} Instance {instance_idx} (MemOS) ===")

    # 1. Load Data & Config
    questions = []
    user_id = "" # Mapping to MemOS user_id/cube
    template_name = None

    # Mappings correspond to ingest.py logic
    if task == "accurate_retrieval":
        data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
        try:
            data = load_benchmark_data(data_path, instance_idx)
            questions = list(data["questions"])
        except Exception as e:
            logger.error(f"Error loading {task}: {e}")
            return
        user_id = f"memos_acc_ret_{instance_idx}"
        template_name = "ruler_qa"

    elif task == "conflict_resolution":
        data_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions = data.get("questions", [])
        except Exception as e:
            logger.error(f"Error loading {task}: {e}")
            return
        user_id = f"memos_conflict_{instance_idx}"
        template_name = "factconsolidation_"

    elif task == "long_range_understanding":
        data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions = data.get("questions", [])
        except Exception as e:
            logger.error(f"Error loading {task}: {e}")
            return
        user_id = f"memos_long_range_{instance_idx}"
        # Logic from mem0 infer.py
        template_name = "detective_qa" if instance_idx >= 100 else None

    elif task == "test_time_learning":
        data_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions = data.get("questions", [])
        except Exception as e:
            logger.error(f"Error loading {task}: {e}")
            return
        user_id = f"memos_ttl_{instance_idx}"
        template_name = "recsys_redial" if instance_idx == 0 else "icl_"
    
    else:
        logger.error(f"Unknown task: {task}")
        return

    # 2. Setup System
    memory, llm = setup_memos_and_llm(user_id)
    if not memory or not llm:
        return

    # 3. Run Adaptors
    results = {}
    
    # Adaptor initialization
    # Assuming adaptors take (llm, memory)
    
    if "all" in adaptors_to_run or "R1" in adaptors_to_run:
        logger.info("Running R1 (Single Turn)...")
        adaptor = SingleTurnAdaptor(llm, memory)
        results["R1"] = evaluate_adaptor("R1", adaptor, questions, limit, template_name)
        
    if "all" in adaptors_to_run or "R2" in adaptors_to_run:
        logger.info("Running R2 (Iterative)...")
        adaptor = IterativeAdaptor(llm, memory)
        results["R2"] = evaluate_adaptor("R2", adaptor, questions, limit, template_name)
        
    if "all" in adaptors_to_run or "R3" in adaptors_to_run:
        logger.info("Running R3 (Plan & Act)...")
        adaptor = PlanAndActAdaptor(llm, memory)
        results["R3"] = evaluate_adaptor("R3", adaptor, questions, limit, template_name)

    # 4. Save Results
    final_report = {
        "dataset": task,
        "backend": "memOS",
        "instance_idx": instance_idx,
        "results": results
    }
    
    save_results(final_report, task, instance_idx, output_suffix)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptors on MemoryAgentBench with MemOS")
    
    parser.add_argument("--task", type=str, required=True, 
                        choices=["accurate_retrieval", "conflict_resolution", "long_range_understanding", "test_time_learning"],
                        help="The task/dataset to evaluate")
                        
    parser.add_argument("--adaptor", nargs='+', default=["all"], choices=["R1", "R2", "R3", "all"], help="Adaptors to run")
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to run (-1 for all)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filename")
    
    args = parser.parse_args()

    # Parse instance indices
    indices = parse_instance_indices(args.instance_idx)
    
    logger.info(f"Task: {args.task}")
    logger.info(f"Target instances: {indices}")
    logger.info(f"Target adaptors: {args.adaptor}")

    for idx in indices:
        evaluate_one_instance(args.task, idx, args.adaptor, args.limit, args.output_suffix)

if __name__ == "__main__":
    main()
