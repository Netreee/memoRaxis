import argparse
import json
import re
import sys
import glob
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Try importing, handle if src not found in path
try:
    from src.benchmark_utils import load_benchmark_data, parse_instance_indices
    from src.llm_interface import OpenAIClient
    from src.config import get_config
except ImportError:
    # If running from root, these imports should work
    pass

# --- Common Utilities ---

def normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def is_correct_mechanical(prediction: str, references: List[str]) -> bool:
    """Evaluation Logic using Mechanical Substring Match with Negative Detection."""
    if references is None: return False
    if isinstance(references, np.ndarray):
        references = references.tolist()
    if not references: return False
    
    negative_patterns = [
        "does not contain any information", "insufficient information",
        "not mentioned in the context", "no information related to",
        "上下文没有提到", "没有找到相关信息", "信息不足", "i don't know", "i cannot answer"
    ]
    
    pred_norm = prediction.lower()
    for pattern in negative_patterns:
        if pattern in pred_norm:
            return False

    norm_prediction = normalize_text(prediction)
    
    flat_refs = []
    for r in references:
        if isinstance(r, list): flat_refs.extend(r)
        else: flat_refs.append(r)
            
    for ref in flat_refs:
        ref_norm = normalize_text(ref)
        if not ref_norm: continue
        if ref_norm in norm_prediction:
            return True
            
    return False

# --- Reporting ---

def _print_report(task_name: str, data: Dict[int, Dict[str, float]], output_file: str = None):
    lines = []
    lines.append(f"=== {task_name} Evaluation Report (N={len(data)}) ===\n")
    lines.append(f"{'Inst':<5} | {'R1':<6} | {'R2':<6} | {'R3':<6} | {'Winner':<6}")
    lines.append("-" * 35)
    
    global_scores = {"R1": [], "R2": [], "R3": []}

    for idx in sorted(data.keys()):
        s = data[idx]
        if not s: continue
        
        r1 = s.get('R1', 0.0)
        r2 = s.get('R2', 0.0)
        r3 = s.get('R3', 0.0)
        
        global_scores['R1'].append(r1)
        global_scores['R2'].append(r2)
        global_scores['R3'].append(r3)
        
        winner = max(s, key=s.get) if s else "N/A"
        lines.append(f"{idx:<5} | {r1:<6.2f} | {r2:<6.2f} | {r3:<6.2f} | {winner:<6}")

    lines.append("\n## Global Average")
    for k in ["R1", "R2", "R3"]:
        if global_scores[k]:
            avg = np.mean(global_scores[k])
            lines.append(f"- {k}: {avg:.4f}")
        else:
            lines.append(f"- {k}: N/A")

    report = "\n".join(lines)
    print(report)
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to {output_file}")
        except Exception as e:
            print(f"Failed to save report: {e}")

# --- Task Specific Analysis ---

def analyze_acc_ret(files: List[str], output_file: str = None):
    print(f"Analyzing Accurate Retrieval results from {len(files)} files...")
    
    data = {} # idx -> {R1: acc, R2: acc, ...}

    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                res_json = json.load(f)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

        instance_idx = res_json.get("instance_idx")
        if instance_idx is None:
            match = re.search(r'mem0_acc_ret_results_(\d+)', fpath)
            if match: instance_idx = int(match.group(1))

        if instance_idx is None: continue

        # Load GT
        try:
             # Note: load_benchmark_data loads 1 instance row correctly
            bench_data = load_benchmark_data("MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet", instance_idx)
            gt_answers = list(bench_data["answers"])
        except Exception as e:
            print(f"Error loading GT for {instance_idx}: {e}")
            continue

        scores = {}
        for adaptor, items in res_json.get("results", {}).items():
            correct = 0
            count = 0 
            for i, item in enumerate(items):
                pred = item.get("answer", "")
                if i < len(gt_answers):
                    if is_correct_mechanical(pred, gt_answers[i]):
                        correct += 1
                    count += 1
            scores[adaptor] = correct / count if count > 0 else 0.0
        
        data[instance_idx] = scores

    _print_report("Accurate Retrieval", data, output_file)

def analyze_conflict(files: List[str], output_file: str = None):
    print(f"Analyzing Conflict Resolution results from {len(files)} files...")
    data = {} 
    
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                res_json = json.load(f)
        except Exception as e: continue

        instance_idx = res_json.get("instance_idx")
        if instance_idx is None:
            match = re.search(r'mem0_conflict_results_(\d+)', fpath)
            if match: instance_idx = int(match.group(1))
        
        if instance_idx is None: continue

        # Load GT
        gt_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"
        if not Path(gt_path).exists():
           print(f"GT not found: {gt_path}")
           continue
           
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            gt_answers = gt_data.get("answers", [])

        scores = {}
        for adaptor, items in res_json.get("results", {}).items():
            correct = 0
            count = 0
            for i, item in enumerate(items):
                if i < len(gt_answers):
                    pred = item.get("answer", "")
                    if is_correct_mechanical(pred, gt_answers[i]):
                        correct += 1
                    count += 1
            scores[adaptor] = correct / count if count > 0 else 0.0
        data[instance_idx] = scores

    _print_report("Conflict Resolution", data, output_file)

def analyze_ttl(files: List[str], output_file: str = None):
    print(f"Analyzing Test Time Learning results from {len(files)} files...")
    
    # Load Entity Map for TTL
    id_map = {}
    try:
        with open("MemoryAgentBench/entity2id.json", 'r', encoding='utf-8') as f:
            ent_data = json.load(f)
        for uri, idx in ent_data.items():
            raw = uri.replace("<http://dbpedia.org/resource/", "").replace(">", "")
            title = raw.replace("_", " ")
            id_map[str(idx)] = title
    except FileNotFoundError:
        print("Warning: MemoryAgentBench/entity2id.json not found. TTL matching might fail.")

    data = {}
    
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                res_json = json.load(f)
        except: continue
        
        instance_idx = res_json.get("instance_idx")
        if instance_idx is None:
             match = re.search(r'mem0_ttl_results_(\d+)', fpath)
             if match: instance_idx = int(match.group(1))
        
        if instance_idx is None: continue
        
        gt_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"
        if not Path(gt_path).exists(): continue
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_inst = json.load(f)
            gt_answers = gt_inst.get("answers", [])

        scores = {}
        for adaptor, items in res_json.get("results", {}).items():
            correct = 0
            count = 0
            for i, item in enumerate(items):
                if i >= len(gt_answers): break
                pred = item.get("answer", "")
                
                # Check match
                gt_ids = gt_answers[i] 
                if not isinstance(gt_ids, list): gt_ids = [gt_ids]
                
                is_hit = False
                for gid in gt_ids:
                    title = id_map.get(str(gid))
                    if title:
                        if title.lower() in pred.lower():
                            is_hit = True
                            break
                        # Base title check
                        base_title = re.sub(r'\s\(\d{4}.*\)', '', title)
                        if base_title.lower() in pred.lower() and len(base_title) > 3:
                             is_hit = True
                             break
                
                if is_hit: correct += 1
                count += 1
            
            scores[adaptor] = correct / count if count > 0 else 0.0
        data[instance_idx] = scores

    _print_report("Test Time Learning", data, output_file)

# --- Long Range Understanding (LLM Evaluation) ---

FLUENCY_PROMPT = """Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.
Rubric:
- Score 0: Incoherent, repetitive, incomplete, or gibberish.
- Score 1: Coherent, non-repetitive, fluent, grammatically correct.
Output json: {{"fluency": 1}}.
Text: "{text}"
"""

RECALL_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary.
Rubric Recall:
- Score: number of key points mostly-supported by the provided summary.
Output json: {{"supported_key_points": [1, 3], "recall": 2}}.
Key points:
{keypoints}
Summary: <start of summary>{summary}<end of summary>
"""

PRECISION_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary.
Rubric Precision:
- Score: number of sentences in provided summary supported by expert summary.
Output json: {{"precision": 7, "sentence_count": 20}}.
Expert summary: <start of summary>{expert_summary}<end of summary>
Provided summary: <start of summary>{summary}<end of summary>
"""

class LRUEvaluator:
    def __init__(self):
        conf = get_config()
        self.llm = OpenAIClient(
            api_key=conf.llm.get("api_key"),
            base_url=conf.llm.get("base_url"),
            model=conf.llm.get("model")
        )

    def evaluate_one(self, prediction: str, keypoints: List[str], expert_summary: str) -> Dict[str, float]:
        # 1. Fluency
        f_prompt = FLUENCY_PROMPT.format(text=prediction[:4000]) # truncated
        try:
            res = self.llm.generate_json(f_prompt)
            fluency = float(res.get("fluency", 0))
        except: fluency = 0.0

        # 2. Recall
        if not keypoints:
            recall = 0.0
        else:
            kp_str = "\n".join([f"{i+1}. {kp}" for i, kp in enumerate(keypoints)])
            r_prompt = RECALL_PROMPT.format(keypoints=kp_str, summary=prediction[:4000])
            try:
                res = self.llm.generate_json(r_prompt)
                rec_count = float(res.get("recall", 0))
            except: rec_count = 0.0
            recall = rec_count / len(keypoints)

        # 3. Precision
        if not expert_summary or not expert_summary.strip():
            precision = 0.0
        else:
            p_prompt = PRECISION_PROMPT.format(expert_summary=expert_summary[:4000], summary=prediction[:4000])
            try:
                res = self.llm.generate_json(p_prompt)
                prec_count = float(res.get("precision", 0))
                sent_count = float(res.get("sentence_count", 1))
            except: 
                prec_count = 0.0
                sent_count = 1.0
            precision = prec_count / sent_count if sent_count > 0 else 0

        if (recall + precision) > 0:
            f1 = fluency * 2 * (recall * precision) / (recall + precision)
        else:
            f1 = 0.0
            
        return {"fluency": fluency, "recall": recall, "precision": precision, "f1": f1}

def analyze_lru(files: List[str], output_file: str = None):
    print(f"Analyzing Long Range Understanding results from {len(files)} files (using LLM evaluation)...")
    
    evaluator = LRUEvaluator()
    data = {} # idx -> {R1: f1, ...}

    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                res_json = json.load(f)
        except: continue
        
        idx = res_json.get("instance_idx")
        if idx is None:
             match = re.search(r'mem0_long_range_results_(\d+)', fpath)
             if match: idx = int(match.group(1))
        
        if idx is None: continue

        gt_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{idx}.json"
        if not Path(gt_path).exists():
            print(f"GT not found for instance {idx}: {gt_path}")
            continue
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            metadata = gt_data.get("metadata") or {}
            keypoints = metadata.get("keypoints", [])
            
            answers = gt_data.get("answers", [])
            expert_summary = ""
            if answers:
                 first = answers[0]
                 if isinstance(first, list) and len(first) > 0: expert_summary = first[0]
                 elif isinstance(first, str): expert_summary = first

            if not keypoints: print(f"Warning: No keypoints found for instance {idx}.")
            if not expert_summary: print(f"Warning: No expert summary found for instance {idx}.")

        inst_scores = {}
        for adaptor, items in res_json.get("results", {}).items():
            if not items: continue
            pred = items[0].get("answer", "")
            
            # Evaluate using LLM
            metrics = evaluator.evaluate_one(pred, keypoints, expert_summary)
            inst_scores[adaptor] = metrics["f1"] 
            
            print(f"  [{adaptor}] F1: {metrics['f1']:.4f} (Rec: {metrics['recall']:.2f}, Prec: {metrics['precision']:.2f}, Flu: {metrics['fluency']})")
        
        data[idx] = inst_scores
        print(f"Evaluated Instance {idx}")

    _print_report("Long Range Understanding", data, output_file)


def main():
    parser = argparse.ArgumentParser(description="Analyze Mem0 Results")
    parser.add_argument("--task", type=str, required=True, 
                        choices=["Accurate_Retrieval", "Conflict_Resolution", "Long_Range_Understanding", "Test_Time_Learning"],
                        help="Task to analyze")
    parser.add_argument("--input", type=str, default="out/mem0", help="Input directory or file pattern")
    parser.add_argument("--output", type=str, default=None, help="Output file for analysis report")
    
    args = parser.parse_args()

    # Determine files
    files = []
    if Path(args.input).is_dir():
        # Default patterns based on task
        if args.task == "Accurate_Retrieval":
            pattern = "mem0_acc_ret_results_*.json"
        elif args.task == "Conflict_Resolution":
            pattern = "mem0_conflict_results_*.json"
        elif args.task == "Long_Range_Understanding":
            pattern = "mem0_long_range_results_*.json"
        elif args.task == "Test_Time_Learning":
            pattern = "mem0_ttl_results_*.json"
        
        files = glob.glob(str(Path(args.input) / pattern))
    else:
        # Check if input is a glob pattern string directly
        files = glob.glob(args.input)

    files.sort()
    
    if not files:
        print(f"No results found for {args.task} in {args.input}")
        return

    if args.task == "Accurate_Retrieval":
        analyze_acc_ret(files, args.output)
    elif args.task == "Conflict_Resolution":
        analyze_conflict(files, args.output)
    elif args.task == "Long_Range_Understanding":
        analyze_lru(files, args.output)
    elif args.task == "Test_Time_Learning":
        analyze_ttl(files, args.output)

if __name__ == "__main__":
    main()
