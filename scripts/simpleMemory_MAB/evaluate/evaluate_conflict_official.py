
import json
import re
import string
import argparse
import glob
import numpy as np
from pathlib import Path

# --- 官方风格的文本处理函数 ---

def normalize_answer(s):
    """ 去小写、去标点、去冠词 """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for idx, ch in enumerate(text) if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

from collections import Counter

# --- 评测主逻辑 ---

def evaluate_conflict_results():
    results_files = glob.glob("out/conflict_res_results_*.json")
    results_files.sort(key=lambda x: int(re.search(r'conflict_res_results_(\d+).json', x).group(1)))
    
    if not results_files:
        print("No conflict result files found.")
        return

    print(f"=== Conflict Resolution Official Evaluation Report (N={len(results_files)}) ===\n")
    print(f"{ 'Inst':<5} | { 'Adaptor':<8} | { 'ExactMatch':<10} | { 'SubMatch':<10} | { 'F1 Score':<10}")
    print("-" * 55)

    global_stats = {}

    for fpath in results_files:
        match = re.search(r'conflict_res_results_(\d+).json', fpath)
        idx = int(match.group(1))
        
        with open(fpath, 'r') as f:
            data = json.load(f)
            
        instance_gt_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{idx}.json"
        with open(instance_gt_path, 'r') as f:
            gt_data = json.load(f)
        
        qa_map = {q: a for q, a in zip(gt_data["questions"], gt_data["answers"])}

        for adaptor, items in data["results"].items():
            if adaptor not in global_stats:
                global_stats[adaptor] = {"em": [], "sub": [], "f1": []}
            
            em_hits = 0
            sub_hits = 0
            f1_scores = []
            total = len(items)
            
            for item in items:
                q = item.get("question")
                pred = item.get("answer", "")
                ref_list = qa_map.get(q, [])
                ref = ref_list[0] if isinstance(ref_list, list) else ref_list
                
                norm_pred = normalize_answer(pred)
                norm_ref = normalize_answer(ref)
                
                # 1. Exact Match
                if norm_pred == norm_ref:
                    em_hits += 1
                
                # 2. Substring Match (宽松)
                if norm_ref in norm_pred:
                    sub_hits += 1
                
                # 3. F1
                f1_scores.append(f1_score(pred, ref))
            
            inst_em = em_hits / total if total > 0 else 0
            inst_sub = sub_hits / total if total > 0 else 0
            inst_f1 = np.mean(f1_scores) if f1_scores else 0
            
            global_stats[adaptor]["em"].append(inst_em)
            global_stats[adaptor]["sub"].append(inst_sub)
            global_stats[adaptor]["f1"].append(inst_f1)
            
            print(f"{idx:<5} | {adaptor:<8} | {inst_em:>10.2%} | {inst_sub:>10.2%} | {inst_f1:>10.4f}")

    print("\n## Global Summary")
    for ad, stats in global_stats.items():
        print(f"Adaptor {ad}:")
        print(f"  - Avg Exact Match: {np.mean(stats['em']):.2%}")
        print(f"  - Avg SubMatch:    {np.mean(stats['sub']):.2%}")
        print(f"  - Avg F1 Score:    {np.mean(stats['f1']):.4f}")

if __name__ == "__main__":
    evaluate_conflict_results()
