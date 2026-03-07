
import json
import argparse
from pathlib import Path
import numpy as np

def analyze_q2_eval(eval_file: str):
    path = Path(eval_file)
    if not path.exists():
        print(f"Error: Evaluation file not found at {eval_file}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"=== Q2 Benchmark Evaluation Report ===\\n")
    
    # 1. Global Summary
    print("## 1. Global Performance")
    print(f"{'Adaptor':<10} | {'Accuracy':<10} | {'Avg Tokens':<12}")
    print("-" * 40)
    
    for adaptor, summary in data.get("summary", {}).items():
        print(f"{adaptor:<10} | {summary['accuracy']:>7.2%} | {summary['avg_tokens']:>10.1f}")

    print("\\n## 2. Performance by Question Type (Negative vs Positive)")
    print("Negative means the ground truth is '无法回答'.")
    print(f"{'Adaptor':<10} | {'Negative Acc':<15} | {'Positive Acc':<15}")
    print("-" * 50)
    
    for adaptor, details in data.get("details", {}).items():
        neg_correct = 0
        neg_total = 0
        pos_correct = 0
        pos_total = 0
        
        for item in details:
            gt = str(item.get("ground_truth", "")).strip()
            score = item.get("score", 0)
            
            if gt == "无法回答":
                neg_total += 1
                if score == 1: neg_correct += 1
            else:
                pos_total += 1
                if score == 1: pos_correct += 1
                
        neg_acc = neg_correct / neg_total if neg_total > 0 else 0
        pos_acc = pos_correct / pos_total if pos_total > 0 else 0
        
        print(f"{adaptor:<10} | {neg_acc:>12.2%} | {pos_acc:>12.2%}")
        
    print("\\n## 3. Breakdown by Cluster")
    
    # Find all unique clusters
    all_clusters = set()
    for details in data.get("details", {}).values():
        for item in details:
            all_clusters.add(item.get("cluster_id", "Unknown"))
            
    all_clusters = sorted(list(all_clusters))
    
    header = f"{'Cluster':<10}"
    adaptors = sorted(data.get("summary", {}).keys())
    for ad in adaptors:
        header += f" | {ad:<8}"
    print(header)
    print("-" * len(header))
    
    for cluster in all_clusters:
        row = f"{cluster:<10}"
        for ad in adaptors:
            details = data["details"].get(ad, [])
            cluster_items = [item for item in details if item.get("cluster_id") == cluster]
            
            if not cluster_items:
                row += f" | {'N/A':<8}"
                continue
                
            correct = sum(1 for item in cluster_items if item.get("score", 0) == 1)
            acc = correct / len(cluster_items)
            row += f" | {acc:>7.2%}"
            
        print(row)

def main():
    parser = argparse.ArgumentParser(description="Analyze Q2 Benchmark Evaluation")
    parser.add_argument("--eval_file", type=str, default="out/q2_benchmark/eval_q2_infer_results.json", help="Path to eval JSON")
    args = parser.parse_args()

    analyze_q2_eval(args.eval_file)

if __name__ == "__main__":
    main()
