
import json
import glob
import numpy as np
import os
import re
from pathlib import Path

def analyze_lru_a():
    # 查找所有已生成的评测文件
    eval_files = glob.glob("out/eval/eval_lru_a_*.json")
    eval_files.sort(key=lambda x: int(re.search(r'eval_lru_a_(\d+).json', x).group(1)))
    
    if not eval_files:
        print("No evaluation files found in out/eval/")
        return
    
    data = {}
    metrics_list = ["fluency", "recall", "precision", "f1"]
    
    # 聚合数据
    for fpath in eval_files:
        match = re.search(r'eval_lru_a_(\d+).json', fpath)
        if not match: continue
        idx = int(match.group(1))
        
        with open(fpath, 'r') as f:
            content = json.load(f)
            
        data[idx] = content["metrics"]

    # 打印报告
    print(f"=== Long Range Understanding (LRU-A) Evaluation Report (N={len(data)}) ===\n")
    
    print("## 1. Global Performance (Avg Metrics)")
    print(f"{ 'Adaptor':<10} | { 'Fluency':<8} | { 'Recall':<8} | { 'Precision':<10} | { 'F1 Score':<8}")
    print("-" * 55)
    
    adaptors = ["R1", "R2", "R3"]
    # 动态检查实际存在的 adaptors
    if data:
        first_metrics = list(data.values())[0]
        adaptors = sorted(first_metrics.keys())

    for ad in adaptors:
        avgs = {m: [] for m in metrics_list}
        for idx in data:
            if ad in data[idx]:
                for m in metrics_list:
                    avgs[m].append(data[idx][ad][m])
        
        if not avgs["f1"]: continue
        
        print(f"{ad:<10} | {np.mean(avgs['fluency']):.2f}     | {np.mean(avgs['recall']):.2%}   | {np.mean(avgs['precision']):.2%}      | {np.mean(avgs['f1']):.4f}")

    print("\n## 2. Detailed Instance Breakdown (F1 Score)")
    header = f"{ 'Inst':<5}"
    for ad in adaptors: header += f" | {ad:<6}"
    header += " | Winner"
    print(header)
    print("-" * (len(header) + 5))
    
    for idx in sorted(data.keys()):
        row = f"{idx:<5}"
        scores = {}
        for ad in adaptors:
            if ad in data[idx]:
                s = data[idx][ad]['f1']
                scores[ad] = s
                row += f" | {s:.4f}"
            else:
                scores[ad] = -1
                row += f" | N/A   "
        
        winner = max(scores, key=scores.get) if scores else "N/A"
        row += f" | {winner}"
        print(row)

if __name__ == "__main__":
    analyze_lru_a()
