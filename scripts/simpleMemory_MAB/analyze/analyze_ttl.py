
import json
import glob
import numpy as np
import re
from pathlib import Path

def load_id_map():
    with open("MemoryAgentBench/entity2id.json", 'r') as f:
        data = json.load(f)
    id_to_title = {}
    for uri, idx in data.items():
        raw = uri.replace("<http://dbpedia.org/resource/", "").replace(">", "")
        title = raw.replace("_", " ")
        id_to_title[str(idx)] = title
    return id_to_title

def analyze_ttl():
    eval_files = glob.glob("out/ttl_results_*.json")
    eval_files.sort(key=lambda x: int(re.search(r'ttl_results_(\d+).json', x).group(1)))
    
    if not eval_files:
        print("No evaluation files found.")
        return

    id_map = load_id_map()
    
    print(f"=== Test Time Learning (TTL) Evaluation Report (N={len(eval_files)}) ===\n")
    print("## 1. Detailed Instance Breakdown (Accuracy)")
    print(f"{ 'Inst':<5} | { 'Type':<15} | { 'R1':<8} | { 'R2':<8} | { 'R3':<8} | { 'Winner':<6}")
    print("-" * 75)

    all_scores = {"R1": [], "R2": [], "R3": []}

    for fpath in eval_files:
        match = re.search(r'ttl_results_(\d+).json', fpath)
        idx = int(match.group(1))
        
        with open(fpath, 'r') as f:
            data = json.load(f)
            
        # Determine type
        # Instance 0 is Movie Rec, 1-5 is Banking (Short Text)
        inst_type = "Movie Rec" if idx == 0 else "Banking77"
        
        instance_scores = {}
        for ad in ["R1", "R2", "R3"]:
            items = data["results"].get(ad, [])
            correct = 0
            total = len(items)
            
            for item in items:
                pred = item.get("answer", "")
                gt_ids = item.get("ground_truth", [])
                
                is_hit = False
                for gid in gt_ids:
                    title = id_map.get(str(gid))
                    if title:
                        if title.lower() in pred.lower():
                            is_hit = True
                            break
                        base_title = re.sub(r'\s\(\d{4}.*\)', '', title)
                        if base_title.lower() in pred.lower() and len(base_title) > 3:
                             is_hit = True
                             break
                
                if is_hit:
                    correct += 1
            
            acc = correct / total if total > 0 else 0.0
            instance_scores[ad] = acc
            all_scores[ad].append(acc)

        winner = max(instance_scores, key=instance_scores.get)
        print(f"{idx:<5} | {inst_type:<15} | {instance_scores['R1']:.2%}   | {instance_scores['R2']:.2%}   | {instance_scores['R3']:.2%}   | {winner}")

    print("\n## 2. Summary by Type")
    
    # Instance 0
    print("### Movie Recommendation (Instance 0)")
    print(f"- R1: {all_scores['R1'][0]:.2%}")
    print(f"- R2: {all_scores['R2'][0]:.2%}")
    print(f"- R3: {all_scores['R3'][0]:.2%}")
    
    # Banking (1-5)
    banking_r1 = np.mean(all_scores['R1'][1:]) if len(all_scores['R1']) > 1 else 0
    banking_r2 = np.mean(all_scores['R2'][1:]) if len(all_scores['R2']) > 1 else 0
    banking_r3 = np.mean(all_scores['R3'][1:]) if len(all_scores['R3']) > 1 else 0
    
    print("\n### Banking Intent (Instance 1-5)")
    print("*Note: Low scores due to missing Entity Mapping for Banking77 labels*")
    print(f"- R1: {banking_r1:.2%}")
    print(f"- R2: {banking_r2:.2%}")
    print(f"- R3: {banking_r3:.2%}")

if __name__ == "__main__":
    analyze_ttl()
