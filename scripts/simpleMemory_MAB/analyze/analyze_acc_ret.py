import re
import numpy as np

def parse_raw_report(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Split by instance blocks
    blocks = content.split('--- [Mechanical Evaluation Result:')
    
    data = {}
    
    for block in blocks:
        if not block.strip():
            continue
            
        # Extract filename (instance id)
        title_match = re.search(r'acc_ret_results_(\d+)\.json', block)
        if not title_match:
            continue
        
        instance_idx = int(title_match.group(1))
        
        # Extract scores
        # Pattern: Adaptor R1 : Accuracy = 95.00% (95/100)
        scores = {}
        for adaptor in ['R1', 'R2', 'R3']:
            pattern = f"Adaptor {adaptor}\s+:\s+Accuracy\s+=\s+([\d\.]+)\%"
            match = re.search(pattern, block)
            if match:
                scores[adaptor] = float(match.group(1))
            else:
                scores[adaptor] = 0.0 # Default if missing
        
        data[instance_idx] = scores
        
    return data

def analyze(data):
    # Calculate average scores per instance (to define zones based on average performance or baseline R1)
    # Strategy: Use R1's score to define difficulty zones.
    # High: R1 > 80, Mid: 60-80, Low: < 60
    
    zones = {
        "High": [],
        "Mid": [],
        "Low": []
    }
    
    all_scores = {"R1": [], "R2": [], "R3": []}
    
    for idx, scores in data.items():
        # Collect for global
        for k in scores:
            all_scores[k].append(scores[k])
            
        # Determine zone by R1 performance (Baseline)
        base_score = scores['R1']
        if base_score >= 80:
            zones["High"].append(scores)
        elif base_score >= 60:
            zones["Mid"].append(scores)
        else:
            zones["Low"].append(scores)

    print(f"=== Accurate Retrieval Evaluation Report (N={len(data)}) ===\n")
    
    # 1. Global
    print("## 1. Global Performance (Avg Accuracy)")
    print(f"- R1 (Single-turn): {np.mean(all_scores['R1']):.2f}%")
    print(f"- R2 (Iterative)  : {np.mean(all_scores['R2']):.2f}%")
    print(f"- R3 (Plan-Act)   : {np.mean(all_scores['R3']):.2f}%")
    print("")

    # 2. Zones Analysis
    print("## 2. Zone Analysis (Based on R1 Baseline Difficulty)")
    
    for zone_name, items in [("High (>80%)", zones["High"]), ("Mid (60-80%)", zones["Mid"]), ("Low (<60%)", zones["Low"])]:
        if not items:
            continue
            
        n = len(items)
        r1_avg = np.mean([x['R1'] for x in items])
        r2_avg = np.mean([x['R2'] for x in items])
        r3_avg = np.mean([x['R3'] for x in items])
        
        print(f"### {zone_name} Zone (N={n})")
        print(f"- R1: {r1_avg:.2f}%")
        print(f"- R2: {r2_avg:.2f}% ({r2_avg - r1_avg:+.2f}% vs R1)")
        print(f"- R3: {r3_avg:.2f}% ({r3_avg - r1_avg:+.2f}% vs R1)")
        
        best_adaptor = "R1"
        if r2_avg > max(r1_avg, r3_avg): best_adaptor = "R2"
        if r3_avg > max(r1_avg, r2_avg): best_adaptor = "R3"
        
        print(f"  -> Winner: {best_adaptor}")
        print("")

    # 3. Detailed Table
    print("## 3. Detailed Instance Breakdown")
    print(f"{'Inst':<5} | {'R1':<6} | {'R2':<6} | {'R3':<6} | {'Winner':<6}")
    print("-" * 35)
    
    sorted_indices = sorted(data.keys())
    for idx in sorted_indices:
        s = data[idx]
        winner = max(s, key=s.get)
        print(f"{idx:<5} | {s['R1']:<6} | {s['R2']:<6} | {s['R3']:<6} | {winner:<6}")

if __name__ == "__main__":
    data = parse_raw_report("acc_ret_summary_raw.txt")
    analyze(data)
