
import json
import random

def find_failures():
    # Load R3 results (Instance 2)
    with open("out/acc_ret_results_2.json", 'r') as f:
        data = json.load(f)
    
    # Load Ground Truth
    with open("MemoryAgentBench/preview_samples/Accurate_Retrieval/instance_2.json", 'r') as f:
        gt_data = json.load(f)
    
    gt_map = {q: a for q, a in zip(gt_data["questions"], gt_data["answers"])}
    
    failures = []
    r3_items = data.get("results", {}).get("R3", [])
    
    for item in r3_items:
        q = item.get("question")
        pred = item.get("answer")
        
        # Simple exact match or inclusion check
        # Since answers are choices, we check if the prediction contains the ground truth text
        # But wait, gt is a list of strings in the original file? No, usually it's a single string for AccRet
        # Let's check gt format
        
        ref = gt_map.get(q)
        if not ref:
            continue
            
        # In AccRet, ref is usually a list like ["Answer Text"]
        if isinstance(ref, list):
            ref_text = ref[0]
        else:
            ref_text = ref
            
        if ref_text not in pred:
            failures.append({
                "question": q,
                "prediction": pred,
                "reference": ref_text
            })
            
    print(f"Found {len(failures)} failures in R3.")
    
    # Sample 5
    sample = failures[:5] # Just take first 5 to align with sequential logs
    
    for i, fail in enumerate(sample):
        print(f"\n=== Failure Case {i+1} ===")
        # Print first line of question to identify it easily
        q_lines = fail['question'].split('\n')
        print(f"Question Header: {q_lines[2] if len(q_lines)>2 else q_lines[0]}") 
        print(f"Prediction: {fail['prediction']}")
        print(f"Reference:  {fail['reference']}")
        print(f"Full Question Key: {fail['question'][:100]}...") # For grep

if __name__ == "__main__":
    find_failures()
