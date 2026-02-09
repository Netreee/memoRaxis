
import json
import random

def sample_r3_cases():
    path = "out/acc_ret_results_2.json"
    with open(path, 'r') as f:
        data = json.load(f)
    
    r3_results = data.get("results", {}).get("R3", [])
    if not r3_results:
        print("No R3 results found.")
        return

    # Sample 5 random indices
    indices = random.sample(range(len(r3_results)), 5)
    
    # Instance 2 is "The Moonstone" or similar narrative text based on previous logs?
    # Actually based on logs it seemed to be Gone with the Wind or similar?
    # Let's see the content.
    
    for i, idx in enumerate(indices):
        item = r3_results[idx]
        print(f"=== Case {i+1} (Index {idx}) ===")
        print(f"Question: {item.get('question')}")
        print(f"Answer: {item.get('answer')}")
        print(f"Steps: {item.get('steps')}")
        
        # Note: The result file might not contain the detailed plan/evidence logs.
        # Those are usually in the log file, not the result json. 
        # The result json only has summary metrics.
        # Wait, the result json structure in `evaluate_accurate_retrieval.py` only saves:
        # question, answer, ground_truth, steps, tokens, replan.
        # It DOES NOT save the plan text or evidence content.
        
        # To do a deep dive, I need to correlate with the log file!
        # The log file `log/20260206-xxxx.log` (if it was logged) would contain the plan.
        # But since I don't know exactly which log file corresponds to this run 
        # (the user implied it might be from yesterday, or the one I saw running earlier which was PID 2486),
        # I should try to grep the log file I saw earlier: `log/20260204-144402.log` or similar? 
        # No, the one I saw running was actively logging.
        
        # Let's first print the question from the JSON, then I can grep the logs for that specific question
        # to find the Plan and Evidence.
        print("-" * 20)

if __name__ == "__main__":
    sample_r3_cases()
