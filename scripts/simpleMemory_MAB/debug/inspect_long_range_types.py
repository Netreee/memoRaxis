
import json
import os
from pathlib import Path

def inspect_types():
    folder = Path("MemoryAgentBench/preview_samples/Long_Range_Understanding")
    
    # Counters
    type_a_count = 0  # 1 question
    type_b_count = 0  # >1 questions
    
    print(f"{'Inst':<5} | {'Q Count':<8} | {'Q1 Preview':<50} | {'Answer Type'}")
    print("-" * 90)

    # Scan all 110 instances (assuming they exist from conversion)
    # Use convert_all_data.py generated files
    instances = sorted([f for f in os.listdir(folder) if f.startswith("instance_") and f.endswith(".json")], 
                       key=lambda x: int(x.split('_')[1].split('.')[0]))

    for filename in instances:
        idx = int(filename.split('_')[1].split('.')[0])
        path = folder / filename
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        questions = data.get("questions", [])
        answers = data.get("answers", [])
        q_count = len(questions)
        
        q1_preview = questions[0][:45] + "..." if questions else "NO QUESTIONS"
        ans_preview = str(answers[0])[:20] + "..." if answers else "NO ANSWERS"
        
        if q_count == 1:
            type_a_count += 1
        elif q_count > 1:
            type_b_count += 1
            
        # Sampling output: Print first 10, last 10, and any Type B found
        if idx < 10 or idx >= 100 or q_count > 1:
            print(f"{idx:<5} | {q_count:<8} | {q1_preview:<50} | {ans_preview}")

    print("-" * 90)
    print(f"Summary:")
    print(f"Type A (1 Question): {type_a_count}")
    print(f"Type B (>1 Question): {type_b_count}")

if __name__ == "__main__":
    inspect_types()
