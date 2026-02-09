
import json
import re

def inspect_raw():
    path = "MemoryAgentBench/preview_samples/Conflict_Resolution/instance_4.json"
    
    with open(path, 'r') as f:
        data = json.load(f)
        
    questions = data["questions"]
    answers = data["answers"]
    context = data["context"]
    
    # Indices of interest from previous analysis:
    # Index 2: Rugby Union
    # Index 17: Association Football
    
    target_indices = [2, 17]
    
    print(f"=== Raw Data Inspection (Instance 4) ===\n")
    
    for idx in target_indices:
        if idx >= len(questions):
            continue
            
        q = questions[idx]
        a = answers[idx]
        
        print(f"--- Q{idx} ---")
        print(f"Question: {q}")
        print(f"GT Answer: {a}")
        
        # Search in Context
        # Extract keyword from question to grep context
        keywords = []
        if "football" in q: keywords = ["association football"]
        if "rugby" in q: keywords = ["rugby union"]
        
        print(f"Context Evidence:")
        found = False
        for line in context.split('\n'):
            for kw in keywords:
                if kw in line:
                    print(f"  > {line.strip()}")
                    found = True
        if not found:
            print("  (No direct keyword match found in context)")
        print("")

if __name__ == "__main__":
    inspect_raw()
