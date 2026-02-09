
import json
import re
from pathlib import Path

def estimate_ttl_calls():
    folder = Path('MemoryAgentBench/preview_samples/Test_Time_Learning')
    target_size = 800
    
    print(f"{'Instance':<10} | {'Type':<10} | {'Strategy':<15} | {'Est. Calls':<10}")
    print("-" * 55)

    total_calls = 0

    for i in range(6):
        path = folder / f'instance_{i}.json'
        if not path.exists():
            continue

        with open(path, 'r') as f:
            data = json.load(f)
        
        context = data['context']
        
        # Check type based on "Dialogue 1:" signature
        if "Dialogue 1:" in context[:100]:
            # Strategy: Regex Split
            dialogues = re.findall(r'Dialogue \d+:', context)
            calls = len(dialogues)
            print(f"Inst {i:<5} | Dialogue   | Regex Split     | {calls:<10}")
        else:
            # Strategy: Accumulate > 800
            char_len = len(context)
            # 粗略估算，加上最后一块
            calls = (char_len // target_size) + 1
            print(f"Inst {i:<5} | ShortText  | Accumulate>800  | {calls:<10}")
            
        total_calls += calls

    print("-" * 55)
    print(f"Grand Total Estimated Embedding Calls: {total_calls}")

if __name__ == "__main__":
    estimate_ttl_calls()
