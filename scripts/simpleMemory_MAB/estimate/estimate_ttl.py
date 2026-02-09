
import json
import re
from pathlib import Path

def estimate_ttl():
    folder = Path('MemoryAgentBench/preview_samples/Test_Time_Learning')
    total_dialogues = 0
    total_chars = 0
    
    print(f"{'Instance':<10} | {'Char Count':<12} | {'Dialogues':<12} | {'Avg Chars/Dial':<15}")
    print("-" * 60)

    for i in range(6):
        path = folder / f'instance_{i}.json'
        if not path.exists():
            continue

        with open(path, 'r') as f:
            data = json.load(f)
        
        context = data['context']
        char_len = len(context)
        
        # Count dialogues
        dialogues = re.findall(r'Dialogue \d+:', context)
        num_dialogues = len(dialogues)
        
        avg_len = char_len / num_dialogues if num_dialogues > 0 else 0
        
        print(f"Inst {i:<5} | {char_len:<12} | {num_dialogues:<12} | {avg_len:.0f}")
        
        total_chars += char_len
        total_dialogues += num_dialogues

    print("-" * 60)
    print(f"Total Dialogues: {total_dialogues}")
    print(f"Total Chars: {total_chars}")

if __name__ == "__main__":
    estimate_ttl()
