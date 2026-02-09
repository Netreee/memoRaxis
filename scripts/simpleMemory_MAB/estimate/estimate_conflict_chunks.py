
import json
import re
import os
from pathlib import Path

def estimate_chunks():
    folder = Path('MemoryAgentBench/preview_samples/Conflict_Resolution')
    total_calls = 0
    target_size = 800

    print(f"{'Instance':<10} | {'Total Facts':<12} | {'Est. Chunks (Calls)':<20} | {'Avg Facts/Chunk':<15}")
    print("-" * 65)

    for i in range(8):
        path = folder / f'instance_{i}.json'
        if not path.exists():
            continue

        with open(path, 'r') as f:
            data = json.load(f)
        
        context = data['context']
        # Split into list of facts. 
        # Pattern matches newline followed by digits and dot.
        # We assume the file starts with "0. " or similar so we handle the first one if needed,
        # but usually context is just a string. 
        # A simple split by newline is safer if lines represent facts.
        
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        chunks = 0
        current_chunk_len = 0
        current_facts_count = 0
        
        facts_per_chunk_list = []

        for line in lines:
            line_len = len(line)
            current_chunk_len += line_len
            current_facts_count += 1
            
            # Strategy: Accumulate until > 800 chars
            if current_chunk_len > target_size:
                chunks += 1
                facts_per_chunk_list.append(current_facts_count)
                
                # Reset for next chunk
                current_chunk_len = 0
                current_facts_count = 0
        
        # Count the last partial chunk if it has any content
        if current_chunk_len > 0:
            chunks += 1
            facts_per_chunk_list.append(current_facts_count)

        avg_facts = sum(facts_per_chunk_list) / len(facts_per_chunk_list) if facts_per_chunk_list else 0
        
        print(f"{f'Inst {i}':<10} | {len(lines):<12} | {chunks:<20} | {avg_facts:.1f}")
        total_calls += chunks

    print("-" * 65)
    print(f"Grand Total Estimated Embedding Calls: {total_calls}")

if __name__ == "__main__":
    estimate_chunks()
