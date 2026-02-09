
import json
from pathlib import Path

def estimate_long_range():
    folder = Path('MemoryAgentBench/preview_samples/Long_Range_Understanding')
    total_chars = 0
    total_instances = 0
    chunk_size = 850
    overlap = 200
    stride = chunk_size - overlap

    print(f"{'Instance':<10} | {'Char Count':<12} | {'Est. Chunks':<12}")
    print("-" * 40)

    # 抽样统计前 10 个和最后 10 个，避免遍历太慢
    sample_indices = list(range(0, 10)) + list(range(100, 110))
    
    for i in sample_indices:
        path = folder / f'instance_{i}.json'
        if not path.exists():
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
        
        char_len = len(data['context'])
        # 简单估算滑动窗口数量
        chunks = (char_len // stride) + 1
        
        print(f"Inst {i:<5} | {char_len:<12} | {chunks:<12}")
        
        total_chars += char_len
        total_instances += 1

    avg_chars = total_chars / total_instances
    est_total_chunks = (avg_chars // stride + 1) * 110
    
    print("-" * 40)
    print(f"Average Length: {avg_chars:.0f} chars")
    print(f"Estimated Total Embedding Calls (for 110 instances): {est_total_chunks:.0f}")

if __name__ == "__main__":
    estimate_long_range()
