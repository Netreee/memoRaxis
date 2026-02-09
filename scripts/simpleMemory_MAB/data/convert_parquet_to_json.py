
import pandas as pd
import json
import re
import os
import numpy as np
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """ 处理 numpy 对象以便 JSON 序列化 """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
    output_dir = Path("MemoryAgentBench/preview_samples/Accurate_Retrieval")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_parquet(data_path)
        print(f"Loaded parquet file with {len(df)} instances.")
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return

    # Task 1: Analyze Instance 0 Document Length
    instance_0_context = df.iloc[0]["context"]
    # Regex split as per benchmark_utils logic
    chunks = re.split(r"Document \d+:\n", instance_0_context)
    valid_chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
    
    if valid_chunks:
        avg_len = sum(len(c) for c in valid_chunks) / len(valid_chunks)
        print(f"\n[Analysis for Instance 0]")
        print(f"Total Documents found: {len(valid_chunks)}")
        print(f"Average Character Count per Document: {avg_len:.2f}")
    else:
        print("\n[Analysis for Instance 0]")
        print("No 'Document N' format found or documents are empty.")

    # Task 2: Convert all instances to JSON
    print(f"\n[Converting Instances to JSON]")
    for i in range(len(df)):
        instance_data = df.iloc[i].to_dict()
        
        # Save to file
        file_name = f"instance_{i}.json"
        file_path = output_dir / file_name
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(instance_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        if i % 5 == 0:
            print(f"Saved {file_name}...")

    print(f"All 22 instances saved to {output_dir}")

if __name__ == "__main__":
    main()
