
import pandas as pd
import json
import numpy as np
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_split(parquet_name, output_folder_name):
    data_path = Path(f"MemoryAgentBench/data/{parquet_name}")
    output_dir = Path(f"MemoryAgentBench/preview_samples/{output_folder_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Skipping {parquet_name}: File not found.")
        return

    try:
        df = pd.read_parquet(data_path)
        print(f"Processing {parquet_name} ({len(df)} instances)...")
        
        for i in range(len(df)):
            instance_data = df.iloc[i].to_dict()
            file_path = output_dir / f"instance_{i}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(instance_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"Successfully converted {parquet_name} to {output_dir}")
    except Exception as e:
        print(f"Error converting {parquet_name}: {e}")

def main():
    splits = {
        "Accurate_Retrieval-00000-of-00001.parquet": "Accurate_Retrieval",
        "Conflict_Resolution-00000-of-00001.parquet": "Conflict_Resolution",
        "Long_Range_Understanding-00000-of-00001.parquet": "Long_Range_Understanding",
        "Test_Time_Learning-00000-of-00001.parquet": "Test_Time_Learning"
    }

    for p, folder in splits.items():
        convert_split(p, folder)

if __name__ == "__main__":
    main()
