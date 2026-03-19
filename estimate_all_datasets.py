
import pandas as pd
import tiktoken
import math
from pathlib import Path

def estimate():
    data_dir = Path("MemoryAgentBench/data")
    datasets = [
        "Accurate_Retrieval-00000-of-00001.parquet",
        "Conflict_Resolution-00000-of-00001.parquet",
        "Long_Range_Understanding-00000-of-00001.parquet",
        "Test_Time_Learning-00000-of-00001.parquet"
    ]
    
    # Tiktoken encoder (using cl100k_base which is standard for GPT-4/embeddings)
    enc = tiktoken.get_encoding("cl100k_base")
    
    CHUNK_SIZE_TOKENS = 2000
    OVERLAP_TOKENS = int(CHUNK_SIZE_TOKENS * 0.05) # 100
    STRIDE = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS # 1900
    
    # Assumptions for Mem0G / GraphRAG LLM extraction
    PROMPT_OVERHEAD_TOKENS = 200
    ESTIMATED_OUTPUT_TOKENS_PER_CHUNK = 300
    
    print(f"=== Dataset Token & Chunk Estimation ===")
    print(f"Config: {CHUNK_SIZE_TOKENS} tokens/chunk, {OVERLAP_TOKENS} overlap (5%)")
    print("-" * 85)
    print(f"{'Dataset':<25} | {'Instances'} | {'Total Chunks'} | {'Est. Input Tokens'} | {'Est. Output Tokens'}")
    print("-" * 85)
    
    grand_chunks = 0
    grand_input = 0
    grand_output = 0

    for file_name in datasets:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"{file_name[:25]:<25} | File not found")
            continue
            
        df = pd.read_parquet(file_path)
        num_instances = len(df)
        
        dataset_chunks = 0
        
        for idx in range(num_instances):
            context = df.iloc[idx]["context"]
            # Convert text to tokens
            tokens = enc.encode(context)
            num_tokens = len(tokens)
            
            # Calculate chunks
            if num_tokens <= CHUNK_SIZE_TOKENS:
                chunks = 1
            else:
                # Number of strides needed to cover the text
                chunks = math.ceil((num_tokens - OVERLAP_TOKENS) / STRIDE)
                
            dataset_chunks += chunks
            
        dataset_input_tokens = dataset_chunks * (CHUNK_SIZE_TOKENS + PROMPT_OVERHEAD_TOKENS)
        dataset_output_tokens = dataset_chunks * ESTIMATED_OUTPUT_TOKENS_PER_CHUNK
        
        name_short = file_name.split('-')[0][:25]
        print(f"{name_short:<25} | {num_instances:<9} | {dataset_chunks:<12} | {dataset_input_tokens:<17} | {dataset_output_tokens:<18}")
        
        grand_chunks += dataset_chunks
        grand_input += dataset_input_tokens
        grand_output += dataset_output_tokens

    print("-" * 85)
    print(f"{'TOTAL':<25} | {'-':<9} | {grand_chunks:<12} | {grand_input:<17} | {grand_output:<18}")

if __name__ == "__main__":
    estimate()
