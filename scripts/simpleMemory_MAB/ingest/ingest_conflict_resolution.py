
import argparse
import sys
import logging
from pathlib import Path
from typing import List

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from src.logger import get_logger
from src.simple_memory import SimpleRAGMemory
from src.benchmark_utils import load_benchmark_data, parse_instance_indices

logger = get_logger()

def chunk_facts(context: str, min_chars: int = 800) -> List[str]:
    """
    Conflict Resolution 专用切分策略：
    按行读取 Fact，累积直到缓冲区字符数 > min_chars，然后作为一个 Chunk。
    """
    lines = [line.strip() for line in context.split('\n') if line.strip()]
    
    chunks = []
    current_chunk_lines = []
    current_length = 0
    
    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)
        
        if current_length > min_chars:
            # 形成一个 chunk
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append(chunk_text)
            # 重置缓冲区
            current_chunk_lines = []
            current_length = 0
            
    # 处理剩余的缓冲区
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        chunks.append(chunk_text)
        
    return chunks

def ingest_one_instance(instance_idx: int, min_chars: int):
    logger.info(f"=== Processing Conflict_Resolution Instance {instance_idx} ===")
    
    # 注意：这里我们读取的是已经转换好的 JSON 文件，而不是 Parquet
    # 因为之前的 convert_all_data.py 已经生成了漂亮的 JSON
    data_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    try:
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    # 使用专用切分策略
    chunks = chunk_facts(data["context"], min_chars=min_chars)

    table_name = f"bench_conflict_{instance_idx}"
    logger.info(f"Ingesting {len(chunks)} chunks into table: {table_name}")
    
    # Initialize Memory
    memory = SimpleRAGMemory(table_name=table_name)
    memory.reset() # 自动删表重建

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        # 元数据记录该 chunk 包含的 fact 范围稍微麻烦点，这里简化记录 chunk id
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 10 == 0:
            print(f"  Ingested {i}/{len(chunks)}...", end="\r", flush=True)
            
    print(f"\nInstance {instance_idx} complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Ingest Conflict_Resolution data")
    parser.add_argument("--instance_idx", type=str, default="0-7", help="Index range (e.g., '0-7')")
    parser.add_argument("--min_chars", type=int, default=800, help="Minimum chars per chunk")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.min_chars)

if __name__ == "__main__":
    main()
