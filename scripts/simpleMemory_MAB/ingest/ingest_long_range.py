
import argparse
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from src.logger import get_logger
from src.simple_memory import SimpleRAGMemory
from src.benchmark_utils import parse_instance_indices, chunk_context

logger = get_logger()

def ingest_one_instance(instance_idx: int, chunk_size: int, overlap: int):
    logger.info(f"=== Processing Long_Range_Understanding Instance {instance_idx} ===")
    
    # 使用预先转换好的 JSON 文件
    data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"
    
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

    # 使用滑动窗口切分
    # 注意：这里我们知道它是小说文本，没有 "Document N:"，chunk_context 会自动回退到滑动窗口
    chunks = chunk_context(data["context"], chunk_size=chunk_size, overlap=overlap)

    table_name = f"bench_long_range_{instance_idx}"
    logger.info(f"Ingesting {len(chunks)} chunks into table: {table_name}")
    
    # Initialize Memory
    memory = SimpleRAGMemory(table_name=table_name)
    memory.reset() # 重置表

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        # 记录 chunk 的相对顺序
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 100 == 0:
            print(f"  Ingested {i}/{len(chunks)}...", end="\r", flush=True)
            
    print(f"\nInstance {instance_idx} complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Ingest Long_Range_Understanding data")
    # 默认 Top 40 (0-39)
    parser.add_argument("--instance_idx", type=str, default="0-39", help="Index range (e.g., '0-39')")
    parser.add_argument("--chunk_size", type=int, default=1200, help="Chunk size for sliding window")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap for sliding window")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    logger.info(f"Config: Chunk Size={args.chunk_size}, Overlap={args.overlap}")

    for idx in indices:
        ingest_one_instance(idx, args.chunk_size, args.overlap)

if __name__ == "__main__":
    main()
