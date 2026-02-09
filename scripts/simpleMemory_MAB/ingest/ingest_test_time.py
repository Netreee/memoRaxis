
import argparse
import sys
import re
import logging
from pathlib import Path
from typing import List

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from src.logger import get_logger
from src.simple_memory import SimpleRAGMemory
from src.benchmark_utils import parse_instance_indices

logger = get_logger()

def chunk_dialogues(context: str) -> List[str]:
    """
    策略 A: 针对 Dialogue N: 格式的正则切分
    """
    # 使用正则 split，保留分隔符内容有点麻烦，不如直接 findall 或者 split 后重组
    # 这里使用 split 并重新加上 Dialogue N: 前缀
    # Split by "Dialogue \d+:\n" but keeping the delimiter is tricky in Python split.
    # We will split by newline before "Dialogue" to keep structure.
    
    parts = re.split(r'\n(Dialogue \d+:)', '\n' + context)
    chunks = []
    # parts[0] is empty or preamble
    # parts[1] is "Dialogue 1:", parts[2] is content...
    
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        full_text = f"{header}\n{body}"
        if len(full_text) > 10: # Filter empty
            chunks.append(full_text)
            
    return chunks

def chunk_accumulation(context: str, min_chars: int = 800) -> List[str]:
    """
    策略 B: 累积切分 (复用 Conflict Resolution 的逻辑)
    """
    lines = [line.strip() for line in context.split('\n') if line.strip()]
    chunks = []
    current_chunk_lines = []
    current_length = 0
    
    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)
        
        if current_length > min_chars:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_length = 0
            
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))
        
    return chunks

def ingest_one_instance(instance_idx: int):
    logger.info(f"=== Processing TTL Instance {instance_idx} ===")
    
    data_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"
    
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

    context = data["context"]
    
    # 自适应选择策略
    if "Dialogue 1:" in context[:500]:
        logger.info("Strategy: Regex Split (Dialogue mode)")
        chunks = chunk_dialogues(context)
    else:
        logger.info("Strategy: Accumulation > 800 chars (ShortText mode)")
        chunks = chunk_accumulation(context, min_chars=800)

    table_name = f"bench_ttl_{instance_idx}"
    logger.info(f"Ingesting {len(chunks)} chunks into table: {table_name}")
    
    # Initialize Memory
    memory = SimpleRAGMemory(table_name=table_name)
    memory.reset() 

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 50 == 0:
            print(f"  Ingested {i}/{len(chunks)}...", end="\r", flush=True)
            
    print(f"\nInstance {instance_idx} complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Ingest Test_Time_Learning data")
    parser.add_argument("--instance_idx", type=str, default="0-5", help="Index range (e.g., '0-5')")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx)

if __name__ == "__main__":
    main()
