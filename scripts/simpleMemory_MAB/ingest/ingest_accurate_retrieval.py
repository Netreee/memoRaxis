import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
sys.path.append(str(Path(__file__).parent))

from src.logger import get_logger
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices
from src.simple_memory import SimpleRAGMemory

logger = get_logger()

def ingest_one_instance(instance_idx: int, chunk_size: int):
    logger.info(f"=== Processing Instance {instance_idx} ===")
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
    
    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    chunks = chunk_context(data["context"], chunk_size=chunk_size)

    table_name = f"bench_acc_ret_{instance_idx}"
    logger.info(f"Ingesting into table: {table_name}")
    
    # Initialize Memory with specific table
    memory = SimpleRAGMemory(table_name=table_name)
    memory.reset() # Always reset on explicit ingest run

    print(f"Starting ingestion of {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"doc_id": i, "instance_idx": instance_idx})
        if i % 10 == 0:
            print(f"Ingested {i}/{len(chunks)} chunks...", end="\r", flush=True)
    print(f"\nIngestion complete. {len(chunks)} records added to {table_name}.")

def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument("--chunk_size", type=int, default=850, help="Fallback chunk size")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.chunk_size)

if __name__ == "__main__":
    main()
