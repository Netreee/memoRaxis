import argparse
import sys
import re
import json
import time
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.logger import get_logger
from src.memos import MemOS
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices

logger = get_logger()

# --- Chunking Strategies (Copied from mem0_MAB/ingest.py) ---

def chunk_facts(context: str, min_chars: int = 800) -> List[str]:
    """
    Accumulate lines until buffer > min_chars. Used for Conflict Resolution.
    """
    lines = [line.strip() for line in context.split('\n') if line.strip()]
    
    chunks = []
    current_chunk_lines = []
    current_length = 0
    
    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)
        
        if current_length > min_chars:
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append(chunk_text)
            current_chunk_lines = []
            current_length = 0
            
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        chunks.append(chunk_text)
        
    return chunks

def chunk_dialogues(context: str) -> List[str]:
    """
    Split by 'Dialogue N:'. Used for Test Time Learning (Dialogue mode).
    """
    parts = re.split(r'\n(Dialogue \d+:)', '\n' + context)
    chunks = []
    
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        full_text = f"{header}\n{body}"
        if len(full_text) > 10: 
            chunks.append(full_text)
            
    return chunks

def chunk_accumulation(context: str, min_chars: int = 800) -> List[str]:
    """
    Alias for chunk_facts logic, used in Test Time Learning (ShortText mode).
    """
    return chunk_facts(context, min_chars)

# --- Data Loading & Configuration ---

DATASETS = {
    "accurate_retrieval": {
        "folder": None, # Uses parquet data loader
        "collection_prefix": "memos_acc_ret",
        "default_chunk_size": 8000
    },
    "conflict_resolution": {
        "folder": "Conflict_Resolution",
        "collection_prefix": "memos_conflict",
        "default_min_chars": 8000
    },
    "long_range_understanding": {
        "folder": "Long_Range_Understanding",
        "collection_prefix": "memos_long_range",
        "default_chunk_size": 24000,
        "default_overlap": 300
    },
    "test_time_learning": {
        "folder": "Test_Time_Learning",
        "collection_prefix": "memos_ttl",
        "default_min_chars": 1600
    }
}

def load_data(dataset: str, instance_idx: int):
    config = DATASETS[dataset]
    
    if dataset == "accurate_retrieval":
        data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
        try:
            return load_benchmark_data(data_path, instance_idx)
        except Exception as e:
            raise RuntimeError(f"Error loading Parquet data for instance {instance_idx}: {e}")
            
    elif config["folder"]:
        data_path = f"MemoryAgentBench/preview_samples/{config['folder']}/instance_{instance_idx}.json"
        path_obj = Path(data_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}. Please run convert_all_data.py first.")
        
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading JSON data from {data_path}: {e}")
            
    else:
        raise ValueError(f"Unknown dataset configuration for {dataset}")

def get_chunks(dataset: str, context: str, args) -> List[str]:
    chunks = []
    
    if dataset == "accurate_retrieval":
        chunk_size = args.chunk_size or DATASETS[dataset]["default_chunk_size"]
        chunks = chunk_context(context, chunk_size=chunk_size)
        
    elif dataset == "conflict_resolution":
        min_chars = args.min_chars or DATASETS[dataset]["default_min_chars"]
        chunks = chunk_facts(context, min_chars=min_chars)
        
    elif dataset == "long_range_understanding":
        chunk_size = args.chunk_size or DATASETS[dataset]["default_chunk_size"]
        overlap = args.overlap if args.overlap is not None else DATASETS[dataset]["default_overlap"]
        chunks = chunk_context(context, chunk_size=chunk_size, overlap=overlap)
        
    elif dataset == "test_time_learning":
        # Adaptive Strategy
        if "Dialogue 1:" in context[:500]:
            logger.info("Strategy: Regex Split (Dialogue mode)")
            chunks = chunk_dialogues(context)
        else:
            logger.info("Strategy: Accumulation (ShortText mode)")
            min_chars = args.min_chars or DATASETS[dataset]["default_min_chars"]
            chunks = chunk_accumulation(context, min_chars=min_chars)
            
    return chunks

def ingest_one_instance(dataset: str, instance_idx: int, args):
    t_start = time.perf_counter()
    logger.info(f"=== Processing [{dataset}] Instance {instance_idx} ===")

    # 1. Load Data
    try:
        data = load_data(dataset, instance_idx)
    except Exception as e:
        logger.error(str(e))
        return

    # 2. Chunk Data
    context = data["context"]
    chunks = get_chunks(dataset, context, args)
    if not chunks:
        logger.warning(f"No chunks generated for instance {instance_idx}")
        return

    # 3. Initialize Memory
    config = DATASETS[dataset]
    user_id = f"{config['collection_prefix']}_{instance_idx}"

    logger.info(f"Ingesting {len(chunks)} chunks into MemOS for user_id: {user_id}")

    try:
        memory = MemOS(user_id=user_id)
    except Exception as e:
        logger.error(f"Failed to initialize MemOS: {e}")
        return

    if args.force:
        logger.info(f"Force flag ignored for MemOS ingestion script.")

    # 4. Ingest Loop
    logger.info(f"Starting ingestion of {len(chunks)} chunks into MemOS...")
    success_count = 0

    for i, chunk in enumerate(chunks):
        try:
            metadata = {
                "chunk_id": i,
                "instance_idx": instance_idx,
                "dataset": dataset,
                "source": "MemoryAgentBench"
            }
            memory.add_memory(data=chunk, metadata=metadata)
            success_count += 1

            update_freq = 100 if len(chunks) > 500 else 10
            if (i + 1) % update_freq == 0:
                logger.info(f"Progress: {i + 1}/{len(chunks)} chunks added")
        except Exception as e:
            logger.error(f"Failed to add chunk {i}: {e}")

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"Instance {instance_idx} complete: {success_count}/{len(chunks)} chunks ingested, "
        f"elapsed={elapsed:.1f}s ({elapsed/60:.1f}min). "
        f"Note: MemOS LLM token count not available externally."
    )

def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data into MemOS")
    
    # Dataset Selection
    parser.add_argument("--dataset", required=True, 
                        choices=["accurate_retrieval", "conflict_resolution", "long_range_understanding", "test_time_learning"],
                        help="Which dataset to ingest")
    
    # Common Args
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument('--force', action='store_true', help='Force re-ingestion (Note: mostly placebo for MemOS script currently)')
    
    # Specific Args (Optional, will default based on dataset if not provided)
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size (Accurate_Retrieval, Long_Range)")
    parser.add_argument("--overlap", type=int, default=None, help="Overlap (Long_Range)")
    parser.add_argument("--min_chars", type=int, default=None, help="Minimum chars per chunk (Conflict_Resolution, TTL)")

    args = parser.parse_args()

    # Parse Indices
    indices = parse_instance_indices(args.instance_idx)

    logger.info(f"Target Dataset: {args.dataset}")
    logger.info(f"Target Instances: {indices}")

    for idx in indices:
        ingest_one_instance(args.dataset, idx, args)

if __name__ == "__main__":
    main()
