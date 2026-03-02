import argparse
import sys
import re
import json
from pathlib import Path
from typing import List, Optional
from mem0 import Memory
from qdrant_client import QdrantClient

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.logger import get_logger
from src.mem0_utils import get_mem0_config
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices

logger = get_logger()

# --- Chunking Strategies ---

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
        "collection_prefix": "mem0_acc_ret",
        "default_chunk_size": 850
    },
    "conflict_resolution": {
        "folder": "Conflict_Resolution",
        "collection_prefix": "mem0_conflict",
        "default_min_chars": 800
    },
    "long_range_understanding": {
        "folder": "Long_Range_Understanding",
        "collection_prefix": "mem0_long_range",
        "default_chunk_size": 1200,
        "default_overlap": 100
    },
    "test_time_learning": {
        "folder": "Test_Time_Learning",
        "collection_prefix": "mem0_ttl",
        "default_min_chars": 800
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
    collection_name = f"{config['collection_prefix']}_{instance_idx}"
    logger.info(f"Ingesting {len(chunks)} chunks into collection: {collection_name}")
    
    mem0_config = get_mem0_config(collection_name)
    try:
        memory = Memory.from_config(mem0_config)
    except Exception as e:
        logger.error(f"Failed to initialize Mem0: {e}")
        return

    # 4. Check Existing Data
    user_id = "ingest_user"
    if not args.force:
        try:
            existing = memory.get_all(user_id=user_id, limit=1)
            results = existing.get("results") if isinstance(existing, dict) else existing
            
            if results and len(results) > 0:
                logger.info(f"Skipping Instance {instance_idx}: Collection {collection_name} already contains data. Use --force to overwrite.")
                return
        except Exception as e:
            logger.debug(f"Pre-flight check failed (likely safe to proceed): {e}")

    # 5. Ingest Loop
    logger.info(f"Starting ingestion of {len(chunks)} chunks into Mem0 (Qdrant)...")
    success_count = 0
    
    for i, chunk in enumerate(chunks):
        try:
            metadata = {
                "chunk_id": i,
                "instance_idx": instance_idx,
                "dataset": dataset,
                "source": "MemoryAgentBench"
            }
            # infer=False ensures we just store the text/vector, not run LLM processing if configured not to
            memory.add(chunk, user_id=user_id, metadata=metadata, infer=False)
            success_count += 1
            
            update_freq = 100 if len(chunks) > 500 else 10
            if (i + 1) % update_freq == 0:
                logger.info(f"Progress: {i + 1}/{len(chunks)} chunks added")
        except Exception as e:
            logger.error(f"Failed to add chunk {i}: {e}")
            
    logger.info(f"Instance {instance_idx} complete: {success_count}/{len(chunks)} chunks ingested.")

def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data into Mem0 (Unified)")
    
    # Dataset Selection
    parser.add_argument("--dataset", required=True, 
                        choices=["accurate_retrieval", "conflict_resolution", "long_range_understanding", "test_time_learning"],
                        help="Which dataset to ingest")
    
    # Common Args
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument('--force', action='store_true', help='Force re-ingestion even if data exists')
    
    # Specific Args (Optional, will default based on dataset if not provided)
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size (Accurate_Retrieval, Long_Range)")
    parser.add_argument("--overlap", type=int, default=None, help="Overlap (Long_Range)")
    parser.add_argument("--min_chars", type=int, default=None, help="Minimum chars per chunk (Conflict_Resolution, TTL)")

    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target Dataset: {args.dataset}")
    logger.info(f"Target Instances: {indices}")

    for idx in indices:
        ingest_one_instance(args.dataset, idx, args)

if __name__ == "__main__":
    main()
