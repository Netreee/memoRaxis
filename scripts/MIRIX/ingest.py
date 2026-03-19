import argparse
import sys
import os
import time
import re
import json
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from MIRIX.remote_client import MirixClient
except ImportError:
    try:
        from mirix.client.remote_client import MirixClient
    except ImportError:
        raise ImportError("Could not import MirixClient. Please ensure MIRIX package is available.")

from src.logger import get_logger
from src.mirix import Mirix
from src.mirix_utils import get_mirix_config, get_mirix_connection_info
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
        "user_prefix": "mirix_acc_ret",
        "default_chunk_size": 8000       # 对齐 mem0, ~2400 tokens/chunk
    },
    "conflict_resolution": {
        "folder": "Conflict_Resolution",
        "user_prefix": "mirix_conf_res",
        "default_min_chars": 8000        # 对齐 mem0
    },
    "long_range_understanding": {
        "folder": "Long_Range_Understanding",
        "user_prefix": "mirix_long_range",
        "default_chunk_size": 24000,     # 对齐 mem0
        "default_overlap": 300
    },
    "test_time_learning": {
        "folder": "Test_Time_Learning",
        "user_prefix": "mirix_ttl",
        "default_min_chars": 12000       # 对齐 mem0; 对话模式按 Dialogue N: 切割, 不受此参数影响
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
            raise FileNotFoundError(f"Data file not found: {data_path}. Please set up data first.")
        
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
    user_id = f"{config['user_prefix']}_{instance_idx}"
    logger.info(f"Ingesting into MIRIX for user: {user_id}")
    
    # Initialize MIRIX Client
    try:
        client = MirixClient(api_key=args.api_key, base_url=args.base_url)

        try:
             client.initialize_meta_agent(config=get_mirix_config())
        except Exception as e:
            logger.error(f"Meta agent initialization failed: {e}")
            
        memory_system = Mirix(client, user_id=user_id)
        
    except Exception as e:
        logger.error(f"Failed to initialize MIRIX: {e}")
        return

    if args.force:
        logger.info(f"Force flag set. Clearing memory for user {user_id}...")
        try:
            memory_system.reset()
            time.sleep(2) 
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")

    # 4. Ingest Loop
    logger.info(f"Starting ingestion of {len(chunks)} chunks into MIRIX...")
    ingest_start = time.time()
    success_count = 0

    for i, chunk in enumerate(chunks):
        if args.max_chunks is not None and i >= args.max_chunks:
            logger.info(f"Reached max_chunks limit ({args.max_chunks}); exiting early.")
            break

        try:
            metadata = {
                "chunk_id": i,
                "instance_idx": instance_idx,
                "dataset": dataset,
                "source": "MemoryAgentBench"
            }

            chunk_t0 = time.time()
            memory_system.add_memory(chunk, metadata=metadata)
            last_chunk_latency = round(time.time() - chunk_t0, 2)

            update_freq = 50 if len(chunks) > 50 else 10
            if (i + 1) % update_freq == 0:
                logger.info(f"Progress: {i + 1}/{len(chunks)} chunks queued | last_chunk_latency={last_chunk_latency}s")
            success_count += 1

        except Exception as e:
            logger.error(f"Failed to add chunk {i}: {e}")

    total_time = round(time.time() - ingest_start, 2)
    logger.info(f"Instance {instance_idx} complete: {success_count} chunks ingested in {total_time}s (token tracking N/A — MIRIX internal LLM calls)")

def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data into MIRIX (Unified)")
    
    # Dataset Selection
    parser.add_argument("--dataset", required=True, 
                        choices=["accurate_retrieval", "conflict_resolution", "long_range_understanding", "test_time_learning"],
                        help="Which dataset to ingest")
    
    # Common Args
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument('--force', action='store_true', help='Force re-ingestion (clears existing memory for user)')
    parser.add_argument("--max_chunks", type=int, default=None, help="Maximum number of chunks to ingest per instance (for testing)")
    
    # Specific Args (Optional, will default based on dataset if not provided)
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size (Accurate_Retrieval, Long_Range)")
    parser.add_argument("--overlap", type=int, default=None, help="Overlap (Long_Range)")
    parser.add_argument("--min_chars", type=int, default=None, help="Minimum chars per chunk (Conflict_Resolution, TTL)")
    
    # MIRIX Connection Args
    parser.add_argument('--api_key', type=str, default=None, help='MIRIX API Key')
    parser.add_argument('--base_url', type=str, default=None, help='MIRIX Base URL')

    args = parser.parse_args()

    # Load connection info from config/config.yaml explicitly via utils
    mirix_conn = get_mirix_connection_info()
    
    # Determine API Key and Base URL: Args > Config
    args.api_key = args.api_key or mirix_conn.get("api_key")
    args.base_url = args.base_url or mirix_conn.get("base_url")

    # Set env vars if provided
    if args.api_key:
        os.environ["MIRIX_API_KEY"] = args.api_key

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target Dataset: {args.dataset}")
    logger.info(f"Target Instances: {indices}")

    for idx in indices:
        ingest_one_instance(args.dataset, idx, args)

if __name__ == "__main__":
    main()