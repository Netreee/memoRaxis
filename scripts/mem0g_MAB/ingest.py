import argparse
import sys
import os
import re # Added for regex operations
import json # Added for json operations
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from mem0 import Memory
from qdrant_client import QdrantClient
try:
    import neo4j
    from neo4j import GraphDatabase, Session, Transaction
except ImportError:
    neo4j = None
    GraphDatabase = None
    Session = None
    Transaction = None

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.logger import get_logger
from src.mem0_utils import get_mem0_config
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices

logger = get_logger()

# --- Patch for Neo4j Syntax Error with Hyphens ---
def patch_neo4j_for_hyphens():
    """
    Monkey-patch neo4j.Session.run and neo4j.Transaction.run to backtick-quote
    relationship types that are invalid Cypher identifiers:
      - types containing hyphens (e.g. IS-A)
      - types starting with a digit (e.g. 15th_busiest_airport_in)
    """
    if not GraphDatabase or not Session or not Transaction:
        return

    # Only patch if not already patched
    if getattr(Session, '_patched_for_hyphens', False):
        return

    original_session_run = Session.run
    original_transaction_run = Transaction.run

    def sanitize_query(query: str) -> str:
        if not query:
            return query

        def replace(match):
            var = match.group(1)
            rel_type = match.group(2)
            return f"-[{var}:`{rel_type}`]"

        # Match unquoted rel types that: (a) contain a hyphen, or (b) start with a digit
        pattern = r"-\[\s*([a-zA-Z0-9_]*)\s*:\s*((?:[a-zA-Z0-9_]*-[a-zA-Z0-9_\-]*)|(?:\d[a-zA-Z0-9_]*))\s*\]"
        return re.sub(pattern, replace, query)

    def patched_session_run(self, query, parameters=None, **kwargs):
        new_query = sanitize_query(query)
        return original_session_run(self, new_query, parameters, **kwargs)

    def patched_transaction_run(self, query, parameters=None, **kwargs):
        new_query = sanitize_query(query)
        return original_transaction_run(self, new_query, parameters, **kwargs)

    Session.run = patched_session_run
    Session._patched_for_hyphens = True
    Transaction.run = patched_transaction_run
    logger.info("Patched Neo4j driver to backtick-quote invalid relationship type identifiers.")

# Apply the patch immediately
patch_neo4j_for_hyphens()

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

# --- Configuration for Datasets ---

DATASETS = {
    "accurate_retrieval": {
        "folder": None, # Uses parquet data loader
        "collection_prefix": "mem0g_acc_ret",
        "dataset_type": "acc_ret",
        "default_chunk_size": 8000
    },
    "conflict_resolution": {
        "folder": "Conflict_Resolution",
        "collection_prefix": "mem0g_conflict",
        "dataset_type": "conflict_res",
        "default_min_chars": 8000
    },
    "long_range_understanding": {
        "folder": "Long_Range_Understanding",
        "collection_prefix": "mem0g_long_range",
        "dataset_type": "long_range",
        "default_chunk_size": 24000,
        "default_overlap": 1000
    },
    "test_time_learning": {
        "folder": "Test_Time_Learning",
        "collection_prefix": "mem0g_ttl",
        "dataset_type": "ttl",
        "default_min_chars": 12000
    }
}


def sanitize_neo4j_db_name(name: str) -> str:
    """Neo4j DB names: lowercase letters/digits/dot/dash; no underscore."""
    candidate = (name or "neo4j").strip().lower().replace("_", "-")
    candidate = re.sub(r"[^a-z0-9.-]", "-", candidate)
    candidate = candidate.strip(".-")
    return candidate or "neo4j"


def ensure_neo4j_database(uri: str, username: str, password: str, db_name: str) -> str:
    """Ensure target Neo4j DB exists, fallback to neo4j on failure."""
    target_db = sanitize_neo4j_db_name(db_name)
    if GraphDatabase is None:
        logger.warning("neo4j driver not installed, fallback to default database 'neo4j'.")
        return "neo4j"

    try:
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            with driver.session(database="system") as session:
                existing_dbs = [record["name"] for record in session.run("SHOW DATABASES")]
                if target_db not in existing_dbs:
                    session.run(f"CREATE DATABASE `{target_db}`")
                    logger.info(f"Created Neo4j database: {target_db}")
                else:
                    logger.info(f"Neo4j database already exists: {target_db}")
        return target_db
    except Exception as e:
        logger.warning(f"Failed to ensure Neo4j database '{target_db}', fallback to 'neo4j'. Error: {e}")
        return "neo4j"

def load_data(dataset: str, instance_idx: int):
    config = DATASETS[dataset]
    
    if dataset == "accurate_retrieval":
        data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
        try:
            return load_benchmark_data(data_path, instance_idx)
        except Exception as e:
            raise RuntimeError(f"Error loading Parquet data for instance {instance_idx}: {e}")
            
    elif config["folder"]:
        # Preview samples usually have instance_N.json structure
        data_path = f"MemoryAgentBench/preview_samples/{config['folder']}/instance_{instance_idx}.json"
        path_obj = Path(data_path)
        if not path_obj.exists():
            # Try alternative path if needed, but current scripts use preview_samples
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
        chunk_size = args.chunk_size if args.chunk_size is not None else DATASETS[dataset]["default_chunk_size"]
        chunks = chunk_context(context, chunk_size=chunk_size)
        
    elif dataset == "conflict_resolution":
        min_chars = args.min_chars if args.min_chars is not None else DATASETS[dataset]["default_min_chars"]
        chunks = chunk_facts(context, min_chars=min_chars)
        
    elif dataset == "long_range_understanding":
        chunk_size = args.chunk_size if args.chunk_size is not None else DATASETS[dataset]["default_chunk_size"]
        overlap = args.overlap if args.overlap is not None else DATASETS[dataset]["default_overlap"]
        chunks = chunk_context(context, chunk_size=chunk_size, overlap=overlap)
        
    elif dataset == "test_time_learning":
        # Adaptive Strategy
        if "Dialogue 1:" in context[:500]:
            logger.info("Strategy: Regex Split (Dialogue mode)")
            chunks = chunk_dialogues(context)
        else:
            logger.info("Strategy: Accumulation (ShortText mode)")
            min_chars = args.min_chars if args.min_chars is not None else DATASETS[dataset]["default_min_chars"]
            chunks = chunk_accumulation(context, min_chars=min_chars)
            
    return chunks

def ingest_one_instance(dataset: str, instance_idx: int, args):
    logger.info(f"=== Processing [{dataset}] Instance {instance_idx} (Mem0G - with Graph) ===")
    
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

    # 3. Initialize Memory (Mem0G)
    config = DATASETS[dataset]
    collection_name = f"{config['collection_prefix']}_{instance_idx}"
    logger.info(f"Ingesting {len(chunks)} chunks into collection: {collection_name}")
    
    mem0_config = get_mem0_config(
        collection_name=collection_name, 
        include_graph=True
    )

    # Neo4j database selection:
    # - default: use collection_name (keeps graph/vector isolation aligned)
    # - override: --neo4j_db
    graph_conf = mem0_config.get("graph_store", {}).get("config", {})
    neo4j_uri = graph_conf.get("url")
    neo4j_user = graph_conf.get("username")
    neo4j_password = graph_conf.get("password")

    desired_db = args.neo4j_db if args.neo4j_db else collection_name
    desired_db = sanitize_neo4j_db_name(desired_db)
    selected_db = desired_db
    if neo4j_uri and neo4j_user and neo4j_password:
        selected_db = ensure_neo4j_database(neo4j_uri, neo4j_user, neo4j_password, desired_db)
    os.environ["NEO4J_DATABASE"] = selected_db
    logger.info(f"Using Qdrant collection: {collection_name}")
    logger.info(f"Using Neo4j database: {selected_db}")
    
    try:
        memory = Memory.from_config(mem0_config)
    except Exception as e:
        logger.error(f"Failed to initialize Mem0G: {e}")
        return

    # 4. Check Existing Data
    user_id = f"instance_{instance_idx}"
    existing_chunk_ids = set()
    if not args.force:
        try:
            # Retrieve all memories for this instance to see which chunks are already done
            existing = memory.get_all(user_id=user_id, limit=2000)
            mems = existing.get("results") if isinstance(existing, dict) else existing
            for m in mems:
                cid = m.get("metadata", {}).get("chunk_id")
                if cid is not None:
                    existing_chunk_ids.add(int(cid))
            
            if existing_chunk_ids:
                logger.info(f"Found {len(existing_chunk_ids)} chunks already ingested for Instance {instance_idx}. Resuming...")
        except Exception as e:
            logger.debug(f"Pre-flight check failed (safe to proceed): {e}")

    # 5. Optional: skip dedup (monkey-patch _search_graph_db → [] for fresh ingest)
    if getattr(args, 'skip_dedup', False) and hasattr(memory, 'graph') and memory.graph is not None:
        from mem0.memory.graph_memory import MemoryGraph
        MemoryGraph._search_graph_db = lambda self, *a, **kw: []
        logger.info("skip_dedup enabled: _search_graph_db patched to return []")

    # 6. Ingest Loop (serial or parallel workers)
    workers = getattr(args, 'workers', 1)
    logger.info(f"Starting ingestion of {len(chunks)} chunks into Mem0G (workers={workers})...")

    success_count = 0
    _lock = threading.Lock()

    def _add_chunk(i: int, chunk: str) -> bool:
        if i in existing_chunk_ids and not args.force:
            logger.info(f"Skipping chunk {i}/{len(chunks)}: Already exists.")
            return True
        metadata = {
            "chunk_id": i,
            "instance_idx": instance_idx,
            "source": "MemoryAgentBench",
            "ingest_type": "mem0g",
            "dataset_type": config["dataset_type"]
        }
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                memory.add(chunk, user_id=user_id, metadata=metadata, infer=False)
                return True
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RateLimitExceeded" in err_str
                if is_rate_limit and attempt < max_retries:
                    delay = 60 * (2 ** attempt)  # 60s, 120s, 240s
                    logger.warning(f"Chunk {i}: 429 rate limit, retry {attempt+1}/{max_retries} in {delay}s")
                    time.sleep(delay)
                    continue
                logger.error(f"Failed to add chunk {i}: {e}")
                return False
        return False

    if workers <= 1:
        for i, chunk in enumerate(chunks):
            ok = _add_chunk(i, chunk)
            if ok:
                success_count += 1
                logger.info(f"Progress: {success_count}/{len(chunks)} chunks added")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(_add_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(future_to_idx):
                ok = future.result()
                with _lock:
                    if ok:
                        success_count += 1
                    done = success_count
                if ok:
                    logger.info(f"Progress: {done}/{len(chunks)} chunks added")

    logger.info(f"Instance {instance_idx} complete: {success_count}/{len(chunks)} chunks ingested.")

def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data into Mem0G (Unified, with Graph)")
    
    # Dataset Selection
    parser.add_argument("--dataset", required=True, 
                        choices=["accurate_retrieval", "conflict_resolution", "long_range_understanding", "test_time_learning"],
                        help="Which dataset to ingest")
    
    # Common Args
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument('--force', action='store_true', help='Force re-ingestion even if data exists')
    
    # Specific Args (Optional)
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size (Accurate_Retrieval, Long_Range)")
    parser.add_argument("--overlap", type=int, default=None, help="Overlap (Long_Range)")
    parser.add_argument("--min_chars", type=int, default=None, help="Minimum chars per chunk (Conflict_Resolution, TTL)")
    parser.add_argument(
        "--neo4j_db",
        type=str,
        default=None,
        help="Neo4j database name override. Default follows collection name per instance.",
    )
    # --neo4j_instance 用于大规模并行入库：启动 3 个独立的 Docker Neo4j CE 容器，
    # 每个容器对应一个端口（7687 / 7688 / 7689），每对进程挂同一个容器，彼此完全隔离。
    # 这样可以消除多进程共享单个 Neo4j 实例时的写锁竞争，将 mem0g 并行效率从 44% 提升到 ~80%。
    # 端口映射：1 → bolt://localhost:7687  2 → bolt://localhost:7688  3 → bolt://localhost:7689
    parser.add_argument(
        "--neo4j_instance",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help=(
            "Docker Neo4j CE instance to connect to (1/2/3). "
            "Maps to bolt://localhost:7687/7688/7689. "
            "Use different instances for parallel ingest workers to avoid write-lock contention."
        ),
    )
    parser.add_argument(
        "--skip_dedup",
        action="store_true",
        help=(
            "Skip graph dedup step (_search_graph_db → []) for fresh ingest. "
            "Removes LLM#3 + N embedding calls per chunk, ~30-40%% speedup."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel chunk-ingest workers (default: 1). Each worker uses ~1 concurrent LLM call.",
    )

    args = parser.parse_args()

    # 根据 --neo4j_instance 设置 NEO4J_URL 环境变量，供 src/mem0_utils.py 读取。
    # 只在未被外部显式设置时覆盖，允许直接用 NEO4J_URL=... 启动进程绕过此逻辑。
    neo4j_port_map = {1: 7687, 2: 7688, 3: 7689}
    if not os.getenv("NEO4J_URL"):
        port = neo4j_port_map[args.neo4j_instance]
        os.environ["NEO4J_URL"] = f"bolt://localhost:{port}"
        logger.info(f"Neo4j instance {args.neo4j_instance} → bolt://localhost:{port}")

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target Dataset: {args.dataset}")
    logger.info(f"Target Instances: {indices}")

    for idx in indices:
        ingest_one_instance(args.dataset, idx, args)

if __name__ == "__main__":
    main()
