import argparse
import os
import sys
import re
import json
import time
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

# ============================================================
# Chunk 策略配置 — 直接在这里改，不需要命令行参数
# ============================================================
DATASETS = {
    "accurate_retrieval": {
        "folder": None,                    # 使用 parquet loader
        "collection_prefix": "mem0_acc_ret",
        "chunk_size": 8000,                 # 按字符数切割
    },
    "conflict_resolution": {
        "folder": "Conflict_Resolution",
        "collection_prefix": "mem0_conflict",
        "min_chars": 8000,                  # 按行累积，超过此长度才切割
    },
    "long_range_understanding": {
        "folder": "Long_Range_Understanding",
        "collection_prefix": "mem0_long_range",
        "chunk_size": 24000,                # 按字符数切割
        "overlap": 300,                    # chunk 间重叠字符数
    },
    "test_time_learning": {
        "folder": "Test_Time_Learning",
        "collection_prefix": "mem0_ttl",
        "min_chars": 12000,                  # 对话模式自动按 Dialogue N: 切割，忽略此参数
    },
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

def get_chunks(dataset: str, context: str) -> List[str]:
    cfg = DATASETS[dataset]
    chunks = []

    if dataset == "accurate_retrieval":
        chunks = chunk_context(context, chunk_size=cfg["chunk_size"])

    elif dataset == "conflict_resolution":
        chunks = chunk_facts(context, min_chars=cfg["min_chars"])

    elif dataset == "long_range_understanding":
        chunks = chunk_context(context, chunk_size=cfg["chunk_size"], overlap=cfg["overlap"])

    elif dataset == "test_time_learning":
        if "Dialogue 1:" in context[:500]:
            logger.info("Strategy: Regex Split (Dialogue mode)")
            chunks = chunk_dialogues(context)
        else:
            logger.info("Strategy: Accumulation (ShortText mode)")
            chunks = chunk_accumulation(context, min_chars=cfg["min_chars"])

    return chunks

# --- Checkpoint Helpers ---

def _ckpt_path(dataset: str, instance_idx: int) -> Path:
    p = Path("checkpoints/mem0_MAB")
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{dataset}_{instance_idx}.json"

def _load_checkpoint(dataset: str, instance_idx: int) -> int:
    """返回已成功 ingest 的 chunk 数量（即下次应从哪个 index 开始）。"""
    p = _ckpt_path(dataset, instance_idx)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f).get("ingested", 0)
        except Exception:
            return 0
    return 0

def _save_checkpoint(dataset: str, instance_idx: int, ingested: int, total: int):
    with open(_ckpt_path(dataset, instance_idx), "w") as f:
        json.dump({"dataset": dataset, "instance_idx": instance_idx,
                   "ingested": ingested, "total": total}, f)

def _clear_checkpoint(dataset: str, instance_idx: int):
    p = _ckpt_path(dataset, instance_idx)
    if p.exists():
        p.unlink()


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
    chunks = get_chunks(dataset, context)
    if not chunks:
        logger.warning(f"No chunks generated for instance {instance_idx}")
        return

    config = DATASETS[dataset]
    collection_name = f"{config['collection_prefix']}_{instance_idx}"
    user_id = "ingest_user"

    # 3. Checkpoint / Force Logic（必须在 Memory 初始化之前处理删除）
    if args.force:
        # 先删 Qdrant collection，再初始化 Memory（否则 from_config 创建 collection 后立刻被删）
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        try:
            qc = QdrantClient(host=qdrant_host, port=qdrant_port)
            if qc.collection_exists(collection_name):
                qc.delete_collection(collection_name)
                logger.info(f"Deleted existing Qdrant collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete Qdrant collection: {e}")
        _clear_checkpoint(dataset, instance_idx)
        start_idx = 0
    else:
        start_idx = _load_checkpoint(dataset, instance_idx)
        if start_idx >= len(chunks):
            logger.info(f"Instance {instance_idx} already fully ingested ({start_idx}/{len(chunks)}). Use --force to re-ingest.")
            return
        if start_idx > 0:
            logger.info(f"Resuming from chunk {start_idx}/{len(chunks)} (checkpoint found).")
        else:
            logger.info(f"Starting fresh ingestion of {len(chunks)} chunks into: {collection_name}")

    # 4. Initialize Memory（删除之后再初始化，确保 collection 是全新的）
    mem0_config = get_mem0_config(collection_name)
    try:
        memory = Memory.from_config(mem0_config)
    except Exception as e:
        logger.error(f"Failed to initialize Mem0: {e}")
        return

    # 5. Ingest Loop（从 start_idx 续跑）
    success_count = 0
    update_freq = 100 if len(chunks) > 500 else 10

    for i in range(start_idx, len(chunks)):
        chunk = chunks[i]
        metadata = {
            "chunk_id": i,
            "instance_idx": instance_idx,
            "dataset": dataset,
            "source": "MemoryAgentBench"
        }

        # --- Retry Logic ---
        # 背景问题：mem0 使用 infer=True 时，每个 chunk 会触发 LLM 提取原子事实，
        # 并并发写入 Qdrant。Qdrant 在构建 HNSW 索引时会进行 segment merge（合并段），
        # 期间向量存储短暂不可写，导致 mem0 内部的 add 请求收到 404 Not Found 错误。
        # 实测在约第 119 个 chunk 时出现这个问题，此后整个 collection 从 Qdrant 磁盘消失。
        #
        # 解决策略：捕获所有异常并重试最多 MAX_RETRIES 次，每次等待 RETRY_SLEEP 秒，
        # 给 Qdrant 足够时间完成 segment merge 并恢复可写状态。
        MAX_RETRIES = 3
        RETRY_SLEEP = 5  # 秒；Qdrant segment merge 通常在几秒内完成

        succeeded = False
        for attempt in range(MAX_RETRIES):
            try:
                memory.add(chunk, user_id=user_id, metadata=metadata, infer=True)
                succeeded = True
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Chunk {i} attempt {attempt + 1}/{MAX_RETRIES} failed "
                        f"(Qdrant 可能正在 segment merge): {e}. "
                        f"等待 {RETRY_SLEEP}s 后重试..."
                    )
                    time.sleep(RETRY_SLEEP)
                else:
                    logger.error(f"Chunk {i} 在 {MAX_RETRIES} 次尝试后仍失败，跳过: {e}")

        if succeeded:
            success_count += 1
            _save_checkpoint(dataset, instance_idx, start_idx + success_count, len(chunks))

        if (i + 1) % update_freq == 0:
            logger.info(f"Progress: {i + 1}/{len(chunks)} chunks added (success={success_count})")

    total_ingested = start_idx + success_count
    logger.info(f"Instance {instance_idx} done: {total_ingested}/{len(chunks)} chunks in Qdrant.")

def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data into Mem0 (Unified)")
    
    # Dataset Selection
    parser.add_argument("--dataset", required=True, 
                        choices=["accurate_retrieval", "conflict_resolution", "long_range_understanding", "test_time_learning"],
                        help="Which dataset to ingest")
    
    # Common Args
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument('--force', action='store_true', help='Force re-ingestion even if data exists')
    
    # Chunk 策略在脚本顶部 DATASETS 字典中配置，无需命令行参数

    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target Dataset: {args.dataset}")
    logger.info(f"Target Instances: {indices}")

    for idx in indices:
        ingest_one_instance(args.dataset, idx, args)

if __name__ == "__main__":
    main()
