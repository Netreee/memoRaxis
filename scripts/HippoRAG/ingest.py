"""HippoRAG ingest — 统一 4 任务 ingest 脚本

用法:
    python scripts/HippoRAG/ingest.py --dataset accurate_retrieval --instance_idx 0-21
    python scripts/HippoRAG/ingest.py --dataset conflict_resolution --instance_idx 0-7
    python scripts/HippoRAG/ingest.py --dataset test_time_learning --instance_idx 1-5
    python scripts/HippoRAG/ingest.py --dataset long_range_understanding --instance_idx 0-39

产物: out/hipporag/indices/hipporag_{prefix}_{idx}/
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices
from src.hipporag_memory import HippoRAGMemory

logger = get_logger()

DATASETS = {
    "accurate_retrieval": {
        "folder": None,
        "index_prefix": "hipporag_acc_ret",
        "chunk_size": 8000,
    },
    "conflict_resolution": {
        "folder": "Conflict_Resolution",
        "index_prefix": "hipporag_conflict",
        "min_chars": 8000,
    },
    "long_range_understanding": {
        "folder": "Long_Range_Understanding",
        "index_prefix": "hipporag_long_range",
        "chunk_size": 24000,
        "overlap": 300,
    },
    "test_time_learning": {
        "folder": "Test_Time_Learning",
        "index_prefix": "hipporag_ttl",
        "min_chars": 12000,
    },
}

INDEX_DIR = Path("out/hipporag/indices")


# --- Chunking (same strategies as RAPTOR for consistency) ---

def chunk_facts(context: str, min_chars: int = 8000) -> List[str]:
    lines = [line.strip() for line in context.split('\n') if line.strip()]
    chunks, buf, buf_len = [], [], 0
    for line in lines:
        buf.append(line)
        buf_len += len(line)
        if buf_len > min_chars:
            chunks.append("\n".join(buf))
            buf, buf_len = [], 0
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def chunk_dialogues(context: str) -> List[str]:
    parts = re.split(r'\n(Dialogue \d+:)', '\n' + context)
    chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        full_text = f"{header}\n{body}"
        if len(full_text) > 10:
            chunks.append(full_text)
    return chunks


# --- Data Loading ---

def load_data(dataset: str, instance_idx: int):
    config = DATASETS[dataset]
    if dataset == "accurate_retrieval":
        return load_benchmark_data(
            "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet",
            instance_idx,
        )
    folder = config["folder"]
    data_path = Path(f"MemoryAgentBench/preview_samples/{folder}/instance_{instance_idx}.json")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_chunks(dataset: str, context: str) -> List[str]:
    cfg = DATASETS[dataset]
    if dataset == "accurate_retrieval":
        return chunk_context(context, chunk_size=cfg["chunk_size"])
    elif dataset == "conflict_resolution":
        return chunk_facts(context, min_chars=cfg["min_chars"])
    elif dataset == "long_range_understanding":
        return chunk_context(context, chunk_size=cfg["chunk_size"], overlap=cfg["overlap"])
    elif dataset == "test_time_learning":
        if "Dialogue 1:" in context[:500]:
            return chunk_dialogues(context)
        else:
            return chunk_facts(context, min_chars=cfg["min_chars"])
    return []


# --- Main Logic ---

def _is_index_complete(index_path: Path) -> bool:
    """Check if HippoRAG index already has a built graph."""
    # HippoRAG creates a working_dir under save_dir with graph.pickle when index is done
    for f in index_path.rglob("graph.pickle"):
        return True
    return False


def ingest_one_instance(dataset: str, instance_idx: int, args):
    cfg = DATASETS[dataset]
    index_path = INDEX_DIR / f"{cfg['index_prefix']}_{instance_idx}"

    if not args.force and _is_index_complete(index_path):
        logger.info(f"[{dataset}] Instance {instance_idx}: index already exists at {index_path}. Use --force to rebuild.")
        return

    logger.info(f"=== Processing [{dataset}] Instance {instance_idx} (HippoRAG) ===")

    try:
        data = load_data(dataset, instance_idx)
    except Exception as e:
        logger.error(str(e))
        return

    chunks = get_chunks(dataset, data["context"])
    if not chunks:
        logger.warning(f"No chunks generated for instance {instance_idx}")
        return

    total_chars = sum(len(c) for c in chunks)
    logger.info(f"Chunks: {len(chunks)} (total {total_chars:,} chars)")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    memory = HippoRAGMemory(
        index_dir=str(index_path),
        openie_mode="online",
        force_rebuild=args.force,
    )

    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx, "dataset": dataset})

    logger.info(f"Building HippoRAG index (NER + triple extraction + graph + embeddings)...")

    t0 = time.time()
    memory.build_index()
    elapsed = time.time() - t0

    logger.info(
        f"[Ingest Summary] dataset={dataset} instance={instance_idx} "
        f"chunks={len(chunks)} chars={total_chars:,} "
        f"elapsed={elapsed:.1f}s ({elapsed/60:.1f}min) "
        f"per_chunk={elapsed/len(chunks):.1f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data into HippoRAG indices")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS.keys()),
                        help="Which dataset to ingest")
    parser.add_argument("--instance_idx", type=str, default="0",
                        help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if index exists")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Target Instances: {indices}")

    for idx in indices:
        ingest_one_instance(args.dataset, idx, args)


if __name__ == "__main__":
    main()
