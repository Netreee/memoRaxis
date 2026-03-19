"""A-MEM ingest — 统一 4 任务 ingest 脚本 (带 per-chunk checkpoint)

用法:
    python scripts/amem_MAB/ingest.py --dataset accurate_retrieval --instance_idx 0-21
    python scripts/amem_MAB/ingest.py --dataset conflict_resolution --instance_idx 0-7
    python scripts/amem_MAB/ingest.py --dataset long_range_understanding --instance_idx 0-39
    python scripts/amem_MAB/ingest.py --dataset test_time_learning --instance_idx 0-5

产物:
    out/amem/chroma/{task_short}_{idx}/     — ChromaDB 持久化目录
    out/amem/chroma/{task_short}_{idx}/memories.pkl — 内存字典快照
    out/amem/ingest_progress/{task_short}_{idx}.json — 进度检查点
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
from src.amem_memory import AMemMemory

logger = get_logger()

DATASETS = {
    "accurate_retrieval": {
        "folder": None,
        "task_short": "acc_ret",
        "chunk_size": 8000,
    },
    "conflict_resolution": {
        "folder": "Conflict_Resolution",
        "task_short": "conflict",
        "min_chars": 8000,
    },
    "long_range_understanding": {
        "folder": "Long_Range_Understanding",
        "task_short": "long_range",
        "chunk_size": 24000,
        "overlap": 300,
    },
    "test_time_learning": {
        "folder": "Test_Time_Learning",
        "task_short": "ttl",
        "min_chars": 12000,
    },
}

CHROMA_BASE = Path("out/amem/chroma")
PROGRESS_DIR = Path("out/amem/ingest_progress")


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
            logger.info("Strategy: Regex Split (Dialogue mode)")
            return chunk_dialogues(context)
        else:
            logger.info("Strategy: Accumulation (ShortText mode)")
            return chunk_facts(context, min_chars=cfg["min_chars"])
    return []


def _progress_path(task_short: str, idx: int) -> Path:
    return PROGRESS_DIR / f"{task_short}_{idx}.json"


def _load_progress(task_short: str, idx: int) -> dict:
    p = _progress_path(task_short, idx)
    if p.exists():
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return {"chunks_done": 0}


def _save_progress(task_short: str, idx: int, progress: dict):
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _progress_path(task_short, idx).with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(progress, f, indent=2)
    tmp.rename(_progress_path(task_short, idx))


def ingest_one_instance(dataset: str, instance_idx: int, args):
    cfg = DATASETS[dataset]
    task_short = cfg["task_short"]
    chroma_dir = CHROMA_BASE / f"{task_short}_{instance_idx}"

    progress = _load_progress(task_short, instance_idx)
    chunks_done = progress.get("chunks_done", 0)

    logger.info(f"=== Processing [{dataset}] Instance {instance_idx} (A-MEM) ===")

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
    total_chunks = len(chunks)
    logger.info(f"Chunks: {total_chunks} (total {total_chars} chars)")

    if chunks_done >= total_chunks and not args.force:
        logger.info(f"Already complete ({chunks_done}/{total_chunks}). Use --force to re-ingest.")
        return

    if chunks_done > 0 and not args.force:
        logger.info(f"Resuming from chunk {chunks_done + 1}/{total_chunks}")

    memory = AMemMemory(
        chroma_dir=str(chroma_dir),
        enable_evolution=args.evolution,
        evo_threshold=args.evo_threshold,
    )

    t0 = time.time()

    for i in range(chunks_done, total_chunks):
        chunk = chunks[i]
        chunk_t0 = time.time()

        try:
            memory.add_memory(chunk, metadata={
                "chunk_id": i,
                "instance_idx": instance_idx,
                "dataset": dataset,
            })
            chunk_elapsed = time.time() - chunk_t0
            logger.info(
                f"  Chunk {i + 1}/{total_chunks} done | "
                f"{chunk_elapsed:.1f}s | memories={memory.memory_count}"
            )
        except Exception as e:
            logger.error(f"  Chunk {i + 1}/{total_chunks} FAILED: {e}")
            memory.save()
            _save_progress(task_short, instance_idx, {
                "chunks_done": i,
                "total_chunks": total_chunks,
                "error": str(e),
            })
            raise

        if (i + 1) % max(1, args.save_every) == 0 or i == total_chunks - 1:
            memory.save()
            _save_progress(task_short, instance_idx, {
                "chunks_done": i + 1,
                "total_chunks": total_chunks,
                "elapsed_s": round(time.time() - t0, 1),
                "llm_calls": memory.get_llm_stats()["llm_calls"],
            })

    elapsed = time.time() - t0
    stats = memory.get_llm_stats()

    memory.save()
    _save_progress(task_short, instance_idx, {
        "chunks_done": total_chunks,
        "total_chunks": total_chunks,
        "elapsed_s": round(elapsed, 1),
        "llm_calls": stats["llm_calls"],
        "status": "complete",
    })

    logger.info(
        f"[Ingest Summary] dataset={dataset} instance={instance_idx} "
        f"chunks={total_chunks} chars={total_chars} "
        f"elapsed={elapsed:.1f}s llm_calls={stats['llm_calls']} "
        f"memories={memory.memory_count}"
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data into A-MEM")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS.keys()),
                        help="Which dataset to ingest")
    parser.add_argument("--instance_idx", type=str, default="0",
                        help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument("--force", action="store_true",
                        help="Force re-ingest even if progress exists")
    parser.add_argument("--evolution", action="store_true", default=True,
                        help="Enable memory evolution (default: True)")
    parser.add_argument("--no-evolution", dest="evolution", action="store_false",
                        help="Disable memory evolution (raw storage only)")
    parser.add_argument("--evo_threshold", type=int, default=100,
                        help="Evolution consolidation threshold (default: 100)")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N chunks (default: 5)")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Target Instances: {indices}")
    logger.info(f"Evolution: {args.evolution} (threshold={args.evo_threshold})")

    for idx in indices:
        ingest_one_instance(args.dataset, idx, args)


if __name__ == "__main__":
    main()
