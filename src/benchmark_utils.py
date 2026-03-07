import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from src.logger import get_logger

logger = get_logger()

def parse_instance_indices(idx_str: str) -> List[int]:
    """
    解析索引范围字符串。
    支持格式: "0", "0-5", "1,3,5", "0-2,5"
    """
    indices = set()
    parts = idx_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                indices.update(range(start, end + 1))
            except ValueError:
                logger.warning(f"Invalid range format: {part}")
        else:
            try:
                indices.add(int(part))
            except ValueError:
                logger.warning(f"Invalid index format: {part}")
    return sorted(list(indices))

def load_benchmark_data(file_path: str, instance_idx: int) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        df = pd.read_parquet(str(path))
        if instance_idx >= len(df):
            raise IndexError(f"Instance index {instance_idx} out of range (total {len(df)})")
        data = df.iloc[instance_idx].to_dict()
        logger.info(f"Loaded instance {instance_idx} from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def chunk_context(context: str, chunk_size: int = 850, overlap: int = 50) -> List[str]:
    """
    将长 Context 切分为文档片段。
    策略：
    1. 优先尝试 "Document N:" 标记切分，然后按 chunk_size 合并碎片。
       - chunk_size 小 → 每个 Document 单独一个 chunk
       - chunk_size 大 → 多个 Document 合并进一个 chunk
    2. 如果没有标记，使用固定长度滑动窗口切分（overlap 仅此模式生效）。
    """
    regex_chunks = re.split(r"Document \d+:\n", context)
    valid_regex_chunks = [c.strip() for c in regex_chunks if len(c.strip()) > 10]

    if len(valid_regex_chunks) > 1:
        # 按 chunk_size 合并：累积 Document 片段直到超过目标大小
        merged = []
        buf = []
        buf_len = 0
        for doc in valid_regex_chunks:
            if buf and buf_len + len(doc) > chunk_size:
                merged.append("\n\n".join(buf))
                buf = []
                buf_len = 0
            buf.append(doc)
            buf_len += len(doc)
        if buf:
            merged.append("\n\n".join(buf))
        logger.info(
            f"Chunking Strategy: Regex split ('Document N:') + merge. "
            f"Result: {len(merged)} chunks (chunk_size={chunk_size}, "
            f"raw_docs={len(valid_regex_chunks)})."
        )
        return merged

    logger.info("Chunking Strategy: Fallback to Fixed-size Sliding Window.")
    chunks = []
    start = 0
    text_len = len(context)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(context[start:end])
        if end == text_len:
            break
        start += chunk_size - overlap

    logger.info(f"Result: {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return chunks
