import os
import json
import argparse
from pathlib import Path
from typing import List
import sys

# 确保能找到 src 目录
sys.path.append(str(Path(__file__).parent))

from src.logger import get_logger
from src.simple_memory import SimpleRAGMemory
from src.benchmark_utils import chunk_context

logger = get_logger()

def ingest_q2_md(base_dir: str, chunk_size: int = 1500, overlap: int = 150):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[错误] Directory not found: {base_path}")
        logger.error(f"Directory not found: {base_path}")
        return

    table_name = "bench_q2_md"
    print(f"Initializing Memory System for table: {table_name}...")
    logger.info(f"Initializing Memory System for table: {table_name}")
    
    memory = SimpleRAGMemory(table_name=table_name)
    print("Resetting table...")
    logger.info("Resetting table...")
    memory.reset()

    # 获取所有的文档文件夹
    doc_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    print(f"Found {len(doc_dirs)} document directories. Starting ingestion...")
    logger.info(f"Found {len(doc_dirs)} document directories.")

    total_chunks_ingested = 0

    for i, doc_dir in enumerate(doc_dirs):
        block_id = doc_dir.name
        md_file = doc_dir / f"{block_id}.md"
        meta_file = doc_dir / "metadata.json"

        if not md_file.exists() or not meta_file.exists():
            logger.warning(f"Skipping {block_id}: Missing .md or metadata.json")
            continue

        # 读取 Metadata
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for {block_id}: {e}")
            continue

        # 读取 Markdown 内容
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except Exception as e:
            logger.error(f"Error reading markdown for {block_id}: {e}")
            continue

        title = metadata.get("title", "Unknown Title")
        ai_summary = metadata.get("ai_summary", "")
        ai_tags = ", ".join(metadata.get("ai_tags", []))

        # 核心策略：全局上下文 Prefix
        context_prefix = f"[Title]: {title}\n[Tags]: {ai_tags}\n[Summary]: {ai_summary}\n\n[Content Segment]:\n"

        # 如果内容太长，进行切片
        raw_chunks = chunk_context(md_content, chunk_size=chunk_size, overlap=overlap)
        
        for chunk_idx, raw_chunk in enumerate(raw_chunks):
            # 将 Prefix 与具体的 Chunk 结合
            enriched_chunk = context_prefix + raw_chunk
            
            # 存储时保留相关的 block_id 和 chunk 索引
            db_metadata = {
                "block_id": block_id,
                "title": title,
                "chunk_idx": chunk_idx,
                "total_chunks": len(raw_chunks)
            }
            
            memory.add_memory(enriched_chunk, metadata=db_metadata)
            total_chunks_ingested += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(doc_dirs):
            print(f"Processed {i + 1}/{len(doc_dirs)} documents... (Total Chunks: {total_chunks_ingested})", end="\r", flush=True)

    print(f"\nIngestion Complete! Total documents processed: {len(doc_dirs)}")
    print(f"Total chunks stored in DB: {total_chunks_ingested}")

def main():
    parser = argparse.ArgumentParser(description="Ingest Q2 Human Readable MD into SimpleMemory")
    parser.add_argument("--data_dir", type=str, default="q2DataBase/data/human_readable_md", help="Path to MD directories")
    parser.add_argument("--chunk_size", type=int, default=1500, help="Chunk size for long MD files")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap between chunks")
    args = parser.parse_args()

    ingest_q2_md(args.data_dir, args.chunk_size, args.overlap)

if __name__ == "__main__":
    main()
