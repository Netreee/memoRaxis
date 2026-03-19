"""
A/B Test: BM25 (retrieve_with_conversation) vs Embedding (search) retrieval
对比两种检索方式在 AR inst 0 前 N 题上的 R1 表现。

Usage:
    python scripts/MIRIX/ab_test_retrieval.py --n_questions 10
    python scripts/MIRIX/ab_test_retrieval.py --n_questions 10 --instance_idx 1
"""

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from MIRIX.remote_client import MirixClient
except ImportError:
    from mirix.client.remote_client import MirixClient

from src.logger import get_logger
from src.config import get_config
from src.memory_interface import BaseMemorySystem, Evidence
from src.mirix import Mirix
from src.mirix_utils import get_mirix_config, get_mirix_connection_info
from src.benchmark_utils import load_benchmark_data
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor
from MemoryAgentBench.utils.templates import get_template

logger = get_logger()


# ── Embedding 检索变体 ────────────────────────────────────────────────────────

class MirixEmbedding(BaseMemorySystem):
    """与 Mirix 相同接口，但 retrieve 走 client.search(method='embedding')"""

    def __init__(self, client: MirixClient, user_id: str = "default_user",
                 similarity_threshold: Optional[float] = None):
        self.client = client
        self.user_id = user_id
        self.similarity_threshold = similarity_threshold

    def add_memory(self, data: str, metadata: Dict[str, Any] = None) -> None:
        raise NotImplementedError("ab_test only tests retrieval")

    def retrieve(self, query: str, top_k: int = 5,
                 user_id: Optional[str] = None) -> List[Evidence]:
        target_user_id = user_id if user_id is not None else self.user_id

        results = self.client.search(
            user_id=target_user_id,
            query=query,
            memory_type="all",
            search_field="null",
            search_method="embedding",
            limit=top_k,
            similarity_threshold=self.similarity_threshold,
        )

        evidences = []
        for item in results.get("results", []):
            content = (item.get("summary") or item.get("details") or
                       item.get("content") or item.get("text") or "")
            if item.get("details") and item.get("summary"):
                content = f"{item['summary']}\n{item['details']}"
            if content:
                meta = {k: v for k, v in item.items()
                        if k not in ("summary", "details", "content", "text")}
                meta["source"] = "mirix_embedding"
                evidences.append(Evidence(content=content, metadata=meta))
        return evidences

    def reset(self) -> None:
        self.client.clear_memory(user_id=self.user_id)


# ── 机械评分 ──────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())

def keyword_hit(pred: str, refs) -> bool:
    pred_n = normalize(str(pred))
    if not hasattr(refs, '__iter__') or isinstance(refs, str):
        refs = [refs]
    for ref in refs:
        if normalize(str(ref)) in pred_n:
            return True
    return False


# ── 主实验 ────────────────────────────────────────────────────────────────────

def run_experiment(instance_idx: int, n_questions: int, top_k: int,
                   sim_threshold: Optional[float]):

    # 加载数据
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
    data = load_benchmark_data(data_path, instance_idx)
    questions = list(data["questions"])[:n_questions]
    answers   = list(data["answers"])[:n_questions]

    # 初始化 MIRIX client（共用同一个 server）
    conn = get_mirix_connection_info()
    api_key  = conn.get("api_key")
    base_url = conn.get("base_url")

    client = MirixClient(api_key=api_key, base_url=base_url)
    try:
        client.initialize_meta_agent(config=get_mirix_config())
    except Exception as e:
        logger.warning(f"Meta agent init: {e}")

    user_id = f"mirix_acc_ret_{instance_idx}"

    bm25_mem = Mirix(client, user_id=user_id)
    emb_mem  = MirixEmbedding(client, user_id=user_id,
                               similarity_threshold=sim_threshold)

    # 初始化 LLM（Fornax）
    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm.get("api_key", ""),
        base_url=conf.llm.get("base_url", ""),
        model=conf.llm.get("model", ""),
        provider=conf.llm.get("provider", "openai"),
        fornax_ak=conf.llm.get("fornax_ak", ""),
        fornax_sk=conf.llm.get("fornax_sk", ""),
        fornax_prompt_key=conf.llm.get("fornax_prompt_key", ""),
    )

    template = get_template("ruler_qa", "query", "Agentic_memory")

    bm25_adaptor = SingleTurnAdaptor(llm, bm25_mem)
    emb_adaptor  = SingleTurnAdaptor(llm, emb_mem)

    print(f"\n{'='*70}")
    print(f"  A/B Test: BM25 vs Embedding | AR inst={instance_idx} | "
          f"N={n_questions} | top_k={top_k}")
    if sim_threshold:
        print(f"  Embedding similarity_threshold={sim_threshold}")
    print(f"{'='*70}")
    print(f"  {'#':<3} | {'BM25':^5} | {'Emb':^5} | {'GT answer':<30} | BM25 answer / Emb answer")
    print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*5}-+-{'-'*30}-+--------")

    bm25_hits = 0
    emb_hits  = 0

    rows = []
    for i, (q, refs) in enumerate(zip(questions, answers)):
        formatted_q = template.format(question=q) if template else q

        # BM25
        try:
            t0 = time.time()
            r_bm25 = bm25_adaptor.run(formatted_q)
            bm25_lat = round(time.time() - t0, 1)
            bm25_ans = r_bm25.answer
        except Exception as e:
            bm25_ans = f"[ERR: {e}]"
            bm25_lat = 0

        # Embedding
        try:
            t0 = time.time()
            r_emb = emb_adaptor.run(formatted_q)
            emb_lat = round(time.time() - t0, 1)
            emb_ans = r_emb.answer
        except Exception as e:
            emb_ans = f"[ERR: {e}]"
            emb_lat = 0

        bm25_ok = keyword_hit(bm25_ans, refs if isinstance(refs, list) else [refs])
        emb_ok  = keyword_hit(emb_ans,  refs if isinstance(refs, list) else [refs])
        if bm25_ok: bm25_hits += 1
        if emb_ok:  emb_hits  += 1

        gt_short = str(refs[0] if isinstance(refs, list) else refs)[:28]
        mark_b = "✓" if bm25_ok else "✗"
        mark_e = "✓" if emb_ok  else "✗"

        print(f"  {i+1:<3} | {mark_b:^5} | {mark_e:^5} | {gt_short:<30} |")
        print(f"       BM25({bm25_lat}s): {bm25_ans[:80]}")
        print(f"       Emb ({emb_lat}s):  {emb_ans[:80]}")
        print()

        rows.append({
            "question": q,
            "gt": refs,
            "bm25_answer": bm25_ans,
            "bm25_hit": bm25_ok,
            "emb_answer": emb_ans,
            "emb_hit": emb_ok,
        })

    total = len(questions)
    print(f"{'='*70}")
    print(f"  Final: BM25 {bm25_hits}/{total} = {bm25_hits/total:.1%}"
          f"  |  Embedding {emb_hits}/{total} = {emb_hits/total:.1%}")
    delta = emb_hits - bm25_hits
    print(f"  Delta: Embedding {'↑' if delta>0 else '↓' if delta<0 else '='}{abs(delta)} questions")
    print(f"{'='*70}\n")

    # 保存结果
    out_path = Path("out/eval/ab_retrieval_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "instance_idx": instance_idx,
            "n_questions": n_questions,
            "top_k": top_k,
            "similarity_threshold": sim_threshold,
            "bm25_acc": bm25_hits / total,
            "emb_acc":  emb_hits  / total,
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=10,
                        help="Number of questions to test (default: 10)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Memories to retrieve per query (default: 10, same as infer)")
    parser.add_argument("--sim_threshold", type=float, default=None,
                        help="Embedding similarity_threshold (0.0-2.0, None=no filter)")
    args = parser.parse_args()

    run_experiment(
        instance_idx=args.instance_idx,
        n_questions=args.n_questions,
        top_k=args.top_k,
        sim_threshold=args.sim_threshold,
    )
