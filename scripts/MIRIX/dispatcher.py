#!/usr/bin/env python3
"""
MIRIX Ingest Dispatcher — 非侵入式调度层

职责：
  1. 按 dataset/instance 顺序调用 ingest.py 的函数投喂 chunks
  2. 通过 trace API 监控队列深度，节流投喂速率
  3. 每个 instance 完成后检测失败 chunks 并重投
  4. checkpoint 断点续跑
  5. 每个 dataset 独立日志

用法：
  python scripts/MIRIX/dispatcher.py                          # 跑全部
  python scripts/MIRIX/dispatcher.py --dataset accurate_retrieval  # 单 dataset
  python scripts/MIRIX/dispatcher.py --resume                 # 从 checkpoint 续跑

架构：
  dispatcher.py  →  ingest.py (投喂入队)
                 →  MIRIX trace API (监控 + 失败检测)
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import requests

from scripts.MIRIX.ingest import (
    DATASETS,
    load_data,
    get_chunks,
    chunk_facts,
    chunk_dialogues,
    chunk_accumulation,
)
from src.mirix_utils import get_mirix_connection_info, get_mirix_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QUEUE_DEPTH_LIMIT = 24          # 当 pending traces > 此值时暂停投喂 (4x workers)
QUEUE_POLL_INTERVAL = 10        # 检查队列深度的间隔 (秒)
DRAIN_POLL_INTERVAL = 15        # 等待队列排空的轮询间隔 (秒)
DRAIN_TIMEOUT = 3600            # 等待队列排空的超时 (秒, 60min — 大 instance 需要更久)
RETRY_LIMIT = 2                 # 失败 chunk 最大重投次数
RETRY_COOLDOWN = 30             # 重投前冷却 (秒)
MAX_PARALLEL_INSTANCES = 4      # 并行 instance 数 (4 feeder × 6 MIRIX workers)
STAGGER_DELAY = 15              # 线程启动交错延迟 (秒, 确保 meta init 不重叠)
CHECKPOINT_DIR = Path("checkpoints/mirix_ingest")
LOG_DIR = Path("logs")

# 全量执行计划: dataset → instance indices
# Round 4: 清理后重跑
# AR inst 5,6,7-15 数据充分 → 保留; inst 1,3,4,16-21 → clear 重跑
# LR 0-39 全部重跑
# MIRIX workers 已从 12 降到 4 以避免 TPM 超限
PLAN = {
    "accurate_retrieval":       [1, 3, 4, 16, 17, 18, 19, 20, 21],
    "long_range_understanding": list(range(40)),
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(dataset: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"mirix_dispatch_{dataset}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(LOG_DIR / f"mirix_ingest_{dataset}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# MIRIX Trace API helpers
# ---------------------------------------------------------------------------
class TraceMonitor:
    """通过 MIRIX REST API 监控队列 trace 状态"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        })

    def get_traces(self, user_id: str = None, status: str = None, limit: int = 200) -> list:
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        if status:
            params["status"] = status
        try:
            resp = self.session.get(f"{self.base_url}/memory/queue-traces", params=params)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("traces", [])
        except Exception:
            return []

    def count_pending(self, user_id: str = None) -> int:
        """未处理的 traces (queued + processing)"""
        queued = self.get_traces(user_id=user_id, status="queued")
        processing = self.get_traces(user_id=user_id, status="processing")
        return len(queued) + len(processing)

    def count_settled(self, user_id: str) -> tuple:
        """返回 (completed, failed) 数量"""
        completed = len(self.get_traces(user_id=user_id, status="completed"))
        failed = len(self.get_traces(user_id=user_id, status="failed"))
        return completed, failed

    def get_failed(self, user_id: str) -> list:
        return self.get_traces(user_id=user_id, status="failed")

    def get_completed_count(self, user_id: str) -> int:
        return len(self.get_traces(user_id=user_id, status="completed"))

    @staticmethod
    def cleanup_stale_traces_db(user_id: str = None):
        """删除 PostgreSQL 中该 user 的全部 traces (容器重启后内存队列丢失, 旧 traces 会干扰)"""
        import subprocess
        where = "WHERE 1=1"
        if user_id:
            where += f" AND user_id = '{user_id}'"
        cmd = f"DELETE FROM memory_queue_traces {where};"
        result = subprocess.run(
            ["docker", "exec", "mirix_pgvector", "psql", "-U", "mirix", "-d", "mirix", "-t", "-c", cmd],
            capture_output=True, text=True, timeout=10,
        )
        count = result.stdout.strip().split()[-1] if result.returncode == 0 else "error"
        return count


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def checkpoint_path(dataset: str, instance_idx: int) -> Path:
    return CHECKPOINT_DIR / f"{dataset}_{instance_idx}.json"


def load_checkpoint(dataset: str, instance_idx: int) -> dict:
    p = checkpoint_path(dataset, instance_idx)
    if p.exists():
        return json.loads(p.read_text())
    return {"queued": 0, "confirmed": 0, "status": "pending"}


def save_checkpoint(dataset: str, instance_idx: int, data: dict):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path(dataset, instance_idx).write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Core: 投喂单个 instance (带节流)
# ---------------------------------------------------------------------------
def feed_instance(
    dataset: str,
    instance_idx: int,
    base_url: str,
    api_key: str,
    monitor: TraceMonitor,
    logger: logging.Logger,
    resume: bool = False,
    force: bool = False,
):
    user_prefix = DATASETS[dataset]["user_prefix"]
    user_id = f"{user_prefix}_{instance_idx}"

    # 清理该 user 的陈旧 traces
    TraceMonitor.cleanup_stale_traces_db(user_id)

    # 加载 checkpoint
    ckpt = load_checkpoint(dataset, instance_idx)
    if ckpt.get("status") == "done" and not force:
        logger.info(f"[{dataset}][inst {instance_idx}] 已完成，跳过")
        return True

    # force 模式: 清理旧 checkpoint, 从头开始
    if force:
        p = checkpoint_path(dataset, instance_idx)
        if p.exists():
            p.unlink()
        ckpt = {"queued": 0, "confirmed": 0, "status": "pending"}

    # 加载数据 & 分 chunk
    try:
        data = load_data(dataset, instance_idx)
    except Exception as e:
        logger.error(f"[{dataset}][inst {instance_idx}] 加载数据失败: {e}")
        return False

    context = data["context"]

    # 复用 ingest.py 的 chunk 逻辑 (通过构造 mock args)
    class _Args:
        chunk_size = None
        overlap = None
        min_chars = None
    chunks = get_chunks(dataset, context, _Args())

    if not chunks:
        logger.warning(f"[{dataset}][inst {instance_idx}] 无 chunks")
        save_checkpoint(dataset, instance_idx, {"queued": 0, "confirmed": 0, "status": "done"})
        return True

    start_from = ckpt.get("queued", 0) if resume else 0
    logger.info(
        f"[{dataset}][inst {instance_idx}] {len(chunks)} chunks, "
        f"从 #{start_from} 开始, user_id={user_id}"
    )

    # 初始化 MIRIX client (直接用 HTTP，避免改 sdk)
    session = requests.Session()
    session.headers.update({
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    })

    # 初始化 meta agent (带重试)
    meta_agent_id = None
    mirix_config = get_mirix_config()
    for attempt in range(3):
        try:
            init_resp = session.post(
                f"{base_url}/agents/meta/initialize",
                json={"config": mirix_config, "update_agents": False},
                timeout=180,
            )
            init_resp.raise_for_status()
            meta_agent_id = init_resp.json().get("id")
            break
        except Exception as e:
            logger.warning(f"[{dataset}][inst {instance_idx}] meta init attempt {attempt+1}/3 失败: {e}")
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
    if not meta_agent_id:
        logger.error(f"[{dataset}][inst {instance_idx}] meta agent 初始化 3 次均失败, 放弃")
        return False

    # 确保 user 存在
    session.post(f"{base_url}/users/create_or_get", json={"user_id": user_id})

    # 投喂循环
    queued_count = start_from
    feed_start = time.time()

    for i in range(start_from, len(chunks)):
        chunk = chunks[i]

        # 节流: 直接看 pending (queued+processing) 数量
        throttle_logged = False
        while True:
            pending = monitor.count_pending(user_id=user_id)
            if pending < QUEUE_DEPTH_LIMIT:
                break
            if not throttle_logged:
                logger.info(f"  [inst {instance_idx}] 节流: pending={pending} >= {QUEUE_DEPTH_LIMIT}, 等待排空...")
                throttle_logged = True
            time.sleep(QUEUE_POLL_INTERVAL)

        # 投喂
        filter_tags = {
            "chunk_id": i,
            "instance_idx": instance_idx,
            "dataset": dataset,
            "source": "MemoryAgentBench",
        }

        messages = [
            {"role": "user", "content": [{"type": "text", "text": chunk}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Acknowledged."}]},
        ]

        payload = {
            "user_id": user_id,
            "meta_agent_id": meta_agent_id,
            "messages": messages,
            "chaining": False,
            "filter_tags": filter_tags,
            "use_cache": True,
        }

        try:
            resp = session.post(f"{base_url}/memory/add", json=payload, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"  chunk {i} 入队失败: {e}")
            continue

        queued_count += 1

        if queued_count % 20 == 0 or i == len(chunks) - 1:
            elapsed = round(time.time() - feed_start, 1)
            rate = queued_count / max(elapsed, 1) * 60
            logger.info(
                f"  进度 {queued_count}/{len(chunks)} | "
                f"elapsed={elapsed}s | rate={rate:.0f} chunks/min"
            )
            save_checkpoint(dataset, instance_idx, {
                "queued": queued_count,
                "confirmed": 0,
                "status": "feeding",
            })

    logger.info(
        f"[{dataset}][inst {instance_idx}] 投喂完成: "
        f"{queued_count}/{len(chunks)} chunks 已入队"
    )

    # 等待全部处理完成 (用 completed >= expected 而非 pending == 0, 避免旧 trace 干扰)
    expected = queued_count
    logger.info(f"[{dataset}][inst {instance_idx}] 等待处理完成 ({expected} chunks)...")
    drain_start = time.time()
    while time.time() - drain_start < DRAIN_TIMEOUT:
        completed_total = monitor.get_completed_count(user_id)
        pending = monitor.count_pending(user_id=user_id)
        if completed_total >= expected or pending == 0:
            break
        elapsed = int(time.time() - drain_start)
        rate = completed_total / max(time.time() - feed_start, 1) * 60
        logger.info(
            f"  completed={completed_total}/{expected} | pending={pending} | "
            f"已等 {elapsed}s | rate={rate:.1f}/min"
        )
        time.sleep(DRAIN_POLL_INTERVAL)
    else:
        logger.warning(f"[{dataset}][inst {instance_idx}] 排空超时 ({DRAIN_TIMEOUT}s)")

    # 最终统计
    final_completed = monitor.get_completed_count(user_id)
    final_failed = len(monitor.get_failed(user_id))
    logger.info(
        f"[{dataset}][inst {instance_idx}] 结果: "
        f"completed={final_completed}, failed={final_failed}"
    )

    # 重投失败 chunks
    failed = monitor.get_failed(user_id)
    if failed:
        logger.info(f"  冷却 {RETRY_COOLDOWN}s 后重投 {len(failed)} 个失败 chunks...")
        time.sleep(RETRY_COOLDOWN)

        for attempt in range(1, RETRY_LIMIT + 1):
            failed = monitor.get_failed(user_id)
            if not failed:
                break
            logger.info(f"  重投 attempt {attempt}/{RETRY_LIMIT}: {len(failed)} 失败")

            for trace in failed:
                # 提取 filter_tags 中的 chunk_id
                tags = trace.get("filter_tags", {})
                chunk_id = tags.get("chunk_id")
                if chunk_id is None:
                    continue
                chunk_id = int(chunk_id)
                if chunk_id >= len(chunks):
                    continue

                ft = {
                    "chunk_id": chunk_id,
                    "instance_idx": instance_idx,
                    "dataset": dataset,
                    "source": "MemoryAgentBench",
                    "retry": attempt,
                }
                msgs = [
                    {"role": "user", "content": [{"type": "text", "text": chunks[chunk_id]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Acknowledged."}]},
                ]
                try:
                    session.post(
                        f"{base_url}/memory/add",
                        json={
                            "user_id": user_id,
                            "meta_agent_id": meta_agent_id,
                            "messages": msgs,
                            "chaining": False,
                            "filter_tags": ft,
                            "use_cache": True,
                        },
                    ).raise_for_status()
                except Exception as e:
                    logger.error(f"    重投 chunk {chunk_id} 失败: {e}")

            # 等重投处理完
            logger.info(f"  等待重投处理 (max {DRAIN_TIMEOUT//2}s)...")
            wait_start = time.time()
            while time.time() - wait_start < DRAIN_TIMEOUT // 2:
                if monitor.count_pending(user_id=user_id) == 0:
                    break
                time.sleep(DRAIN_POLL_INTERVAL)

    # 最终统计: completed >= 需要的 chunk 数即视为完成
    final_completed = monitor.get_completed_count(user_id)
    final_failed = len(monitor.get_failed(user_id))
    final_pending = monitor.count_pending(user_id=user_id)
    status = "done" if final_completed >= len(chunks) else "incomplete"

    save_checkpoint(dataset, instance_idx, {
        "queued": queued_count,
        "confirmed": final_completed,
        "failed": final_failed,
        "pending": final_pending,
        "needed": len(chunks),
        "status": status,
    })

    logger.info(
        f"[{dataset}][inst {instance_idx}] 完成: "
        f"status={status}, confirmed={final_completed}/{len(chunks)}, "
        f"failed={final_failed}, pending={final_pending}"
    )
    return status == "done"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_dataset_parallel(
    dataset: str,
    indices: list,
    base_url: str,
    api_key: str,
    resume: bool = False,
    force: bool = False,
) -> dict:
    """并行处理一个 dataset 的所有 instances (利用 user_id partition 并行)"""
    logger = setup_logger(dataset)
    logger.info(f"===== {dataset}: {len(indices)} instances (parallel, max {MAX_PARALLEL_INSTANCES}) =====")

    results = {"success": 0, "failed": 0}

    def _run_one(idx):
        # 每个线程独立的 monitor (独立 session)
        mon = TraceMonitor(base_url, api_key)
        return idx, feed_instance(
            dataset=dataset,
            instance_idx=idx,
            base_url=base_url,
            api_key=api_key,
            monitor=mon,
            logger=logger,
            resume=resume,
            force=force,
        )

    batch_size = min(MAX_PARALLEL_INSTANCES, len(indices))
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_run_one, idx): idx for idx in indices}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, ok = future.result()
                if ok:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                logger.info(
                    f"[{dataset}][inst {idx}] 线程结束 | "
                    f"当前进度: {results['success']}✓ {results['failed']}✗ / {len(indices)}"
                )
            except Exception as e:
                results["failed"] += 1
                logger.error(f"[{dataset}][inst {idx}] 线程异常: {e}")

    logger.info(f"===== {dataset} 完成: {results} =====\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="MIRIX Ingest Dispatcher")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(PLAN.keys()),
                        help="只跑单个 dataset (默认全部)")
    parser.add_argument("--instance_idx", type=str, default=None,
                        help="指定 instance 范围 (e.g., '0-5', '1,3')")
    parser.add_argument("--resume", action="store_true",
                        help="从 checkpoint 续跑")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划, 不执行")
    parser.add_argument("--serial", action="store_true",
                        help="串行模式 (调试用, 默认并行)")
    parser.add_argument("--force", action="store_true",
                        help="强制清理 checkpoint 重跑 (不跳过已完成的 instance)")
    args = parser.parse_args()

    # 连接信息
    mirix_conn = get_mirix_connection_info()
    base_url = mirix_conn.get("base_url", "http://localhost:8531")
    api_key = mirix_conn.get("api_key", "")

    if not api_key:
        print("错误: config/config.yaml 中缺少 mirix.api_key")
        sys.exit(1)

    # 构建执行计划
    if args.dataset:
        plan = {args.dataset: PLAN[args.dataset]}
    else:
        plan = PLAN

    if args.instance_idx:
        from src.benchmark_utils import parse_instance_indices
        indices = parse_instance_indices(args.instance_idx)
        plan = {ds: [i for i in idxs if i in indices] for ds, idxs in plan.items()}

    # 统计 (精确计算 chunk 数)
    total_instances = sum(len(idxs) for idxs in plan.values())
    total_chunks = 0
    max_inst_chunks = 0
    class _Args:
        chunk_size = None; overlap = None; min_chars = None
    for ds, idxs in plan.items():
        ds_chunks = 0
        for i in idxs:
            try:
                d = load_data(ds, i)
                c = get_chunks(ds, d["context"], _Args())
                ds_chunks += len(c)
                max_inst_chunks = max(max_inst_chunks, len(c))
            except Exception:
                pass
        total_chunks += ds_chunks
        print(f"  {ds}: {len(idxs)} instances, {ds_chunks} chunks")
    print(f"总计: {total_instances} instances, {total_chunks} chunks")
    print(f"模式: {'串行' if args.serial else f'并行 (max {MAX_PARALLEL_INSTANCES} workers)'}")
    # 瓶颈是最大单 instance (串行)
    bottleneck_h = max_inst_chunks * 2.2 / 60
    print(f"瓶颈: 最大单 instance {max_inst_chunks} chunks → ~{bottleneck_h:.1f}h (串行)")
    print(f"小 instance 预计: ~{total_chunks * 2.2 / max(1, min(total_instances, MAX_PARALLEL_INSTANCES)) / 60:.1f}h")

    if args.dry_run:
        return

    # 清理陈旧 traces (容器重启后内存队列丢失, 但 DB 中 trace 仍为 queued/processing)
    cleaned = TraceMonitor.cleanup_stale_traces_db()
    print(f"清理陈旧 traces: {cleaned}")

    # 执行
    overall_start = time.time()
    all_results = {"success": 0, "failed": 0}

    if args.serial:
        # 串行模式 (调试)
        for dataset, indices in plan.items():
            logger = setup_logger(dataset)
            monitor = TraceMonitor(base_url, api_key)
            logger.info(f"===== {dataset}: {len(indices)} instances (serial) =====")
            for idx in indices:
                ok = feed_instance(
                    dataset=dataset, instance_idx=idx,
                    base_url=base_url, api_key=api_key,
                    monitor=monitor, logger=logger,
                    resume=args.resume, force=args.force,
                )
                all_results["success" if ok else "failed"] += 1
            logger.info(f"===== {dataset} 完成 =====\n")
    else:
        # 全并行模式: 所有 dataset 的所有 instance 一起跑
        # 不同 user_id 会被 hash 到不同 partition, 最大化 worker 利用率
        all_jobs = []
        loggers = {}
        for dataset, indices in plan.items():
            loggers[dataset] = setup_logger(dataset)
            loggers[dataset].info(f"===== {dataset}: {len(indices)} instances =====")
            for idx in indices:
                all_jobs.append((dataset, idx))

        print(f"启动 {len(all_jobs)} 个任务 (max {MAX_PARALLEL_INSTANCES} 并行, 交错 {STAGGER_DELAY}s)...")

        # 交错启动: 用 queue 代替全量 submit, 避免初始化风暴
        import queue as _queue
        job_queue = _queue.Queue()
        for job in all_jobs:
            job_queue.put(job)

        def _run_job_staggered(job_idx):
            """从 queue 中不断取任务, 直到 queue 为空"""
            # 交错延迟: 第 N 个 worker 延迟 N*STAGGER_DELAY 秒启动
            time.sleep(job_idx * STAGGER_DELAY)
            while True:
                try:
                    ds, idx = job_queue.get_nowait()
                except _queue.Empty:
                    return []  # 所有任务已取完
                mon = TraceMonitor(base_url, api_key)
                ok = feed_instance(
                    dataset=ds, instance_idx=idx,
                    base_url=base_url, api_key=api_key,
                    monitor=mon, logger=loggers[ds],
                    resume=args.resume, force=args.force,
                )
                job_results.append((ds, idx, ok))
                loggers[ds].info(
                    f"[{ds}][inst {idx}] 完毕 | "
                    f"全局 {len(job_results)}/{len(all_jobs)} | "
                    f"elapsed {(time.time()-overall_start)/3600:.1f}h"
                )

        job_results = []
        workers = min(MAX_PARALLEL_INSTANCES, len(all_jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_job_staggered, i) for i in range(workers)]
            for future in as_completed(futures):
                try:
                    future.result()  # 只等待 worker 退出, 结果在 job_results 中
                except Exception as e:
                    print(f"Worker 异常: {e}")

        # 汇总结果
        for ds, idx, ok in job_results:
            all_results["success" if ok else "failed"] += 1

    elapsed = round(time.time() - overall_start, 1)
    print(f"\n全部完成: {all_results} | 总耗时 {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
