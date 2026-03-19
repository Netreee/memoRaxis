#!/usr/bin/env bash
# overnight.sh - MemGPT 全量 ingest+infer 自动化恢复脚本
# 设计：ingest 不用 LLM，infer 用 LLM，两组可同时跑互不干扰
# LLM 并发限制：最多 4 个 infer worker 同时运行，防止 429
set -euo pipefail
cd "$(dirname "$0")"

LOG=overnight_run.log
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p gaoang/out/infer_logs

log "======================================================"
log "  overnight.sh 启动"
log "======================================================"

# ── Phase 1: 并发 re-ingest 所有缺失 agent（不消耗 LLM）──────
log ""
log "=== Phase 1: 并发 re-ingest 缺失 agents ==="

nohup python3 gaoang/scripts/simpleMemory_MAB/ingest/ingest_conflict_resolution.py \
    --instance_idx 0-7 \
    > gaoang/out/ingest_logs/cr_full.log 2>&1 &
CR_INGEST=$!
log "  CR   ingest PID=$CR_INGEST  (inst 0-7,  ~3880 chunks, ~19 min)"

nohup python3 gaoang/scripts/simpleMemory_MAB/ingest/ingest_long_range.py \
    --instance_idx 0-10 \
    > gaoang/out/ingest_logs/lr_missing.log 2>&1 &
LR_INGEST=$!
log "  LR   ingest PID=$LR_INGEST  (inst 0-10, ~8731 chunks, ~44 min)"

nohup python3 gaoang/scripts/simpleMemory_MAB/ingest/ingest_test_time.py \
    --instance_idx 0-2 \
    > gaoang/out/ingest_logs/ttl_missing.log 2>&1 &
TTL_INGEST=$!
log "  TTL  ingest PID=$TTL_INGEST (inst 0-2,  ~8036 chunks, ~40 min)"

nohup python3 gaoang/scripts/simpleMemory_MAB/ingest/ingest_accurate_retrieval.py \
    --instance_idx 0-3 \
    > gaoang/out/ingest_logs/ar_missing.log 2>&1 &
AR_INGEST=$!
log "  AR   ingest PID=$AR_INGEST  (inst 0-3,  ~11360 chunks, ~57 min)"

# ── Phase 2: 同步开始对已有 agent 做 infer（R1，4 worker）────
log ""
log "=== Phase 2: 对已有 agents 启动 infer R1 (4 workers) ==="

# AR 分 3 组，每组 6 个 instance，各跑约 2.5h
nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
    --instance_idx 4,5,6,7,8,9 --adaptor R1 --limit -1 --output_suffix memgpt \
    > gaoang/out/infer_logs/ar_4_9.log 2>&1 &
AR_INFER_A=$!
log "  AR-A infer PID=$AR_INFER_A (inst 4-9)"

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
    --instance_idx 10,11,12,13,14,15 --adaptor R1 --limit -1 --output_suffix memgpt \
    > gaoang/out/infer_logs/ar_10_15.log 2>&1 &
AR_INFER_B=$!
log "  AR-B infer PID=$AR_INFER_B (inst 10-15)"

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
    --instance_idx 16,17,18,19,20,21 --adaptor R1 --limit -1 --output_suffix memgpt \
    > gaoang/out/infer_logs/ar_16_21.log 2>&1 &
AR_INFER_C=$!
log "  AR-C infer PID=$AR_INFER_C (inst 16-21)"

# LR inst 11-39 已有 agent，1 worker 跑完约 1.8h
nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_long_range.py \
    --instance_idx 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 \
    --adaptor R1 --limit -1 --output_suffix memgpt \
    > gaoang/out/infer_logs/lr_11_39.log 2>&1 &
LR_INFER_A=$!
log "  LR-A infer PID=$LR_INFER_A (inst 11-39)"

log "  4 infer workers 已启动，同步等待 re-ingest 完成..."

# ── Phase 3: 等 re-ingest 完成，继续补跑剩余 infer ───────────
log ""
log "=== Phase 3: 等待 re-ingest 完成 ==="

wait $CR_INGEST && log "  ✓ CR  ingest done" || log "  ✗ CR  ingest failed"
wait $LR_INGEST && log "  ✓ LR  ingest done" || log "  ✗ LR  ingest failed"
wait $TTL_INGEST && log "  ✓ TTL ingest done" || log "  ✗ TTL ingest failed"
wait $AR_INGEST && log "  ✓ AR  ingest done" || log "  ✗ AR  ingest failed"

log ""
log "=== Phase 3b: 对新 ingest 完成的 agents 启动 infer R1 ==="

# TTL inst 3-5 已有，0-2 刚 ingest 完，一起跑
nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_test_time.py \
    --instance_idx 0,1,2,3,4,5 --adaptor R1 --limit -1 --output_suffix memgpt \
    > gaoang/out/infer_logs/ttl_all.log 2>&1 &
TTL_INFER=$!
log "  TTL infer PID=$TTL_INFER (inst 0-5)"

# LR inst 0-10 刚 ingest 完
nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_long_range.py \
    --instance_idx 0,1,2,3,4,5,6,7,8,9,10 \
    --adaptor R1 --limit -1 --output_suffix memgpt \
    > gaoang/out/infer_logs/lr_0_10.log 2>&1 &
LR_INFER_B=$!
log "  LR-B infer PID=$LR_INFER_B (inst 0-10)"

# ── Phase 4: 等 LR-A 跑完（最短）再加 CR ─────────────────────
log ""
log "=== Phase 4: 等 LR inst11-39 infer 完成，再启动 CR ==="
wait $LR_INFER_A && log "  ✓ LR-A infer done" || log "  ✗ LR-A infer failed"

# CR 分 2 组，各跑约 2.8h
nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_conflict_resolution.py \
    --instance_idx 0,1,2,3 --adaptor R1 --limit -1 --output_suffix memgpt \
    > gaoang/out/infer_logs/cr_0_3.log 2>&1 &
CR_INFER_A=$!
log "  CR-A infer PID=$CR_INFER_A (inst 0-3)"

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_conflict_resolution.py \
    --instance_idx 4,5,6,7 --adaptor R1 --limit -1 --output_suffix memgpt \
    > gaoang/out/infer_logs/cr_4_7.log 2>&1 &
CR_INFER_B=$!
log "  CR-B infer PID=$CR_INFER_B (inst 4-7)"

# AR inst 0-3 刚 ingest 完，各自单独跑（最大的实例）
for i in 0 1 2 3; do
    nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
        --instance_idx $i --adaptor R1 --limit -1 --output_suffix memgpt \
        > gaoang/out/infer_logs/ar_${i}.log 2>&1 &
    log "  AR-$i infer PID=$! (inst $i)"
done

# ── Phase 5: 等所有 infer 完成 ───────────────────────────────
log ""
log "=== Phase 5: 等待所有 infer 完成 ==="
wait $AR_INFER_A $AR_INFER_B $AR_INFER_C $LR_INFER_B $TTL_INFER $CR_INFER_A $CR_INFER_B
log "  ✓ 所有 MemGPT infer R1 完成！"

# ── Phase 6: 统计结果 ─────────────────────────────────────────
log ""
log "=== Phase 6: 结果统计 ==="
python3 - <<'PYEOF'
import glob, json

for pat, label, total in [
    ("out/acc_ret_results_*memgpt*.json",     "AR  (22)", 22),
    ("out/conflict_res_results_*memgpt*.json","CR  ( 8)",  8),
    ("out/long_range_results_*memgpt*.json",  "LR  (40)", 40),
    ("out/ttl_results_*memgpt*.json",         "TTL ( 6)",  6),
]:
    files = glob.glob(pat)
    ok, err = 0, 0
    for f in files:
        d = json.load(open(f))
        for rows in d.get("results", {}).values():
            ok  += sum(1 for r in rows if r.get("answer") and not r.get("error"))
            err += sum(1 for r in rows if r.get("error"))
    print(f"  {label}: {len(files)}/{total} instances, {ok} answers, {err} errors")
PYEOF

log "======================================================"
log "  overnight.sh 全部完成 $(date)"
log "======================================================"
