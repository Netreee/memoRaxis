#!/usr/bin/env bash
# finish_memgpt.sh - 顺序完成剩余 MemGPT 工作
# Phase 1: AR R2 + AR R3 (2并行) → ~20:39
# Phase 2: CR R2 + CR R3 (2并行) → ~21:00
# + TTL inst0 R2/R3 (快速，同时启动)
set -euo pipefail
cd "$(dirname "$0")"

LOG=finish_memgpt.log
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "====== finish_memgpt.sh 启动 ======"

# ── 立即: simpleMemory TTL inst 1,3,5 (快，不影响TPM) ──────────────
log "启动 simpleMemory TTL inst 1,3,5"
nohup python3 scripts/simpleMemory_MAB/infer/infer_test_time.py \
    --instance_idx 1,3,5 \
    > out/sm_ttl_135_fix.log 2>&1 &
SM_PID=$!
log "  simpleMemory TTL PID=$SM_PID"

# ── 立即: MemGPT TTL inst0 R2/R3 (2题，快速) ──────────────────────
log "启动 MemGPT TTL inst0 R2"
nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_test_time.py \
    --instance_idx 0 --adaptor R2 \
    > gaoang/out/ttl_inst0_r2.log 2>&1 &
TTL_R2_PID=$!
log "  MemGPT TTL R2 PID=$TTL_R2_PID"

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_test_time.py \
    --instance_idx 0 --adaptor R3 \
    > gaoang/out/ttl_inst0_r3.log 2>&1 &
TTL_R3_PID=$!
log "  MemGPT TTL R3 PID=$TTL_R3_PID"

# ── Phase 1: AR R2 + AR R3 (2并行) ───────────────────────────────
log ""
log "=== Phase 1: MemGPT AR R2 + R3 (2并行) ==="
nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
    --instance_idx 2,3,6,7,8,9,13,14,15,21 --adaptor R2 \
    > gaoang/out/ar_r2_phase1.log 2>&1 &
AR_R2_PID=$!
log "  AR R2 (10题) PID=$AR_R2_PID"

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
    --instance_idx 1,2,3,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21 --adaptor R3 \
    > gaoang/out/ar_r3_phase1.log 2>&1 &
AR_R3_PID=$!
log "  AR R3 (18题) PID=$AR_R3_PID"

log "  等待 AR 完成..."
wait $AR_R2_PID && log "  ✓ AR R2 done" || log "  ✗ AR R2 failed"
wait $AR_R3_PID && log "  ✓ AR R3 done" || log "  ✗ AR R3 failed"
log "=== Phase 1 完成 ==="

# ── Phase 2: CR R2 + CR R3 (2并行) ───────────────────────────────
log ""
log "=== Phase 2: MemGPT CR R2 + R3 (2并行) ==="
nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_conflict_resolution.py \
    --instance_idx 1,2,3,4,5,6,7 --adaptor R2 \
    > gaoang/out/cr_r2_phase2.log 2>&1 &
CR_R2_PID=$!
log "  CR R2 (7题) PID=$CR_R2_PID"

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_conflict_resolution.py \
    --instance_idx 1,2,3,4,5,6,7 --adaptor R3 \
    > gaoang/out/cr_r3_phase2.log 2>&1 &
CR_R3_PID=$!
log "  CR R3 (7题) PID=$CR_R3_PID"

log "  等待 CR 完成..."
wait $CR_R2_PID && log "  ✓ CR R2 done" || log "  ✗ CR R2 failed"
wait $CR_R3_PID && log "  ✓ CR R3 done" || log "  ✗ CR R3 failed"
log "=== Phase 2 完成 ==="

log ""
log "====== 全部完成 $(date) ======"
