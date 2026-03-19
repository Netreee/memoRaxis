#!/usr/bin/env bash
# resume_all.sh - 从断点恢复所有剩余工作（修复版）
# Phase 1: MemGPT AR R2+R3 (2并行) + simpleMemory TTL
# Phase 2: MemGPT CR R2+R3 + TTL inst0
# Phase 3: Raptor ingest (CR→TTL→AR→LR)
set -euo pipefail
cd /Users/bytedance/proj/memoRaxis

LOG=resume_all.log
# 注意：不用 exec 重定向，避免 PID 捕获被污染
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======================================================"
log "  resume_all.sh 启动"
log "======================================================"

mkdir -p gaoang/out out/raptor_trees gaoang/out/raptor_ingest_logs

# ── 立即: simpleMemory TTL 1,3,5 (纯embedding，不占LLM) ────────────
log "启动 simpleMemory TTL inst 1,3,5"
nohup python3 scripts/simpleMemory_MAB/infer/infer_test_time.py \
    --instance_idx 1,3,5 \
    >> out/sm_ttl_135_resume.log 2>&1 &
SM_PID=$!
log "  simpleMemory TTL PID=$SM_PID"

# ── Phase 1: MemGPT AR R2 + R3 (严格2并行) ────────────────────────
log ""
log "=== Phase 1: MemGPT AR R2(10题) + R3(18题) ==="

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
    --instance_idx 2,3,6,7,8,9,13,14,15,21 --adaptor R2 \
    >> gaoang/out/ar_r2_resume.log 2>&1 &
AR_R2_PID=$!
log "  AR R2 (10题) PID=$AR_R2_PID"

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
    --instance_idx 1,2,3,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21 --adaptor R3 \
    >> gaoang/out/ar_r3_resume.log 2>&1 &
AR_R3_PID=$!
log "  AR R3 (18题) PID=$AR_R3_PID"

log "  等待 AR 完成..."
wait $AR_R2_PID && log "  ✓ AR R2 done" || log "  ✗ AR R2 exited non-zero"
wait $AR_R3_PID && log "  ✓ AR R3 done" || log "  ✗ AR R3 exited non-zero"
log "=== Phase 1 完成 ==="

# ── Phase 2: MemGPT CR R2+R3 + TTL inst0 ──────────────────────────
log ""
log "=== Phase 2: MemGPT CR R2(7题) + R3(7题) + TTL inst0 ==="

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_conflict_resolution.py \
    --instance_idx 1,2,3,4,5,6,7 --adaptor R2 \
    >> gaoang/out/cr_r2_resume.log 2>&1 &
CR_R2_PID=$!
log "  CR R2 (7题) PID=$CR_R2_PID"

nohup python3 gaoang/scripts/simpleMemory_MAB/infer/infer_conflict_resolution.py \
    --instance_idx 1,2,3,4,5,6,7 --adaptor R3 \
    >> gaoang/out/cr_r3_resume.log 2>&1 &
CR_R3_PID=$!
log "  CR R3 (7题) PID=$CR_R3_PID"

# TTL inst0 串行跑（2题，小任务）
nohup bash -c "
  python3 gaoang/scripts/simpleMemory_MAB/infer/infer_test_time.py --instance_idx 0 --adaptor R2 >> gaoang/out/ttl_inst0_resume.log 2>&1 &&
  python3 gaoang/scripts/simpleMemory_MAB/infer/infer_test_time.py --instance_idx 0 --adaptor R3 >> gaoang/out/ttl_inst0_resume.log 2>&1
" &
TTL_PID=$!
log "  TTL inst0 R2+R3 PID=$TTL_PID"

log "  等待 CR + TTL 完成..."
wait $CR_R2_PID && log "  ✓ CR R2 done" || log "  ✗ CR R2 exited non-zero"
wait $CR_R3_PID && log "  ✓ CR R3 done" || log "  ✗ CR R3 exited non-zero"
wait $TTL_PID   && log "  ✓ TTL inst0 done" || log "  ✗ TTL inst0 exited non-zero"
log "=== Phase 2 完成 - MemGPT 全部收工 ==="

# ── Phase 3: Raptor ingest (MemGPT退出后LLM全让给Raptor) ──────────
log ""
log "=== Phase 3: Raptor ingest ==="

raptor_batch() {
    local label=$1 dataset=$2; shift 2
    local pids=()
    log "  批次 $label: instances=($*)"
    for inst in "$@"; do
        nohup python3 gaoang/scripts/Raptor_MAB/ingest/ingest_${dataset}.py \
            --instance_idx "$inst" --save_dir out/raptor_trees \
            >> gaoang/out/raptor_ingest_logs/${dataset}_${inst}.log 2>&1 &
        pids+=($!)
        log "    inst $inst PID=$!"
    done
    for pid in "${pids[@]}"; do
        wait "$pid" && log "    ✓ PID=$pid done" || log "    ✗ PID=$pid failed"
    done
    log "  批次 $label 完成"
}

# 小实例先跑
raptor_batch "CR"      conflict_resolution  3 4 5 6 7
raptor_batch "TTL0"    test_time            0
# AR missing (15-21 中等，0-6 大)
raptor_batch "AR-sm"   accurate_retrieval   15 16 17 18 19 20 21
raptor_batch "AR-lg-1" accurate_retrieval    0  1
raptor_batch "AR-lg-2" accurate_retrieval    2  3
raptor_batch "AR-lg-3" accurate_retrieval    4  5
raptor_batch "AR-lg-4" accurate_retrieval    6
# LR 全部 0-39，每批4并行
for batch in "0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15" \
             "16 17 18 19" "20 21 22 23" "24 25 26 27" \
             "28 29 30 31" "32 33 34 35" "36 37 38 39"; do
    # shellcheck disable=SC2086
    raptor_batch "LR-$(echo $batch | awk '{print $1}')" long_range $batch
done

log ""
log "=== 最终统计 ==="
python3 - <<'PYEOF'
from pathlib import Path
for pat, label, total in [
    ("raptor_ttl_*.pkl","TTL",6), ("raptor_acc_ret_*.pkl","AR",22),
    ("raptor_conflict_*.pkl","CR",8), ("raptor_long_range_*.pkl","LR",40),
]:
    n = len(list(Path("out/raptor_trees").glob(pat)))
    print(f"  Raptor {label}: {n}/{total}")
PYEOF

log "======================================================"
log "  resume_all.sh 全部完成 $(date)"
log "======================================================"
