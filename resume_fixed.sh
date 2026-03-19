#!/usr/bin/env bash
# resume_fixed.sh - 修复版，全部使用正确 output_suffix，严格2个LLM流
# Stream A: AR R1(14) → AR R2(4) → CR R2(7) → TTL R2(1)  ~64min
# Stream B: AR R3(14) → CR R3(7) → TTL R3(1)             ~94min
# Background: simpleMemory TTL (纯embedding)
set -euo pipefail
cd /Users/bytedance/proj/memoRaxis

LOG=resume_fixed.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======================================================"
log "  resume_fixed.sh 启动（修复版）"
log "======================================================"

mkdir -p gaoang/out out/raptor_trees gaoang/out/raptor_ingest_logs

# ── 立即: simpleMemory TTL (纯embedding，不占LLM TPM) ──────────────
log "启动 simpleMemory TTL inst 1,3,5"
nohup python3 scripts/simpleMemory_MAB/infer/infer_test_time.py \
    --instance_idx 1,3,5 \
    >> out/sm_ttl_135_final.log 2>&1 &
log "  simpleMemory TTL PID=$!"

# ── Stream A: AR R1(14) → AR R2(4) → CR R2(7) → TTL R2(1) ────────
log ""
log "=== Stream A 启动 ==="
(
  log "[A] AR R1 (14题, suffix=memgpt_r1b)"
  python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
      --instance_idx 1,2,3,5,6,7,8,9,10,11,13,14,15,21 \
      --adaptor R1 --output_suffix memgpt_r1b \
      >> gaoang/out/ar_r1b_stream.log 2>&1
  log "[A] ✓ AR R1 done"

  log "[A] AR R2 (4题: 2,3,15,21, suffix=memgpt_r2)"
  python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
      --instance_idx 2,3,15,21 \
      --adaptor R2 --output_suffix memgpt_r2 \
      >> gaoang/out/ar_r2_stream.log 2>&1
  log "[A] ✓ AR R2 done"

  log "[A] CR R2 (7题: 1-7, suffix=memgpt_r2)"
  python3 gaoang/scripts/simpleMemory_MAB/infer/infer_conflict_resolution.py \
      --instance_idx 1,2,3,4,5,6,7 \
      --adaptor R2 --output_suffix memgpt_r2 \
      >> gaoang/out/cr_r2_stream.log 2>&1
  log "[A] ✓ CR R2 done"

  log "[A] TTL inst0 R2 (suffix=memgpt_r2)"
  python3 gaoang/scripts/simpleMemory_MAB/infer/infer_test_time.py \
      --instance_idx 0 \
      --adaptor R2 --output_suffix memgpt_r2 \
      >> gaoang/out/ttl_r2_stream.log 2>&1
  log "[A] ✓ TTL R2 done"

  log "[A] Stream A 全部完成"
) &
STREAM_A_PID=$!
log "Stream A PID=$STREAM_A_PID"

# ── Stream B: AR R3(14) → CR R3(7) → TTL R3(1) ────────────────────
log ""
log "=== Stream B 启动 ==="
(
  log "[B] AR R3 (14题: 6-11,12,15-21, suffix=memgpt_r3)"
  python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
      --instance_idx 6,7,8,9,10,11,12,15,16,17,18,19,20,21 \
      --adaptor R3 --output_suffix memgpt_r3 \
      >> gaoang/out/ar_r3_stream.log 2>&1
  log "[B] ✓ AR R3 done"

  log "[B] CR R3 (7题: 1-7, suffix=memgpt_r3)"
  python3 gaoang/scripts/simpleMemory_MAB/infer/infer_conflict_resolution.py \
      --instance_idx 1,2,3,4,5,6,7 \
      --adaptor R3 --output_suffix memgpt_r3 \
      >> gaoang/out/cr_r3_stream.log 2>&1
  log "[B] ✓ CR R3 done"

  log "[B] TTL inst0 R3 (suffix=memgpt_r3)"
  python3 gaoang/scripts/simpleMemory_MAB/infer/infer_test_time.py \
      --instance_idx 0 \
      --adaptor R3 --output_suffix memgpt_r3 \
      >> gaoang/out/ttl_r3_stream.log 2>&1
  log "[B] ✓ TTL R3 done"

  log "[B] Stream B 全部完成"
) &
STREAM_B_PID=$!
log "Stream B PID=$STREAM_B_PID"

log ""
log "等待两个 Stream 完成..."
wait $STREAM_A_PID && log "✓ Stream A done" || log "✗ Stream A exited non-zero"
wait $STREAM_B_PID && log "✓ Stream B done" || log "✗ Stream B exited non-zero"
log "=== MemGPT 全部收工 ==="

# ── Phase 3: Raptor ingest ─────────────────────────────────────────
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

raptor_batch "CR"      conflict_resolution  3 4 5 6 7
raptor_batch "TTL0"    test_time            0
raptor_batch "AR-sm"   accurate_retrieval   15 16 17 18 19 20 21
raptor_batch "AR-lg-1" accurate_retrieval    0  1
raptor_batch "AR-lg-2" accurate_retrieval    2  3
raptor_batch "AR-lg-3" accurate_retrieval    4  5
raptor_batch "AR-lg-4" accurate_retrieval    6
for batch in "0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15" \
             "16 17 18 19" "20 21 22 23" "24 25 26 27" \
             "28 29 30 31" "32 33 34 35" "36 37 38 39"; do
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
log "  resume_fixed.sh 全部完成 $(date)"
log "======================================================"
