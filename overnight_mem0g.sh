#!/usr/bin/env bash
# overnight_mem0g.sh - mem0g AR ingest + infer R1（严格串行，避免Neo4j超时）
# 注意: inst0 infer 已在运行 (PID 27644)，此脚本跳过 inst0
set -euo pipefail
cd /Users/bytedance/proj/memoRaxis

mkdir -p logs/overnight results/mem0g/AR

LOG=logs/overnight/mem0g.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======================================================"
log "  overnight_mem0g.sh 启动（串行版，跳过inst0）"
log "======================================================"

mem0g_ingest_done() {
    local inst=$1
    local cnt
    cnt=$(curl -s "http://localhost:6333/collections/mem0g_acc_ret_${inst}" 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('result',{}).get('points_count',0))" 2>/dev/null || echo 0)
    [ "${cnt:-0}" -gt 5 ]  # 需要>5个点才算真正ingested
}

run_one() {
    local inst=$1
    local dst="out/mem0g/mem0g_acc_ret_results_${inst}.json"

    # ingest
    if mem0g_ingest_done $inst; then
        log "[INGEST-SKIP] inst${inst} 已有数据（>5 points）"
    else
        log "[INGEST] inst${inst} 开始"
        python3 scripts/mem0g_MAB/ingest.py \
            --dataset accurate_retrieval --instance_idx "${inst}" \
            >> "logs/overnight/ingest_ar_${inst}.log" 2>&1
        local ec=$?
        # 验证ingest结果
        if mem0g_ingest_done $inst; then
            log "[INGEST] ✅ inst${inst} 完成"
        else
            log "[INGEST] ✗ inst${inst} 失败（exit=$ec），跳过infer"
            return 0
        fi
    fi

    # infer R1
    if [ -f "$dst" ]; then
        log "[INFER-SKIP] inst${inst} 已有结果"
        return 0
    fi
    log "[INFER R1] inst${inst} 开始"
    python3 scripts/mem0g_MAB/infer.py \
        --task Accurate_Retrieval --instance_idx "${inst}" \
        --adaptor R1 --limit -1 \
        >> "logs/overnight/infer_ar_r1_${inst}.log" 2>&1 \
        && log "[INFER R1] ✅ inst${inst} 完成" \
        || log "[INFER R1] ✗ inst${inst} 失败"
}

# inst0 已在跑（PID 27644），跳过
# inst1,5,7 已有infer结果，mem0g_ingest_done会验证，infer会跳过
# 先跑已ingested的 inst1,5,6,7 的infer（或跳过已有结果的）
log "=== 先处理已ingested实例 ==="
for inst in 1 5 6 7; do
    run_one $inst
done

# 再串行ingest+infer剩余实例
log ""
log "=== 串行处理剩余实例 ==="
for inst in 2 3 4 8 9 10 11 12 13 14 15 16 17 18 19 20 21; do
    run_one $inst
done

log ""
log "=== 最终统计 ==="
python3 - <<'PYEOF'
from pathlib import Path
done = sorted(Path("results/mem0g/AR").glob("instance_*.json"))
print(f"  mem0g AR infer done: {len(done)}/22")
for p in done: print(f"    {p.name}")
PYEOF

log "======================================================"
log "  overnight_mem0g.sh 全部完成 $(date)"
log "======================================================"
