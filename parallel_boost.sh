#!/usr/bin/env bash
# parallel_boost.sh - 在不破坏run_plan.sh的前提下开启额外并行流
# R1 Stream C: inst14,15,17,18
# R1 Stream D: inst19,20,21
# R2 Stream D: inst13,14,15
# R2 Stream E: inst17,18,19,20,21
set -euo pipefail
cd /Users/bytedance/proj/memoRaxis

LOG=logs/boost.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

already_done() {
    local dst="results/memgpt/${1}_inst$(printf '%02d' $2)_${3}.json"
    [ -f "$dst" ]
}

audit_and_copy() 2{
    local ds=$1 inst=$2 adaptor=$3 suffix=$4
    local src="out/acc_ret_results_${inst}_mgpt_${suffix}.json"
    local dst="results/memgpt/ar_inst$(printf '%02d' ${inst})_$(echo ${adaptor} | tr 'A-Z' 'a-z').json"
    # 验证100q（LR=1q，TTL0=200q）
    local expected=100
    local count
    count=$(python3 -c "
import json, sys
try:
    d=json.load(open('$src'))
    r=d.get('results',{})
    k=list(r.keys())[0] if r else None
    qs=r[k] if k else d.get('data',[])
    print(len(qs) if isinstance(qs,list) else 0)
except: print(0)
" 2>/dev/null || echo 0)
    if [ "$count" -ge "$expected" ]; then
        cp "$src" "$dst"
        log "[AUDIT] ✅ $dst (${count}q)"
    else
        log "[AUDIT] ✗ $src (${count}q < $expected，未归档)"
    fi
}

run_ar() {
    local inst=$1 adaptor=$2
    local suffix=$(echo $adaptor | tr 'A-Z' 'a-z')
    local ds_name
    case $adaptor in
        R1) ds_name="acc_ret" ;;
        R2) ds_name="acc_ret" ;;
        R3) ds_name="acc_ret" ;;
    esac
    already_done "ar" $inst $adaptor && { log "[SKIP] ar inst${inst} ${adaptor} 已完成"; return 0; }
    log "[AR] inst${inst} ${adaptor} 开始"
    python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
        --instance_idx "$inst" --adaptor "$adaptor" \
        --output_suffix "mgpt_${suffix}" --limit -1 \
        >> "logs/boost_ar_${inst}_${suffix}.log" 2>&1 \
        && audit_and_copy "acc_ret" $inst $adaptor $suffix \
        || log "[AR] ✗ inst${inst} ${adaptor}"
}

log "====== parallel_boost.sh 启动 ======"
log "R1 Stream C: 14→15→17→18"
log "R1 Stream D: 19→20→21"
log "R2 Stream D: 13→14→15"
log "R2 Stream E: 17→18→19→20→21"

# R1 Stream C
(
    for inst in 14 15 17 18; do run_ar $inst R1; done
    log "[C] R1 Stream C 完成"
) &
STREAM_C=$!
log "Stream C PID=$STREAM_C"

# R1 Stream D
(
    for inst in 19 20 21; do run_ar $inst R1; done
    log "[D] R1 Stream D 完成"
) &
STREAM_D=$!
log "Stream D PID=$STREAM_D"

# R2 Stream D（等5s错开，避免同时发起LLM）
sleep 5
(
    for inst in 13 14 15; do run_ar $inst R2; done
    log "[E] R2 Stream D 完成"
) &
STREAM_E=$!
log "Stream E PID=$STREAM_E"

# R2 Stream E
sleep 5
(
    for inst in 17 18 19 20 21; do run_ar $inst R2; done
    log "[F] R2 Stream E 完成"
) &
STREAM_F=$!
log "Stream F PID=$STREAM_F"

log "4个新 stream 全部启动，等待完成..."
wait $STREAM_C && log "✅ Stream C done" || log "✗ Stream C failed"
wait $STREAM_D && log "✅ Stream D done" || log "✗ Stream D failed"
wait $STREAM_E && log "✅ Stream E done" || log "✗ Stream E failed"
wait $STREAM_F && log "✅ Stream F done" || log "✗ Stream F failed"

log "====== parallel_boost.sh 全部完成 ======"
