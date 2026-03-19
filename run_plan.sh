#!/usr/bin/env bash
# run_plan.sh — 革新版，规避所有已知坑
# 坑列表:
#  [P1] Embedding proxy 必须在跑 (port 8284)
#  [P2] gaoang AR/LR/TTL 必须 --limit -1 (默认5题)
#  [P3] output_suffix 区分 adaptor，避免文件覆盖
#  [P4] 单 instance 单次跑（无 checkpoint，减少崩溃损失）
#  [P5] () & + $! 捕获 PID（禁止 exec>tee 污染）
#  [P6] simpleMemory 需 PYTHONPATH
#  [P7] TTL ICL inst(1-5): 只跑 R1，跳过 R2/R3
#  [P8] 完成后立即 audit → results/
#  [P9] infer 前必须确认 ingest 已完成

set -euo pipefail
cd /Users/bytedance/proj/memoRaxis

mkdir -p logs results/memgpt results/simpleMemory
LOG=logs/run_plan_$(date +%m%d_%H%M).log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======================================================"
log "  run_plan.sh  $(date)"
log "======================================================"

# ── [P1] Proxy 检查 & 自动启动 ───────────────────────────────────────
ensure_proxy() {
    curl -sf http://127.0.0.1:8284/v1/models >/dev/null 2>&1 && return 0
    log "[PROXY] 未运行，启动中..."
    nohup python3 gaoang/embedding_proxy.py >> logs/embedding_proxy.log 2>&1 &
    sleep 3
    curl -sf http://127.0.0.1:8284/v1/models >/dev/null 2>&1 || { log "[PROXY] 启动失败！"; exit 1; }
    log "[PROXY] ✅ 已启动"
}

# ── [P8] Audit 函数 ──────────────────────────────────────────────────
audit_memgpt() {
    # audit_memgpt <file> <ds_short> <inst> <adaptor> <expected_q>
    local file=$1 ds=$2 inst=$3 adaptor=$4 expected=${5:-100}
    [ -f "$file" ] || { log "[AUDIT] ⚠️  文件不存在: $file"; return 1; }
    read n errs < <(python3 -c "
import json,sys
d=json.load(open('$file'))
rows=d.get('results',{}).get('$adaptor',[])
print(len(rows), sum(1 for r in rows if r.get('error')))
" 2>/dev/null || echo "0 -1")
    local dst="results/memgpt/${ds}_inst$(printf '%02d' $inst)_$(echo $adaptor | tr 'A-Z' 'a-z').json"
    if [ "$n" -ge "$expected" ] && [ "$errs" -eq 0 ]; then
        cp "$file" "$dst"
        log "[AUDIT] ✅ $dst (${n}q, 0err)"
    else
        log "[AUDIT] ⚠️  不达标 ${file} (${n}q, ${errs}err)，跳过"
        return 1
    fi
}

already_done() {
    # already_done <ds_short> <inst> <adaptor>
    local dst="results/memgpt/${1}_inst$(printf '%02d' $2)_$(echo $3 | tr 'A-Z' 'a-z').json"
    [ -f "$dst" ] && { log "[SKIP] 已有 $dst"; return 0; }
    return 1
}

# ── Ingest helper ────────────────────────────────────────────────────
ingest_if_needed() {
    # ingest_if_needed <dataset_name> <inst>
    local dataset=$1 inst=$2
    local agent_prefix
    case $dataset in
        accurate_retrieval)   agent_prefix="bench_acc_ret_" ;;
        conflict_resolution)  agent_prefix="bench_conflict_" ;;
        long_range)           agent_prefix="bench_long_range_" ;;
        test_time)            agent_prefix="bench_ttl_" ;;
    esac
    # 检查 Letta agent 是否存在
    local exists
    exists=$(curl -sL "http://127.0.0.1:8283/v1/agents/?name=${agent_prefix}${inst}" 2>/dev/null | \
        python3 -c "import json,sys; d=json.load(sys.stdin); a=d if isinstance(d,list) else d.get('data',d.get('agents',[])); print('yes' if any('${agent_prefix}${inst}'==x.get('name') for x in a) else 'no')" 2>/dev/null || echo "no")
    if [ "$exists" = "yes" ]; then
        log "[INGEST] ${agent_prefix}${inst} 已存在，跳过"
        return 0
    fi
    log "[INGEST] ${agent_prefix}${inst} 不存在，开始 ingest..."
    python3 gaoang/scripts/simpleMemory_MAB/ingest/ingest_${dataset}.py \
        --instance_idx "$inst" >> "logs/ingest_${dataset}_${inst}.log" 2>&1
    log "[INGEST] ✅ ${agent_prefix}${inst} 完成"
}

# ══════════════════════════════════════════════════════════════════════
# PHASE 0: simpleMemory 补跑（有 checkpoint，安全）
# ══════════════════════════════════════════════════════════════════════
log ""
log "=== PHASE 0: simpleMemory 补跑 ==="

# [P6][P7] TTL inst5 R1 only (ICL，只有 R1 有意义)
if [ ! -f "out/ttl_results_5.json" ]; then
    log "[SM-TTL] inst5 R1 启动"
    PYTHONPATH=/Users/bytedance/proj/memoRaxis \
        python3 scripts/simpleMemory_MAB/infer/infer_test_time.py \
        --instance_idx 5 --adaptor R1 \
        >> logs/sm_ttl_5.log 2>&1
    log "[SM-TTL] ✅ inst5 R1 done"
    cp out/ttl_results_5.json results/simpleMemory/
fi

# TTL inst0 R3 resume (recsys 任务，checkpoint 存在)
log "[SM-TTL] inst0 R3 resume (recsys)"
PYTHONPATH=/Users/bytedance/proj/memoRaxis \
    python3 scripts/simpleMemory_MAB/infer/infer_test_time.py \
    --instance_idx 0 --adaptor R3 \
    >> logs/sm_ttl_0_r3.log 2>&1 &
SM_TTL0=$!

# AR inst17-21 resume (checkpoint 存在，~40题/实例)
# 检查是否已有 R1+R2+R3 完整结果
_sm_ar_done=true
for _i in 17 18 19 20 21; do
    _cnt=$(python3 -c "import json; d=json.load(open('results/simpleMemory/acc_ret_results_${_i}.json')); r=d.get('results',d); print(sum(len(v) for v in r.values() if isinstance(v,list)))" 2>/dev/null || echo 0)
    [ "${_cnt:-0}" -ge 60 ] || { _sm_ar_done=false; break; }
done
if $_sm_ar_done; then
    log "[SM-AR] inst17-21 已完成（R1+R2+R3），跳过"
    SM_AR=""
else
    log "[SM-AR] inst17-21 resume"
    PYTHONPATH=/Users/bytedance/proj/memoRaxis \
        python3 scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
        --instance_idx 17,18,19,20,21 --limit -1 \
        >> logs/sm_ar_17_21.log 2>&1 &
    SM_AR=$!
fi

wait $SM_TTL0 && log "[SM-TTL] ✅ inst0 R3 done" || log "[SM-TTL] ✗ inst0 R3 failed"
[ -n "$SM_AR" ] && { wait $SM_AR && log "[SM-AR] ✅ AR 17-21 done" || log "[SM-AR] ✗ AR 17-21 failed"; }

# 更新 results/simpleMemory/（simpleMemory 有自己的 results 目录）
cp out/simpleMemory_MAB/results/ttl_results_0.json results/simpleMemory/ 2>/dev/null || true
for i in 17 18 19 20 21; do
    cp "out/simpleMemory_MAB/results/acc_ret_results_${i}.json" results/simpleMemory/ 2>/dev/null || true
done
log "=== PHASE 0 完成 ==="

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: MemGPT AR — 先 ingest 缺失，再 infer
# ══════════════════════════════════════════════════════════════════════
log ""
log "=== PHASE 1: MemGPT AR ==="
ensure_proxy

# AR 缺失 ingest: inst 3-11
for inst in 3 4 5 6 7 8 9 10 11; do
    ingest_if_needed accurate_retrieval $inst
done

# 每个 instance 单独跑，[P2] --limit -1，[P3] suffix，[P8] audit
run_ar() {
    local inst=$1 adaptor=$2
    local suffix="mgpt_$(echo $adaptor | tr 'A-Z' 'a-z')"
    already_done "ar" $inst $adaptor && return 0
    ensure_proxy
    local out="out/acc_ret_results_${inst}_${suffix}.json"
    log "[AR] inst${inst} ${adaptor} 开始"
    python3 gaoang/scripts/simpleMemory_MAB/infer/infer_accurate_retrieval.py \
        --instance_idx "$inst" --adaptor "$adaptor" \
        --output_suffix "$suffix" --limit -1 \
        >> "logs/ar_${inst}_${suffix}.log" 2>&1
    local expected_q=100; [[ $inst -ge 17 ]] && expected_q=60
    audit_memgpt "$out" "ar" $inst $adaptor $expected_q \
        || log "[AR] ✗ inst${inst} ${adaptor}"
}

# 已完成: AR R1: 0,4,12,16  R2: 0,1,4,5,10,11,12,16  R3: 0,2,4,13
# 需补:
AR_R1_MISSING="1 2 3 5 6 7 8 9 10 11 13 14 15 17 18 19 20 21"
AR_R2_MISSING="2 3 6 7 8 9 13 14 15 17 18 19 20 21"
AR_R3_MISSING="1 3 5 6 7 8 9 10 11 12 14 15 16 17 18 19 20 21"

# [P5] 两 stream 并行（R1 stream A, R2 stream B）
(
    for inst in $AR_R1_MISSING; do run_ar $inst R1; done
    log "[A] AR R1 全部完成"
) &
STREAM_A=$!

(
    for inst in $AR_R2_MISSING; do run_ar $inst R2; done
    log "[B] AR R2 全部完成"
) &
STREAM_B=$!

wait $STREAM_A
wait $STREAM_B

# R3 单流（最慢）
for inst in $AR_R3_MISSING; do run_ar $inst R3; done

log "=== PHASE 1 完成 ==="

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: MemGPT LR — ingest 缺失 inst 8-19，再 infer
# ══════════════════════════════════════════════════════════════════════
log ""
log "=== PHASE 2: MemGPT LR ==="
ensure_proxy

for inst in 8 9 10 11 12 13 14 15 16 17 18 19; do
    ingest_if_needed long_range $inst
done

run_lr() {
    local inst=$1 adaptor=$2
    local suffix="mgpt_$(echo $adaptor | tr 'A-Z' 'a-z')"
    already_done "lr" $inst $adaptor && return 0
    ensure_proxy
    local out="out/long_range_results_${inst}_${suffix}.json"
    log "[LR] inst${inst} ${adaptor} 开始"
    python3 gaoang/scripts/simpleMemory_MAB/infer/infer_long_range.py \
        --instance_idx "$inst" --adaptor "$adaptor" \
        --output_suffix "$suffix" --limit -1 \
        >> "logs/lr_${inst}_${suffix}.log" 2>&1 \
        && audit_memgpt "$out" "lr" $inst $adaptor 1 \
        || log "[LR] ✗ inst${inst} ${adaptor}"
}

(for inst in $(seq 0 39); do run_lr $inst R1; done) &
STREAM_A=$!
(for inst in $(seq 0 39); do run_lr $inst R2; done) &
STREAM_B=$!
wait $STREAM_A; wait $STREAM_B
for inst in $(seq 0 39); do run_lr $inst R3; done

log "=== PHASE 2 完成 ==="

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: MemGPT CR R2/R3 (最重，100题/inst，放最后)
# ══════════════════════════════════════════════════════════════════════
log ""
log "=== PHASE 3: MemGPT CR R2+R3 (CR 已全部 ingest) ==="
ensure_proxy

run_cr() {
    local inst=$1 adaptor=$2
    local suffix="mgpt_$(echo $adaptor | tr 'A-Z' 'a-z')"
    already_done "cr" $inst $adaptor && return 0
    ensure_proxy
    local out="out/conflict_res_results_${inst}_${suffix}.json"
    log "[CR] inst${inst} ${adaptor} 开始"
    python3 gaoang/scripts/simpleMemory_MAB/infer/infer_conflict_resolution.py \
        --instance_idx "$inst" --adaptor "$adaptor" \
        --output_suffix "$suffix" \
        >> "logs/cr_${inst}_${suffix}.log" 2>&1 \
        && audit_memgpt "$out" "cr" $inst $adaptor 100 \
        || log "[CR] ✗ inst${inst} ${adaptor}"
}

(for inst in 0 1 2 3 4 5 6 7; do run_cr $inst R2; done) &
STREAM_A=$!
(for inst in 0 1 2 3 4 5 6 7; do run_cr $inst R3; done) &
STREAM_B=$!
wait $STREAM_A; wait $STREAM_B

log "=== PHASE 3 完成 ==="

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: MemGPT TTL (inst 0 recsys R1/R2/R3，ingest inst 1-5 可选)
# ══════════════════════════════════════════════════════════════════════
log ""
log "=== PHASE 4: MemGPT TTL inst0 (recsys) ==="
ensure_proxy

for adaptor in R1 R2 R3; do
    already_done "ttl" 0 $adaptor && continue
    local suffix="mgpt_$(echo $adaptor | tr 'A-Z' 'a-z')"
    python3 gaoang/scripts/simpleMemory_MAB/infer/infer_test_time.py \
        --instance_idx 0 --adaptor "$adaptor" \
        --output_suffix "$suffix" --limit -1 \
        >> "logs/ttl_0_${suffix}.log" 2>&1 \
        && audit_memgpt "out/ttl_results_0_${suffix}.json" "ttl" 0 $adaptor 200 \
        || log "[TTL] ✗ inst0 ${adaptor}"
done

log ""
log "======================================================"
log "  run_plan.sh 全部完成  $(date)"
log "======================================================"
