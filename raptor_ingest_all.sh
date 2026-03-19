#!/usr/bin/env bash
# raptor_ingest_all.sh — RAPTOR 全量 ingest 调度脚本
#
# 并行策略: 3 个进程组并发，每组内串行跑实例，每个实例内部 4-worker 并行 summarization
# 峰值并发 LLM 请求: 3x4 = 12，约 43K TPM，留 30%+ 缓冲
#
# 分组（按估算 summarization 调用数均衡）:
#   Group A: CR 4-7 + AR 0-6          ~4,642 calls
#   Group B: AR 15-21 + LR 0-12       ~6,103 calls
#   Group C: LR 13-39                 ~6,304 calls
#
# 已完成的 tree 会自动跳过（ingest.py 内置 skip 逻辑）
# 预估总耗时: ~22h（受限于最慢的 Group C）
#
# 用法:
#   nohup bash raptor_ingest_all.sh > raptor_ingest_all.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
INGEST="scripts/RAPTOR/ingest.py"
LOGDIR="logs/raptor_ingest"
mkdir -p "$LOGDIR"

T_START=$(date +%s)
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "======================================================"
log "  RAPTOR 全量 ingest 启动"
log "  3 进程组 x 4 workers/实例"
log "======================================================"

# ── Group A: CR 4-7 → AR 0-6 ─────────────────────────────
run_group_a() {
    log "[A] 开始: CR 4-7"
    $PYTHON $INGEST --dataset conflict_resolution --instance_idx 4-7 \
        2>&1 | tee "$LOGDIR/group_a_cr.log"

    log "[A] 开始: AR 0-6"
    $PYTHON $INGEST --dataset accurate_retrieval --instance_idx 0-6 \
        2>&1 | tee "$LOGDIR/group_a_ar.log"

    log "[A] 全部完成"
}

# ── Group B: AR 15-21 → LR 0-12 ──────────────────────────
run_group_b() {
    log "[B] 开始: AR 15-21"
    $PYTHON $INGEST --dataset accurate_retrieval --instance_idx 15-21 \
        2>&1 | tee "$LOGDIR/group_b_ar.log"

    log "[B] 开始: LR 0-12"
    $PYTHON $INGEST --dataset long_range_understanding --instance_idx 0-12 \
        2>&1 | tee "$LOGDIR/group_b_lr.log"

    log "[B] 全部完成"
}

# ── Group C: LR 13-39 ────────────────────────────────────
run_group_c() {
    log "[C] 开始: LR 13-39"
    $PYTHON $INGEST --dataset long_range_understanding --instance_idx 13-39 \
        2>&1 | tee "$LOGDIR/group_c_lr.log"

    log "[C] 全部完成"
}

# ── 启动 3 个进程组 ──────────────────────────────────────
run_group_a > "$LOGDIR/group_a.log" 2>&1 &
PID_A=$!
log "Group A 启动 (PID=$PID_A): CR 4-7, AR 0-6"

run_group_b > "$LOGDIR/group_b.log" 2>&1 &
PID_B=$!
log "Group B 启动 (PID=$PID_B): AR 15-21, LR 0-12"

run_group_c > "$LOGDIR/group_c.log" 2>&1 &
PID_C=$!
log "Group C 启动 (PID=$PID_C): LR 13-39"

log ""
log "等待所有进程组完成..."
log "监控: tail -f $LOGDIR/group_{a,b,c}.log"
log ""

# ── 等待并记录完成状态 ───────────────────────────────────
FAIL=0
for label_pid in "A:$PID_A" "B:$PID_B" "C:$PID_C"; do
    label="${label_pid%%:*}"
    pid="${label_pid##*:}"
    if wait "$pid"; then
        log "Group $label (PID=$pid) 完成"
    else
        log "Group $label (PID=$pid) 异常退出 (exit=$?)"
        FAIL=$((FAIL + 1))
    fi
done

# ── 汇总 ─────────────────────────────────────────────────
T_END=$(date +%s)
ELAPSED=$(( T_END - T_START ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

log ""
log "======================================================"
log "  汇总 (耗时 ${HOURS}h${MINS}m)"
log "======================================================"

# 统计已有 tree 数量
$PYTHON -c "
from pathlib import Path
tree_dir = Path('out/raptor/trees')
for pattern, label, target in [
    ('raptor_conflict_*.pkl',    'CR  (0-7) ', 8),
    ('raptor_acc_ret_*.pkl',     'AR  (0-21)', 22),
    ('raptor_ttl_*.pkl',         'TTL (1-5) ', 5),
    ('raptor_long_range_*.pkl',  'LR  (0-39)', 40),
]:
    found = sorted(tree_dir.glob(pattern))
    status = 'DONE' if len(found) >= target else 'PARTIAL'
    print(f'  {label}: {len(found):>2}/{target} trees  [{status}]')
    if len(found) < target:
        # show which are missing
        import re
        existing = set()
        for f in found:
            m = re.search(r'_(\d+)\.pkl$', f.name)
            if m: existing.add(int(m.group(1)))
        if label.startswith('TTL'):
            expected = set(range(1, target+1))
        else:
            expected = set(range(target))
        missing = sorted(expected - existing)
        if missing:
            print(f'           missing: {missing}')
"

# 提取各实例的 ingest summary
log ""
log "--- 各实例耗时 & token 统计 ---"
grep -h '\[Ingest Summary\]' "$LOGDIR"/group_*.log 2>/dev/null | sort || log "(无 summary 记录)"

if [ $FAIL -gt 0 ]; then
    log ""
    log "WARNING: $FAIL 个进程组异常退出，检查日志: $LOGDIR/group_*.log"
fi

log ""
log "完成 $(date)"
