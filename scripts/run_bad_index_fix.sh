#!/bin/bash
# Sequential rebuild of bad indices (e=0 graphs)
# Order: small first, long_range_9 (biggest) last
# Runs on top of AR+LR, keeping total concurrent processes ≤ 3

VENV_PY=/data00/home/ziqian/proj/memoRaxis/.venv/bin/python
LOG_DIR=/data00/home/ziqian/proj/memoRaxis/logs
IDX_DIR=/data00/home/ziqian/proj/memoRaxis/out/hipporag/indices
SCRIPT=/data00/home/ziqian/proj/memoRaxis/scripts/HippoRAG/ingest.py

run_fix() {
    local dataset=$1
    local idx=$2
    local dir_name=$3
    echo "[fix] $(date '+%H:%M:%S') Starting $dir_name (dataset=$dataset inst=$idx)"
    rm -rf "${IDX_DIR}/${dir_name}"
    $VENV_PY $SCRIPT --dataset $dataset --instance_idx $idx >> "${LOG_DIR}/hippo_fix_${dir_name}.log" 2>&1
    echo "[fix] $(date '+%H:%M:%S') Done $dir_name"
}

echo "[fix] $(date '+%H:%M:%S') === Bad index fix sequence started ==="

# 1. Smallest first
run_fix test_time_learning    5 hipporag_ttl_5
run_fix long_range_understanding 4 hipporag_long_range_4
run_fix conflict_resolution   5 hipporag_conflict_5
run_fix conflict_resolution   6 hipporag_conflict_6

# 2. Biggest last (long_range_9, ~98 chunks)
run_fix long_range_understanding 9 hipporag_long_range_9

echo "[fix] $(date '+%H:%M:%S') === All done ==="
