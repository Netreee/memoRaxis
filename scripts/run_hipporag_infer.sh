#!/bin/bash
# Launch HippoRAG infer for all 4 benchmark tasks in parallel background processes
# AR/CR/TTL/LR indices (standard ranges) confirmed good before running this script.

set -euo pipefail

VENV_PY=/data00/home/ziqian/proj/memoRaxis/.venv/bin/python
LOG_DIR=/data00/home/ziqian/proj/memoRaxis/logs
SCRIPT=/data00/home/ziqian/proj/memoRaxis/scripts/HippoRAG/infer.py
cd /data00/home/ziqian/proj/memoRaxis

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TS] === HippoRAG infer launch ==="

launch_if_not_running() {
    local task=$1 idx=$2 log=$3
    if pgrep -f "HippoRAG/infer.py.*${task}" > /dev/null 2>&1; then
        echo "[$task] already running, skip"
        return
    fi
    nohup $VENV_PY $SCRIPT --task "$task" --instance_idx "$idx" --adaptor all \
        >> "$LOG_DIR/$log" 2>&1 &
    echo "[$task] launched PID=$! log=$log"
}

launch_if_not_running Accurate_Retrieval       0-21  hippo_infer_ar.log
launch_if_not_running Conflict_Resolution      0-7   hippo_infer_cr.log
launch_if_not_running Test_Time_Learning       1-5   hippo_infer_ttl.log
launch_if_not_running Long_Range_Understanding 0-39  hippo_infer_lr.log

echo "[$(date '+%H:%M:%S')] All 4 tasks launched."
