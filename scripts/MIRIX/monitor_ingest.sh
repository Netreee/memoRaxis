#!/bin/bash
# MIRIX Ingest 监控脚本
# 每隔一段时间检查 dispatcher 进度和 MIRIX 健康状况
# 用法: bash scripts/MIRIX/monitor_ingest.sh

API_KEY="b37k12KQiS2XQlpnH7HfK5Hzj2_R61e4u0_d2ibfQao"
BASE_URL="http://localhost:8531"
LOG_DIR="logs"
CKPT_DIR="checkpoints/mirix_ingest"

echo "========================================"
echo "  MIRIX Ingest Monitor"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 1. MIRIX 健康检查
echo -e "\n[Health Check]"
health=$(curl -s "$BASE_URL/health" 2>/dev/null)
if echo "$health" | grep -q "healthy"; then
    echo "  MIRIX API: ✓ healthy"
else
    echo "  MIRIX API: ✗ UNHEALTHY ($health)"
fi

# Docker 资源
echo -e "\n[Docker Resources]"
docker stats mirix_api --no-stream --format "  CPU: {{.CPUPerc}} | MEM: {{.MemUsage}} ({{.MemPerc}})" 2>/dev/null

# 2. Checkpoint 统计
echo -e "\n[Checkpoint Summary]"
for ds in conflict_resolution test_time_learning accurate_retrieval; do
    done_count=$(ls "$CKPT_DIR"/${ds}_*.json 2>/dev/null | xargs grep -l '"done"' 2>/dev/null | wc -l | tr -d ' ')
    total=$(ls "$CKPT_DIR"/${ds}_*.json 2>/dev/null | wc -l | tr -d ' ')
    feeding=$(ls "$CKPT_DIR"/${ds}_*.json 2>/dev/null | xargs grep -l '"feeding"' 2>/dev/null | wc -l | tr -d ' ')
    echo "  $ds: done=$done_count, feeding=$feeding, total_ckpt=$total"
done

# 3. 最新日志行
echo -e "\n[Latest Log Lines]"
for log in "$LOG_DIR"/mirix_ingest_*.log; do
    [ -f "$log" ] || continue
    ds=$(basename "$log" .log | sed 's/mirix_ingest_//')
    last=$(tail -1 "$log")
    echo "  [$ds] $last"
done

# 4. Trace 统计 (每个活跃 user)
echo -e "\n[Trace Summary]"
for prefix in mirix_conf_res mirix_ttl mirix_acc_ret; do
    for i in $(seq 0 25); do
        uid="${prefix}_${i}"
        completed=$(curl -s "$BASE_URL/memory/queue-traces?user_id=$uid&status=completed&limit=1" -H "X-API-Key: $API_KEY" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else 0)" 2>/dev/null)
        if [ "$completed" -gt 0 ] 2>/dev/null; then
            failed=$(curl -s "$BASE_URL/memory/queue-traces?user_id=$uid&status=failed&limit=200" -H "X-API-Key: $API_KEY" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else 0)" 2>/dev/null)
            processing=$(curl -s "$BASE_URL/memory/queue-traces?user_id=$uid&status=processing&limit=200" -H "X-API-Key: $API_KEY" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else 0)" 2>/dev/null)
            [ "$failed" -gt 0 ] 2>/dev/null && flag=" ⚠️ FAILED=$failed" || flag=""
            [ "$processing" -gt 0 ] 2>/dev/null && pflag=" 🔄processing=$processing" || pflag=""
            echo "  $uid: completed=$completed$pflag$flag"
        fi
    done
done

# 5. 检查 dispatcher 进程是否在跑
echo -e "\n[Process]"
dpid=$(pgrep -f "dispatcher.py" 2>/dev/null)
if [ -n "$dpid" ]; then
    echo "  dispatcher PID: $dpid ✓ running"
else
    echo "  dispatcher: NOT running"
fi

echo -e "\n========================================"
