#!/bin/bash
# 自动定时监控脚本
# 每 30 分钟跑一次 monitor, 结果追加到 logs/mirix_auto_check.log
# 用法: nohup bash scripts/MIRIX/auto_check.sh &

LOG="logs/mirix_auto_check.log"
INTERVAL=1800  # 30 分钟

echo "自动监控启动: $(date)" >> "$LOG"
echo "检查间隔: ${INTERVAL}s" >> "$LOG"

while true; do
    echo -e "\n\n" >> "$LOG"
    bash scripts/MIRIX/monitor_ingest.sh >> "$LOG" 2>&1

    # 检查 dispatcher 是否还在运行
    if ! pgrep -f "dispatcher.py" > /dev/null 2>&1; then
        echo "[AUTO_CHECK] ⚠️ dispatcher 进程已退出 @ $(date)" >> "$LOG"
        # 不退出, 继续监控以便查看最终状态
    fi

    # 检查是否有大规模失败
    fail_count=$(ls checkpoints/mirix_ingest/*.json 2>/dev/null | xargs grep -l '"failed": [1-9]' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$fail_count" -gt 3 ]; then
        echo "[AUTO_CHECK] ⚠️ 大规模失败: ${fail_count} 个 instance 有失败 @ $(date)" >> "$LOG"
    fi

    sleep $INTERVAL
done
