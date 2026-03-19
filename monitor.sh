#!/usr/bin/env bash
cd /Users/bytedance/proj/memoRaxis
TS="$(date '+%m-%d %H:%M')"
LOG=logs/monitor.log

{
echo ""
echo "[$TS] ===== MONITOR ====="

# 进程健康
procs=$(ps aux | grep -E "(python3|run_plan)" | grep -v grep | grep -v monitor | wc -l)
echo "[$TS] 活跃进程: $procs"
ps aux | grep -E "(python3.*infer|python3.*run_plan|bash.*run_plan)" | grep -v grep | \
  awk -v ts="[$TS]" '{print ts" ",$2,$11,$12,$13}'

# Proxy
curl -sf http://127.0.0.1:8284/v1/models >/dev/null 2>&1 \
  && echo "[$TS] proxy: ✅" \
  || echo "[$TS] proxy: ❌ ALARM"

# results/ 收录数
ar=$(ls results/memgpt/ar_*.json 2>/dev/null | wc -l | tr -d ' ')
cr=$(ls results/memgpt/cr_*.json 2>/dev/null | wc -l | tr -d ' ')
lr=$(ls results/memgpt/lr_*.json 2>/dev/null | wc -l | tr -d ' ')
echo "[$TS] memgpt results: AR=$ar/66 CR=$cr/24 LR=$lr/120"

# PHASE 0 进度 (simpleMemory)
python3 -c "
import json
tasks = [
  ('out/ttl_results_5.json', 'TTL5-R1', 'R1', 100),
  ('out/simpleMemory_MAB/results/ttl_results_0.json', 'TTL0-R3', 'R3', 200),
]
for path, label, adaptor, total in tasks:
    try:
        rows = json.load(open(path)).get('results',{}).get(adaptor,[])
        print(f'[$TS] {label}: {len(rows)}/{total}')
    except: print(f'[$TS] {label}: 未写出')
for i in [17,18,19,20,21]:
    try:
        d = json.load(open(f'out/simpleMemory_MAB/results/acc_ret_results_{i}.json'))
        total = min(len(rows) for a, rows in d.get('results',{}).items() if isinstance(rows,list))
        print(f'[$TS] SM-AR-inst{i}: {total}/100')
    except: pass
" TS="$TS" 2>/dev/null | sed "s/\['\$TS'\]/[$TS]/g"

# 最新 gaoang log 活动时间
latest=$(ls -t gaoang/log/*.log 2>/dev/null | head -1)
if [ -n "$latest" ]; then
    last_line=$(tail -1 "$latest")
    echo "[$TS] gaoang最新: $(basename $latest) → $last_line"
fi

echo "[$TS] run_plan log末行: $(tail -1 logs/run_plan_master.log 2>/dev/null)"
} | tee -a "$LOG"
