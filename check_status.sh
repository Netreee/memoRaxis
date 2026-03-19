#!/usr/bin/env bash
cd /Users/bytedance/proj/memoRaxis
TS="[$(date '+%m-%d %H:%M')]"

echo ""
echo "$TS ========== STATUS CHECK =========="

# 进程
echo "--- 进程 ---"
ps aux | grep -E "(python3|run_plan)" | grep -v grep | grep -v "check_status" | \
  awk '{printf "  PID=%s CPU=%s MEM=%s %s\n", $2, $3, $4, substr($0, index($0,$11))}'

# PHASE 0 进度
echo "--- PHASE 0 simpleMemory ---"
python3 -c "
import json
tasks = [
  ('out/ttl_results_5.json', 'TTL inst5', 'R1', 100),
  ('out/simpleMemory_MAB/results/ttl_results_0.json', 'TTL inst0', 'R3', 200),
]
for path, label, adaptor, total in tasks:
    try:
        d = json.load(open(path))
        rows = d.get('results',{}).get(adaptor, [])
        if isinstance(rows, list):
            print(f'  {label} {adaptor}: {len(rows)}/{total}')
        else:
            print(f'  {label} {adaptor}: 格式异常')
    except: print(f'  {label}: 未写出')

for i in [17,18,19,20,21]:
    try:
        d = json.load(open(f'out/simpleMemory_MAB/results/acc_ret_results_{i}.json'))
        for a, rows in d.get('results',{}).items():
            if isinstance(rows, list) and len(rows) > 60:
                print(f'  SM AR inst{i} {a}: {len(rows)}/100')
    except: pass
" 2>/dev/null

# PHASE 1 MemGPT AR 进度
echo "--- PHASE 1 MemGPT AR (results/memgpt/) ---"
done=$(ls results/memgpt/ar_*.json 2>/dev/null | wc -l)
echo "  已收录: ${done}/66"
ls results/memgpt/ar_*.json 2>/dev/null | xargs -I{} basename {} .json | sort

# 当前 gaoang log 最新活动
echo "--- 最新 gaoang 日志 ---"
latest=$(ls -t gaoang/log/*.log 2>/dev/null | head -1)
if [ -n "$latest" ]; then
    echo "  $latest"
    tail -3 "$latest"
fi

echo "--- proxy ---"
curl -sf http://127.0.0.1:8284/v1/models >/dev/null 2>&1 && echo "  ✅ proxy 正常" || echo "  ❌ proxy 异常!"
echo "=================================="
