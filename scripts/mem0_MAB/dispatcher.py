import subprocess
import concurrent.futures
import time
from datetime import datetime
import sys

# ==========================================
# 任务队列配置
# ==========================================
TASKS = []

# 1. Accurate Retrieval (共 22 个)
for i in range(22):
    TASKS.append(("accurate_retrieval", i))

# 2. Conflict Resolution (共 8 个)
for i in range(8):
    TASKS.append(("conflict_resolution", i))

# 3. Long Range Understanding (共 110 个)
for i in range(110):
    TASKS.append(("long_range_understanding", i))

# 4. Test Time Learning (共 6 个)
for i in range(1,6):
    TASKS.append(("test_time_learning", i))

# ==========================================
# 并发配置
# ==========================================
MAX_WORKERS = 20

def run_task(task):
    dataset, idx = task
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] 🚀 开始执行: {dataset} [实例 {idx}]", flush=True)
    
    cmd = [
        sys.executable, "scripts/mem0_MAB/ingest.py",
        "--dataset", dataset,
        "--instance_idx", str(idx)
    ]
    
    try:
        # 隐藏标准输出以避免控制台爆炸，依赖 ingest.py 自带的 logger 写入 log 文件夹
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds())
        print(f"[{end_time.strftime('%H:%M:%S')}] ✅ 成功完成: {dataset} [实例 {idx}] (耗时 {duration} 秒)", flush=True)
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        print(f"[{end_time.strftime('%H:%M:%S')}] ❌ 执行失败: {dataset} [实例 {idx}] (错误码 {e.returncode})", flush=True)

def main():
    print(f"==================================================")
    print(f"🌟 Mem0 (纯向量) 智能并发调度器启动")
    print(f"⚙️  并发线程数: {MAX_WORKERS}")
    print(f"📋 队列总任务数: {len(TASKS)}")
    print(f"==================================================", flush=True)
    
    # 使用线程池进行调度
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_task, TASKS)
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 所有队列任务已全部执行完毕！")

if __name__ == "__main__":
    main()
