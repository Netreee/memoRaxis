#!/usr/bin/env python3
"""MIRIX infer dispatcher — 4-parallel, AR+CR+LR only (TTL skipped), embedding retrieval."""
import subprocess, time, sys
from datetime import datetime
from pathlib import Path
WORKERS = 4
ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")
INFER  = str(Path(__file__).parent / "infer.py")
LOG_DIR = ROOT / "logs/mirix_infer_emb"
DISPATCH_LOG = ROOT / "logs/mirix_infer_emb_dispatch.log"
JOBS = (
    [("Accurate_Retrieval",       i) for i in range(22)] +
    [("Conflict_Resolution",      i) for i in range(8)]  +
    [("Long_Range_Understanding", i) for i in range(40)]
    
)  # 70 jobs total (TTL skipped)
def log(msg):
    line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(DISPATCH_LOG, "a") as f:
        f.write(line + "\n")
def start(task, idx):
    log_file = LOG_DIR / f"{task}_{idx}.log"
    proc = subprocess.Popen(
        [PYTHON, INFER, "--task", task, "--adaptor", "R1", "R2", "R3",
         "--instance_idx", str(idx)],
        stdout=open(log_file, "w"), stderr=subprocess.STDOUT, cwd=str(ROOT)
    )
    log(f"START  {task:<35} inst={idx:<3} PID={proc.pid}")
    return proc, task, idx
def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"=== MIRIX infer dispatcher: {len(JOBS)} jobs, {WORKERS} workers ===")
    queue = list(JOBS)
    active, done = [], 0
    while queue or active:
        while queue and len(active) < WORKERS:
            active.append(start(*queue.pop(0)))
        time.sleep(5)
        still = []
        for proc, task, idx in active:
            rc = proc.poll()
            if rc is None:
                still.append((proc, task, idx))
            else:
                done += 1
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                log(f"DONE   {task:<35} inst={idx:<3} {status}  [{done}/{len(JOBS)}]")
        active = still
    log(f"=== All {len(JOBS)} jobs finished ===")
if __name__ == "__main__":
    main()
