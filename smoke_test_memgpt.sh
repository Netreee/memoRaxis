#!/usr/bin/env bash
# MemGPT Smoke Test: ingest + infer + quick eval for each dataset (1 instance each)
# Run from memoRaxis root: bash smoke_test_memgpt.sh
set -e
cd "$(dirname "$0")"

LOG=gaoang/out/smoke_test.log
mkdir -p gaoang/out
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo " MemGPT Smoke Test  $(date)"
echo "=========================================="

INGEST_DIR=gaoang/scripts/simpleMemory_MAB/ingest
INFER_DIR=gaoang/scripts/simpleMemory_MAB/infer
EVAL_DIR=gaoang/scripts/simpleMemory_MAB/evaluate

# ── 1. Conflict Resolution (inst_0, ~32 chunks, 5 questions) ──────────────────
echo ""
echo ">>> [1/4] Conflict_Resolution inst_0"
echo "--- INGEST ---"
python3 $INGEST_DIR/ingest_conflict_resolution.py --instance_idx 0

echo "--- INFER R1 (limit 5) ---"
python3 $INFER_DIR/infer_conflict_resolution.py \
    --instance_idx 0 --adaptor R1 --limit 5 --output_suffix memgpt

echo "--- QUICK EVAL ---"
python3 - <<'PYEOF'
import json, glob
files = sorted(glob.glob("gaoang/out/conflict_res_results_0_memgpt.json"))
if not files:
    print("  No result file found for CR inst_0")
else:
    d = json.load(open(files[-1]))
    for adaptor, rows in d.get("results", {}).items():
        answered = [r for r in rows if "answer" in r and not r.get("error")]
        errors   = [r for r in rows if r.get("error")]
        print(f"  {adaptor}: answered={len(answered)}, errors={len(errors)}")
        for r in answered[:2]:
            print(f"    Q: {r['question'][:60]}")
            print(f"    A: {str(r['answer'])[:80]}")
            print(f"    GT: {str(r['ground_truth'])[:80]}")
PYEOF

# ── 2. Accurate Retrieval (inst_10, max_chunks=30, 5 questions) ───────────────
echo ""
echo ">>> [2/4] Accurate_Retrieval inst_10 (max_chunks=30)"
echo "--- INGEST ---"
python3 $INGEST_DIR/ingest_accurate_retrieval.py --instance_idx 10 --max_chunks 30

echo "--- INFER R1 (limit 5) ---"
python3 $INFER_DIR/infer_accurate_retrieval.py \
    --instance_idx 10 --adaptor R1 --limit 5 --output_suffix memgpt

echo "--- QUICK EVAL ---"
python3 - <<'PYEOF'
import json, glob
files = sorted(glob.glob("gaoang/out/acc_ret_results_10_memgpt.json"))
if not files:
    print("  No result file found for AR inst_10")
else:
    d = json.load(open(files[-1]))
    for adaptor, rows in d.get("results", {}).items():
        answered = [r for r in rows if "answer" in r and not r.get("error")]
        errors   = [r for r in rows if r.get("error")]
        print(f"  {adaptor}: answered={len(answered)}, errors={len(errors)}")
        for r in answered[:2]:
            print(f"    Q: {r['question'][:60]}")
            print(f"    A: {str(r['answer'])[:80]}")
PYEOF

# ── 3. Long Range Understanding (inst_1, max_chunks=50, 1 question) ───────────
echo ""
echo ">>> [3/4] Long_Range_Understanding inst_1 (max_chunks=50)"
echo "--- INGEST ---"
python3 $INGEST_DIR/ingest_long_range.py --instance_idx 1 --max_chunks 50

echo "--- INFER R1 (1 question) ---"
python3 $INFER_DIR/infer_long_range.py \
    --instance_idx 1 --adaptor R1 --output_suffix memgpt

echo "--- QUICK EVAL ---"
python3 - <<'PYEOF'
import json, glob
files = sorted(glob.glob("gaoang/out/long_range_results_1_memgpt.json"))
if not files:
    print("  No result file found for LR inst_1")
else:
    d = json.load(open(files[-1]))
    for adaptor, rows in d.get("results", {}).items():
        answered = [r for r in rows if "answer" in r and not r.get("error")]
        errors   = [r for r in rows if r.get("error")]
        print(f"  {adaptor}: answered={len(answered)}, errors={len(errors)}")
        for r in answered[:1]:
            print(f"    Q: {r['question'][:80]}")
            print(f"    A: {str(r['answer'])[:120]}")
            print(f"    GT: {str(r['ground_truth'])[:120]}")
PYEOF

# ── 4. Test Time Learning (inst_1, max_chunks=30, 3 questions) ────────────────
echo ""
echo ">>> [4/4] Test_Time_Learning inst_1 (max_chunks=30)"
echo "--- INGEST ---"
python3 $INGEST_DIR/ingest_test_time.py --instance_idx 1 --max_chunks 30

echo "--- INFER R1 (limit 3) ---"
python3 $INFER_DIR/infer_test_time.py \
    --instance_idx 1 --adaptor R1 --limit 3 --output_suffix memgpt

echo "--- QUICK EVAL ---"
python3 - <<'PYEOF'
import json, glob
files = sorted(glob.glob("gaoang/out/ttl_results_1_memgpt.json"))
if not files:
    print("  No result file found for TTL inst_1")
else:
    d = json.load(open(files[-1]))
    for adaptor, rows in d.get("results", {}).items():
        answered = [r for r in rows if "answer" in r and not r.get("error")]
        errors   = [r for r in rows if r.get("error")]
        print(f"  {adaptor}: answered={len(answered)}, errors={len(errors)}")
        for r in answered[:2]:
            print(f"    Q: {r['question'][:60]}")
            print(f"    A: {str(r['answer'])[:80]}")
            print(f"    GT: {str(r['ground_truth'])[:80]}")
PYEOF

echo ""
echo "=========================================="
echo " Smoke Test DONE  $(date)"
echo "=========================================="
