"""
LR (Long Range Understanding) LLM-as-Judge — 全系统评测
使用 DeepSeek API，对 simpleMem / mem0 / mem0g / memGPT 的 LR 结果统一评分。

Ground truth 直接嵌在各 result 文件的 ground_truth 字段中。

Usage:
  python3 scripts/lr_judge_all.py [--dry-run] [--system simpleMem] [--adaptor R2]
"""

import json
import glob
import os
import re
import time
import argparse
import numpy as np
from pathlib import Path
from openai import OpenAI

DEEPSEEK_API_KEY = "sk-31bbfe45316a4672aafca84e8b9828c9"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

JUDGE_PROMPT = """你是一个严谨的评测裁判，负责评估长文本摘要任务中Agent回答的质量。

【问题】: {question}
【标准答案（关键内容要点）】: {reference}
【Agent预测】: {prediction}

评估标准：
1. 得 1 分：Agent 的回答包含了标准答案中的核心情节、人物关系和主要事件，整体内容准确。细节可以不完全一致，但核心内容不能缺失或错误。
2. 得 0 分：Agent 回答严重偏离标准答案（如主要人物错误、核心情节缺失或颠倒），或回答"无法作答"/"信息不足"。

请直接输出 JSON 格式：
{{"score": 1, "reason": "简短理由"}} 或 {{"score": 0, "reason": "简短理由"}}
不要输出任何其他内容。"""


def get_lr_files(system: str):
    """返回 {instance_idx: {adaptor: filepath}} 的映射"""
    file_map = {}

    if system == "simpleMem":
        for f in glob.glob("out/simpleMemory_MAB/results/long_range_results_*.json"):
            idx = int(f.split("long_range_results_")[1].split(".")[0])
            file_map[idx] = {"merged": f}  # R1/R2/R3 all in one file

    elif system == "mem0":
        for f in glob.glob("out/mem0/mem0_long_range_results_*.json"):
            fname = os.path.basename(f)
            # skip _r2, _r3 files in this pass
            if "_r2" in fname or "_r3" in fname:
                continue
            idx = int(fname.replace("mem0_long_range_results_", "").replace(".json", ""))
            if idx not in file_map:
                file_map[idx] = {}
            file_map[idx]["R1"] = f
        for suffix, ad in [("_r2", "R2"), ("_r3", "R3")]:
            for f in glob.glob(f"out/mem0/mem0_long_range_results_*{suffix}.json"):
                fname = os.path.basename(f)
                idx = int(fname.replace("mem0_long_range_results_", "").replace(f"{suffix}.json", ""))
                if idx not in file_map:
                    file_map[idx] = {}
                file_map[idx][ad] = f

    elif system == "mem0g":
        for f in glob.glob("out/mem0g/mem0g_long_range_results_*.json"):
            idx = int(f.split("mem0g_long_range_results_")[1].split(".")[0])
            file_map[idx] = {"merged": f}

    elif system == "memGPT":
        # R1: _memgpt.json (no suffix), R2: _memgpt_r2.json, R3: _memgpt_r3.json
        # Note: "_mgpt_" (old Mac runs) vs "_memgpt" (canonical) — filter by exact suffix
        for f in glob.glob("out/long_range_results_*_memgpt.json"):
            if "_mgpt_" in f or "_r1fix" in f:
                continue
            idx = int(f.split("long_range_results_")[1].split("_memgpt")[0])
            if idx not in file_map:
                file_map[idx] = {}
            file_map[idx]["R1"] = f
        for ad, pattern in [("R2", "out/long_range_results_*_memgpt_r2.json"),
                             ("R3", "out/long_range_results_*_memgpt_r3.json")]:
            for f in glob.glob(pattern):
                if "_mgpt_" in f:
                    continue
                idx = int(f.split("long_range_results_")[1].split("_memgpt")[0])
                if idx not in file_map:
                    file_map[idx] = {}
                file_map[idx][ad] = f

    return file_map


def load_items(filepath: str, adaptor: str):
    """从文件加载指定 adaptor 的 (question, answer, ground_truth) 列表"""
    data = json.load(open(filepath))
    results = data.get("results", data)

    # merged file (simpleMem/mem0g): results has R1/R2/R3 keys
    if adaptor in results and isinstance(results[adaptor], list):
        return results[adaptor]

    # per-adaptor file: results might directly be the list, or nested
    for key in [adaptor, "R1", "R2", "R3"]:
        if key in results and isinstance(results[key], list):
            return results[key]

    return []


def judge_one(client, question, reference, prediction, dry_run=False):
    if dry_run:
        return {"score": 1, "reason": "dry-run"}
    if not prediction or len(str(prediction).strip()) < 10:
        return {"score": 0, "reason": "空或过短的预测"}

    prompt = JUDGE_PROMPT.format(
        question=str(question)[:300],
        reference=str(reference)[:500],
        prediction=str(prediction)[:2000],
    )

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            # strip markdown code block if present
            text = re.sub(r"^```json?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            result = json.loads(text)
            result["score"] = int(result.get("score", 0))
            return result
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f"  [Judge] attempt {attempt+1} failed: {e} — wait {wait}s")
            time.sleep(wait)

    return {"score": 0, "reason": "judge failed after retries"}


def eval_lr_system(system: str, filter_adaptor=None, dry_run=False):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    file_map = get_lr_files(system)

    print(f"\n{'='*60}")
    print(f"  LR Judge — {system} ({len(file_map)} instances)")
    print(f"{'='*60}")

    all_scores = {}  # adaptor -> list of per-instance accuracy

    out_dir = Path("out/eval/lr")
    out_dir.mkdir(parents=True, exist_ok=True)

    # GT cache: load from preview_samples when ground_truth not embedded in result
    gt_cache = {}
    def get_gt(idx):
        if idx not in gt_cache:
            gt_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{idx}.json"
            try:
                d = json.load(open(gt_path))
                qa = list(zip(d["questions"], d["answers"]))
                gt_cache[idx] = {q: a for q, a in qa}
            except Exception:
                gt_cache[idx] = {}
        return gt_cache[idx]

    for idx in sorted(file_map.keys()):
        entry = file_map[idx]

        # figure out which adaptors to evaluate
        if "merged" in entry:
            adaptors_to_check = ["R1", "R2", "R3"]
            filepath_map = {ad: entry["merged"] for ad in adaptors_to_check}
        else:
            adaptors_to_check = [ad for ad in ["R1", "R2", "R3"] if ad in entry]
            filepath_map = {ad: entry[ad] for ad in adaptors_to_check}

        if filter_adaptor:
            adaptors_to_check = [a for a in adaptors_to_check if a == filter_adaptor]

        for ad in adaptors_to_check:
            items = load_items(filepath_map[ad], ad)
            if not items:
                continue

            # Check for existing eval to resume
            eval_cache_path = out_dir / f"lr_eval_{system}_inst{idx}_{ad}.json"
            cached = {}
            if eval_cache_path.exists():
                try:
                    cached = json.load(open(eval_cache_path))
                    if cached.get("completed"):
                        score = cached["accuracy"]
                        if ad not in all_scores:
                            all_scores[ad] = []
                        all_scores[ad].append(score)
                        print(f"  inst{idx:<3} | {ad} | {score:.2%} (cached)")
                        continue
                except Exception:
                    cached = {}

            total, correct = 0, 0
            details = []
            for i, item in enumerate(items):
                q = item.get("question", "")
                pred = item.get("answer", "")
                gt = item.get("ground_truth", [])
                if isinstance(gt, list):
                    ref = gt[0] if gt else ""
                else:
                    ref = str(gt) if gt else ""

                # Fallback: load GT from preview_samples if not embedded
                if not ref:
                    q_gt_map = get_gt(idx)
                    ref_list = q_gt_map.get(q, [])
                    ref = ref_list[0] if isinstance(ref_list, list) and ref_list else str(ref_list)

                if not ref:
                    continue

                result = judge_one(client, q, ref, pred, dry_run=dry_run)
                score_val = result.get("score", 0)
                correct += score_val
                total += 1
                details.append({
                    "q_idx": i,
                    "question": str(q)[:100],
                    "score": score_val,
                    "reason": result.get("reason", ""),
                })
                print(f"  inst{idx} {ad} Q{i+1}/{len(items)}: score={score_val}", flush=True)

            acc = correct / total if total > 0 else 0.0
            if ad not in all_scores:
                all_scores[ad] = []
            all_scores[ad].append(acc)
            print(f"  inst{idx:<3} | {ad} | {acc:.2%} ({correct}/{total})")

            # Save per-instance eval result
            json.dump({"system": system, "instance_idx": idx, "adaptor": ad,
                       "accuracy": acc, "correct": correct, "total": total,
                       "completed": True, "details": details},
                      open(eval_cache_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n  --- {system} LR Average ---")
    summary = {}
    for ad in ["R1", "R2", "R3"]:
        if ad in all_scores and all_scores[ad]:
            avg = np.mean(all_scores[ad])
            print(f"  {ad}: {avg:.2%} (n={len(all_scores[ad])})")
            summary[ad] = {"accuracy": float(avg), "n_instances": len(all_scores[ad])}
        else:
            print(f"  {ad}: -")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", nargs="+",
                        default=["simpleMem", "mem0", "mem0g", "memGPT"],
                        help="Systems to evaluate")
    parser.add_argument("--adaptor", default=None, choices=["R1", "R2", "R3"],
                        help="Only evaluate this adaptor")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip actual LLM calls, return score=1 for all")
    args = parser.parse_args()

    all_summary = {}
    for system in args.system:
        summary = eval_lr_system(system, filter_adaptor=args.adaptor, dry_run=args.dry_run)
        all_summary[system] = summary

    # Final table
    print(f"\n{'='*70}")
    print(f"  LR JUDGE FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'System':<12} | {'R1':<12} | {'R2':<12} | {'R3':<12}")
    print("-" * 55)
    for system in args.system:
        s = all_summary.get(system, {})
        r1 = f"{s['R1']['accuracy']:.2%} (n={s['R1']['n_instances']})" if "R1" in s else "-"
        r2 = f"{s['R2']['accuracy']:.2%} (n={s['R2']['n_instances']})" if "R2" in s else "-"
        r3 = f"{s['R3']['accuracy']:.2%} (n={s['R3']['n_instances']})" if "R3" in s else "-"
        print(f"{system:<12} | {r1:<12} | {r2:<12} | {r3:<12}")

    # Save aggregated results
    out_path = "out/eval/lr_judge_summary.json"
    json.dump(all_summary, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
