"""
全系统 Mechanical Eval — simpleMem / mem0 / mem0g
不使用 LLM，纯规则匹配，秒级完成。

评测:
  - AR: 关键词包含匹配 (evaluate_mechanical 逻辑)
  - CR: ExactMatch / SubstringMatch / F1 (evaluate_conflict_official 逻辑)
  - TTL: entity ID 匹配 (evaluate_ttl_mechanical 逻辑)
  - LR: 跳过 (需要 LLM as judge)
"""

import json
import re
import string
import glob
import os
import numpy as np
from collections import Counter
from pathlib import Path

# ==============================
# 通用工具
# ==============================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def load_merged_results(system, dataset_prefix, instance_idx):
    """加载并合并 R1/R2/R3 结果，返回统一的 {R1:[...], R2:[...], R3:[...]} 格式"""
    if system == "simpleMem":
        if dataset_prefix == "acc_ret":
            path = f"out/simpleMemory_MAB/results/acc_ret_results_{instance_idx}.json"
        elif dataset_prefix == "conflict":
            path = f"out/simpleMemory_MAB/results/conflict_res_results_{instance_idx}.json"
        elif dataset_prefix == "ttl":
            path = f"out/ttl_results_{instance_idx}.json"
        else:
            return None
        if not os.path.exists(path):
            return None
        return json.load(open(path))["results"]

    elif system == "mem0":
        merged = {}
        # R1 from base file
        r1_path = f"out/mem0/mem0_{dataset_prefix}_results_{instance_idx}.json"
        if os.path.exists(r1_path):
            d = json.load(open(r1_path))
            if "R1" in d.get("results", {}):
                merged["R1"] = d["results"]["R1"]
        # R2 from dedicated file (canonical)
        r2_path = f"out/mem0/mem0_{dataset_prefix}_results_{instance_idx}_r2.json"
        if os.path.exists(r2_path):
            d = json.load(open(r2_path))
            if "R2" in d.get("results", {}):
                merged["R2"] = d["results"]["R2"]
        # R3 from dedicated file (canonical)
        r3_path = f"out/mem0/mem0_{dataset_prefix}_results_{instance_idx}_r3.json"
        if os.path.exists(r3_path):
            d = json.load(open(r3_path))
            if "R3" in d.get("results", {}):
                merged["R3"] = d["results"]["R3"]
        return merged if merged else None

    elif system == "mem0g":
        if dataset_prefix == "acc_ret":
            path = f"out/mem0g/mem0g_acc_ret_results_{instance_idx}.json"
        elif dataset_prefix == "conflict":
            path = f"out/mem0g/mem0g_conflict_results_{instance_idx}.json"
        elif dataset_prefix == "ttl":
            path = f"out/mem0g/mem0g_ttl_results_{instance_idx}.json"
        else:
            return None
        if not os.path.exists(path):
            return None
        return json.load(open(path))["results"]

    elif system == "amem":
        task_map = {"acc_ret": "acc_ret", "conflict": "conflict", "ttl": "ttl"}
        short = task_map.get(dataset_prefix)
        if not short:
            return None
        path = f"out/amem/amem_{short}_{instance_idx}.json"
        if not os.path.exists(path):
            return None
        return json.load(open(path))["results"]

    elif system == "hipporag":
        path = f"out/hipporag/hipporag_{dataset_prefix}_{instance_idx}.json"
        if not os.path.exists(path):
            return None
        return json.load(open(path))["results"]

    elif system == "memGPT":
        merged = {}
        if dataset_prefix == "acc_ret":
            # R1: only in _memgpt.json, only if complete (100Q)
            r1_path = f"out/acc_ret_results_{instance_idx}_memgpt.json"
            if os.path.exists(r1_path):
                r1_items = json.load(open(r1_path))["results"].get("R1", [])
                if len(r1_items) >= 100:
                    merged["R1"] = r1_items
            for ad, suffix in [("R2", "_memgpt_r2"), ("R3", "_memgpt_r3")]:
                p = f"out/acc_ret_results_{instance_idx}{suffix}.json"
                if os.path.exists(p):
                    merged[ad] = json.load(open(p))["results"].get(ad, [])
        elif dataset_prefix == "conflict":
            r1_path = f"out/conflict_res_results_{instance_idx}_memgpt.json"
            if os.path.exists(r1_path):
                merged["R1"] = json.load(open(r1_path))["results"].get("R1", [])
            for ad, suffix in [("R2", "_memgpt_r2"), ("R3", "_memgpt_r3")]:
                p = f"out/conflict_res_results_{instance_idx}{suffix}.json"
                if os.path.exists(p):
                    merged[ad] = json.load(open(p))["results"].get(ad, [])
        elif dataset_prefix == "ttl":
            r1_path = f"out/ttl_results_{instance_idx}_memgpt.json"
            if os.path.exists(r1_path):
                merged["R1"] = json.load(open(r1_path))["results"].get("R1", [])
            for ad, suffix in [("R2", "_memgpt_r2"), ("R3", "_memgpt_r3")]:
                p = f"out/ttl_results_{instance_idx}{suffix}.json"
                if os.path.exists(p):
                    merged[ad] = json.load(open(p))["results"].get(ad, [])
        return merged if merged else None

    return None

# ==============================
# AR Eval: 关键词包含匹配
# ==============================

def eval_ar(system, instance_range):
    print(f"\n{'='*60}")
    print(f"  AR Mechanical Eval — {system}")
    print(f"{'='*60}")

    global_stats = {}
    for idx in instance_range:
        gt_path = f"MemoryAgentBench/preview_samples/Accurate_Retrieval/instance_{idx}.json"
        if not os.path.exists(gt_path):
            continue
        gt = json.load(open(gt_path))
        qa_map = {q: a for q, a in zip(gt["questions"], gt["answers"])}

        results = load_merged_results(system, "acc_ret", idx)
        if not results:
            continue

        for adaptor in ["R1", "R2", "R3"]:
            if adaptor not in results:
                continue
            items = results[adaptor]
            correct = 0
            total = len(items)
            for item in items:
                pred = item.get("answer", "")
                refs = qa_map.get(item.get("question", ""), [])
                if not pred:
                    continue
                # 拒答检测
                neg_patterns = ["does not contain any information", "insufficient information",
                                "not mentioned in the context", "no information related to",
                                "上下文没有提到", "没有找到相关信息", "信息不足"]
                refused = any(p in pred.lower() for p in neg_patterns)
                if refused:
                    continue
                # 关键词匹配
                for ref in refs:
                    ref_norm = normalize_answer(ref)
                    if ref_norm in normalize_answer(pred):
                        correct += 1
                        break

            acc = correct / total if total > 0 else 0
            if adaptor not in global_stats:
                global_stats[adaptor] = []
            global_stats[adaptor].append(acc)
            print(f"  inst{idx:<3} | {adaptor} | {acc:.2%} ({correct}/{total})")

    print(f"\n  --- {system} AR Average ---")
    for ad in ["R1", "R2", "R3"]:
        if ad in global_stats and global_stats[ad]:
            print(f"  {ad}: {np.mean(global_stats[ad]):.2%} (n={len(global_stats[ad])})")
    return global_stats

# ==============================
# CR Eval: EM / SubMatch / F1
# ==============================

def eval_cr(system, instance_range):
    print(f"\n{'='*60}")
    print(f"  CR Official Eval — {system}")
    print(f"{'='*60}")

    global_stats = {}
    for idx in instance_range:
        gt_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{idx}.json"
        if not os.path.exists(gt_path):
            continue
        gt = json.load(open(gt_path))
        qa_map = {q: a for q, a in zip(gt["questions"], gt["answers"])}

        results = load_merged_results(system, "conflict", idx)
        if not results:
            continue

        for adaptor in ["R1", "R2", "R3"]:
            if adaptor not in results:
                continue
            items = results[adaptor]
            if adaptor not in global_stats:
                global_stats[adaptor] = {"em": [], "sub": [], "f1": []}

            em_hits = 0; sub_hits = 0; f1_scores = []
            total = len(items)
            for item in items:
                pred = item.get("answer", "")
                ref_list = qa_map.get(item.get("question", ""), [])
                ref = ref_list[0] if isinstance(ref_list, list) and ref_list else (ref_list if isinstance(ref_list, str) else "")
                norm_pred = normalize_answer(pred)
                norm_ref = normalize_answer(ref)
                if norm_pred == norm_ref:
                    em_hits += 1
                if norm_ref in norm_pred:
                    sub_hits += 1
                f1_scores.append(f1_score(pred, ref))

            em = em_hits / total if total > 0 else 0
            sub = sub_hits / total if total > 0 else 0
            avg_f1 = np.mean(f1_scores) if f1_scores else 0
            global_stats[adaptor]["em"].append(em)
            global_stats[adaptor]["sub"].append(sub)
            global_stats[adaptor]["f1"].append(avg_f1)
            print(f"  inst{idx:<3} | {adaptor} | EM={em:.2%} Sub={sub:.2%} F1={avg_f1:.4f}")

    print(f"\n  --- {system} CR Average ---")
    for ad in ["R1", "R2", "R3"]:
        if ad in global_stats:
            s = global_stats[ad]
            print(f"  {ad}: EM={np.mean(s['em']):.2%}  Sub={np.mean(s['sub']):.2%}  F1={np.mean(s['f1']):.4f} (n={len(s['em'])})")
    return global_stats

# ==============================
# TTL Eval: Entity ID 匹配
# ==============================

def load_entity_id_map():
    with open("MemoryAgentBench/entity2id.json", 'r') as f:
        data = json.load(f)
    id_to_title = {}
    for uri, idx in data.items():
        raw = uri.replace("<http://dbpedia.org/resource/", "").replace(">", "")
        title = raw.replace("_", " ")
        id_to_title[str(idx)] = title
    return id_to_title

def eval_ttl(system, instance_range):
    print(f"\n{'='*60}")
    print(f"  TTL Mechanical Eval — {system}")
    print(f"{'='*60}")

    id_map = load_entity_id_map()
    global_stats = {}

    for idx in instance_range:
        results = load_merged_results(system, "ttl", idx)
        if not results:
            continue

        # Load GT for fallback (systems that don't embed ground_truth in results)
        gt_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{idx}.json"
        qa_map = {}
        if os.path.exists(gt_path):
            gt = json.load(open(gt_path))
            qa_map = {q: a for q, a in zip(gt.get("questions", []), gt.get("answers", []))}

        for adaptor in ["R1", "R2", "R3"]:
            if adaptor not in results:
                continue
            items = results[adaptor]
            if adaptor not in global_stats:
                global_stats[adaptor] = []

            correct = 0; total = 0
            for item in items:
                pred = item.get("answer", "").strip()
                gt_ids = item.get("ground_truth") or qa_map.get(item.get("question", ""), [])
                if not isinstance(gt_ids, list):
                    gt_ids = [gt_ids] if gt_ids else []
                if idx == 0:
                    # 电影推荐：ID -> Title 匹配
                    is_hit = False
                    for gid in gt_ids:
                        title = id_map.get(str(gid))
                        if title:
                            if title.lower() in pred.lower():
                                is_hit = True; break
                            base_title = re.sub(r'\s\(\d{4}.*\)', '', title)
                            if base_title.lower() in pred.lower() and len(base_title) > 3:
                                is_hit = True; break
                    if is_hit: correct += 1
                else:
                    # Banking77：数字 ID 比对
                    match = re.search(r'\b(\d+)\b', pred)
                    if match:
                        pred_id = match.group(1)
                        if pred_id in [str(g) for g in gt_ids]:
                            correct += 1
                total += 1

            acc = correct / total if total > 0 else 0
            global_stats[adaptor].append(acc)
            print(f"  inst{idx:<3} | {adaptor} | {acc:.2%} ({correct}/{total})")

    print(f"\n  --- {system} TTL Average ---")
    for ad in ["R1", "R2", "R3"]:
        if ad in global_stats and global_stats[ad]:
            print(f"  {ad}: {np.mean(global_stats[ad]):.2%} (n={len(global_stats[ad])})")
    return global_stats

# ==============================
# Main
# ==============================

if __name__ == "__main__":
    all_results = {}

    for system, ar_range, cr_range, ttl_range in [
        ("simpleMem", range(22), range(8), range(6)),
        ("mem0",      range(22), range(8), range(6)),
        ("mem0g",     range(22), range(8), range(6)),
        ("memGPT",    range(22), range(8), range(6)),
        ("amem",      range(22), range(8), range(1, 6)),
        ("hipporag",  range(22), range(8), range(1, 6)),
    ]:
        print(f"\n{'#'*60}")
        print(f"  SYSTEM: {system}")
        print(f"{'#'*60}")

        ar = eval_ar(system, ar_range)
        cr = eval_cr(system, cr_range)
        ttl = eval_ttl(system, ttl_range)
        all_results[system] = {"AR": ar, "CR": cr, "TTL": ttl}

    # === 最终对比表 ===
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'System':<12} | {'Task':<5} | {'Metric':<10} | {'R1':<10} | {'R2':<10} | {'R3':<10}")
    print("-" * 70)

    for system in ["simpleMem", "mem0", "mem0g", "memGPT", "amem", "hipporag"]:
        res = all_results[system]
        # AR
        ar = res["AR"]
        for ad in ["R1","R2","R3"]:
            val = f"{np.mean(ar[ad]):.2%}" if ad in ar and ar[ad] else "-"
            if ad == "R1":
                print(f"{system:<12} | {'AR':<5} | {'Accuracy':<10} | {val:<10}", end="")
            else:
                print(f" | {val:<10}", end="")
        print()

        # CR - SubMatch
        cr = res["CR"]
        for metric_key, metric_name in [("sub", "SubMatch"), ("f1", "F1")]:
            for ad in ["R1","R2","R3"]:
                val = f"{np.mean(cr[ad][metric_key]):.2%}" if ad in cr and cr[ad][metric_key] else "-"
                if metric_key == "f1":
                    val = f"{np.mean(cr[ad][metric_key]):.4f}" if ad in cr and cr[ad][metric_key] else "-"
                if ad == "R1":
                    print(f"{system:<12} | {'CR':<5} | {metric_name:<10} | {val:<10}", end="")
                else:
                    print(f" | {val:<10}", end="")
            print()

        # TTL
        ttl = res["TTL"]
        for ad in ["R1","R2","R3"]:
            val = f"{np.mean(ttl[ad]):.2%}" if ad in ttl and ttl[ad] else "-"
            if ad == "R1":
                print(f"{system:<12} | {'TTL':<5} | {'Accuracy':<10} | {val:<10}", end="")
            else:
                print(f" | {val:<10}", end="")
        print()
        print("-" * 70)

    # Save JSON
    output = {}
    for system in ["simpleMem", "mem0", "mem0g", "memGPT", "amem", "hipporag"]:
        output[system] = {}
        res = all_results[system]
        for task in ["AR", "CR", "TTL"]:
            output[system][task] = {}
            for ad in ["R1", "R2", "R3"]:
                if task == "CR" and ad in res[task]:
                    output[system][task][ad] = {
                        "em": float(np.mean(res[task][ad]["em"])) if res[task][ad]["em"] else None,
                        "sub": float(np.mean(res[task][ad]["sub"])) if res[task][ad]["sub"] else None,
                        "f1": float(np.mean(res[task][ad]["f1"])) if res[task][ad]["f1"] else None,
                        "n_instances": len(res[task][ad]["em"]),
                    }
                elif ad in res[task] and res[task][ad]:
                    output[system][task][ad] = {
                        "accuracy": float(np.mean(res[task][ad])),
                        "n_instances": len(res[task][ad]),
                    }

    os.makedirs("out/eval", exist_ok=True)
    with open("out/eval/mechanical_eval_all.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to out/eval/mechanical_eval_all.json")
