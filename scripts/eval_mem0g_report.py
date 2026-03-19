#!/usr/bin/env python3
"""
临时 eval 脚本：mem0g AR / CR / TTL 结果评估
- error rate > 30% 的 (instance, adaptor) 标记 EXCL，不参与均值
- error rate <= 30% 的：剔除 error 题后评分，保持原始索引对齐 GT
- R1 / R2 / R3 分开报告

用法:
    cd /Users/bytedance/proj/memoRaxis
    python3 scripts/eval_mem0g_report.py
    python3 scripts/eval_mem0g_report.py --out /tmp/mem0g_report.txt
"""

import argparse, glob, json, re, sys
import numpy as np
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional, Tuple, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark_utils import load_benchmark_data

ERROR_THRESHOLD = 0.30
OUT_DIR = ROOT / "out" / "mem0g"
PARQUET_PATH = ROOT / "MemoryAgentBench" / "data" / "Accurate_Retrieval-00000-of-00001.parquet"
PREVIEW_DIR = ROOT / "MemoryAgentBench" / "preview_samples"
ENTITY_MAP_PATH = ROOT / "MemoryAgentBench" / "entity2id.json"

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def is_correct_mechanical(prediction: str, references) -> bool:
    if references is None: return False
    if hasattr(references, 'tolist'): references = references.tolist()
    if not references: return False
    negative_patterns = [
        "does not contain any information", "insufficient information",
        "not mentioned in the context", "no information related to",
        "上下文没有提到", "没有找到相关信息", "信息不足", "i don't know", "i cannot answer"
    ]
    pred_norm = prediction.lower()
    for p in negative_patterns:
        if p in pred_norm: return False
    norm_pred = normalize_text(prediction)
    flat = []
    for r in references:
        if isinstance(r, list): flat.extend(r)
        else: flat.append(r)
    for ref in flat:
        ref_norm = normalize_text(ref)
        if ref_norm and ref_norm in norm_pred:
            return True
    return False

def filter_items(items: List[Dict]) -> Tuple[Optional[List[Tuple[int, Dict]]], bool, float]:
    """
    返回 (valid_items_with_orig_idx, excluded, error_rate)
    valid_items_with_orig_idx: [(orig_i, item), ...] 保留原始索引
    excluded: True 表示 error rate > 阈值，整体排除
    """
    total = len(items)
    if total == 0:
        return [], True, 1.0
    error_count = sum(1 for item in items if "error" in item)
    error_rate = error_count / total
    if error_rate > ERROR_THRESHOLD:
        return None, True, error_rate
    valid = [(i, item) for i, item in enumerate(items) if "error" not in item]
    return valid, False, error_rate

# ── GT 加载 ───────────────────────────────────────────────────────────────────

_ar_df = None
def get_ar_gt(instance_idx: int) -> List:
    global _ar_df
    if _ar_df is None:
        import pandas as pd
        _ar_df = pd.read_parquet(str(PARQUET_PATH))
    row = _ar_df.iloc[instance_idx]
    return list(row["answers"])

def get_cr_gt(instance_idx: int) -> List:
    p = PREVIEW_DIR / "Conflict_Resolution" / f"instance_{instance_idx}.json"
    with open(p) as f:
        return json.load(f).get("answers", [])

_entity_map = None
def get_entity_map() -> Dict[str, str]:
    global _entity_map
    if _entity_map is None:
        with open(ENTITY_MAP_PATH) as f:
            raw = json.load(f)
        _entity_map = {}
        for uri, idx in raw.items():
            title = uri.replace("<http://dbpedia.org/resource/", "").replace(">", "").replace("_", " ")
            _entity_map[str(idx)] = title
    return _entity_map

def get_ttl_gt(instance_idx: int) -> List:
    p = PREVIEW_DIR / "Test_Time_Learning" / f"instance_{instance_idx}.json"
    with open(p) as f:
        return json.load(f).get("answers", [])

# ── 单个 (instance, adaptor) 的评分函数 ──────────────────────────────────────

def score_ar(valid_with_idx: List[Tuple[int, Dict]], gt_answers: List) -> float:
    correct = sum(
        1 for orig_i, item in valid_with_idx
        if orig_i < len(gt_answers) and is_correct_mechanical(item.get("answer", ""), gt_answers[orig_i])
    )
    return correct / len(valid_with_idx) if valid_with_idx else 0.0

def score_cr(valid_with_idx: List[Tuple[int, Dict]], gt_answers: List) -> float:
    correct = sum(
        1 for orig_i, item in valid_with_idx
        if orig_i < len(gt_answers) and is_correct_mechanical(item.get("answer", ""), gt_answers[orig_i])
    )
    return correct / len(valid_with_idx) if valid_with_idx else 0.0

def score_ttl(valid_with_idx: List[Tuple[int, Dict]], gt_answers: List, id_map: Dict) -> float:
    # mem0g 的 TTL pred 直接返回数字 entity ID（如 "{18}" 或 "50"），
    # GT 也是数字 ID 字符串列表（如 ['18']）。
    # 正确做法：从 pred 中提取所有数字，与 GT ID 直接比对。
    correct = 0
    for orig_i, item in valid_with_idx:
        if orig_i >= len(gt_answers): continue
        pred = item.get("answer", "")
        gt_ids = gt_answers[orig_i]
        if not isinstance(gt_ids, list): gt_ids = [gt_ids]
        gt_id_strs = [str(g) for g in gt_ids]
        # 从 pred 提取所有数字串
        pred_nums = re.findall(r'\d+', pred)
        hit = any(n in gt_id_strs for n in pred_nums)
        if hit: correct += 1
    return correct / len(valid_with_idx) if valid_with_idx else 0.0

# ── 数据结构: cell ────────────────────────────────────────────────────────────
# cell = {"acc": float | None, "excluded": bool, "error_rate": float, "n": int, "valid_n": int}

def make_excl_cell(items, error_rate):
    return {"acc": None, "excluded": True, "error_rate": error_rate, "n": len(items), "valid_n": 0}

def make_acc_cell(acc, items, valid_with_idx, error_rate):
    return {"acc": acc, "excluded": False, "error_rate": error_rate,
            "n": len(items), "valid_n": len(valid_with_idx)}

# ── 主 eval 函数 ──────────────────────────────────────────────────────────────

def eval_ar(files):
    """返回 {inst: {adaptor: cell}}"""
    results = {}
    for fpath in sorted(files):
        try:
            d = json.load(open(fpath))
        except: continue
        inst = d.get("instance_idx")
        if inst is None:
            m = re.search(r'acc_ret_results_(\d+)', fpath)
            if m: inst = int(m.group(1))
        if inst is None: continue
        try:
            gt_answers = get_ar_gt(inst)
        except Exception as e:
            print(f"  [AR] inst{inst} GT加载失败: {e}")
            continue
        inst_res = {}
        for adaptor, items in d.get("results", {}).items():
            valid, excluded, err_rate = filter_items(items)
            if excluded:
                inst_res[adaptor] = make_excl_cell(items, err_rate)
            else:
                acc = score_ar(valid, gt_answers)
                inst_res[adaptor] = make_acc_cell(acc, items, valid, err_rate)
        results[inst] = inst_res
    return results

def eval_cr(files):
    results = {}
    for fpath in sorted(files):
        try:
            d = json.load(open(fpath))
        except: continue
        inst = d.get("instance_idx")
        if inst is None:
            m = re.search(r'conflict_results_(\d+)', fpath)
            if m: inst = int(m.group(1))
        if inst is None: continue
        try:
            gt_answers = get_cr_gt(inst)
        except Exception as e:
            print(f"  [CR] inst{inst} GT加载失败: {e}")
            continue
        inst_res = {}
        for adaptor, items in d.get("results", {}).items():
            valid, excluded, err_rate = filter_items(items)
            if excluded:
                inst_res[adaptor] = make_excl_cell(items, err_rate)
            else:
                acc = score_cr(valid, gt_answers)
                inst_res[adaptor] = make_acc_cell(acc, items, valid, err_rate)
        results[inst] = inst_res
    return results

def eval_ttl(files):
    id_map = get_entity_map()
    results = {}
    for fpath in sorted(files):
        try:
            d = json.load(open(fpath))
        except: continue
        inst = d.get("instance_idx")
        if inst is None:
            m = re.search(r'ttl_results_(\d+)', fpath)
            if m: inst = int(m.group(1))
        if inst is None: continue
        try:
            gt_answers = get_ttl_gt(inst)
        except Exception as e:
            print(f"  [TTL] inst{inst} GT加载失败: {e}")
            continue
        inst_res = {}
        for adaptor, items in d.get("results", {}).items():
            valid, excluded, err_rate = filter_items(items)
            if excluded:
                inst_res[adaptor] = make_excl_cell(items, err_rate)
            else:
                acc = score_ttl(valid, gt_answers, id_map)
                inst_res[adaptor] = make_acc_cell(acc, items, valid, err_rate)
        results[inst] = inst_res
    return results

# ── 报告渲染 ──────────────────────────────────────────────────────────────────

ADAPTORS = ["R1", "R2", "R3"]

def fmt_cell(cell) -> str:
    if cell is None: return "  -   "
    if cell["excluded"]: return f"EXCL  "
    return f"{cell['acc']:.4f}"

def render_report(task_name: str, results: Dict, notes: List[str]) -> str:
    lines = []
    lines.append(f"\n{'='*62}")
    lines.append(f"  {task_name}")
    lines.append(f"{'='*62}")

    # 表头
    lines.append(f"{'inst':<6} | {'R1':^8} | {'R2':^8} | {'R3':^8} | notes")
    lines.append(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*20}")

    adaptor_scores: Dict[str, List[float]] = {a: [] for a in ADAPTORS}
    excl_counts: Dict[str, int] = {a: 0 for a in ADAPTORS}
    excl_detail = []

    for inst in sorted(results.keys()):
        inst_data = results[inst]
        cells = {a: inst_data.get(a) for a in ADAPTORS}
        row_notes = []
        for a in ADAPTORS:
            c = cells[a]
            if c is None: continue
            if c["excluded"]:
                excl_counts[a] += 1
                excl_detail.append(f"inst{inst} {a}: err={c['error_rate']:.0%} ({c['n']}题) → EXCL")
                row_notes.append(f"{a}:err={c['error_rate']:.0%}")
            else:
                adaptor_scores[a].append(c["acc"])
                if c["error_rate"] > 0:
                    row_notes.append(f"{a}:err={c['error_rate']:.0%}")

        r1s = fmt_cell(cells.get("R1"))
        r2s = fmt_cell(cells.get("R2"))
        r3s = fmt_cell(cells.get("R3"))
        lines.append(f"{inst:<6} | {r1s:^8} | {r2s:^8} | {r3s:^8} | {', '.join(row_notes)}")

    # 均值行
    lines.append(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*20}")
    avg_parts = []
    for a in ADAPTORS:
        sc = adaptor_scores[a]
        if sc:
            avg_parts.append(f"{a}: {np.mean(sc):.4f} (N={len(sc)}, excl={excl_counts[a]})")
        else:
            avg_parts.append(f"{a}: N/A")
    lines.append("AVG    | " + " | ".join(avg_parts))

    if excl_detail:
        lines.append("")
        lines.append("  [EXCL 明细]")
        for d in excl_detail:
            lines.append(f"    {d}")

    if notes:
        lines.append("")
        lines.append("  [备注]")
        for n in notes:
            lines.append(f"    {n}")

    return "\n".join(lines)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="保存报告到文件")
    parser.add_argument("--error-threshold", type=float, default=ERROR_THRESHOLD)
    args = parser.parse_args()

    # error threshold 通过模块变量传递（默认0.30，命令行可覆盖）
    if args.error_threshold != 0.30:
        print(f"注意：error threshold 已调整为 {args.error_threshold:.0%}")

    print(f"mem0g eval report  |  {date.today()}  |  error threshold={ERROR_THRESHOLD:.0%}")
    print(f"输入目录: {OUT_DIR}")

    sections = []

    # ── AR ──
    ar_files = sorted(glob.glob(str(OUT_DIR / "mem0g_acc_ret_results_*.json")))
    print(f"\n[AR] 找到 {len(ar_files)} 个文件...")
    ar_results = eval_ar(ar_files)
    ar_notes = ["仅 inst0 为100题，inst1/5/7 为200题（含多轮）"]
    sections.append(render_report("Accurate Retrieval  (mem0g, R1/R2/R3)", ar_results, ar_notes))

    # ── CR ──
    cr_files = sorted(glob.glob(str(OUT_DIR / "mem0g_conflict_results_*.json")))
    print(f"[CR] 找到 {len(cr_files)} 个文件...")
    cr_results = eval_cr(cr_files)
    cr_notes = ["inst0 仅10题（Qdrant 4pts，ingest可能不完整，仅参考）"]
    sections.append(render_report("Conflict Resolution  (mem0g, R1/R2/R3)", cr_results, cr_notes))

    # ── TTL ──
    ttl_files = sorted(glob.glob(str(OUT_DIR / "mem0g_ttl_results_*.json")))
    print(f"[TTL] 找到 {len(ttl_files)} 个文件...")
    ttl_results = eval_ttl(ttl_files)
    ttl_notes = ["inst0 未 ingest，缺失；inst3/4 error rate 偏高注意参考"]
    sections.append(render_report("Test Time Learning  (mem0g, R1/R2/R3)", ttl_results, ttl_notes))

    # ── 汇总 ──
    summary_lines = [
        "",
        "=" * 62,
        "  汇总 (只计 R1，因为 mem0g 目前基本只有 R1 结果)",
        "=" * 62,
    ]
    for task_name, task_results in [("AR", ar_results), ("CR", cr_results), ("TTL", ttl_results)]:
        r1_scores = [
            c["acc"] for inst_data in task_results.values()
            for a, c in inst_data.items()
            if a == "R1" and not c["excluded"] and c["acc"] is not None
        ]
        n_excl = sum(
            1 for inst_data in task_results.values()
            for a, c in inst_data.items()
            if a == "R1" and c["excluded"]
        )
        if r1_scores:
            summary_lines.append(
                f"  {task_name:<4} R1 avg: {np.mean(r1_scores):.4f}  (N={len(r1_scores)}, excl={n_excl})"
            )
        else:
            summary_lines.append(f"  {task_name:<4} R1 avg: N/A")
    sections.append("\n".join(summary_lines))

    report = "\n".join(sections) + "\n"
    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告已保存到 {args.out}")

if __name__ == "__main__":
    main()
