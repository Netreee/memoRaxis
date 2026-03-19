#!/usr/bin/env python3
"""
mem0g 综合报告：Ingest审计 + Infer审计 + Eval结果（AR/CR/TTL/LR）
用法:
    cd /Users/bytedance/proj/memoRaxis
    python3 scripts/mem0g_full_report.py
    python3 scripts/mem0g_full_report.py --out /tmp/mem0g_full_report.txt
"""

import argparse, glob, json, re, sys, urllib.request
import numpy as np
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmark_utils import load_benchmark_data

ERROR_THRESHOLD = 0.30
OUT_DIR    = ROOT / "out" / "mem0g"
PARQUET    = ROOT / "MemoryAgentBench" / "data" / "Accurate_Retrieval-00000-of-00001.parquet"
PREVIEW    = ROOT / "MemoryAgentBench" / "preview_samples"
ENTITY_MAP = ROOT / "MemoryAgentBench" / "entity2id.json"
ADAPTORS   = ["R1", "R2", "R3"]

# ── 总实例数 ─────────────────────────────────────────────────────────────────
TOTAL = {"AR": 22, "CR": 8, "LR": 40, "TTL": 6}

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Ingest 审计（Qdrant points）
# ═══════════════════════════════════════════════════════════════════════════════

def get_qdrant_points() -> Dict[str, int]:
    """返回 {collection_name: points_count}"""
    try:
        r = urllib.request.urlopen("http://localhost:6333/collections", timeout=5)
        cols = json.load(r)["result"]["collections"]
    except Exception as e:
        print(f"  [WARN] Qdrant 连接失败: {e}")
        return {}
    result = {}
    for c in cols:
        name = c["name"]
        if not name.startswith("mem0g"): continue
        try:
            r2 = urllib.request.urlopen(f"http://localhost:6333/collections/{name}", timeout=5)
            pts = json.load(r2)["result"]["points_count"]
            result[name] = pts
        except:
            result[name] = -1
    return result

def ingest_status(pts: int) -> str:
    if pts < 0:   return "ERR"
    if pts == 0:  return "empty"
    if pts <= 5:  return "⚠️ sparse"
    return "✅"

DS_COL_PREFIX = {
    "AR":  "mem0g_acc_ret",
    "CR":  "mem0g_conflict",
    "LR":  "mem0g_long_range",
    "TTL": "mem0g_ttl",
}

def build_ingest_table(qdrant: Dict[str, int]) -> Dict[str, Dict[int, int]]:
    """返回 {ds: {inst: points}}"""
    table = {ds: {} for ds in DS_COL_PREFIX}
    for name, pts in qdrant.items():
        for ds, prefix in DS_COL_PREFIX.items():
            if name.startswith(prefix + "_"):
                inst_str = name[len(prefix)+1:]
                try:
                    table[ds][int(inst_str)] = pts
                except ValueError:
                    pass
    return table

# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Infer 审计（result 文件统计）
# ═══════════════════════════════════════════════════════════════════════════════

FILE_PATTERN = {
    "AR":  "mem0g_acc_ret_results_*.json",
    "CR":  "mem0g_conflict_results_*.json",
    "LR":  "mem0g_long_range_results_*.json",
    "TTL": "mem0g_ttl_results_*.json",
}
INST_RE = {
    "AR":  r"acc_ret_results_(\d+)",
    "CR":  r"conflict_results_(\d+)",
    "LR":  r"long_range_results_(\d+)",
    "TTL": r"ttl_results_(\d+)",
}

def build_infer_audit() -> Dict[str, Dict[int, Dict[str, dict]]]:
    """返回 {ds: {inst: {adaptor: {n, errs, err_rate}}}}"""
    audit = {ds: {} for ds in FILE_PATTERN}
    for ds, pat in FILE_PATTERN.items():
        for fpath in sorted(glob.glob(str(OUT_DIR / pat))):
            m = re.search(INST_RE[ds], fpath)
            if not m: continue
            inst = int(m.group(1))
            try:
                d = json.load(open(fpath))
            except: continue
            inst_audit = {}
            for adaptor, items in d.get("results", {}).items():
                n = len(items)
                errs = sum(1 for x in items if "error" in x)
                inst_audit[adaptor] = {"n": n, "errs": errs,
                                       "err_rate": errs/n if n else 0}
            audit[ds][inst] = inst_audit
    return audit

# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: Eval（AR / CR / TTL 机械匹配，LR 读外部文件）
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_text(t):
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r'[^\w\s]', '', t)
    return " ".join(t.split())

def is_correct_mechanical(pred, refs):
    if refs is None: return False
    if hasattr(refs, 'tolist'): refs = refs.tolist()
    if not refs: return False
    neg = ["does not contain any information","insufficient information",
           "not mentioned in the context","no information related to",
           "上下文没有提到","没有找到相关信息","信息不足","i don't know","i cannot answer"]
    pl = pred.lower()
    for p in neg:
        if p in pl: return False
    np_ = normalize_text(pred)
    flat = []
    for r in refs:
        if isinstance(r, list): flat.extend(r)
        else: flat.append(r)
    for r in flat:
        rn = normalize_text(r)
        if rn and rn in np_: return True
    return False

def filter_items(items):
    total = len(items)
    if total == 0: return [], True, 1.0
    errs = sum(1 for x in items if "error" in x)
    rate = errs / total
    if rate > ERROR_THRESHOLD: return None, True, rate
    valid = [(i, x) for i, x in enumerate(items) if "error" not in x]
    return valid, False, rate

# cell = {acc, excluded, err_rate, n, valid_n}
def make_cell(acc, excluded, err_rate, n, valid_n):
    return dict(acc=acc, excluded=excluded, err_rate=err_rate, n=n, valid_n=valid_n)

_ar_df = None
def ar_gt(inst):
    global _ar_df
    if _ar_df is None:
        import pandas as pd
        _ar_df = pd.read_parquet(str(PARQUET))
    return list(_ar_df.iloc[inst]["answers"])

def cr_gt(inst):
    p = PREVIEW / "Conflict_Resolution" / f"instance_{inst}.json"
    return json.load(open(p)).get("answers", [])

_emap = None
def entity_map():
    global _emap
    if _emap is None:
        raw = json.load(open(ENTITY_MAP))
        _emap = {}
        for uri, idx in raw.items():
            t = uri.replace("<http://dbpedia.org/resource/","").replace(">","").replace("_"," ")
            _emap[str(idx)] = t
    return _emap

def ttl_gt(inst):
    p = PREVIEW / "Test_Time_Learning" / f"instance_{inst}.json"
    return json.load(open(p)).get("answers", [])

def score_ar(valid, gt):
    c = sum(1 for i, x in valid if i < len(gt) and is_correct_mechanical(x.get("answer",""), gt[i]))
    return c / len(valid) if valid else 0.0

def score_cr(valid, gt):
    c = sum(1 for i, x in valid if i < len(gt) and is_correct_mechanical(x.get("answer",""), gt[i]))
    return c / len(valid) if valid else 0.0

def score_ttl(valid, gt, em):
    correct = 0
    for orig_i, item in valid:
        if orig_i >= len(gt): continue
        pred = item.get("answer","")
        gt_ids = [str(g) for g in (gt[orig_i] if isinstance(gt[orig_i], list) else [gt[orig_i]])]
        if any(n in gt_ids for n in re.findall(r'\d+', pred)):
            correct += 1
    return correct / len(valid) if valid else 0.0

def eval_dataset(ds: str) -> Dict[int, Dict[str, dict]]:
    em = entity_map() if ds == "TTL" else None
    results = {}
    for fpath in sorted(glob.glob(str(OUT_DIR / FILE_PATTERN[ds]))):
        m = re.search(INST_RE[ds], fpath)
        if not m: continue
        inst = int(m.group(1))
        try:
            d = json.load(open(fpath))
        except: continue
        try:
            if ds == "AR":  gt = ar_gt(inst)
            elif ds == "CR": gt = cr_gt(inst)
            elif ds == "TTL": gt = ttl_gt(inst)
            else: gt = None
        except Exception as e:
            print(f"  [WARN] {ds} inst{inst} GT 加载失败: {e}")
            continue
        inst_res = {}
        for adaptor, items in d.get("results", {}).items():
            valid, excluded, err_rate = filter_items(items)
            n = len(items)
            if excluded:
                inst_res[adaptor] = make_cell(None, True, err_rate, n, 0)
            else:
                if ds == "AR":  acc = score_ar(valid, gt)
                elif ds == "CR": acc = score_cr(valid, gt)
                elif ds == "TTL": acc = score_ttl(valid, gt, em)
                else: acc = None
                inst_res[adaptor] = make_cell(acc, False, err_rate, n, len(valid))
        results[inst] = inst_res
    return results

def parse_lr_results(lr_file: str) -> Dict[int, Dict[str, float]]:
    """从 analyze.py 输出的文本报告解析 LR F1 得分"""
    results = {}
    try:
        text = open(lr_file).read()
    except FileNotFoundError:
        return {}
    # 匹配格式如: "0     | 0.45  | 0.32  | 0.10  | R1"
    for line in text.splitlines():
        m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', line)
        if m:
            inst = int(m.group(1))
            results[inst] = {
                "R1": float(m.group(2)),
                "R2": float(m.group(3)),
                "R3": float(m.group(4)),
            }
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# Part 4: 报告渲染
# ═══════════════════════════════════════════════════════════════════════════════

W = 70

def hline(c="─"): return c * W

def section(title):
    return f"\n{'═'*W}\n  {title}\n{'═'*W}"

def render_ingest_audit(ingest_table, total_map):
    lines = [section("INGEST 审计  （Qdrant points per instance）")]
    summary = []
    for ds in ["AR", "CR", "LR", "TTL"]:
        total = total_map[ds]
        data  = ingest_table.get(ds, {})
        existing  = {k: v for k, v in data.items() if v > 5}
        sparse    = {k: v for k, v in data.items() if 0 < v <= 5}
        empty_col = {k: v for k, v in data.items() if v == 0}
        missing   = total - len(data)
        lines.append(f"\n  [{ds}]  共{total}个实例  |  ✅ ingested={len(existing)}  ⚠️ sparse={len(sparse)}  ❌ empty={len(empty_col)}  — missing={missing}")
        lines.append(f"  {'inst':<6} {'points':>6}  {'status':<12}")
        lines.append("  " + "─"*28)
        for inst in sorted(data.keys()):
            pts = data[inst]
            st  = ingest_status(pts)
            lines.append(f"  inst{inst:<4} {pts:>6}  {st}")
        summary.append(f"  {ds:<4}  ✅{len(existing)}/{total}  ⚠️{len(sparse)}  ❌{len(empty_col)}  —{missing}")
    lines.append(f"\n  {'─'*40}")
    lines.append("  汇总（>5pts 算有效）：")
    lines.extend(summary)
    return "\n".join(lines)

def render_infer_audit(infer_audit, total_map):
    lines = [section("INFER 审计  （questions / errors per (inst, adaptor)）")]
    for ds in ["AR", "CR", "LR", "TTL"]:
        total = total_map[ds]
        data  = infer_audit.get(ds, {})
        n_files = len(data)
        lines.append(f"\n  [{ds}]  共{total}个实例  |  有结果文件: {n_files}/{total}")
        lines.append(f"  {'inst':<6} {'adaptor':<8} {'n':>5} {'errs':>5} {'err%':>6}  {'flag'}")
        lines.append("  " + "─"*42)
        for inst in sorted(data.keys()):
            for adaptor in ADAPTORS:
                if adaptor not in data[inst]: continue
                a = data[inst][adaptor]
                flag = "EXCL" if a["err_rate"] > ERROR_THRESHOLD else ("⚠️ " if a["err_rate"] > 0 else "")
                lines.append(f"  inst{inst:<4} {adaptor:<8} {a['n']:>5} {a['errs']:>5} {a['err_rate']:>5.0%}  {flag}")
    return "\n".join(lines)

def render_eval_table(ds, results, metric_label="acc"):
    lines = []
    lines.append(f"\n  [{ds}]  (metric: {metric_label}, error_threshold={ERROR_THRESHOLD:.0%})")
    lines.append(f"  {'inst':<6} {'R1':^9} {'R2':^9} {'R3':^9}  notes")
    lines.append("  " + "─"*50)
    per_adaptor: Dict[str, List[float]] = {a: [] for a in ADAPTORS}
    excl: Dict[str, int] = {a: 0 for a in ADAPTORS}
    excl_notes = []
    for inst in sorted(results.keys()):
        row = results[inst]
        cells = []
        row_notes = []
        for a in ADAPTORS:
            c = row.get(a)
            if c is None:
                cells.append("   -   ")
            elif c["excluded"]:
                cells.append(" EXCL  ")
                excl[a] += 1
                excl_notes.append(f"inst{inst} {a}: err={c['err_rate']:.0%}({c['n']}q)")
                if c["err_rate"] > 0: row_notes.append(f"{a}:err={c['err_rate']:.0%}")
            else:
                cells.append(f" {c['acc']:.4f}")
                per_adaptor[a].append(c["acc"])
                if c["err_rate"] > 0: row_notes.append(f"{a}:err={c['err_rate']:.0%}")
        lines.append(f"  inst{inst:<4} {''.join(f'{x:^9}' for x in cells)}  {', '.join(row_notes)}")
    lines.append("  " + "─"*50)
    avg_parts = []
    for a in ADAPTORS:
        sc = per_adaptor[a]
        if sc: avg_parts.append(f"{a}:{np.mean(sc):.4f}(N={len(sc)},excl={excl[a]})")
        else:  avg_parts.append(f"{a}:N/A")
    lines.append(f"  AVG    {' | '.join(avg_parts)}")
    if excl_notes:
        lines.append(f"  EXCL → {', '.join(excl_notes)}")
    return "\n".join(lines), {a: (np.mean(per_adaptor[a]) if per_adaptor[a] else None) for a in ADAPTORS}

def render_lr_eval(lr_results, infer_audit):
    lines = []
    lines.append(f"\n  [LR]  (metric: F1 = fluency × 2PR/(P+R), LLM evaluated)")
    lines.append(f"  {'inst':<6} {'R1':^9} {'R2':^9} {'R3':^9}")
    lines.append("  " + "─"*40)
    per_adaptor: Dict[str, List[float]] = {a: [] for a in ADAPTORS}
    # 构建实际存在的 (inst, adaptor) 集合，用于区分 "真0分" 和 "缺失被填0"
    has_infer = {}
    for inst, adata in infer_audit.get("LR", {}).items():
        has_infer[inst] = set(adata.keys())
    for inst in sorted(lr_results.keys()):
        row = lr_results[inst]
        cells = []
        for a in ADAPTORS:
            v = row.get(a)
            infer_exists = a in has_infer.get(inst, set())
            if not infer_exists:
                cells.append("   -   ")
            elif v is None:
                ia = infer_audit.get("LR", {}).get(inst, {}).get(a)
                if ia and ia["err_rate"] > ERROR_THRESHOLD:
                    cells.append(" EXCL  ")
                else:
                    cells.append("   -   ")
            else:
                cells.append(f" {v:.4f}")
                per_adaptor[a].append(v)
        lines.append(f"  inst{inst:<4} {''.join(f'{x:^9}' for x in cells)}")
    lines.append("  " + "─"*40)
    avg_parts = []
    for a in ADAPTORS:
        sc = per_adaptor[a]
        if sc: avg_parts.append(f"{a}:{np.mean(sc):.4f}(N={len(sc)})")
        else:  avg_parts.append(f"{a}:N/A")
    lines.append(f"  AVG    {' | '.join(avg_parts)}")
    return "\n".join(lines), {a: (np.mean(per_adaptor[a]) if per_adaptor[a] else None) for a in ADAPTORS}

def render_summary(avgs):
    lines = [section("汇总  （各 Task 各 Adaptor 平均分）")]
    lines.append(f"\n  {'Task':<6} {'R1':^10} {'R2':^10} {'R3':^10}  metric")
    lines.append("  " + "─"*48)
    metric_label = {"AR":"acc(mech)","CR":"acc(mech)","TTL":"acc(id)","LR":"F1(LLM)"}
    for ds in ["AR", "CR", "TTL", "LR"]:
        a = avgs.get(ds, {})
        def fmt(v): return f"{v:.4f}" if v is not None else "  N/A "
        lines.append(f"  {ds:<6} {fmt(a.get('R1')):^10} {fmt(a.get('R2')):^10} {fmt(a.get('R3')):^10}  {metric_label[ds]}")
    lines.append("")
    lines.append("  注：")
    lines.append("  - CR 评分可能偏高（时间序信息全链路丢失，部分高分为运气）")
    lines.append("  - TTL inst5 得分低是数据难度（49标签 vs 其他6标签），非系统问题")
    lines.append("  - LR 仅13/40实例有结果，代表性有限")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--lr-file", default="/tmp/mem0g_lr_eval.txt",
                        help="LR eval 结果文件（analyze.py 输出）")
    args = parser.parse_args()

    print(f"[1/5] 读取 Qdrant ingest 数据...")
    qdrant = get_qdrant_points()
    ingest_table = build_ingest_table(qdrant)

    print(f"[2/5] 读取 infer 审计数据...")
    infer_audit = build_infer_audit()

    print(f"[3/5] 运行 AR / CR / TTL eval...")
    ar_results  = eval_dataset("AR")
    cr_results  = eval_dataset("CR")
    ttl_results = eval_dataset("TTL")

    print(f"[4/5] 读取 LR eval 结果（{args.lr_file}）...")
    lr_results = parse_lr_results(args.lr_file)
    if not lr_results:
        print("  [WARN] LR 结果文件未就绪或为空，LR 部分将显示为 pending")

    print(f"[5/5] 生成报告...")

    header = [
        hline("═"),
        f"  mem0g 综合评估报告  |  {date.today()}",
        f"  error threshold = {ERROR_THRESHOLD:.0%}  |  datasets: AR(22) CR(8) LR(40) TTL(6)",
        hline("═"),
    ]

    ingest_sec = render_ingest_audit(ingest_table, TOTAL)
    infer_sec  = render_infer_audit(infer_audit, TOTAL)

    eval_lines = [section("EVAL 结果")]
    avgs = {}

    ar_txt,  avgs["AR"]  = render_eval_table("AR",  ar_results,  "accuracy (mechanical)")
    cr_txt,  avgs["CR"]  = render_eval_table("CR",  cr_results,  "accuracy (mechanical)")
    ttl_txt, avgs["TTL"] = render_eval_table("TTL", ttl_results, "accuracy (entity-id match)")
    eval_lines.extend([ar_txt, cr_txt, ttl_txt])

    if lr_results:
        lr_txt, avgs["LR"] = render_lr_eval(lr_results, infer_audit)
        eval_lines.append(lr_txt)
    else:
        eval_lines.append("\n  [LR]  ⏳ eval 尚未完成，请稍后重新运行")
        avgs["LR"] = {}

    summary_sec = render_summary(avgs)

    report = "\n".join(header) + "\n" + ingest_sec + "\n" + infer_sec + "\n" + "\n".join(eval_lines) + "\n" + summary_sec + "\n"
    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n报告已保存到 {args.out}")

if __name__ == "__main__":
    main()
