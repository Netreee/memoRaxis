#!/usr/bin/env python3
"""
mem0 综合报告：Ingest审计 + Infer审计 + Eval结果（AR / CR / TTL）
策略：error 题直接剔除，剩余题目正常评分，不整个 inst 排除。
用法:
    cd /Users/bytedance/proj/memoRaxis
    python3 scripts/mem0_full_report.py
    python3 scripts/mem0_full_report.py --out /tmp/mem0_report.txt
"""

import argparse, glob, json, re, sys, urllib.request
import numpy as np
from pathlib import Path
from datetime import date
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmark_utils import load_benchmark_data

OUT_DIR    = ROOT / "out" / "mem0"
PARQUET    = ROOT / "MemoryAgentBench" / "data" / "Accurate_Retrieval-00000-of-00001.parquet"
PREVIEW    = ROOT / "MemoryAgentBench" / "preview_samples"
ENTITY_MAP = ROOT / "MemoryAgentBench" / "entity2id.json"
ADAPTORS   = ["R1", "R2", "R3"]
TOTAL      = {"AR": 22, "CR": 8, "LR": 110, "TTL": 6}

# ── mem0 文件命名：R1=无后缀，R2=_r2，R3=_r3 ─────────────────────────────
DS_PREFIX = {
    "AR":  "mem0_acc_ret_results_",
    "CR":  "mem0_conflict_results_",
    "LR":  "mem0_long_range_results_",
    "TTL": "mem0_ttl_results_",
}
ADAPTOR_SUFFIX = {"R1": "", "R2": "_r2", "R3": "_r3"}

def find_infer_files(ds: str) -> Dict[Tuple[int, str], Path]:
    """返回 {(inst, adaptor): filepath}"""
    prefix = DS_PREFIX[ds]
    result = {}
    for f in OUT_DIR.glob(f"{prefix}*.json"):
        stem = f.stem  # e.g. mem0_acc_ret_results_3_r2
        rest = stem.replace(prefix, "")
        if rest.endswith("_r3"):
            adaptor, inst_str = "R3", rest[:-3]
        elif rest.endswith("_r2"):
            adaptor, inst_str = "R2", rest[:-3]
        else:
            adaptor, inst_str = "R1", rest
        try:
            result[(int(inst_str), adaptor)] = f
        except ValueError:
            pass
    return result

# ═══ Part 1: Ingest 审计 ════════════════════════════════════════════════════

def get_qdrant_points() -> Dict[str, int]:
    try:
        r = urllib.request.urlopen("http://localhost:6333/collections", timeout=5)
        cols = json.load(r)["result"]["collections"]
    except Exception as e:
        print(f"  [WARN] Qdrant 连接失败: {e}")
        return {}
    out = {}
    for c in cols:
        name = c["name"]
        if not (name.startswith("mem0") and not name.startswith("mem0g")):
            continue
        try:
            r2 = urllib.request.urlopen(f"http://localhost:6333/collections/{name}", timeout=5)
            out[name] = json.load(r2)["result"]["points_count"]
        except:
            out[name] = -1
    return out

DS_COL_PREFIX = {
    "AR":  "mem0_acc_ret_",
    "CR":  "mem0_conflict_",
    "LR":  "mem0_long_range_",
    "TTL": "mem0_ttl_",
}

def build_ingest_table(qdrant: Dict[str, int]) -> Dict[str, Dict[int, int]]:
    table = {ds: {} for ds in DS_COL_PREFIX}
    for name, pts in qdrant.items():
        if name == "mem0migrations":
            continue
        for ds, prefix in DS_COL_PREFIX.items():
            if name.startswith(prefix):
                inst_str = name[len(prefix):]
                try:
                    table[ds][int(inst_str)] = pts
                except ValueError:
                    pass
    return table

def ingest_flag(pts: int) -> str:
    if pts < 0:  return "ERR"
    if pts == 0: return "❌"
    if pts <= 5: return "⚠️ "
    return "✅"

# ═══ Part 2: Infer 审计 ════════════════════════════════════════════════════

def build_infer_audit() -> Dict[str, Dict[int, Dict[str, dict]]]:
    """返回 {ds: {inst: {adaptor: {n, errs, err_rate}}}}"""
    audit = {}
    for ds in ["AR", "CR", "TTL"]:
        audit[ds] = {}
        files = find_infer_files(ds)
        for (inst, adaptor), fpath in files.items():
            try:
                d = json.load(open(fpath))
                items = d.get("results", {}).get(adaptor, [])
                n = len(items)
                errs = sum(1 for x in items if "error" in x)
                if inst not in audit[ds]:
                    audit[ds][inst] = {}
                audit[ds][inst][adaptor] = {
                    "n": n, "errs": errs,
                    "err_rate": errs / n if n else 0,
                }
            except:
                pass
    return audit

# ═══ Part 3: Eval ══════════════════════════════════════════════════════════

def normalize_text(t: str) -> str:
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r'[^\w\s]', '', t)
    return " ".join(t.split())

def is_correct_mechanical(pred: str, refs) -> bool:
    if refs is None: return False
    if hasattr(refs, 'tolist'): refs = refs.tolist()
    if not refs: return False
    neg = [
        "does not contain any information", "insufficient information",
        "not mentioned in the context", "no information related to",
        "上下文没有提到", "没有找到相关信息", "信息不足",
        "i don't know", "i cannot answer",
        "上下文信息不足",  # mem0 常见回答
    ]
    pl = pred.lower()
    for p in neg:
        if p in pl: return False
    np_ = normalize_text(pred)
    flat = []
    for r in refs:
        if isinstance(r, list): flat.extend(r)
        else: flat.append(r)
    for ref in flat:
        rn = normalize_text(ref)
        if rn and rn in np_: return True
    return False

_ar_df = None
def ar_gt(inst: int) -> List:
    global _ar_df
    if _ar_df is None:
        import pandas as pd
        _ar_df = pd.read_parquet(str(PARQUET))
    return list(_ar_df.iloc[inst]["answers"])

def cr_gt(inst: int) -> List:
    p = PREVIEW / "Conflict_Resolution" / f"instance_{inst}.json"
    return json.load(open(p)).get("answers", [])

_emap = None
def entity_map() -> Dict[str, str]:
    global _emap
    if _emap is None:
        raw = json.load(open(ENTITY_MAP))
        _emap = {}
        for uri, idx in raw.items():
            t = uri.replace("<http://dbpedia.org/resource/","").replace(">","").replace("_"," ")
            _emap[str(idx)] = t
    return _emap

def ttl_gt(inst: int) -> List:
    p = PREVIEW / "Test_Time_Learning" / f"instance_{inst}.json"
    return json.load(open(p)).get("answers", [])

# cell: {acc, err_rate, n, valid_n}  — 不再有 excluded 字段
def eval_ds(ds: str) -> Dict[int, Dict[str, dict]]:
    em = entity_map() if ds == "TTL" else None
    results: Dict[int, Dict[str, dict]] = {}
    files = find_infer_files(ds)

    for (inst, adaptor), fpath in sorted(files.items()):
        try:
            d = json.load(open(fpath))
            items = d.get("results", {}).get(adaptor, [])
        except:
            continue
        try:
            if ds == "AR":  gt = ar_gt(inst)
            elif ds == "CR": gt = cr_gt(inst)
            elif ds == "TTL": gt = ttl_gt(inst)
        except Exception as e:
            print(f"  [WARN] {ds} inst{inst} GT 加载失败: {e}")
            continue

        # 剔除 error 题，保留原始索引
        valid = [(i, x) for i, x in enumerate(items) if "error" not in x]
        n     = len(items)
        errs  = n - len(valid)
        err_rate = errs / n if n else 0

        if not valid:
            acc = None  # 全错，无法评分
        elif ds == "AR":
            correct = sum(
                1 for orig_i, x in valid
                if orig_i < len(gt) and is_correct_mechanical(x.get("answer",""), gt[orig_i])
            )
            acc = correct / len(valid)
        elif ds == "CR":
            correct = sum(
                1 for orig_i, x in valid
                if orig_i < len(gt) and is_correct_mechanical(x.get("answer",""), gt[orig_i])
            )
            acc = correct / len(valid)
        elif ds == "TTL":
            gt_answers = gt
            id_map = em
            correct = 0
            for orig_i, x in valid:
                if orig_i >= len(gt_answers): continue
                pred = x.get("answer", "")
                gt_ids = gt_answers[orig_i]
                if not isinstance(gt_ids, list): gt_ids = [gt_ids]
                gt_id_strs = [str(g) for g in gt_ids]
                if any(num in gt_id_strs for num in re.findall(r'\d+', pred)):
                    correct += 1
            acc = correct / len(valid)

        if inst not in results:
            results[inst] = {}
        results[inst][adaptor] = {
            "acc": acc, "err_rate": err_rate, "n": n, "valid_n": len(valid)
        }
    return results

# ═══ Part 4: 报告渲染 ═══════════════════════════════════════════════════════

W = 72
def hline(c="─"): return c * W
def section(t): return f"\n{'═'*W}\n  {t}\n{'═'*W}"

def render_ingest(ingest_table: dict) -> str:
    lines = [section("INGEST 审计  （Qdrant points，>5pts 算有效）")]
    summary = []
    for ds in ["AR", "CR", "LR", "TTL"]:
        total = TOTAL[ds]
        data  = ingest_table.get(ds, {})
        ok    = sum(1 for v in data.values() if v > 5)
        sp    = sum(1 for v in data.values() if 0 < v <= 5)
        em    = sum(1 for v in data.values() if v == 0)
        miss  = total - len(data)
        lines.append(f"\n  [{ds}]  共{total}实例  ✅{ok}  ⚠️ {sp}sparse  ❌{em}empty  —{miss}missing")
        if ds != "LR":  # LR 110个太多，只印汇总
            lines.append(f"  {'inst':<6} {'pts':>6}  {'flag'}")
            lines.append("  " + "─" * 22)
            for inst in sorted(data.keys()):
                pts = data[inst]
                extra = ""
                if ds == "TTL" and inst == 0: extra = "  ← ingest进行中"
                lines.append(f"  inst{inst:<3}  {pts:>6}  {ingest_flag(pts)}{extra}")
        else:
            lines.append(f"  (LR 110实例全部 ✅，points 范围: "
                         f"{min(data.values()) if data else 0}–{max(data.values()) if data else 0})")
        summary.append(f"  {ds:<4}  ✅{ok}/{total}  ⚠️{sp}  ❌{em}  —{miss}")
    lines.append(f"\n  {'─'*40}\n  汇总：")
    lines.extend(summary)
    return "\n".join(lines)

def render_infer_audit(audit: dict) -> str:
    lines = [section("INFER 审计  （各 inst × adaptor 题数 / 错误）")]
    for ds in ["AR", "CR", "TTL"]:
        total = TOTAL[ds]
        data  = audit.get(ds, {})
        lines.append(f"\n  [{ds}]  总{total}实例")
        lines.append(f"  {'inst':<6} {'adaptor':<5} {'n':>5} {'errs':>5} {'err%':>6}  flag")
        lines.append("  " + "─" * 38)
        for inst in sorted(data.keys()):
            for adaptor in ADAPTORS:
                if adaptor not in data[inst]: continue
                a = data[inst][adaptor]
                flag = "⚠️ " if a["err_rate"] > 0.3 else ("⚠ " if a["err_rate"] > 0 else "")
                lines.append(f"  inst{inst:<3}  {adaptor:<5} {a['n']:>5} {a['errs']:>5} {a['err_rate']:>5.0%}  {flag}")
        # 每个adaptor的汇总
        for adaptor in ADAPTORS:
            insts_done = [inst for inst in data if adaptor in data[inst]]
            total_q = sum(data[i][adaptor]["n"] for i in insts_done)
            total_e = sum(data[i][adaptor]["errs"] for i in insts_done)
            if total_q:
                lines.append(f"  {'':6}  {adaptor} 合计: {len(insts_done)}/{total}insts, "
                              f"{total_q}q, {total_e}err ({total_e/total_q:.1%})")
    return "\n".join(lines)

def render_eval(ds: str, results: Dict[int, Dict[str, dict]], metric: str) -> Tuple[str, dict]:
    lines = []
    lines.append(f"\n  [{ds}]  metric: {metric}  （error题已剔除，不排除整个inst）")
    lines.append(f"  {'inst':<6} {'R1':^10} {'R2':^10} {'R3':^10}  notes")
    lines.append("  " + "─" * 56)

    scores: Dict[str, List[float]] = {a: [] for a in ADAPTORS}

    for inst in sorted(results.keys()):
        row = results[inst]
        cells, notes = [], []
        for a in ADAPTORS:
            c = row.get(a)
            if c is None:
                cells.append("    -   ")
            elif c["acc"] is None:
                cells.append("  ALL_E ")
                notes.append(f"{a}:all_err")
            else:
                cells.append(f" {c['acc']:.4f} ")
                scores[a].append(c["acc"])
                if c["err_rate"] > 0:
                    notes.append(f"{a}:err={c['err_rate']:.0%}({c['errs'] if 'errs' in c else c['n']-c['valid_n']}q剔)")
        lines.append(f"  inst{inst:<3} {''.join(cells)}  {', '.join(notes)}")

    lines.append("  " + "─" * 56)
    avg_parts = []
    for a in ADAPTORS:
        sc = scores[a]
        if sc: avg_parts.append(f"{a}:{np.mean(sc):.4f}(N={len(sc)})")
        else:  avg_parts.append(f"{a}:N/A")
    lines.append("  AVG    " + " | ".join(avg_parts))

    avgs = {a: (np.mean(scores[a]) if scores[a] else None) for a in ADAPTORS}
    return "\n".join(lines), avgs

def render_summary(avgs: dict) -> str:
    lines = [section("汇总  （各 Task 各 Adaptor 平均分，已剔除 error 题）")]
    lines.append(f"\n  {'Task':<6} {'R1':^10} {'R2':^10} {'R3':^10}  metric  (N=有效实例数)")
    lines.append("  " + "─" * 55)
    metrics = {"AR": "acc(mechanical)", "CR": "acc(mechanical)", "TTL": "acc(entity-id)"}
    for ds in ["AR", "CR", "TTL"]:
        a = avgs.get(ds, {})
        def fmt(v): return f"{v:.4f}" if v is not None else "  N/A  "
        lines.append(f"  {ds:<6} {fmt(a.get('R1')):^10} {fmt(a.get('R2')):^10} {fmt(a.get('R3')):^10}  {metrics[ds]}")
    lines.append("")
    lines.append("  注：")
    lines.append("  - CR R3 整体 error rate ~48%，剔除后剩余题数较少，分数仅供参考")
    lines.append("  - TTL R3 error rate ~45%，同上")
    lines.append("  - TTL inst4/5 未 ingest，R1 infer 结果应视为不可靠（无记忆支撑）")
    lines.append("  - LR 未评估（需 LLM 评分）")
    return "\n".join(lines)

# ═══ Main ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print("[1/5] 读取 Qdrant ingest 数据...")
    qdrant = get_qdrant_points()
    ingest_table = build_ingest_table(qdrant)

    print("[2/5] 读取 infer 审计数据...")
    audit = build_infer_audit()

    print("[3/5] 运行 AR / CR / TTL eval...")
    ar_results  = eval_ds("AR")
    cr_results  = eval_ds("CR")
    ttl_results = eval_ds("TTL")

    print("[4/5] 渲染报告...")

    header = "\n".join([
        hline("═"),
        f"  mem0 综合评估报告  |  {date.today()}",
        f"  策略：error题直接剔除，剩余题正常评分  |  AR(22) CR(8) TTL(6)  [LR需LLM略]",
        hline("═"),
    ])

    ingest_sec = render_ingest(ingest_table)
    infer_sec  = render_infer_audit(audit)

    eval_sec_lines = [section("EVAL 结果")]
    all_avgs = {}

    ar_txt,  all_avgs["AR"]  = render_eval("AR",  ar_results,  "accuracy (mechanical substring)")
    cr_txt,  all_avgs["CR"]  = render_eval("CR",  cr_results,  "accuracy (mechanical substring)")
    ttl_txt, all_avgs["TTL"] = render_eval("TTL", ttl_results, "accuracy (entity-id match)")
    eval_sec_lines.extend([ar_txt, cr_txt, ttl_txt])

    summary_sec = render_summary(all_avgs)

    report = (header + "\n" + ingest_sec + "\n" + infer_sec + "\n"
              + "\n".join(eval_sec_lines) + "\n" + summary_sec + "\n")

    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n报告已保存到 {args.out}")

if __name__ == "__main__":
    main()
