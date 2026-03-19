#!/usr/bin/env python3
"""
三系统综合对比报告：mem0g / mem0 / simpleMemory
用法:
    cd /Users/bytedance/proj/memoRaxis
    python3 scripts/combined_report.py
    python3 scripts/combined_report.py --out /tmp/combined_report.txt
"""
import argparse, glob, json, re, sys
import numpy as np
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmark_utils import load_benchmark_data

ADAPTORS   = ["R1", "R2", "R3"]
PREVIEW    = ROOT / "MemoryAgentBench" / "preview_samples"
PARQUET    = ROOT / "MemoryAgentBench" / "data" / "Accurate_Retrieval-00000-of-00001.parquet"
ENTITY_MAP = ROOT / "MemoryAgentBench" / "entity2id.json"

MEM0G_THRESHOLD = 0.30   # error > 30% → EXCL 整个 (inst,adaptor)
# mem0 / simpleMemory: 只剔除 error 题，不整个排除

W = 74

def hline(c="─"): return c * W
def section(title): return f"\n{'═'*W}\n  {title}\n{'═'*W}"

# ─── GT loaders ──────────────────────────────────────────────────────────────
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

# ─── Scoring helpers ──────────────────────────────────────────────────────────
def normalize_text(t):
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r'[^\w\s]', '', t)
    return " ".join(t.split())

NEG_PATTERNS = [
    "does not contain any information","insufficient information",
    "not mentioned in the context","no information related to",
    "上下文没有提到","没有找到相关信息","信息不足","上下文信息不足",
    "i don't know","i cannot answer",
]

def is_correct_mechanical(pred, refs):
    if refs is None: return False
    if hasattr(refs, 'tolist'): refs = refs.tolist()
    if not refs: return False
    pl = pred.lower()
    for p in NEG_PATTERNS:
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

def score_ttl_items(valid_with_idx, gt):
    correct = 0
    for orig_i, item in valid_with_idx:
        if orig_i >= len(gt): continue
        pred = item.get("answer","")
        gt_ids = [str(g) for g in (gt[orig_i] if isinstance(gt[orig_i], list) else [gt[orig_i]])]
        if any(n in gt_ids for n in re.findall(r'\d+', pred)):
            correct += 1
    return correct / len(valid_with_idx) if valid_with_idx else 0.0

def score_mech_items(valid_with_idx, gt):
    c = sum(1 for i, x in valid_with_idx if i < len(gt) and
            is_correct_mechanical(x.get("answer",""), gt[i]))
    return c / len(valid_with_idx) if valid_with_idx else 0.0

# ─── Per-system eval ──────────────────────────────────────────────────────────
# Returns {inst: {adaptor: {acc, err_rate, n, valid_n, excluded}}}

def eval_files(files_dict, ds, threshold=None, exclude_whole=False):
    """
    files_dict: {inst: filepath_or_dict}
      - filepath: JSON file path
      - dict: already-loaded {"results": {adaptor: items}}
    threshold: error rate threshold (None = never exclude)
    exclude_whole: if True and err_rate > threshold → exclude cell entirely
    """
    em = entity_map() if ds == "TTL" else None
    results = {}
    for inst, fpath_or_dict in sorted(files_dict.items()):
        if isinstance(fpath_or_dict, dict):
            d = fpath_or_dict
        else:
            try:
                d = json.load(open(fpath_or_dict))
            except: continue
        try:
            if ds == "AR":   gt = ar_gt(inst)
            elif ds == "CR": gt = cr_gt(inst)
            elif ds == "TTL": gt = ttl_gt(inst)
            else: gt = None
        except Exception as e:
            print(f"  [WARN] {ds} inst{inst} GT error: {e}")
            continue
        inst_res = {}
        for adaptor, items in d.get("results", {}).items():
            n = len(items)
            errs = sum(1 for x in items if "error" in x)
            err_rate = errs / n if n else 0
            excluded = exclude_whole and threshold is not None and err_rate > threshold
            if excluded:
                inst_res[adaptor] = dict(acc=None, err_rate=err_rate, n=n, valid_n=0, excluded=True)
            else:
                valid = [(i, x) for i, x in enumerate(items) if "error" not in x]
                if ds == "AR":   acc = score_mech_items(valid, gt)
                elif ds == "CR": acc = score_mech_items(valid, gt)
                elif ds == "TTL": acc = score_ttl_items(valid, gt)
                else: acc = None
                inst_res[adaptor] = dict(acc=acc, err_rate=err_rate, n=n, valid_n=len(valid), excluded=False)
        results[inst] = inst_res
    return results

# ─── File discovery ───────────────────────────────────────────────────────────
def find_mem0g_files(ds):
    patterns = {"AR":"mem0g_acc_ret_results_*.json","CR":"mem0g_conflict_results_*.json",
                "LR":"mem0g_long_range_results_*.json","TTL":"mem0g_ttl_results_*.json"}
    regexes  = {"AR":r"acc_ret_results_(\d+)","CR":r"conflict_results_(\d+)",
                "LR":r"long_range_results_(\d+)","TTL":r"ttl_results_(\d+)"}
    out = {}
    for fpath in glob.glob(str(ROOT/"out"/"mem0g"/patterns[ds])):
        m = re.search(regexes[ds], fpath)
        if m: out[int(m.group(1))] = fpath
    return out

def find_mem0_files(ds):
    """返回 {inst: merged_results_dict}，合并 R1/R2/R3 各 adaptor 分散文件"""
    prefix_map = {"AR":"acc_ret_","CR":"conflict_","LR":"long_range_","TTL":"ttl_"}
    prefix = prefix_map[ds]
    base_dir = ROOT / "out" / "mem0"
    merged = {}  # {inst: {"results": {adaptor: items}}}
    for fpath in sorted(glob.glob(str(base_dir / f"mem0_{prefix}results_*.json"))):
        stem = Path(fpath).stem
        rest = stem.replace(f"mem0_{prefix}results_", "")
        if rest.endswith("_r3"):   adaptor, inst_str = "R3", rest[:-3]
        elif rest.endswith("_r2"): adaptor, inst_str = "R2", rest[:-3]
        else:                      adaptor, inst_str = "R1", rest
        try:
            inst = int(inst_str)
        except ValueError:
            continue
        try:
            d = json.load(open(fpath))
        except: continue
        if inst not in merged:
            merged[inst] = {"results": {}}
        for a, items in d.get("results", {}).items():
            merged[inst]["results"][a] = items
    return merged  # {inst: dict} — eval_files also accepts dict values

def find_sm_files(ds):
    sm_dir = ROOT / "results" / "simpleMemory"
    patterns = {"AR":"acc_ret_results_*.json","CR":"conflict_res_results_*.json",
                "LR":"long_range_results_*.json","TTL":"ttl_results_*.json"}
    regexes  = {"AR":r"acc_ret_results_(\d+)","CR":r"conflict_res_results_(\d+)",
                "LR":r"long_range_results_(\d+)","TTL":r"ttl_results_(\d+)"}
    out = {}
    for fpath in glob.glob(str(sm_dir / patterns[ds])):
        m = re.search(regexes[ds], fpath)
        if m: out[int(m.group(1))] = fpath
    return out

# ─── Summary stats ────────────────────────────────────────────────────────────
def avg_by_adaptor(results):
    """Returns {adaptor: (mean, N, n_excl)} or None if no data"""
    per = {a: [] for a in ADAPTORS}
    excl = {a: 0 for a in ADAPTORS}
    for inst_data in results.values():
        for a, cell in inst_data.items():
            if cell["excluded"]:
                excl[a] += 1
            elif cell["acc"] is not None:
                per[a].append(cell["acc"])
    return {a: (np.mean(per[a]), len(per[a]), excl[a]) if per[a] else (None, 0, excl[a])
            for a in ADAPTORS}

# ─── Render helpers ───────────────────────────────────────────────────────────
def fmt_cell(v, excluded=False):
    if excluded: return " EXCL "
    if v is None: return "  -   "
    return f"{v:.4f}"

def render_per_inst_table(results, title, note=""):
    lines = [f"\n  [{title}]"]
    if note: lines.append(f"  ({note})")
    lines.append(f"  {'inst':<7} {'R1':^8} {'R2':^8} {'R3':^8}  err-notes")
    lines.append("  " + "─"*52)
    for inst in sorted(results.keys()):
        row = results[inst]
        cells = []
        enotes = []
        for a in ADAPTORS:
            c = row.get(a)
            if c is None:
                cells.append("  -   ")
            else:
                cells.append(fmt_cell(c["acc"], c["excluded"]))
                if c["err_rate"] > 0:
                    enotes.append(f"{a}:{c['err_rate']:.0%}({c['errs'] if 'errs' in c else int(c['n']*c['err_rate'])}q)")
        lines.append(f"  inst{inst:<5} {''.join(f'{x:^8}' for x in cells)}  {', '.join(enotes)}")
    lines.append("  " + "─"*52)
    avgs = avg_by_adaptor(results)
    parts = []
    for a in ADAPTORS:
        mean, n, ex = avgs[a]
        if mean is not None: parts.append(f"{a}:{mean:.4f}(N={n},excl={ex})")
        else: parts.append(f"{a}:N/A")
    lines.append(f"  AVG  {' | '.join(parts)}")
    return "\n".join(lines)

def render_comparison_table(systems_avgs, ds, metric_label):
    """
    systems_avgs: {system_name: {adaptor: (mean, N, excl)}}
    """
    lines = []
    lines.append(f"\n  [{ds}]  metric: {metric_label}")
    lines.append(f"  {'System':<14} {'R1':^10} {'R2':^10} {'R3':^10}  (N=有效inst数)")
    lines.append("  " + "─"*52)
    for sys_name, avgs in systems_avgs.items():
        parts = []
        for a in ADAPTORS:
            mean, n, ex = avgs.get(a, (None, 0, 0))
            if mean is not None: parts.append(f"{mean:.4f}({n})")
            else: parts.append("  N/A   ")
        lines.append(f"  {sys_name:<14} {''.join(f'{x:^10}' for x in parts)}")
    return "\n".join(lines)

def parse_lr_results(lr_file):
    results = {}
    try:
        text = open(lr_file).read()
    except FileNotFoundError:
        return {}
    for line in text.splitlines():
        m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', line)
        if m:
            inst = int(m.group(1))
            results[inst] = {"R1": float(m.group(2)), "R2": float(m.group(3)), "R3": float(m.group(4))}
    return results

def lr_avgs_from_parsed(lr_results, has_infer_set):
    """
    has_infer_set: {inst: set of adaptors with actual infer results}
    """
    per = {a: [] for a in ADAPTORS}
    for inst, row in lr_results.items():
        for a in ADAPTORS:
            if a not in has_infer_set.get(inst, set()): continue
            v = row.get(a)
            if v is not None:
                per[a].append(v)
    return {a: (np.mean(per[a]), len(per[a]), 0) if per[a] else (None, 0, 0) for a in ADAPTORS}

def build_has_infer(files_dict, ds):
    """Returns {inst: set_of_adaptors}"""
    result = {}
    for inst, fpath_or_dict in files_dict.items():
        if isinstance(fpath_or_dict, dict):
            d = fpath_or_dict
        else:
            try:
                d = json.load(open(fpath_or_dict))
            except: continue
        result[inst] = set(d.get("results", {}).keys())
    return result

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--mem0g-lr",  default="/tmp/mem0g_lr_eval.txt")
    parser.add_argument("--sm-lr",     default="/tmp/sm_lr_eval.txt")
    parser.add_argument("--mem0-lr",   default="/tmp/mem0_lr_eval.txt")
    args = parser.parse_args()

    print("正在计算各系统评分...")

    # ── Discover files ─────────────────────────────────────────────────────────
    files = {
        "mem0g":  {ds: find_mem0g_files(ds) for ds in ["AR","CR","TTL","LR"]},
        "mem0":   {ds: find_mem0_files(ds)  for ds in ["AR","CR","TTL","LR"]},
        "simpleMem": {ds: find_sm_files(ds) for ds in ["AR","CR","TTL","LR"]},
    }

    # ── Eval AR / CR / TTL ─────────────────────────────────────────────────────
    results = {}
    for sys in ["mem0g", "mem0", "simpleMem"]:
        excl = (sys == "mem0g")
        thresh = MEM0G_THRESHOLD if excl else None
        results[sys] = {}
        for ds in ["AR","CR","TTL"]:
            results[sys][ds] = eval_files(files[sys][ds], ds, thresh, excl)

    # ── LR (use pre-computed LLM scores where available) ──────────────────────
    lr_data = {
        "mem0g":    parse_lr_results(args.mem0g_lr),
        "simpleMem": parse_lr_results(args.sm_lr),
        "mem0":     parse_lr_results(args.mem0_lr),
    }
    lr_has = {
        "mem0g":    build_has_infer(files["mem0g"]["LR"], "LR"),
        "simpleMem": build_has_infer(files["simpleMem"]["LR"], "LR"),
        "mem0":     build_has_infer(files["mem0"]["LR"], "LR"),
    }

    # ── Build comparison avgs ─────────────────────────────────────────────────
    comp = {}
    for ds in ["AR","CR","TTL"]:
        comp[ds] = {}
        for sys in ["mem0g","mem0","simpleMem"]:
            comp[ds][sys] = avg_by_adaptor(results[sys][ds])

    comp["LR"] = {}
    for sys in ["mem0g","mem0","simpleMem"]:
        comp["LR"][sys] = lr_avgs_from_parsed(lr_data[sys], lr_has[sys])

    # ─── Render ────────────────────────────────────────────────────────────────
    lines = []
    lines.append("═"*W)
    lines.append(f"  三系统综合对比报告  |  {date.today()}")
    lines.append(f"  系统: mem0g (excl整inst>30%err) | mem0 (只剔除err题) | simpleMemory (只剔除err题)")
    lines.append("═"*W)

    # ════ Section 1: 汇总对比表 ════
    lines.append(section("汇总对比  （各系统各Task各Adaptor均分）"))
    metric_labels = {"AR":"acc(mech)","CR":"acc(mech)","TTL":"acc(entity-id)","LR":"F1(LLM,partial)"}
    for ds in ["AR","CR","TTL","LR"]:
        lines.append(render_comparison_table(comp[ds], ds, metric_labels[ds]))

    lines.append(f"\n  注：")
    lines.append(f"  - mem0g: AR仅5/22 inst, TTL仅5/6 inst (数据覆盖有限)")
    lines.append(f"  - mem0g CR: inst0仅有R3(10题)；inst5 R1被排除(err=49%)")
    lines.append(f"  - mem0g LR: 13/40 inst (R3仅6 inst)")
    lines.append(f"  - mem0 TTL: inst4/5无有效记忆(ingest=0)，R1结果不可信")
    lines.append(f"  - simpleMem LR: {'有' if lr_data['simpleMem'] else '无'} LLM eval 结果")

    # ════ Section 2: AR 详细 ════
    lines.append(section("AR (Accurate Retrieval) 各inst详细"))
    for sys, label in [("mem0g","mem0g (excl>30%)"), ("mem0","mem0 (filter-err)"), ("simpleMem","simpleMemory")]:
        note = "仅5个ingest inst" if sys=="mem0g" else ""
        lines.append(render_per_inst_table(results[sys]["AR"], f"AR {label}", note))

    # ════ Section 3: CR 详细 ════
    lines.append(section("CR (Conflict Resolution) 各inst详细"))
    for sys, label in [("mem0g","mem0g"), ("mem0","mem0"), ("simpleMem","simpleMemory")]:
        lines.append(render_per_inst_table(results[sys]["CR"], f"CR {label}"))

    # ════ Section 4: TTL 详细 ════
    lines.append(section("TTL (Test-Time Learning) 各inst详细"))
    for sys, label in [("mem0g","mem0g"), ("mem0","mem0"), ("simpleMem","simpleMemory")]:
        lines.append(render_per_inst_table(results[sys]["TTL"], f"TTL {label}"))

    # ════ Section 5: LR ════
    lines.append(section("LR (Long Range Understanding) — LLM F1 (where available)"))
    for sys in ["mem0g", "mem0", "simpleMem"]:
        lr = lr_data[sys]
        has = lr_has[sys]
        if not lr:
            lines.append(f"\n  [{sys}]  LR eval 未就绪 (--{sys.replace('0','0').lower()}-lr 未找到)")
            continue
        lines.append(f"\n  [{sys}]  ({len(lr)} inst evaluated)")
        lines.append(f"  {'inst':<7} {'R1':^8} {'R2':^8} {'R3':^8}")
        lines.append("  " + "─"*36)
        per = {a: [] for a in ADAPTORS}
        for inst in sorted(lr.keys()):
            row = lr[inst]
            cells = []
            for a in ADAPTORS:
                if a not in has.get(inst, set()):
                    cells.append("  -   ")
                else:
                    v = row.get(a, 0.0)
                    cells.append(f"{v:.4f}")
                    per[a].append(v)
            lines.append(f"  inst{inst:<5} {''.join(f'{x:^8}' for x in cells)}")
        lines.append("  " + "─"*36)
        avg_parts = [f"{a}:{np.mean(per[a]):.4f}(N={len(per[a])})" if per[a] else f"{a}:N/A"
                     for a in ADAPTORS]
        lines.append(f"  AVG  {' | '.join(avg_parts)}")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告已保存到 {args.out}")

if __name__ == "__main__":
    main()
