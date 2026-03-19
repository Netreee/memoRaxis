#!/usr/bin/env python3
"""
consolidate_results.py
将各系统的推理结果汇总到 results/ 目录，原始文件不动。
命名规则：results/{system}/{dataset}/instance_{i}.json
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

DATASET_MAP = {
    "Accurate_Retrieval":       "AR",
    "Conflict_Resolution":      "CR",
    "Long_Range_Understanding": "LR",
    "Test_Time_Learning":       "TTL",
}

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def copy_system(src_dir: Path, system: str, file_prefix: str, dataset_key: str):
    """复制 mem0 / mem0g 格式（已包含 R1/R2/R3）。"""
    count = 0
    for f in sorted(src_dir.glob(f"{file_prefix}*.json")):
        try:
            d = json.load(f.open())
            ds = d.get("dataset", "")
            inst = d.get("instance_idx")
            if inst is None:
                continue
            short = DATASET_MAP.get(ds, ds)
            dest = ensure(RESULTS / system / short) / f"instance_{inst}.json"
            if dest.exists():
                # 合并 adaptors（不覆盖已有）
                existing = json.loads(dest.read_text())
                existing.setdefault("results", {}).update(d.get("results", {}))
                write_json(dest, existing)
            else:
                shutil.copy2(f, dest)
            count += 1
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
    print(f"  [{system}] {count} files processed from {src_dir}")

def merge_memgpt():
    """合并 out/ 下的 memgpt / memgpt_r2 / memgpt_r3 文件到单个 instance json。"""
    src = ROOT / "out"
    # 收集所有 memgpt 文件，按 (dataset_short, inst) 分组
    groups = defaultdict(dict)  # (ds, inst) -> {R1: data, R2: data, R3: data}

    patterns = [
        # (glob, adaptor)
        ("*_memgpt.json",    "R1"),
        ("*_memgpt_r2.json", "R2"),
        ("*_memgpt_r3.json", "R3"),
    ]
    ds_prefix = {
        "acc_ret":      "AR",
        "conflict_res": "CR",
        "long_range":   "LR",
        "ttl":          "TTL",
    }

    for glob_pat, adaptor in patterns:
        for f in sorted(src.glob(glob_pat)):
            try:
                d = json.load(f.open())
                ds_raw = d.get("dataset", "")
                inst = d.get("instance_idx")
                short = DATASET_MAP.get(ds_raw)
                if short is None:
                    # fallback: guess from filename
                    stem = f.stem
                    for k, v in ds_prefix.items():
                        if stem.startswith(k):
                            short = v; break
                if short is None or inst is None:
                    continue
                rows = d.get("results", {})
                # 取第一个 key（通常 R1/R2/R3）
                for k, v in rows.items():
                    groups[(short, inst)][k] = v
            except Exception as e:
                print(f"  SKIP {f.name}: {e}")

    count = 0
    for (short, inst), adaptor_results in groups.items():
        dest = ensure(RESULTS / "memgpt" / short) / f"instance_{inst}.json"
        if dest.exists():
            existing = json.loads(dest.read_text())
            existing.setdefault("results", {}).update(adaptor_results)
            write_json(dest, existing)
        else:
            write_json(dest, {
                "dataset": {v: k for k, v in DATASET_MAP.items()}.get(short, short),
                "instance_idx": inst,
                "system": "memgpt",
                "results": adaptor_results,
            })
        count += 1
    print(f"  [memgpt] {count} instances merged")

def copy_raptor():
    """复制 raptor 结果。"""
    src = ROOT / "out"
    ds_prefix = {
        "acc_ret":      "AR",
        "conflict_res": "CR",
        "long_range":   "LR",
        "ttl":          "TTL",
    }
    count = 0
    for f in sorted(src.glob("*_raptor.json")):
        try:
            d = json.load(f.open())
            ds_raw = d.get("dataset", "")
            inst = d.get("instance_idx")
            short = DATASET_MAP.get(ds_raw)
            if short is None:
                for k, v in ds_prefix.items():
                    if f.stem.startswith(k): short = v; break
            if short is None or inst is None:
                continue
            dest = ensure(RESULTS / "raptor" / short) / f"instance_{inst}.json"
            shutil.copy2(f, dest)
            count += 1
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
    # also check gaoang/out
    src2 = ROOT / "gaoang" / "out"
    for f in sorted(src2.glob("*_raptor.json")) if src2.exists() else []:
        try:
            d = json.load(f.open())
            ds_raw = d.get("dataset", "")
            inst = d.get("instance_idx")
            short = DATASET_MAP.get(ds_raw)
            if short is None or inst is None:
                continue
            dest = ensure(RESULTS / "raptor" / short) / f"instance_{inst}.json"
            if not dest.exists():
                shutil.copy2(f, dest)
                count += 1
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
    print(f"  [raptor] {count} files processed")

def report():
    print("\n=== 汇总报告 ===")
    for system_dir in sorted(RESULTS.iterdir()):
        if not system_dir.is_dir():
            continue
        system = system_dir.name
        for ds_dir in sorted(system_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            files = sorted(ds_dir.glob("instance_*.json"))
            adaptors = set()
            for f in files:
                try:
                    d = json.loads(f.read_text())
                    adaptors.update(d.get("results", {}).keys())
                except:
                    pass
            print(f"  {system:10s} {ds_dir.name:5s}: {len(files):3d} instances  adaptors={sorted(adaptors)}")

if __name__ == "__main__":
    print("=== consolidate_results.py ===")
    print(f"目标目录: {RESULTS}")
    RESULTS.mkdir(exist_ok=True)

    # mem0
    copy_system(ROOT / "out" / "mem0", "mem0", "mem0_", "mem0")
    # mem0g
    copy_system(ROOT / "out" / "mem0g", "mem0g", "mem0g_", "mem0g")
    # memgpt（需合并 R1/R2/R3）
    merge_memgpt()
    # raptor
    copy_raptor()

    report()
    print("\n完成！原始文件未修改。")
