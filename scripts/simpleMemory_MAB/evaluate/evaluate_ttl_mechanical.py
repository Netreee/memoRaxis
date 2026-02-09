
import json
import argparse
import re
from pathlib import Path

def load_id_map():
    with open("MemoryAgentBench/entity2id.json", 'r') as f:
        data = json.load(f)
    
    # Invert: ID -> Clean Title
    id_to_title = {}
    for uri, idx in data.items():
        # Clean URI: <http://dbpedia.org/resource/Water_(1985_film)> -> Water (1985 film)
        raw = uri.replace("<http://dbpedia.org/resource/", "").replace(">", "")
        title = raw.replace("_", " ")
        id_to_title[str(idx)] = title
        
    return id_to_title

def normalize(text):
    return re.sub(r'[^\w\s]', '', text).lower().strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_pattern", type=str, default="out/ttl_results_*.json")
    args = parser.parse_args()

    import glob
    files = glob.glob(args.results_pattern)
    files.sort()
    
    if not files:
        print("No files found.")
        return

    id_map = load_id_map()
    print(f"Loaded {len(id_map)} entity mappings.")

    print(f"\n{'Inst':<5} | {'Adaptor':<8} | {'Accuracy':<10}")
    print("-" * 30)

    global_stats = {"R1": [], "R2": [], "R3": []}

    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        instance_idx = data["instance_idx"]
        results = data["results"]
        
        for adaptor, items in results.items():
            correct = 0
            total = 0
            
            for item in items:
                pred = item.get("answer", "").strip()
                gt_ids = item.get("ground_truth", [])
                
                # --- 评测逻辑分流 ---
                if instance_idx == 0:
                    # 电影推荐：需映射 ID -> Title
                    is_hit = False
                    for gid in gt_ids:
                        title = id_map.get(str(gid))
                        if title:
                            # 宽松匹配
                            if title.lower() in pred.lower():
                                is_hit = True
                                break
                            base_title = re.sub(r'\s\(\d{4}.*\)', '', title)
                            if base_title.lower() in pred.lower() and len(base_title) > 3:
                                 is_hit = True
                                 break
                    if is_hit: correct += 1
                    
                else:
                    # Banking77：直接比对数字 ID
                    # 模型输出可能是 "28" 或 "Label 28" 或 "28."，尝试提取数字
                    match = re.search(r'\b(\d+)\b', pred)
                    if match:
                        pred_id = match.group(1)
                        # GT 是 ["28"]
                        if pred_id in [str(g) for g in gt_ids]:
                            correct += 1
                
                total += 1
            
            acc = correct / total if total > 0 else 0
            global_stats[adaptor].append(acc)
            print(f"{instance_idx:<5} | {adaptor:<8} | {acc:.2%}")

    print("\n=== Global Average Accuracy ===")
    for ad, scores in global_stats.items():
        if scores:
            print(f"{ad}: {sum(scores)/len(scores):.2%}")

if __name__ == "__main__":
    main()
