
import json

def find_case():
    # 注意：我们要找的是 conflict_res_results_4.json
    with open("out/conflict_res_results_4.json", 'r') as f:
        data = json.load(f)
    
    r1_results = data["results"]["R1"]
    r3_results = data["results"]["R3"]
    
    # 我们需要 GT 来判断对错，或者直接对比 R1 和 R3 的答案
    # 既然 R1 的 EM 很高，我们找 R1 != R3 且 R1 包含关键信息的
    
    for i in range(len(r1_results)):
        q = r1_results[i]["question"]
        ans1 = r1_results[i]["answer"]
        ans3 = r3_results[i]["answer"]
        
        # 简单粗暴：如果 R1 长度适中且 R3 回答“信息不足”或明显跑偏
        if ans1.lower() != ans3.lower() and len(ans1) > 5:
            # 进一步检查 R3 是否包含拒答词
            if "insufficient" in ans3.lower() or "not contain" in ans3.lower() or "无法确定" in ans3:
                print(f"Index {i} matches!")
                print(f"Q: {q[:100]}...")
                print(f"R1: {ans1}")
                print(f"R3 (Old): {ans3}")
                print("-" * 20)

if __name__ == "__main__":
    find_case()
