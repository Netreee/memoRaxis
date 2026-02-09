
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

def normalize_text(text: str) -> str:
    """ 基础文本归一化：转小写，去除标点，去除多余空格 """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def is_correct_mechanical(prediction: str, references: List[str]) -> bool:
    """ 
    机械评测逻辑：
    1. 检查预测中是否包含拒答词
    2. 检查预测中是否包含标准答案中的任意一个关键词/短语
    """ 
    # 1. 拒答检测：如果 Agent 明确说没找到信息，即使它靠常识猜对了，也算错（符合 RAG 评测逻辑）
    negative_patterns = [
        "does not contain any information",
        "insufficient information",
        "not mentioned in the context",
        "no information related to",
        "上下文没有提到",
        "没有找到相关信息",
        "信息不足"
    ]
    
    pred_norm = prediction.lower()
    for pattern in negative_patterns:
        if pattern in pred_norm:
            # 特殊情况：如果它说“没找到 A，但答案是 B”，且 B 在参考答案里，这里依然判错
            # 因为这说明检索失效了
            return False

    # 2. 关键词匹配
    for ref in references:
        ref_norm = normalize_text(ref)
        # 简单包含判定 (也可以改用更复杂的 F1 Score 或 EM)
        if ref_norm in normalize_text(prediction):
            return True
            
    return False

def main():
    parser = argparse.ArgumentParser(description="Mechanical Scorer for Accurate_Retrieval")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--instance", type=str, required=True, help="Path to ground truth JSON")
    args = parser.parse_args()

    with open(args.results, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    with open(args.instance, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    qa_map = {}
    for q, a in zip(ground_truth['questions'], ground_truth['answers']):
        qa_map[q] = a

    summary = {}
    
    for adaptor_name, predictions in results_data['results'].items():
        correct_count = 0
        total_count = len(predictions)
        
        for item in predictions:
            q = item.get('question', '')
            # 如果没有 answer 字段（例如报错了），则默认为空字符串，自动判错
            pred = item.get('answer', '')
            ref = qa_map.get(q, [])
            
            if is_correct_mechanical(pred, ref):
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        summary[adaptor_name] = {
            "accuracy": accuracy,
            "total_questions": total_count,
            "correct_count": correct_count
        }

    print(f"\n--- [Mechanical Evaluation Result: {Path(args.results).name}] ---")
    for adaptor, stats in summary.items():
        print(f"Adaptor {adaptor:3}: Accuracy = {stats['accuracy']:.2%} ({stats['correct_count']}/{stats['total_questions']})")

if __name__ == "__main__":
    main()
