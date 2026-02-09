# -*- coding: utf-8 -*-
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.logger import get_logger
from src.config import get_config
from src.llm_interface import OpenAIClient

logger = get_logger()

# 使用三引号包裹多行字符串，注意内部不要出现冲突的引号
JUDGE_PROMPT_TEMPLATE = """

你是一个严谨的评测裁判。你需要根据【标准答案】来评估【Agent预测】的准确性。

【问题】: {question}
【标准答案】: {reference_answers}
【Agent预测】: {prediction}

评估标准：
1. 如果 Agent 准确地给出了标准答案中的信息，得 1 分。
2. 如果 Agent 回答“信息不足”或“无法确定”，但标准答案中确实存在正确信息，得 0 分。
3. 如果 Agent 给出了错误的、与事实不符的信息，得 0 分。
4. 对于事实性问题，Agent 的回答可以比标准答案更详细，只要核心事实正确即可。

请直接输出 JSON 格式的结果：
{{"score": 0, "reason": "简短的理由"}} 或 {{"score": 1, "reason": "简短的理由"}}
不要输出任何其他多余文字。
"""

class LLMJudge:
    def __init__(self):
        conf = get_config()
        self.llm = OpenAIClient(
            api_key=conf.llm["api_key"],
            base_url=conf.llm["base_url"],
            model=conf.llm["model"]
        )

    def judge(self, question: str, reference: List[str], prediction: str) -> Dict[str, Any]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            reference_answers=", ".join(reference),
            prediction=prediction
        )
        try:
            # generate_json 内部会处理 Markdown 块和解析
            result = self.llm.generate_json(prompt)
            if not result:
                return {"score": 0, "reason": "LLM 返回了空结果"}
            
            # 兼容性处理：确保 score 字段存在且为数字
            score = result.get("score", 0)
            try:
                result["score"] = int(score)
            except (ValueError, TypeError):
                result["score"] = 0
                
            return result
        except Exception as e:
            logger.error(f"Judge failed: {e}")
            return {"score": 0, "reason": f"Error: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="LLM as a Judge for MemoryAgentBench")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--instance", type=str, required=True, help="Path to ground truth instance JSON")
    args = parser.parse_args()

    # 1. 加载数据
    if not Path(args.results).exists():
        print(f"错误: 结果文件不存在 {args.results}")
        return
    if not Path(args.instance).exists():
        print(f"错误: 实例文件不存在 {args.instance}")
        return

    with open(args.results, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    with open(args.instance, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # 创建问题到答案的映射
    qa_map = {}
    for q, a in zip(ground_truth['questions'], ground_truth['answers']):
        qa_map[q] = a

    judge_engine = LLMJudge()
    
    final_eval = {
        "dataset": results_data.get("dataset", "unknown"),
        "instance_idx": results_data.get("instance_idx", 0),
        "summary": {},
        "details": {}
    }

    # 2. 遍历每个适配器的结果
    for adaptor_name, predictions in results_data['results'].items():
        logger.info(f"正在评测适配器: {adaptor_name}...")
        total_score = 0
        count = 0
        adaptor_details = []

        for item in predictions:
            q = item['question']
            pred = item['answer']
            ref = qa_map.get(q, [])

            if not ref:
                continue

            # 调用 LLM 打分
            eval_res = judge_engine.judge(q, ref, pred)
            
            score = eval_res.get("score", 0)
            total_score += score
            count += 1

            adaptor_details.append({
                "question": q,
                "prediction": pred,
                "reference": ref,
                "score": score,
                "reason": eval_res.get("reason", ""),
                "tokens": item.get("tokens", 0),
                "steps": item.get("steps", 0)
            })
            
            if count % 10 == 0:
                print(f"[{adaptor_name}] 已评测 {count}/{len(predictions)} 个问题...", flush=True)

        accuracy = total_score / count if count > 0 else 0
        final_eval["summary"][adaptor_name] = {
            "accuracy": accuracy,
            "total_questions": count,
            "avg_tokens": sum(d['tokens'] for d in adaptor_details) / count if count > 0 else 0
        }
        final_eval["details"][adaptor_name] = adaptor_details

    # 3. 保存评估结果
    output_dir = Path("out/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_stem = Path(args.results).stem
    output_file = output_dir / f"eval_{result_stem}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_eval, f, indent=2, ensure_ascii=False)

    print(f"\n评测完成！摘要结果: {final_eval['summary']}")
    print(f"详细报告已保存至: {output_file}")

if __name__ == "__main__":
    main()