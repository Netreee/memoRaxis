
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.logger import get_logger
from src.config import get_config
from src.llm_interface import OpenAIClient

logger = get_logger()

# 专门针对 Q2 设定的 Judge Prompt
JUDGE_PROMPT_TEMPLATE = """
你是一个严谨的评测裁判。你需要根据【标准答案】来评估【Agent预测】的准确性。

【问题】: {question}
【标准答案】: {ground_truth}
【Agent预测】: {prediction}

评估标准：
1. 核心事实匹配：如果 Agent 准确地给出了标准答案中的核心信息，得 1 分。
2. 细节宽容：对于事实性问题，Agent 的回答可以比标准答案更详细，只要核心事实没有错误或遗漏即可。
3. 如果标准答案包含多个点（例如一个链接和一个数字），Agent 必须全部答对才能得 1 分；漏答或错答任何一个核心点均得 0 分。
4. 任何与事实不符、产生幻觉的回答，得 0 分。

请直接输出 JSON 格式的结果：
{{"score": 0, "reason": "简短的理由"}} 或 {{"score": 1, "reason": "简短的理由"}}
不要输出任何其他多余文字。
"""

class Q2Judge:
    def __init__(self):
        conf = get_config()
        self.llm = OpenAIClient(
            api_key=conf.llm["api_key"],
            base_url=conf.llm["base_url"],
            model=conf.llm["model"]
        )

    def judge(self, question: str, gt: str, pred: str) -> Dict[str, Any]:
        # Rule-based fast pass for negative samples
        gt_clean = gt.strip()
        pred_clean = pred.strip()
        
        if gt_clean == "无法回答":
            if "无法回答" in pred_clean:
                return {"score": 1, "reason": "Rule-based: 正确识别为无法回答"}
            else:
                return {"score": 0, "reason": "Rule-based: 应该是无法回答，但模型给出了猜测"}
        
        if "无法回答" in pred_clean and gt_clean != "无法回答":
            return {"score": 0, "reason": "Rule-based: 模型放弃回答，但其实有答案"}

        # LLM based evaluation for complex answers
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=gt,
            prediction=pred
        )
        try:
            result = self.llm.generate_json(prompt)
            if not result:
                return {"score": 0, "reason": "LLM Judge returned empty"}
            
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
    parser = argparse.ArgumentParser(description="Evaluate Q2 Benchmark Results")
    parser.add_argument("--results_file", type=str, default="out/q2_benchmark/q2_infer_results.json", help="Path to infer results JSON")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    judge_engine = Q2Judge()
    
    final_eval = {
        "dataset": "Q2_Benchmark",
        "summary": {},
        "details": {}
    }

    # 遍历每个适配器的结果
    for adaptor_name, predictions in results_data.get('results', {}).items():
        logger.info(f"Judging {adaptor_name}...")
        total_score = 0
        count = 0
        adaptor_details = []

        for item in predictions:
            q = item.get('question', '')
            pred = item.get('answer', '')
            gt = item.get('ground_truth', '')
            cluster_id = item.get('cluster_id', '')

            # 调用混合打分逻辑 (Rule + LLM)
            eval_res = judge_engine.judge(q, gt, pred)
            
            score = eval_res.get("score", 0)
            total_score += score
            count += 1

            adaptor_details.append({
                "question": q,
                "cluster_id": cluster_id,
                "prediction": pred,
                "ground_truth": gt,
                "score": score,
                "reason": eval_res.get("reason", ""),
                "tokens": item.get("tokens", 0),
                "steps": item.get("steps", 0)
            })
            
            if count % 10 == 0 or count == len(predictions):
                print(f"[{adaptor_name}] Evaluated {count}/{len(predictions)}...", flush=True)

        accuracy = total_score / count if count > 0 else 0
        final_eval["summary"][adaptor_name] = {
            "accuracy": accuracy,
            "total_questions": count,
            "avg_tokens": sum(d['tokens'] for d in adaptor_details) / count if count > 0 else 0
        }
        final_eval["details"][adaptor_name] = adaptor_details

    # 保存评估结果
    output_dir = Path("out/q2_benchmark")
    output_file = output_dir / f"eval_{results_path.stem}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_eval, f, indent=2, ensure_ascii=False)

    print(f"\\nEvaluation complete. Summary: {final_eval['summary']}")
    print(f"Detailed report saved to {output_file}")

if __name__ == "__main__":
    main()
