
import json
import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# 复用项目现有的基础设施
from src.logger import get_logger
from src.config import get_config
from src.llm_interface import OpenAIClient

logger = get_logger()

# --- 官方 PROMPTS (直接复用) ---
FLUENCY_PROMPT = """Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish.
- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers.

Now, read the provided text, and evaluate the fluency using the rubric. Then output your score in the following json format: {{"fluency": 1}}.

Text: "{text}"
"""

RECALL_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel. It should discuss the plots and characters of the story. The text should contain all the given key points.

Below is your grading rubric:
Recall:
- Evaluate the provided summary by deciding if each of the key points is present in the provided summary. A key point is considered present if its factual information is mostly-supported by the provided summary.
- Score: the number of key points mostly-supported by the provided summary.

Now, read the provided summary and key points, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"supported_key_points": [1, 3], "recall": 2}}.

Key points:
{keypoints}

Summary: <start of summary>{summary}<end of summary>
"""

PRECISION_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel.

Below is your grading rubric:
Precision:
- Evaluate the provided summary by deciding if each sentence in the provided summary is supported by the information provided in the expert summary.
- Score: the number of sentences in the provided summary that are supported by the expert summary.

Now, read the provided summary and expert summary, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"precision": 7, "sentence_count": 20}}.

Expert summary: <start of summary>{expert_summary}<end of summary>

Provided summary: <start of summary>{summary}<end of summary>
"""

class SummarizationJudge:
    def __init__(self):
        conf = get_config()
        self.llm = OpenAIClient(
            api_key=conf.llm["api_key"],
            base_url=conf.llm["base_url"],
            model=conf.llm["model"]
        )

    def judge_fluency(self, text: str) -> int:
        prompt = FLUENCY_PROMPT.format(text=text)
        res = self.llm.generate_json(prompt)
        return int(res.get("fluency", 0))

    def judge_recall(self, summary: str, keypoints: List[str]) -> int:
        kp_str = "\n".join([f"{i+1}. {kp}" for i, kp in enumerate(keypoints)])
        prompt = RECALL_PROMPT.format(keypoints=kp_str, summary=summary)
        res = self.llm.generate_json(prompt)
        return int(res.get("recall", 0))

    def judge_precision(self, summary: str, expert_summary: str) -> tuple[int, int]:
        prompt = PRECISION_PROMPT.format(expert_summary=expert_summary, summary=summary)
        res = self.llm.generate_json(prompt)
        return int(res.get("precision", 0)), int(res.get("sentence_count", 1))

def main():
    parser = argparse.ArgumentParser(description="LRU-A Summarization Evaluator")
    parser.add_argument("--results", type=str, required=True, help="Path to infer results JSON")
    parser.add_argument("--instance_folder", type=str, default="MemoryAgentBench/preview_samples/Long_Range_Understanding")
    args = parser.parse_args()

    with open(args.results, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    instance_idx = results_data["instance_idx"]
    instance_path = Path(args.instance_folder) / f"instance_{instance_idx}.json"
    
    with open(instance_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # 提取评测所需的 GT 信息
    keypoints = ground_truth["metadata"].get("keypoints", [])
    expert_summary = ground_truth["answers"][0] if ground_truth["answers"] else ""

    judge = SummarizationJudge()
    
    final_eval = {
        "dataset": "LRU-A",
        "instance_idx": instance_idx,
        "metrics": {}
    }

    for adaptor_name, predictions in results_data['results'].items():
        logger.info(f"Judging {adaptor_name}...")
        
        # LRU-A 每本书只有一个问题
        if not predictions: continue
        pred_item = predictions[0]
        prediction = pred_item["answer"]

        # 评分
        f_score = judge.judge_fluency(prediction)
        r_found = judge.judge_recall(prediction, keypoints)
        p_found, p_total = judge.judge_precision(prediction, expert_summary)

        # 计算比例
        recall = r_found / len(keypoints) if keypoints else 0
        precision = p_found / p_total if p_total > 0 else 0
        f1 = f_score * 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

        final_eval["metrics"][adaptor_name] = {
            "fluency": f_score,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "raw": {
                "recall_found": r_found,
                "recall_total": len(keypoints),
                "precision_found": p_found,
                "precision_total": p_total
            }
        }
        logger.info(f"[{adaptor_name}] F1: {f1:.4f}")

    # 保存结果
    output_dir = Path("out/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"eval_lru_a_{instance_idx}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_eval, f, indent=2, ensure_ascii=False)

    print(f"\nLRU-A Evaluation Report for Instance {instance_idx} saved to {output_file}")

if __name__ == "__main__":
    main()
