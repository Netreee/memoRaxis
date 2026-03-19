import sys
from pathlib import Path
import psycopg2

sys.path.append(str(Path(__file__).parent))
from src.config import get_config
from src.simple_memory import SimpleRAGMemory

def check_evidence():
    memory = SimpleRAGMemory(table_name="bench_q2_md")
    
    q_r1 = "Which model, listed as a priority for the Axon project's evaluation, is also included in the evaluation of AI application content copying into IM input box?\\n\\n【重要指令】：请仔细阅读提供的所有上下文。如果上下文中完全没有足够的信息来回答这个问题，请务必直接输出“无法回答”四个字，不要进行任何猜测或输出其他解释。"
    q_r2 = "Axon项目优先级模型 AI应用内容复制IM输入框评估"
    
    print("=== R1 Evidence ===")
    evidences_r1 = memory.retrieve(q_r1, top_k=5)
    for i, ev in enumerate(evidences_r1):
        content = ev.content
        has_deepseek = "DeepSeek" in content or "deepseek" in content.lower()
        print(f"[{i+1}] Score: {ev.metadata.get('score')} | Has DeepSeek? {has_deepseek}")
        if has_deepseek:
            # print a snippet around 'DeepSeek'
            idx = content.lower().find("deepseek")
            start = max(0, idx - 100)
            end = min(len(content), idx + 100)
            print(f"    Snippet: ...{content[start:end]}...")
            
    print("\n=== R2 Evidence ===")
    evidences_r2 = memory.retrieve(q_r2, top_k=3)
    for i, ev in enumerate(evidences_r2):
        content = ev.content
        has_deepseek = "DeepSeek" in content or "deepseek" in content.lower()
        print(f"[{i+1}] Score: {ev.metadata.get('score')} | Has DeepSeek? {has_deepseek}")
        if has_deepseek:
            idx = content.lower().find("deepseek")
            start = max(0, idx - 100)
            end = min(len(content), idx + 100)
            print(f"    Snippet: ...{content[start:end]}...")

if __name__ == "__main__":
    check_evidence()
