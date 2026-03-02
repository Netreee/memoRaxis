import sys
from pathlib import Path
import psycopg2

sys.path.append(str(Path(__file__).parent))

from src.config import get_config
from src.llm_interface import OpenAIClient
from src.simple_memory import SimpleRAGMemory

def verify():
    print("=== Infrastructure Verification ===")
    conf = get_config()
    
    print(f"1. Testing DB connection to: {conf.database.get('url').split('@')[-1]}...")
    try:
        conn = psycopg2.connect(conf.database.get('url'))
        conn.close()
        print("   [OK] Database connected.")
    except Exception as e:
        print(f"   [FAIL] Database connection failed: {e}")
        return

    print(f"2. Testing LLM Backbone ({conf.llm.get('model')})...")
    try:
        llm = OpenAIClient(
            api_key=conf.llm.get('api_key'),
            base_url=conf.llm.get('base_url'),
            model=conf.llm.get('model')
        )
        resp = llm.generate("Hello, reply with 'Ready'.")
        print(f"   [OK] LLM responded: {resp.strip()}")
    except Exception as e:
        print(f"   [FAIL] LLM call failed: {e}")
        return

    print(f"3. Testing Embedding Service ({conf.embedding.get('model')})...")
    try:
        memory = SimpleRAGMemory(table_name="verify_tmp")
        vec = memory._get_embedding("Checking infra")
        print(f"   [OK] Embedding dimension: {len(vec)}")
    except Exception as e:
        print(f"   [FAIL] Embedding service failed: {e}")
        return

    print("\n--- ALL SYSTEMS OPERATIONAL ---")

if __name__ == "__main__":
    verify()
