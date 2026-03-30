#!/usr/bin/env python3
"""HippoRAG smoke test — minimal index + retrieve via Fornax LLM + embedding proxy.

运行前需确保：
  - gaoang/embedding_proxy.py 已在 8284 端口运行
  - 本脚本会自动启动 fornax_openai_server.py (port 8285)
"""

import sys, os, time, socket, subprocess

sys.path.insert(0, 'external/hipporag_repo/src')
# src 作为包根导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 配置（从 config.yaml 读取）----
from src.config import get_config
cfg = get_config()
llm_conf = cfg.llm
emb_conf = cfg.embedding

FORNAX_AK = llm_conf.get("fornax_ak", "")
FORNAX_SK = llm_conf.get("fornax_sk", "")
FORNAX_PROMPT_KEY = llm_conf.get("fornax_prompt_key", "")
FORNAX_PORT = 8285
EMBED_PROXY_URL = "http://127.0.0.1:8284/v1"
ARK_API_KEY = emb_conf.get("api_key", "")

os.environ.setdefault("OPENAI_API_KEY", ARK_API_KEY or "dummy")


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


# ---- 1. 检查 / 启动 embedding proxy (8284) ----
if not port_open(8284):
    print("[smoke] Starting embedding proxy on port 8284 ...")
    env = os.environ.copy()
    subprocess.Popen(
        [sys.executable, "gaoang/embedding_proxy.py"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        if port_open(8284):
            break
        time.sleep(0.5)
    else:
        sys.exit("[smoke] ERROR: embedding proxy failed to start")
    print("[smoke] Embedding proxy ready")
else:
    print("[smoke] Embedding proxy already running on 8284")

# ---- 2. 检查 / 启动 Fornax server (8285) ----
if not port_open(FORNAX_PORT):
    print(f"[smoke] Starting Fornax server on port {FORNAX_PORT} ...")
    env = os.environ.copy()
    env["FORNAX_AK"] = FORNAX_AK
    env["FORNAX_SK"] = FORNAX_SK
    env["FORNAX_PROMPT_KEY"] = FORNAX_PROMPT_KEY
    env["FORNAX_OPENAI_PORT"] = str(FORNAX_PORT)
    subprocess.Popen(
        [sys.executable, "fornax/fornax_openai_server.py"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        if port_open(FORNAX_PORT):
            break
        time.sleep(0.5)
    else:
        sys.exit("[smoke] ERROR: Fornax server failed to start")
    print("[smoke] Fornax server ready")
else:
    print(f"[smoke] Fornax server already running on {FORNAX_PORT}")

# ---- 3. 初始化 HippoRAG ----
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

print("\n=== [1/3] Initializing HippoRAG ===")
t0 = time.time()
config = BaseConfig(
    llm_name=FORNAX_PROMPT_KEY,
    llm_base_url=f"http://127.0.0.1:{FORNAX_PORT}/v1",
    embedding_model_name="text-embedding-3-small",
    embedding_base_url=EMBED_PROXY_URL,
    openie_mode="online",
    save_dir="out/hipporag_smoke",
    force_index_from_scratch=True,
    force_openie_from_scratch=True,
    seed=42,
)
hippo = HippoRAG(global_config=config)
print(f"Init: {time.time()-t0:.1f}s")

# ---- 4. Index ----
docs = [
    "Oliver Badman is a politician from London.",
    "George Rankin is a politician who served in the Australian Senate.",
    "Thomas Marwick was a Scottish politician active in the early 20th century.",
    "Cinderella attended the royal ball and met the prince.",
    "The prince used the lost glass slipper to search the entire kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort was born in Montebello, a small town.",
    "Montebello is located in Rockland County, New York.",
]

print(f"\n=== [2/3] Indexing {len(docs)} docs ===")
t1 = time.time()
hippo.index(docs)
t_index = time.time() - t1
print(f"Index: {t_index:.1f}s")

# ---- 5. Retrieve ----
queries = [
    "What is George Rankin's occupation?",
    "What county is Erik Hort's birthplace in?",
]

print(f"\n=== [3/3] Retrieving {len(queries)} queries ===")
t2 = time.time()
results = hippo.retrieve(queries, num_to_retrieve=3)
t_retrieve = time.time() - t2
print(f"Retrieve: {t_retrieve:.1f}s")

for i, (q, sol) in enumerate(zip(queries, results)):
    print(f"\nQ{i}: {q}")
    docs_out = getattr(sol, "docs", None)
    if docs_out is None: docs_out = []
    scores = getattr(sol, "doc_scores", None)
    if scores is None: scores = []
    for j, doc in enumerate(docs_out[:3]):
        s = f"{scores[j]:.4f}" if j < len(scores) else "?"
        print(f"  [{j}] (score={s}) {doc[:100]}")

print(f"\n{'='*50}")
print(f"SMOKE TEST PASSED")
print(f"  Init:     {time.time()-t0 - t_index - t_retrieve:.1f}s")
print(f"  Index:    {t_index:.1f}s  ({len(docs)} docs)")
print(f"  Retrieve: {t_retrieve:.1f}s  ({len(queries)} queries)")
print(f"  Total:    {time.time()-t0:.1f}s")
print(f"{'='*50}")
