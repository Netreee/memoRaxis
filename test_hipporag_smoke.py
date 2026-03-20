#!/usr/bin/env python3
"""HippoRAG smoke test — minimal index + retrieve via Ark LLM + embedding proxy."""

import sys, os, time

# HippoRAG source
sys.path.insert(0, 'external/hipporag_repo/src')
os.environ.setdefault('OPENAI_API_KEY', os.getenv('ARK_API_KEY', ''))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# -- Config --
config = BaseConfig(
    llm_name='ep-20251113195357-4gftp',
    llm_base_url='https://ark-cn-beijing.bytedance.net/api/v3',
    embedding_model_name='text-embedding-3-small',  # triggers OpenAIEmbeddingModel
    embedding_base_url='http://127.0.0.1:8284/v1',  # local proxy → Ark multimodal
    openie_mode='online',
    save_dir='out/hipporag_smoke',
    force_index_from_scratch=True,
    force_openie_from_scratch=True,
    seed=42,
)

print("=== Initializing HippoRAG ===")
t0 = time.time()
hippo = HippoRAG(global_config=config)
print(f"Init took {time.time()-t0:.1f}s")

# -- Minimal docs --
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

print(f"\n=== Indexing {len(docs)} docs ===")
t0 = time.time()
hippo.index(docs)
print(f"Index took {time.time()-t0:.1f}s")

# -- Retrieve --
queries = [
    "What is George Rankin's occupation?",
    "What county is Erik Hort's birthplace in?",
]

print(f"\n=== Retrieving {len(queries)} queries ===")
t0 = time.time()
results = hippo.retrieve(queries, num_to_retrieve=3)
print(f"Retrieve took {time.time()-t0:.1f}s")

for i, (q, sol) in enumerate(zip(queries, results)):
    print(f"\nQ{i}: {q}")
    docs_retrieved = getattr(sol, 'docs', None)
    if docs_retrieved is None: docs_retrieved = []
    scores = getattr(sol, 'doc_scores', None)
    if scores is None: scores = []
    for j, doc in enumerate(docs_retrieved[:3]):
        s = scores[j] if j < len(scores) else '?'
        print(f"  [{j}] (score={s}) {doc[:100]}")

print("\n=== SMOKE TEST PASSED ===")
