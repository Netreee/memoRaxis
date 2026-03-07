---
license: mit
task_categories:
- question-answering
- zero-shot-classification
- summarization
- text-classification
- text-generation
tags:
- llm-agents
- memory
- benchmark
- rag
library_name: datasets
dataset_info:
  features:
  - name: context
    dtype: string
  - name: questions
    sequence: string
  - name: answers
    sequence:
      sequence: string
  - name: metadata
    struct:
    - name: demo
      dtype: string
    - name: haystack_sessions
      list:
        list:
          list:
          - name: content
            dtype: string
          - name: has_answer
            dtype: bool
          - name: role
            dtype: string
    - name: keypoints
      sequence: string
    - name: previous_events
      sequence: string
    - name: qa_pair_ids
      sequence: string
    - name: question_dates
      sequence: string
    - name: question_ids
      sequence: string
    - name: question_types
      sequence: string
    - name: source
      dtype: string
  splits:
  - name: Accurate_Retrieval
    num_bytes: 19889235.616438355
    num_examples: 22
  - name: Test_Time_Learning
    num_bytes: 5424336.98630137
    num_examples: 6
  - name: Long_Range_Understanding
    num_bytes: 99446178.08219178
    num_examples: 110
  - name: Conflict_Resolution
    num_bytes: 7232449.315068494
    num_examples: 8
  download_size: 74805902
  dataset_size: 131992200.0
configs:
- config_name: default
  data_files:
  - split: Accurate_Retrieval
    path: data/Accurate_Retrieval-*
  - split: Test_Time_Learning
    path: data/Test_Time_Learning-*
  - split: Long_Range_Understanding
    path: data/Long_Range_Understanding-*
  - split: Conflict_Resolution
    path: data/Conflict_Resolution-*
---
# 🚧 Update

- [x] (Sep 29th, 2025) We updated our paper, where we removed some in-efficient and high-cost samples. We also added a sub-sample of DetectiveQA. 
    
- [x] (July 7th, 2025) We released the initial version of our datasets.

- [x] (July 22nd, 2025) We modify the datasets slightly, adding the keypoints in LRU and change the ```uuid``` into ```qa_pair_ids```. The ```question_ids``` is only used in Longmemeval task.

- [x] (July 26th, 2025) We fixed bug on ```qa_pair_ids```.

- [x] (Aug.5th, 2025) We removed the ```ruler_niah``` and some other datasets not used in main experiments. We will release a subset for ablation study in future.


# ⚙️ MemoryAgentBench: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions

This repository contains the MemoryAgentBench dataset, designed for evaluating the memory capabilities of LLM agents.

📄 Paper: https://arxiv.org/pdf/2507.05257

💻 Code:  https://github.com/HUST-AI-HYZ/MemoryAgentBench

MemoryAgentBench is a unified benchmark framework for comprehensively evaluating the memory capabilities of LLM agents: through four core competencies (Accurate Retrieval, Test-Time Learning, Long-Range Understanding, and Conflict Resolution) and incremental multi-turn interaction design, it reveals existing limitations and shortcomings of current memory agents and compares performance differences across various memory agents.

## Four Core Competencies for Evaluation
What capabilities does AI need to truly "remember"?  We argue  that merely storing and retrieving information is far from sufficient. The memory system needs to possess four key competencies:
### 1. Accurate Retrieval (AR)
This is the most fundamental capability—precisely **locating required information** from massive dialogue histories. For instance, when you ask about a detail mentioned 3 hours ago after hours of conversation with an AI, can it quickly and accurately find it? This requires not only single-hop retrieval but also multi-hop reasoning capabilities.
### 2. Test-Time Learning (TTL)
Truly intelligent systems should be able to continuously **learn new skills during interactions**. For example, if you teach an AI a new classification method through a few examples, can it flexibly apply this in subsequent conversations? This "learning-while-using" capability is crucial for building adaptive AI.
### 3. Long-Range Understanding (LRU)
Unlike fragmented information retrieval, long-range understanding requires AI to form **global cognition**. Just like after reading a novel, you not only remember specific plot points but also understand the overall narrative and character relationships. AI needs to abstract high-level understanding from long conversations.
### 4. Conflict Resolution (CR) 
Information in the real world is dynamic. When users say "I changed jobs" or "this theory has been disproven," AI must **identify and update** outdated information rather than simply accumulating old and new knowledge.

## Careful Dataset Design
From "feeding data" to "simulating real interactions," MemoryAgentBench demonstrates ingenuity in dataset design: The research team both adapted existing datasets and created two new ones. All data is split into chunks to **simulate real multi-turn interaction scenarios**—just like your daily conversations with an AI assistant, where information accumulates gradually rather than being injected all at once.
### 1. Newly Constructed Datasets:

**EventQA:** Requires AI to understand temporal event chains in novels and predict "what happens next".

**FactConsolidation:** Specifically designed to test conflict resolution capabilities, including single-hop and multi-hop difficulty levels.

Notably, the team adopted a **"inject once, query multiple times"** design philosophy—one long text corresponds to multiple questions, significantly improving evaluation efficiency.

### 2. Unified Evaluation Protocol:
Memory Construction Phase → Incremental chunk input → Build/Update memory
Query Execution Phase → Pose questions → Answer based on memory → Evaluate accuracy


## Key Findings 🔍
### 1. RAG is Not a Silver Bullet 🎯 
RAG shows clear advantages in accurate retrieval tasks—even simple BM25 methods significantly outperform the GPT-4o-mini baseline (100% vs 22.8% on NIAH-MQ task). However, they have a fatal weakness: poor performance on tasks requiring global understanding, as RAG can only retrieve local information fragments.
### 2. Long Context ≠ Universal Solution 🔑
Although GPT-4.1-mini supports million-level tokens, it doesn't achieve top performance across various tasks. For instance, it only achieves 45.8% accuracy on ∞Bench-QA, and computational overhead increases linearly with context length.
### 3. Commercial Systems Fall Short of Expectations 😔 
Three primary factors lead to poor performance of commercial memory systems. First, severe information loss—Mem0 compresses information by extracting "facts," resulting in substantial context loss. Second, limited retrieval mechanisms—while MemGPT supports multiple retrieval iterations, it lacks temporal and structural metadata. Third, absence of global perspective—these methods cannot reconstruct complete documents, performing particularly poorly on long-range understanding tasks.
### 4. Conflict Resolution Remains Challenging ⚠️
For single-hop conflict resolution, memory agents built with GPT-4o achieve only 60% accuracy. In multi-hop conflict resolution scenarios, all methods achieve single-digit accuracy rates (at most 7%), highlighting this as a critical bottleneck for current memory systems.
### 5. Ablation Studies Reveal Optimization Directions 🔬
**Balancing Chunk Size**: Smaller chunks (512 tokens) benefit accurate retrieval tasks (RULER-QA accuracy reaches 90%), while larger chunks (4096 tokens) better preserve semantic coherence for continuous text understanding. Dynamic chunk size adjustment based on task type is recommended.

**Marginal Effects of Top-K**: Increasing K from 2 to 10 yields significant performance gains for accurate retrieval tasks (BM25 improves from 49.5% to 61%), but shows limited impact on learning tasks, indicating that simply increasing retrieval volume is not a panacea.

**Computational Latency Gaps**: The computational overhead difference between simple and complex systems is staggering—Mem0's memory construction time is 20,000x that of BM25. When using 512 tokens for memory input, Cognee requires 3.3 hours to process a single long-context sample. From a practical deployment perspective, commercial systems must find a balance between performance and efficiency.


## Conclusion 📌
MemoryAgentBench demonstrates significant progress in systematically evaluating LLM memory mechanisms—through comprehensive assessment of four core competencies, it reveals for the first time the limitations of current state-of-the-art methods in dynamic memory updates and long-range consistency, providing a standardized evaluation framework for building AI agents with genuine memory capabilities. In future, we will **collect more realistic real-world conversation data** to further enrich the benchmark's diversity and authenticity, and explore comprehensive memory architectures that can balance accurate retrieval, test-time learning, long-range understanding, and conflict resolution.

## Sample Usage

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("ai-hyz/MemoryAgentBench")

# Access a specific split, e.g., 'Accurate_Retrieval'
accurate_retrieval_split = dataset["Accurate_Retrieval"]
print(f"Number of examples in Accurate_Retrieval split: {len(accurate_retrieval_split)}")
print(f"First example from Accurate_Retrieval split: {accurate_retrieval_split[0]}")

# Access another split, e.g., 'Test_Time_Learning'
test_time_learning_split = dataset["Test_Time_Learning"]
print(f"Number of examples in Test_Time_Learning split: {len(test_time_learning_split)}")
print(f"First example from Test_Time_Learning split: {test_time_learning_split[0]}")
```