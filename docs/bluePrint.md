# 工程蓝图：Agent 推理范式适配器 (Reasoning Adaptors)

## 1. 项目背景与核心哲学 (Context & Philosophy)

### 1.1 核心目标
本项目旨在探究 **“记忆架构 ($\mathcal{M}$)”** 与 **“推理范式 ($\mathcal{R}$)”** 在动态复杂任务中的协同效应。我们需要实现一套 **推理适配器（Reasoning Adaptors）**，它们作为“大脑（Brain）”，通过统一的接口调用不同的“海马体（Memory Systems）”。

### 1.2 关键架构假设：Memory as a "Search Engine"
为了解耦 $\mathcal{M}$ 和 $\mathcal{R}$，我们必须严格遵守 **“M轴实现即 QA 机器论”的修正版** —— **“M轴实现即带状态的搜索引擎（Stateful Retriever）”**。

*   **错误的做法**：调用 `memory_system.chat(user_query)`，让记忆系统内部完成推理和生成。这会污染实验变量。
*   **正确的做法**：
    1.  **Adaptor ($\mathcal{R}$)** 负责所有的逻辑控制（规划、循环、判断、最终生成）。
    2.  **Memory ($\mathcal{M}$)** 仅负责 **存储（Storage）** 和 **提供证据（Evidence Provisioning）**。
    3.  **交互协议**：Adaptor 发出 `Query`，Memory 返回 `List[Context/Evidence]`。

---

## 2. 接口定义 (Interface Specifications)

### 2.1 抽象记忆体接口 (The Abstract Memory Base)
所有具体的记忆系统（Mem0, Zep, MemoRAG 等）都必须封装进这个统一的 Wrapper。

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Evidence(BaseModel):
    content: str
    metadata: Dict[str, Any] = {} # e.g., timestamp, source_doc_id, recall_score

class BaseMemorySystem(ABC):
    """
    The 'Black Box' Interface. 
    M-Axis implementations must adapt to this.
    """
    
    @abstractmethod
    def add_memory(self, data: str, metadata: Dict[str, Any]):
        """Ingestion phase: store facts/docs."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """
        CRITICAL: This function must NOT return a final answer.
        It must return raw context, chunks, graph nodes, or clues found in memory.
        """
        pass
        
    def reset(self):
        """Clear state for next experiment."""
        pass
```
这一部分我们不需要实现，只是作为给你参考，我们只之后提到的adaptor都要与类似这样的memorySystem的实例对接。
---

## 3. 三种推理范式适配器 (Reasoning Paradigms Implementation)

我们需要实现三个 `Adaptor` 类，它们共享同一个初始化参数 `llm_client` 和 `memory_system: BaseMemorySystem`。

### 3.1 $\mathcal{R}_1$: Single-turn Adaptor (One-Shot)
**学术定义**：Static Linear RAG。无反思、无循环，一次检索直接生成 [Source 8, 14, 430]。
**对标基线**：*PlanRAG* 中的 `SingleRAG-LM`。

**逻辑流程**：
1.  **Input**: 用户指令 $T$。
2.  **Action**: 直接调用 `memory.retrieve(T)`。
3.  **Synthesis**: 将检索到的 `List[Evidence]` 拼接进 Prompt。
4.  **Output**: LLM 生成最终答案。

**Prompt 模板**：
```text
Task: {user_task}
Context from Memory:
{evidence_list}

Instruction: Based ONLY on the context above, answer the task directly.
```

### 3.2 $\mathcal{R}_2$: Iterative Adaptor (Decision-Driven)
**学术定义**：Dynamic Iterative RAG。Agent 自主决定“信息是否足够”，进行“检索-阅读-判断”的循环 [Source 1, 9, 107]。
**对标系统**：*Auto-RAG*。

**逻辑流程** (参考 *Auto-RAG*):
1.  **Input**: 用户指令 $T$。
2.  **Loop** (Max steps = 5):
    *   **State Analysis**: LLM 基于当前 $T$ 和已有的 `Context`，判断：
        *   A: 信息足够 $\rightarrow$ **BREAK**。
        *   B: 需要更多信息 $\rightarrow$ 生成一个新的 search query $q_t$。
    *   **Action**: 如果是 B，调用 `memory.retrieve(q_t)`。
    *   **Update**: 将新证据追加到 `Context`。
3.  **Synthesis**: 基于最终积累的 `Context` 生成答案。

**关键代码逻辑 (伪代码)**：
```python
context = []
for _ in range(MAX_STEPS):
    # Decision Step
    decision = llm.decide(task, context) 
    if decision.action == "ANSWER":
        return llm.generate(task, context)
    
    # Retrieval Step
    new_evidence = memory.retrieve(decision.search_query)
    context.append(new_evidence)
```

### 3.3 $\mathcal{R}_3$: Plan-then-Act Adaptor (Global Planning)
**学术定义**：Strategic Planning RAG。先生成全局蓝图，再分步执行，支持重规划 (Re-planning) [Source 1, 11, 381]。
**对标系统**：*PlanRAG*。

**逻辑流程** (参考 *PlanRAG*):
1.  **Phase 1 - Planning**: LLM 接收 $T$，生成一个有序计划 $P = [Step_1, Step_2, ..., Step_N]$。
2.  **Phase 2 - Execution Loop**:
    *   遍历 $Step_i$ in $P$:
        *   **Action**: 将 $Step_i$ 转化为 Query，调用 `memory.retrieve(Query)`。
        *   **Observation**: 获得 Evidence。
        *   **Re-plan Check (关键)**: 检查 Observation 是否满足 $Step_i$ 的预期？
            *   如果不满足（如“未找到文档”），触发 LLM 修改后续计划 $P[i+1:]$。
        *   **Accumulate**: 保存当前步的 Evidence。
3.  **Phase 3 - Synthesis**: 综合所有步骤的 Evidence 生成答案。

**Prompt 模板 (Re-planning)**:
```text
Current Plan: {plan}
Current Step: {step}
Observation: {evidence}

Judge: Does the observation provide enough info for this step? 
If NO, generate a revised plan for the remaining steps.
If YES, output "CONTINUE".
```

---

## 4. 关键注意事项 (Implementation Notes)

1.  **MemoRAG 的特殊处理 (The MemoRAG Exception)**:
    *   虽然我们要把 M 当黑箱，但 *MemoRAG* [Source 92, 100] 的特性是先生成 "Global Clues" (Draft Answer)。
    *   **工程适配**：在 `retrieve()` 方法中，如果后端是 MemoRAG，将其生成的 "Clues" 视为一段高权重的 **Evidence (Text)** 返回给 Adaptor。不要让 MemoRAG 内部自动拿着 Clues 去递归检索，这个控制权要交给 $\mathcal{R}_2$ 或 $\mathcal{R}_3$ 的 Adaptor。

2.  **KEDKG/Zep 的特殊处理**:
    *   对于 *KEDKG* [Source 57] 和 *Zep* [Source 410]，它们支持图查询或时序过滤。
    *   **工程适配**：在 $\mathcal{R}_3$ 的 Plan 阶段，允许生成的 Step 包含特定的过滤参数（例如 `{"time_range": "last_week"}`），并透传给 `memory.retrieve(query, filters=...)`。

3.  **日志记录 (Observability)**:
    *   为了验证协同效应，必须记录以下指标：
        *   **Steps Taken**: R2/R3 到底跑了多少轮？
        *   **Token Consumption**: 每一轮消耗多少？
        *   **Re-plan Count**: R3 触发了多少次重规划？（这直接反映了 M 的质量，M 越差，Re-plan 越多）。

---

## 5. 交付物清单 (Deliverables)

1.  `memory_interface.py`: 定义 BaseMemorySystem。
2.  `adaptors.py`: 包含 `SingleTurnAdaptor`, `IterativeAdaptor`, `PlanAndActAdaptor` 类。
3.  `prompts.yaml`: 集中管理三种范式所需的所有 Prompt Templates。
4.  `main.py`: 一个简单的 Driver 脚本，展示如何实例化一个 Adaptor，挂载一个 MockMemory，并运行一个测试 Query。