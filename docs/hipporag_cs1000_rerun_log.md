# HippoRAG cs1000 重跑记录

> 记录日期：2026-03-29
> 背景：HippoRAG 第一轮评测结果异常低，排查后发现根本原因，重新 ingest+infer。

---

## 一、第一轮：默认 chunk_size=8000

### 1.1 Ingest

使用 HippoRAG 默认配置（`chunk_size=8000`），对 MemoryAgentBench 全部 4 个任务完成 ingest：

| Task | 实例数 | 索引路径 |
|------|--------|----------|
| Accurate_Retrieval (AR) | 22 | `out/hipporag/indices/hipporag_acc_ret_{i}/` |
| Conflict_Resolution (CR) | 8 | `out/hipporag/indices/hipporag_conflict_{i}/` |
| Long_Range_Understanding (LR) | 40 | `out/hipporag/indices/hipporag_long_range_{i}/` |
| Test_Time_Learning (TTL) | 5+1 | `out/hipporag/indices/hipporag_ttl_{i}/` |

### 1.2 Infer

对全部 4 任务、R1/R2/R3 三种 adaptor 完成 infer，输出到 `out/hipporag/hipporag_{task}_{i}.json`。

### 1.3 Eval 结果

运行 `scripts/eval_mechanical_all.py` 和 `scripts/lr_judge_all.py` 后，HippoRAG 得分异常低：

**Mechanical Eval（AR / CR / TTL）：**

| Backend | AR R1 | AR R2 | AR R3 | CR R1 F1 | CR R2 F1 | CR R3 F1 | TTL R1 | TTL R2 | TTL R3 |
|---------|-------|-------|-------|----------|----------|----------|--------|--------|--------|
| **HippoRAG (cs8000)** | **2.3%** | **5.6%** | **11.2%** | **0.8%** | **0.6%** | **1.1%** | **1.4%** | **2.0%** | **11.6%** |
| simpleMem (naive RAG) | 60.8% | 71.2% | 64.4% | 17.7% | 18.9% | 10.1% | 57.8% | 53.4% | 54.1% |
| mem0 | 55.1% | 54.3% | 52.7% | 3.8% | 3.7% | 2.2% | 0.4% | 1.0% | 1.8% |
| mem0g | 65.9% | — | — | 35.2% | 45.4% | 51.7% | 62.4% | 62.4% | — |

**问题**：HippoRAG（理论上是图 RAG，应优于 naive RAG）在所有任务上大幅低于 simpleMem，AR 仅 2-11%，完全不正常。

---

## 二、排查根本原因

### 2.1 现象

HippoRAG 的核心依赖是 NER（命名实体识别）+ OpenIE 三元组抽取，通过 LLM 将 passage 解析成知识图谱节点和边。如果 NER 失败，图谱就是空的，检索自然返回无关内容。

### 2.2 根因定位

检查各实例的 `openie_results_ner_*.json` 文件，发现大量 passage 的 triple 列表为空。

**根因**：Ark LLM（`ep-20251113195357-4gftp`）对超长 passage 拒绝返回有效 JSON。

- `chunk_size=8000` 时，单个 passage 约 8000 字符
- Ark LLM 遇到超长内容时会拒绝 JSON 格式响应（或返回截断/无效 JSON）
- 导致 NER 提取失败率 20%~89%（因实例而异）
- 知识图谱极度稀疏，检索命中率几乎为零

### 2.3 验证

将 `chunk_size` 改为 `1000`，对 AR inst0 重新 ingest：

- NER 成功率：**100%**（所有 chunk 均成功提取三元组）
- AR inst0 cs1000 图谱规模：34,837 phrase nodes，2,303 passage nodes，39,767 extracted triples
- AR inst0 cs8000 图谱：passage 数相同但 triple 极稀疏
- AR inst0 cs1000 R3 得分：~87%（vs cs8000 的 ~11%）

结论：**chunk_size=1000 完全修复 NER 问题**，需对全部 4 任务重新 ingest + infer。

---

## 三、第二轮：chunk_size=1000

### 3.1 设计改动

在 `scripts/HippoRAG/ingest.py` 中新增 `--chunk_size` 参数，支持覆盖默认值：

- 索引命名：`hipporag_{task}_cs1000_{i}/`（在原名中插入 `cs{N}`）
- 输出命名：`hipporag_{task}_{i}_cs1000.json`（通过 `--output_suffix cs1000`）

**TTL inst0 特殊处理**：TTL inst0 为对话格式，使用 `chunk_dialogues()` 分块，`chunk_size` 参数对其无效——原有 `hipporag_ttl_0` 索引可直接复用，无需重新 ingest。

### 3.2 Ingest 过程

按任务并行启动：

```bash
# AR inst0-21
nohup .venv/bin/python scripts/HippoRAG/ingest.py \
  --dataset accurate_retrieval --instance_idx 0-21 --chunk_size 1000 \
  >> logs/hipporag_ingest_ar_cs1000.log 2>&1 &

# CR + TTL（串行）
nohup bash -c ".venv/bin/python scripts/HippoRAG/ingest.py \
  --dataset conflict_resolution --instance_idx 0-7 --chunk_size 1000 && \
  .venv/bin/python scripts/HippoRAG/ingest.py \
  --dataset test_time_learning --instance_idx 1-5 --chunk_size 1000" \
  > logs/hipporag_ingest_cr_ttl_cs1000.log 2>&1 &

# LR inst0-39
nohup .venv/bin/python scripts/HippoRAG/ingest.py \
  --dataset long_range_understanding --instance_idx 0-39 --chunk_size 1000 \
  > logs/hipporag_ingest_lr_cs1000.log 2>&1 &
```

**遇到的问题：**

1. **嵌入 API 连接断开**：ingest 进程在 `Batch Encoding` 步骤因 `openai.APIConnectionError` 崩溃（非代码 bug，属网络瞬断）。重启后 LLM cache 保留（NER/triple 不需重跑），仅重新跑 embedding 步骤。
2. **AR OOM**：AR inst1 在 KNN 步骤被系统 OOM killer 静默 kill（无 Python traceback），原因是 cs1000 生成的 embedding index 较大，系统内存不足。随着其他 ingest 任务完成、内存释放后自然恢复。

**Ingest 完成情况（截至 2026-03-29）：**

| Task | 完成 | 进行中 |
|------|------|--------|
| AR | inst0-2（3/22）| inst3+（NER→triple→embedding→graph） |
| CR | inst0-7（8/8 ✅）| — |
| LR | inst0-9（10/40）| inst10+（NER in progress） |
| TTL | inst0-5（6/6 ✅，inst0复用原索引）| — |

### 3.3 Infer 过程

每完成一批索引，立即启动对应的 infer（有 checkpointing，中断可续跑）：

```bash
# CR inst0-2（最早完成）
nohup .venv/bin/python scripts/HippoRAG/infer.py \
  --task Conflict_Resolution --instance_idx 0-2 \
  --adaptor all --chunk_size 1000 --output_suffix cs1000 --limit -1 \
  > logs/hipporag_infer_cr_cs1000.log 2>&1 &

# CR inst3-7
nohup .venv/bin/python scripts/HippoRAG/infer.py \
  --task Conflict_Resolution --instance_idx 3-7 \
  --adaptor all --chunk_size 1000 --output_suffix cs1000 --limit -1 \
  >> logs/hipporag_infer_cr2_cs1000.log 2>&1 &

# TTL inst1-5
nohup .venv/bin/python scripts/HippoRAG/infer.py \
  --task Test_Time_Learning --instance_idx 1-5 \
  --adaptor all --chunk_size 1000 --output_suffix cs1000 --limit -1 \
  >> logs/hipporag_infer_ttl_cs1000.log 2>&1 &

# AR inst1-2（LR inst0-9 索引完成后）
nohup .venv/bin/python scripts/HippoRAG/infer.py \
  --task Accurate_Retrieval --instance_idx 1-2 \
  --adaptor all --chunk_size 1000 --output_suffix cs1000 --limit -1 \
  > logs/hipporag_infer_ar_cs1000.log 2>&1 &

# LR inst0-9（索引完成后）
nohup .venv/bin/python scripts/HippoRAG/infer.py \
  --task Long_Range_Understanding --instance_idx 0-9 \
  --adaptor all --chunk_size 1000 --output_suffix cs1000 --limit -1 \
  > logs/hipporag_infer_lr_cs1000.log 2>&1 &
```

**Infer 完成情况（截至 2026-03-29 19:00）：**

| Task | 完整完成（R1/R2/R3 全部）| 部分完成 | 待启动 |
|------|--------------------------|----------|--------|
| AR | inst0 | inst1（R1✓，R2/R3运行中） | inst3-21（等ingest） |
| CR | inst0-3 | inst4（R1✓，R2/R3运行中） | — |
| LR | inst0-4 | inst5-9（运行中） | inst10-39（等ingest） |
| TTL | inst1 | inst2（R1✓，R2/R3运行中） | — |

**4 个 infer 进程并行运行中：**

| PID | 任务 |
|-----|------|
| 320482 | CR inst3-7 |
| 320483 | TTL inst1-5 |
| 578019 | AR inst1-2 |
| 578066 | LR inst0-9 |

---

## 四、当前状态与后续计划

### 4.1 Ingest 剩余工作

- **AR**：inst3-21（19 个实例），当前 inst3 在跑
- **LR**：inst10-39（30 个实例），当前 inst10 在跑

### 4.2 Infer 剩余工作

随 ingest 完成，陆续补充启动：
- AR inst3-21
- LR inst10-39
- TTL inst0（使用原有 `hipporag_ttl_0` 索引，可直接启动）

### 4.3 Eval

全部 infer 完成后：
1. 更新 `scripts/eval_mechanical_all.py`，将 `hipporag_cs1000` 加入 AR / CR / TTL 评测路径
2. 更新 `scripts/lr_judge_all.py`，加入 LR cs1000 评测路径
3. 运行评测，与 cs8000 及其他 backend 对比

### 4.4 预期结论

基于 cs1000 验证实验（AR inst0 R3: ~87% vs cs8000 的 11%），预期 cs1000 结果将显著高于 cs8000，且与 mem0g / simpleMem 处于同一量级，届时可作为有效结果纳入最终报告。
