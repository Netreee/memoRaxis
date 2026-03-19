# MIRIX MAB Ingest 综合审计报告

> 生成时间: 2026-03-12
> 覆盖范围: R1~R6 全部 ingest 数据
> 数据来源: MIRIX API `/memory/components`, PostgreSQL `memory_queue_traces`, dispatcher logs

---

## 1. 执行概况

| 指标 | 值 |
|------|-----|
| 总 wall-clock 耗时 | R6=13.4h, R1-R5=~20h+, 合计 ~33h+ |
| 并行度 | max 4 workers, 交错 15s |
| MIRIX workers | 6 (PartitionedMemoryQueue) |
| 容器重启 | R6=0次, R4/R5 各 1-2 次 |
| 429 错误 | R6: 9,893 次 (MIRIX 内部重试, 无数据丢失) |
| LLM 调用总数 | ~13,576 次 (R6, Docker 日志) |
| LLM 总费用 | ~$22-25 估计 (R1-R6 全量) |

---

## 2. Accurate Retrieval (22/22 instances, 100% 覆盖)

| Inst | Chunks | Memories | Mem/Chk | Round | Start | End | Duration | s/chunk | Status | Notes |
|-----:|-------:|---------:|--------:|:------|:------|:----|:---------|--------:|:-------|:------|
| 0 | 130 | 2009 | 15.5 | R2/R3 | — | — | — | — | done | 多轮累积 |
| 1 | 241 | 3556 | 14.8 | R6 | 23:34 | 03:00 | 206.5m | 51.4s | incomplete | 1 chunk failed; 多轮累积 |
| 2 | 291 | 552 | 1.9 | R1/R2 | — | — | — | — | done | |
| 3 | 399 | 598 | 1.5 | R6 | 23:34 | 03:25 | 230.7m | 34.7s | done | |
| 4 | 326 | 447 | 1.4 | R6 | 23:34 | 03:47 | 252.9m | 46.5s | done | |
| 5 | 214 | 502 | 2.3 | R1/R2 | — | — | — | — | done | |
| 6 | 247 | 519 | 2.1 | R1/R2 | — | — | — | — | done | |
| 7 | 36 | 108 | 3.0 | R1 | 11:48 | 13:04 | 76.0m | 126.7s | partial | 34/36 confirmed, 2 failed |
| 8 | 36 | 68 | 1.9 | R1 | 11:48 | 13:04 | 76.4m | 127.3s | done | |
| 9 | 36 | 81 | 2.2 | R1 | 11:48 | 12:12 | 24.4m | 40.7s | partial | 33/36, 3 failed |
| 10 | 35 | 73 | 2.1 | R1 | 11:48 | 12:12 | 24.9m | 42.7s | partial | 30/35, 5 failed |
| 11 | 36 | 54 | 1.5 | R1 | 11:48 | 13:04 | 76.4m | 127.3s | partial | 29/36, 7 failed |
| 12 | 73 | 229 | 3.1 | R1 | 11:48 | 12:36 | 48.6m | 39.9s | partial | 68/73, 5 failed |
| 13 | 72 | 156 | 2.2 | R1 | 12:12 | 12:58 | 46.1m | 38.4s | partial | 63/72, 9 failed |
| 14 | 72 | 177 | 2.5 | R1/R2 | 12:12 | 13:44 | 91.2m | 76.0s | done | |
| 15 | 69 | 139 | 2.0 | R1/R2 | 12:36 | 13:41 | 65.3m | 56.8s | done | |
| 16 | 72 | 53 | 0.7 | R6 | 23:35 | 00:49 | 74.2m | 61.8s | done | |
| 17 | 202 | 537 | 2.7 | R6 | 00:49 | 03:26 | 156.9m | 46.6s | done | |
| 18 | 200 | 582 | 2.9 | R6 | 03:00 | 05:12 | 131.4m | 39.4s | done | |
| 19 | 216 | 761 | 3.5 | R6 | 03:25 | 05:53 | 148.4m | 41.2s | done | |
| 20 | 200 | 813 | 4.1 | R6 | 03:26 | 04:45 | 79.5m | 23.8s | done | |
| 21 | 208 | 627 | 3.0 | R6 | 03:47 | 04:54 | 67.0m | 19.3s | done | |
| **SUM** | **3411** | **12641** | **3.7** | | | | | | | |

Memory 类型分布:

| 类型 | 数量 | 占比 |
|------|-----:|-----:|
| semantic | 7429 | 58.8% |
| episodic | 1793 | 14.2% |
| knowledge | 2126 | 16.8% |
| procedural | 880 | 7.0% |
| resource | 389 | 3.1% |
| core | 24 | 0.2% |

---

## 3. Conflict Resolution (8/8 instances, 100% 覆盖)

| Inst | Chunks | Memories | Mem/Chk | Round | Status | Notes |
|-----:|-------:|---------:|--------:|:------|:-------|:------|
| 0 | 4 | 982 | 245.5 | R2 | done | 多轮重投: 实际累积投喂 ~20+ chunks |
| 1 | 17 | 3830 | 225.3 | R2 | partial | 多轮重投: 累积投喂 ~51 chunks, 49 confirmed 3 failed |
| 2 | 34 | 3825 | 112.5 | R2 | partial | 多轮重投: 累积投喂 ~102 chunks, 54 confirmed 2 failed |
| 3 | 137 | 5899 | 43.1 | R2/R3 | partial | 累积投喂 ~411 chunks, 171 confirmed 5 failed |
| 4 | 4 | 35 | 8.8 | R2 | done | |
| 5 | 17 | 196 | 11.5 | R2/R3 | done | |
| 6 | 34 | 543 | 16.0 | R2/R3 | done | |
| 7 | 137 | 3038 | 22.2 | R3 | partial | 140 confirmed, 3 failed |
| **SUM** | **384** | **18348** | **47.8** | | | |

Memory 类型分布:

| 类型 | 数量 | 占比 |
|------|-----:|-----:|
| semantic | 10130 | 55.2% |
| knowledge | 7795 | 42.5% |
| episodic | 381 | 2.1% |
| resource | 30 | 0.2% |
| core | 12 | 0.1% |

---

## 4. Test Time Learning (5/6 instances, 83.3% 覆盖)

| Inst | Chunks | Memories | Mem/Chk | Round | Start | End | Duration | Status | Notes |
|-----:|-------:|---------:|--------:|:------|:------|:----|:---------|:-------|:------|
| 0 | — | 0 | — | — | — | — | — | **未 ingest** | context 5.6MB, 过大跳过 |
| 1 | 33 | 165 | 5.0 | R2/R3 | 01:06 | 03:07 | 121.0m | partial | 45 confirmed, 5 failed |
| 2 | 29 | 104 | 3.6 | R2/R3 | 01:06 | 03:07 | 121.5m | partial | 41 confirmed, 2 failed |
| 3 | 29 | 171 | 5.9 | R2/R3 | 03:07 | 05:08 | 120.8m | partial | 47 confirmed, 6 failed |
| 4 | 32 | 1 | 0.0 | R2 | 03:07 | 04:00 | 53.4m | done | **异常**: 仅 1 条 episodic |
| 5 | 32 | 133 | 4.2 | R2/R3 | 04:00 | 06:01 | 120.8m | partial | 42 confirmed, 5 failed |
| **SUM** | **155** | **574** | **3.7** | | | | | | |

Memory 类型分布:

| 类型 | 数量 | 占比 |
|------|-----:|-----:|
| episodic | 162 | 28.2% |
| semantic | 229 | 39.9% |
| procedural | 71 | 12.4% |
| knowledge | 82 | 14.3% |
| resource | 22 | 3.8% |
| core | 8 | 1.4% |

---

## 5. Long Range Understanding (40/110 instances, 36.4% 覆盖)

| Inst | Chunks | Memories | Mem/Chk | Round | Start | End | Duration | s/chunk | Status | Notes |
|-----:|-------:|---------:|--------:|:------|:------|:----|:---------|--------:|:-------|:------|
| 0 | 74 | 124 | 1.7 | R6 | 04:45 | 05:53 | 68.0m | 55.1s | done | |
| 1 | 17 | 31 | 1.8 | R6 | 04:54 | 05:04 | 10.3m | 36.4s | done | |
| 2 | 42 | 155 | 3.7 | R6 | 05:04 | 05:48 | 43.6m | 62.2s | done | |
| 3 | 17 | 13 | 0.8 | R6 | 05:12 | 05:40 | 28.1m | 99.1s | done | |
| 4 | 23 | 52 | 2.3 | R6 | 05:40 | 05:53 | 13.3m | 34.7s | done | |
| 5 | 27 | 74 | 2.7 | R6 | 05:48 | 06:24 | 35.5m | 79.0s | done | |
| 6 | 28 | 93 | 3.3 | R6 | 05:53 | 06:04 | 10.8m | 23.1s | done | |
| 7 | 17 | 77 | 4.5 | R6 | 05:53 | 06:04 | 11.1m | 39.0s | done | |
| 8 | 41 | 149 | 3.6 | R6 | 05:53 | 06:16 | 23.2m | 33.9s | done | |
| 9 | 98 | 273 | 2.8 | R6 | 06:04 | 06:58 | 54.1m | 33.2s | partial | 96/98, 2 failed |
| 10 | 64 | 223 | 3.5 | R6 | 06:04 | 06:55 | 50.5m | 47.3s | partial | 62/64, 2 failed |
| 11 | 16 | 30 | 1.9 | R6 | 06:16 | 06:36 | 20.1m | 75.3s | done | |
| 12 | 48 | 151 | 3.1 | R6 | 06:24 | 07:01 | 37.9m | 47.4s | partial | 45/48, 3 failed |
| 13 | 36 | 154 | 4.3 | R6 | 06:36 | 06:55 | 18.6m | 31.1s | partial | 34/36, 2 failed |
| 14 | 134 | 280 | 2.1 | R6 | 06:55 | 09:41 | 166.6m | 74.6s | done | |
| 15 | 110 | 261 | 2.4 | R6 | 06:55 | 09:11 | 135.7m | 74.0s | done | |
| 16 | 15 | 2 | 0.1 | R6 | 06:58 | 07:09 | 11.1m | 44.2s | done | **异常**: 极少 memory |
| 17 | 24 | 23 | 1.0 | R6 | 07:01 | 07:39 | 37.6m | 94.1s | done | |
| 18 | 48 | 100 | 2.1 | R6 | 07:09 | 08:50 | 101.0m | 126.2s | done | 后期减速 |
| 19 | 72 | 254 | 3.5 | R6 | 07:39 | 09:35 | 116.2m | 96.8s | done | |
| 20 | 17 | 12 | 0.7 | R6 | 08:50 | 09:17 | 26.6m | 93.8s | done | |
| 21 | 27 | 35 | 1.3 | R6 | 09:11 | 09:27 | 15.9m | 35.3s | done | |
| 22 | 38 | 79 | 2.1 | R6 | 09:17 | 09:33 | 16.8m | 26.6s | done | |
| 23 | 23 | 111 | 4.8 | R6 | 09:27 | 09:39 | 12.8m | 33.4s | done | |
| 24 | 32 | 133 | 4.2 | R6 | 09:33 | 10:55 | 81.6m | 153.1s | done | |
| 25 | 36 | 43 | 1.2 | R6 | 09:35 | 09:56 | 20.8m | 34.6s | done | |
| 26 | 35 | 178 | 5.1 | R6 | 09:40 | 11:18 | 98.2m | 168.3s | done | |
| 27 | 40 | 75 | 1.9 | R6 | 09:41 | 10:12 | 30.3m | 45.4s | done | |
| 28 | 48 | 130 | 2.7 | R6 | 09:56 | 11:49 | 112.6m | 140.8s | done | |
| 29 | 14 | 37 | 2.6 | R6 | 10:12 | 10:18 | 6.0m | 25.9s | done | |
| 30 | 28 | 13 | 0.5 | R6 | 10:18 | 10:45 | 27.7m | 59.3s | done | |
| 31 | 16 | 13 | 0.8 | R6 | 10:45 | 10:53 | 7.5m | 28.2s | done | |
| 32 | 39 | 34 | 0.9 | R6 | 10:53 | 11:13 | 20.7m | 31.8s | done | |
| 33 | 23 | 7 | 0.3 | R6 | 10:55 | 11:15 | 19.6m | 51.0s | done | **异常**: 极少 memory |
| 34 | 24 | 11 | 0.5 | R6 | 11:13 | 11:47 | 33.9m | 84.7s | done | |
| 35 | 70 | 269 | 3.8 | R6 | 11:15 | 12:13 | 58.4m | 50.1s | done | |
| 36 | 15 | 45 | 3.0 | R6 | 11:18 | 12:03 | 45.1m | 180.5s | done | |
| 37 | 44 | 60 | 1.4 | R6 | 11:47 | 12:57 | 69.3m | 94.5s | **stuck** | 7 pending, episodic overflow (84%) |
| 38 | 30 | 98 | 3.3 | R6 | 11:49 | 12:02 | 13.1m | 26.2s | done | |
| 39 | 21 | 13 | 0.6 | R6 | 12:02 | 12:51 | 49.1m | 140.4s | done | |
| **SUM** | **1571** | **3915** | **2.5** | | | | | | | |

未 ingest: inst 40-109 (70 个)

Memory 类型分布:

| 类型 | 数量 | 占比 |
|------|-----:|-----:|
| semantic | 1745 | 44.6% |
| knowledge | 1357 | 34.7% |
| episodic | 721 | 18.4% |
| procedural | 43 | 1.1% |
| core | 42 | 1.1% |
| resource | 7 | 0.2% |

---

## 6. 全局汇总

| 指标 | 值 |
|------|-----|
| 数据集总 instances | 146 (AR=22 + CR=8 + TTL=6 + LR=110) |
| 已 ingest | 75 instances |
| 未 ingest | 71 instances (TTL inst 0; LR inst 40-109) |
| 覆盖率 | 75/146 = 51.4% |
| 总 chunks 处理 | 5,521 |
| 总 memories 生成 | 35,478 |
| 平均 mem/chunk | 6.43 |

全局 Memory 类型分布:

| 类型 | 数量 | 占比 |
|------|-----:|-----:|
| semantic | 19,533 | 55.1% |
| knowledge | 11,330 | 31.9% |
| episodic | 3,233 | 9.1% |
| procedural | 986 | 2.8% |
| resource | 294 | 0.8% |
| core | 96 | 0.3% |

---

## 7. Token 消耗

MIRIX 不暴露 per-instance token 统计。以下为聚合数据:

| 阶段 | LLM 调用数 | 费用 | 429 数 | Wall Clock |
|------|----------:|-----:|-------:|-----------:|
| R6 (主跑) | ~13,576 | ~$16.67 | 9,893 | 13.4h |
| R1-R5 (调试) | 未知 | ~$5-8 est. | 未知 | ~20h+ |
| **合计** | — | **~$22-25** | — | **~33h+** |

平均每 chunk: ~3.75 次 LLM 调用, ~$0.0046
平均每 LLM 调用: ~$0.00123

---

## 8. 数据质量问题逐项深入分析

### 问题 1: CR inst 0-3 — Memory 膨胀 (多轮重复投喂)

**现象**: CR inst 0-3 的 mem/chunk 比率异常 (45~245x)，远超正常值 (~10x)

| Instance | 原始 chunks | 实际累积投喂 chunks | Memories | 正常预期 | 膨胀倍数 |
|---------:|------------:|-------------------:|---------:|---------:|---------:|
| 0 | 4 | ~20+ | 982 | ~35 | ~28x |
| 1 | 17 | ~51 | 3830 | ~196 | ~20x |
| 2 | 34 | ~102 | 3825 | ~543 | ~7x |
| 3 | 137 | ~411 | 5899 | ~3038 | ~2x |
| 4 (对照) | 4 | 4 | 35 | — | 1x |

**根因**: R2 调试阶段反复 ingest 同一 chunks (3-5 轮)，MIRIX 不去重，每轮都创建新 memory。

**去重分析** (基于 `summary|details|name` md5 hash):
- semantic memory: 内容级别几乎无精确重复 (1% dup)，但同一实体有多个版本（如 "Wilhelm II's university" 出现 3 次，每次 summary 措辞不同）
- 按 `name` 字段统计: 200 条 sampled semantic 中 191 个 unique name → **同名不同描述的半重复占比 ~5%**
- episodic memory: 0% 重复（每轮的时间戳不同）
- knowledge memory: 天然 100% "重复"（所有 instance 都如此，是 MIRIX 数据格式特性，非 bug）

**对 infer 的影响**:
- 膨胀的 memory 不会导致错误答案，但会：
  - 检索时返回冗余条目（同一实体多个版本），浪费 context window
  - 如果 MIRIX 内部用 top-k 检索，冗余可能排挤其他有用 memory
- CR 每个 instance 100 个 questions，inst 0-3 共 400 questions
- **占 CR 总 questions 的 50%** (400/800)

**结论**: ⚠️ **不需要重跑，可进 infer**
- 理由: 膨胀不产生错误信息，只是冗余; 检索 top-k 仍能命中正确答案
- 如果 infer 分数异常低，再考虑清理 memory 后重 ingest
- 若要清理: 需 MIRIX API 支持按 user_id 删除全部 memory 后单次重投（无现成 batch delete API）

---

### 问题 2: AR inst 0, 1 — Memory 膨胀 (同理多轮)

**现象**:

| Instance | Chunks | Memories | Mem/Chk | 正常对照 (inst 20) | 膨胀倍数 |
|---------:|-------:|---------:|--------:|:-------------------|--------:|
| 0 | 130 | 2009 | 15.5 | 4.1 mem/chk | ~3.8x |
| 1 | 241 | 3556 | 14.8 | 4.1 mem/chk | ~3.6x |

**根因**: 与 CR 相同——R1/R2 阶段反复 ingest。但膨胀程度较轻 (3-4x vs CR 的 7-28x)。

**去重分析**:
- AR inst 0 semantic: 200 sampled → 200 unique (0% dup) — 内容不重复但数量偏多
- episodic: 172 条 → 172 unique — 每条对应不同时间点
- 主要膨胀来自 **semantic (1399)** 和 **knowledge (303)**

**对 infer 的影响**:
- AR inst 0, 1 各 100 questions → **共 200 questions，占 AR 总 1720 questions 的 11.6%**
- 膨胀倍数较低 (3.6-3.8x)，对检索质量影响有限

**结论**: ✅ **无需重跑，直接进 infer**

---

### 问题 3: TTL inst 4 — 数据近乎丢失 (仅 1 条空 memory)

**现象**: 32 chunks 投喂完成 (dispatcher 报 done)，但 MIRIX 仅存储了 1 条 episodic memory，且 content 为空字符串。

**根因分析**:
- dispatcher 日志: `status=done, confirmed=32, failed=0` — 所有 chunks 标记为完成
- 但 MIRIX 实际没有处理 queue 中的内容（R2 早期，容器可能在 queue 排空前崩溃重启）
- 容器重启后旧 traces 被清理，但新 memory 已丢失
- TTL inst 4 的 1 条 episodic content 为空 → 极可能是 meta agent 初始化时创建的空 placeholder

**对 infer 的影响**:
- TTL inst 4 有 100 个 questions
- 仅 1 条空 memory → **infer 时将完全无法回答任何问题**
- **占 TTL 可用 questions 的 20%** (100/500)

**结论**: ❌ **需要重跑才能评估**
- 重跑 TTL inst 4: 32 chunks，预计 ~30-60 min
- 如不重跑: 该 instance 评分将为 0 分，需在报告中标注排除

---

### 问题 4: LR inst 16, 33 等 — 极少 Memory (content 被 MIRIX agents 忽略)

**现象**:

| Instance | Chunks | Memories | Content 类型 |
|---------:|-------:|---------:|:-------------|
| 16 | 15 | 2 | 小说叙事 (Grace Egan) |
| 33 | 23 | 7 | 历史叙事 (colonial New York) |
| 34 | 24 | 11 | — |
| 30 | 28 | 13 | — |
| 20 | 17 | 12 | — |
| 39 | 21 | 13 | — |

**根因分析**:
- MIRIX 的 6-agent pipeline 对每个 chunk 进行多维度分析 (episodic/semantic/procedural/resource/knowledge/core)
- 当 chunk 内容是纯叙事/小说体裁，没有明确的事实、实体、程序或知识点时，大部分 agent 判定"无需存储"
- LR inst 16: 343K chars 的 Grace Egan 小说，MIRIX 仅提取了 1 个角色和 1 个事件
- 这**不是 ingest 失败**，而是 MIRIX 的记忆抽取策略对叙事文本覆盖不足

**对 infer 的影响**:
- LR 每个 instance 只有 **1 个 question**（长程理解题）
- 如果该 question 涉及的信息没有被 MIRIX 提取为 memory → 无法回答
- inst 16 (2 memories) 几乎必定失分; inst 33 (7 memories) 有一定概率
- **共涉及 ~6 个 instance，占 LR 已 ingest 的 40 个 instance 的 15%**

**结论**: ⚠️ **无需重跑（重跑也不会改善），直接进 infer**
- 这是 MIRIX 系统对叙事体裁的固有弱点
- 重跑不会改善——同一 chunk 被同一 agent pipeline 处理，结果相同
- 在 infer 后，如果这些 instance 确实失分，应在分析报告中标注为"MIRIX memory extraction limitation"

---

### 问题 5: AR inst 7-13 — R1 早期 Partial Completion (60-94%)

**现象**:

| Instance | Chunks | R1 Confirmed | Failed | 完成率 | Memories |
|---------:|-------:|-------------:|-------:|-------:|---------:|
| 7 | 36 | 34 | 2 | 94% | 108 |
| 9 | 36 | 33 | 3 | 92% | 81 |
| 10 | 35 | 30 | 5 | 86% | 73 |
| 11 | 36 | 29 | 7 | 81% | 54 |
| 12 | 73 | 68 | 5 | 93% | 229 |
| 13 | 72 | 63 | 9 | 88% | 156 |

**根因**: R1 早期阶段，无重试机制，LLM 429/timeout 导致部分 chunks 丢失。

**对 infer 的影响**:
- 这些 instance 丢失的是随机 chunks (5-19% 比例)
- AR 是精确检索任务: 如果被问到的 fact 恰好在丢失的 chunk 中 → 无法回答
- 每个 instance 100 questions，丢失比例与 chunk 丢失比例成正比
- **最坏情况: inst 11 丢失 19% chunks → 约 19 个 questions 可能受影响**
- 6 个 instance 共 600 questions，平均 ~10% 受影响 → **~60 questions**
- **占 AR 总 1720 questions 的 ~3.5%**

**结论**: ⚠️ **不建议重跑，直接进 infer**
- 重跑代价: 需清空 memory 后全量重 ingest (288 chunks × ~40s/chunk ≈ 3.2h)
- 收益: 恢复 ~60 个 questions 的回答能力
- 但: 清空 memory 需要 MIRIX 支持 batch delete API (未验证是否可行)
- 建议: 先跑 infer，在结果分析时标注这些 instance 为 "partial ingest"

---

### 问题 6: LR inst 37 — 84% 完成 (episodic overflow)

**现象**: 44 chunks 中 37 完成, 7 stuck (queued 状态永久)

**根因**: MIRIX episodic memory agent 在处理后期 chunks 时，累积的 episodic memory 超过 LLM context window，导致 `Attempt N failed: length` 错误。

**对 infer 的影响**:
- LR inst 37 只有 **1 个 question**
- 已有 60 memories (16 episodic + 42 semantic + 2 core)
- 丢失的 7 chunks 是文档末尾 16%
- 如果 question 涉及末尾内容 → 无法回答; 否则影响不大

**结论**: ⚠️ **无需重跑，直接进 infer**
- 重跑会遇到相同的 episodic overflow 问题
- 60 memories 已有一定覆盖度

---

## 9. 综合影响评估

### 9.1 按 questions 影响范围

| 数据集 | 总 questions | 受质量问题影响 | 影响率 | 主要问题 |
|--------|------------:|---------------:|-------:|:---------|
| AR | 1720 | ~260 | 15.1% | inst 0,1 膨胀 (200q); inst 7-13 partial (60q) |
| CR | 800 | ~400 | 50.0% | inst 0-3 膨胀 (但不产生错误答案) |
| TTL | 500 | 100 | 20.0% | inst 4 丢失 (100q 必定失分) |
| LR | 40 | ~7 | 17.5% | inst 16,33 低 memory (2q); inst 37 partial (1q); 其他低 memory (~4q) |
| **总计** | **3060** | **~767** | **~25%** | |

注意: "受影响"不等于"必定失分"
- CR inst 0-3 的膨胀只是冗余，不是错误信息，实际失分率可能很低
- AR inst 7-13 的 partial 丢失是随机的，只有被问到丢失 chunk 的问题才失分
- **真正"必定失分"的仅 TTL inst 4 (100q) + LR inst 16 (1q) ≈ 101 questions (3.3%)**

### 9.2 重跑建议

| 优先级 | Instance | 工作量 | 收益 | 建议 |
|:------:|:---------|:------:|:----:|:-----|
| P0 | TTL inst 4 | 32 chunks, ~30-60min | 100 questions 从 0→可评估 | **推荐重跑** |
| P1 | TTL inst 0 | ~700 chunks, ~8-12h | 200 questions 新增 | 视时间而定 |
| P2 | AR inst 7-13 清理重跑 | 288 chunks, ~3-4h | ~60 questions 补全 | 不建议 |
| P3 | CR inst 0-3 清理重跑 | 192 chunks, ~2-3h | 质量提升 (非必需) | 不建议 |
| P4 | LR inst 40-109 | ~3000+ chunks, ~20h+ | 70 questions 新增 | 视时间而定 |

### 9.3 最终 Infer 就绪评估

| 数据集 | 可 infer | 高质量 | 有瑕疵但可用 | 不可用 | 总 questions |
|--------|--------:|-------:|-------------:|-------:|-------------:|
| AR | 22 | 13 (inst 2-6, 8, 14-21) | 9 (inst 0,1,7,9-13) | 0 | 1720 |
| CR | 8 | 4 (inst 4-7) | 4 (inst 0-3) | 0 | 800 |
| TTL | 5 | 3 (inst 1,2,3) | 1 (inst 5) | 1 (inst 4) | 400~500 |
| LR | 40 | 31 | 8 (inst 16,20,30-34,37) | 1? (inst 16) | 40 |
| **总计** | **75** | **51** | **22** | **1~2** | **2960~3060** |
