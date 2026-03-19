# memoRaxis MAB 审计报告
# 更新时间: 2026-03-18
# 上次报告: 2026-03-12

## 数据集规模

| 数据集 | 总实例数 | simpleMem/mem0 使用 | mem0g 使用 | A-MEM 使用 |
|--------|---------|-------------------|-----------|-----------|
| Accurate Retrieval (AR) | 22 | 22 | 22 (目标) | 22 (目标) |
| Conflict Resolution (CR) | 8 | 8 | 8 | 8 |
| Long Range Understanding (LR) | 110 | 40 (SM) / 110 (mem0) | 16 (0-15) | 40 |
| Test Time Learning (TTL) | 6 | 6 | 6 | 6 |

---

## 一、Ingest 状态

### SimpleMemory (PostgreSQL + pgvector, port 5432)

| 数据集 | 状态 | 备注 |
|--------|------|------|
| AR | ✅ 22/22 完成 | |
| CR | ✅ 8/8 完成 | |
| LR | ✅ 40/40 完成 | |
| TTL | ✅ 6/6 完成 | |

### Mem0 (Qdrant, port 6333 via mem0g_qdrant 容器)

| 数据集 | collection 数 | 状态 | 备注 |
|--------|--------------|------|------|
| AR | 22/22 | ✅ 完成 | |
| CR | 8/8 | ✅ 完成 | |
| LR | 110/110 | ✅ 完成 | 多做了全部 110 个 |
| TTL | 6/6 | ⚠️ inst0 残缺 | mem0_ttl_0 仅 2233/8928 chunks (25%)，需 --force 重跑 |

### Mem0G (Qdrant + Neo4j, port 6333 + 7687)

| 数据集 | Qdrant collection | 缺失实例 | 状态 |
|--------|-------------------|----------|------|
| AR | 16/22 | 10,12,14,16,18,20 | ❌ 缺 6 个 (inst8 正在 ingest) |
| CR | 8/8 | - | ✅ 完成 |
| LR | 16/16 | - | ✅ 完成 (0-15) |
| TTL | 5/6 | 0 | ❌ 缺 1 个 |

**Neo4j 状态**：仅实例 1 (bolt://localhost:7687) 在线。
- 实例 2 (port 7688) 和实例 3 (port 7689) 已停止
- 所有图数据混在默认 neo4j 库中

**Ingest 变更 (3/9→3/12)**：
- mem0g AR: 14→16 collections (+inst 3, 4 完成; inst 8 进行中)
- mem0g LR: 13→16 (补齐 0-15)

### A-MEM (ChromaDB, out/amem/chroma/)

| 数据集 | 实例数 | chunks/实例(示例) | 状态 | 备注 |
|--------|--------|-----------------|------|------|
| AR | 1/22 | 130 (inst0) | 🔄 进行中 | inst0 完成，no-evolution，24.4s |
| CR | 0/8 | - | ⏳ 待跑 | |
| LR | 0/40 | - | ⏳ 待跑 | |
| TTL | 0/6 | - | ⏳ 待跑 | |

> LLM backbone: Fornax (prompt_key=lark.memo.main)；embedding: all-MiniLM-L6-v2 本地
> 存储路径：`out/amem/chroma/{task_short}_{idx}/`（ChromaDB + memories.pkl）

**Ingest 变更 (3/12→3/18)**：
- A-MEM 系统接入完成；AR inst0 验证通过（130 chunks，0 错误）

---

## 二、Infer 状态

### SimpleMemory

**AR** (22/22 instances) ✅ 完成
| Adaptor | 完成 | 错误数/总题数 |
|---------|------|-------------|
| R1 | 22/22 | 0/2000 (0.0%) |
| R2 | 22/22 | 1/2000 (0.1%) |
| R3 | 22/22 | 1/2000 (0.1%) |

**CR** (8/8 instances) ✅ 完成
| Adaptor | 完成 | 错误数/总题数 |
|---------|------|-------------|
| R1 | 8/8 | 0/800 (0.0%) |
| R2 | 8/8 | 5/800 (0.6%) |
| R3 | 8/8 | 1/800 (0.1%) |

**LR** (40/40 instances) ✅ 完成
| Adaptor | 完成 | 错误数/总题数 |
|---------|------|-------------|
| R1 | 40/40 | 0/40 (0.0%) |
| R2 | 40/40 | 0/40 (0.0%) |
| R3 | 40/40 | 0/40 (0.0%) |

**TTL** (6/6 instances) ✅ 完成
| Adaptor | 完成 | 错误数/总题数 | 备注 |
|---------|------|-------------|------|
| R1 | 6/6 | 0/700 (0.0%) | |
| R2 | 6/6 | 88/710 (12.4%) | inst3 仅 10/100 题 |
| R3 | 6/6 | 69/700 (9.9%) | |

> SM TTL 规范文件位置：`out/ttl_results_{0-5}.json`
> `out/simpleMemory_MAB/results/ttl_results_*.json` 为旧版部分副本，以 `out/ttl_results_*.json` 为准。

**Infer 变更 (3/9→3/12)**：
- AR inst2: R1+R2 从缺失→补齐 ✅
- TTL inst1,3,5: 从缺失→补齐 ✅

---

### Mem0

**AR** (22/22 instances) ✅ 完成
| Adaptor | 完成 | 错误数/总题数 |
|---------|------|-------------|
| R1 | 22/22 | 3/2000 (0.1%) |
| R2 | 22/22 | 180/2000 (9.0%) |
| R3 | 22/22 | 290/2000 (14.5%) |

**CR** (8/8 instances) ✅ 完成
| Adaptor | 完成 | 错误数/总题数 |
|---------|------|-------------|
| R1 | 8/8 | 0/800 (0.0%) |
| R2 | 8/8 | 18/800 (2.2%) |
| R3 | 8/8 | 298/800 (37.2%) |

**LR** (110/110 instances) ✅ 完成
| Adaptor | 完成 | 错误数/总题数 |
|---------|------|-------------|
| R1 | 110/110 | 0/171 (0.0%) |
| R2 | 110/110 | 28/171 (16.4%) |
| R3 | 110/110 | 53/171 (31.0%) |

**TTL** (5/6 instances) ⚠️ inst0 缺失
| Adaptor | 完成 | 错误数/总题数 | 缺失实例 |
|---------|------|-------------|----------|
| R1 | 5/6 | 0/500 (0.0%) | inst0 |
| R2 | 5/6 | 35/500 (7.0%) | inst0 |
| R3 | 5/6 | 139/500 (27.8%) | inst0 |

> mem0 TTL inst0: ingest 残缺 (25%)，需先 `--force` 完整 re-ingest 再跑 R1/R2/R3

**Infer 变更 (3/9→3/12)**：
- AR R2: inst16 补齐，21→22/22 ✅
- AR R3: inst16 补齐，21→22/22 ✅
- CR R2: inst6,7 补齐，6→8/8 ✅
- CR R3: inst6,7 补齐，6→8/8 ✅
- TTL R2: inst4,5 补齐，3→5/6
- TTL R3: inst4,5 补齐，3→5/6

> **注意**：部分 R1 文件（inst 0,1,2,8-10,12,13,17-21）内嵌了旧版 R2/R3 数据（早期高并行跑出，429 错误率高）。
> 规范数据以独立的 `_r2.json` 和 `_r3.json` 文件为准（3/11-12 重跑，错误率低于旧版）。

---

### Mem0G

**AR** (8/22 instances with R1) — 进行中
| Adaptor | 完成实例 | 错误数/总题数 | 缺失 |
|---------|---------|-------------|------|
| R1 | 8 (inst 0-7) | 54/800 (6.8%) | inst 8-21 (部分未 ingest) |
| R2 | 3 (inst 1,5,7) | 1/300 (0.3%) | 大部分未跑 |
| R3 | 0 | - | 全部未跑 |

**CR** (8/8 instances) — R3 部分缺失
| Adaptor | 完成实例 | 错误数/总题数 | 缺失 |
|---------|---------|-------------|------|
| R1 | 8/8 | 84/800 (10.5%) | - |
| R2 | 8/8 | 38/800 (4.8%) | - |
| R3 | 5/8 | 31/410 (7.6%) | inst 5,6,7 |

**LR** (16/16 instances) ✅ 完成
| Adaptor | 完成实例 | 错误数/总题数 |
|---------|---------|-------------|
| R1 | 16/16 | 0/16 (0.0%) |
| R2 | 16/16 | 1/16 (6.2%) |
| R3 | 16/16 | 2/16 (12.5%) |

**TTL** (5/6 instances) — R3 全缺
| Adaptor | 完成实例 | 错误数/总题数 | 缺失 |
|---------|---------|-------------|------|
| R1 | 5/6 | 7/500 (1.4%) | inst0 (未 ingest) |
| R2 | 5/6 | 101/500 (20.2%) | inst0 |
| R3 | 0/6 | - | 全部未跑 |

**Infer 变更 (3/9→3/12)**：
- AR R1: 5→8 (补充 inst 3,4) — 但 R2/R3 仍严重不足
- CR R1/R2: 7→8/8 完成
- LR: 13→16/16 完成 ✅

### A-MEM

**AR** (1/22 instances) — 验证阶段
| Adaptor | 完成 | 错误数/总题数 | 备注 |
|---------|------|-------------|------|
| R1 | 1/22 (3题) | 0/3 (0.0%) | inst0 前3题验证通过 |
| R2 | 0/22 | - | 待跑 |
| R3 | 0/22 | - | 待跑 |

> 审计字段：question/answer/steps/tokens/latency_s/replan 全部写入 ✅
> Template：ruler_qa (rag_agent) 渲染正常 ✅
> 已知问题：no-evolution 模式下 top_k=10 导致每题消耗约 16K tokens；检索相关性一般

**Infer 变更 (3/12→3/18)**：
- A-MEM infer pipeline 验证通过；AR inst0 R1 前3题完成

---

## 三、最终汇总表

| 系统 | 数据集 | 实例数 | R1 (err%) | R2 (err%) | R3 (err%) | 完整? |
|------|--------|--------|-----------|-----------|-----------|-------|
| simpleMem | AR | 22 | 22 (0.0%) | 22 (0.1%) | 22 (0.1%) | ✅ |
| simpleMem | CR | 8 | 8 (0.0%) | 8 (0.6%) | 8 (0.1%) | ✅ |
| simpleMem | LR | 40 | 40 (0.0%) | 40 (0.0%) | 40 (0.0%) | ✅ |
| simpleMem | TTL | 6 | 6 (0.0%) | 6 (12.4%) | 6 (9.9%) | ✅ |
| mem0 | AR | 22 | 22 (0.1%) | 22 (9.0%) | 22 (14.5%) | ✅ |
| mem0 | CR | 8 | 8 (0.0%) | 8 (2.2%) | 8 (37.2%) | ✅ |
| mem0 | LR | 110 | 110 (0.0%) | 110 (16.4%) | 110 (31.0%) | ✅ |
| mem0 | TTL | 5/6 | 5 (0.0%) | 5 (7.0%) | 5 (27.8%) | ⚠️ inst0 |
| mem0g | AR | 8/22 | 8 (6.8%) | 3 (0.3%) | 0 (-) | ❌ |
| mem0g | CR | 8 | 8 (10.5%) | 8 (4.8%) | 5 (7.6%) | ❌ |
| mem0g | LR | 16 | 16 (0.0%) | 16 (6.2%) | 16 (12.5%) | ✅ |
| mem0g | TTL | 5/6 | 5 (1.4%) | 5 (20.2%) | 0 (-) | ❌ |
| **A-MEM** | **AR** | **1/22** | **1 (3题验证)** | - | - | 🔄 |
| **A-MEM** | **CR** | 0/8 | - | - | - | ⏳ |
| **A-MEM** | **LR** | 0/40 | - | - | - | ⏳ |
| **A-MEM** | **TTL** | 0/6 | - | - | - | ⏳ |

---

## 四、结果文件位置

### SimpleMemory
- AR: `out/simpleMemory_MAB/results/acc_ret_results_{i}.json` (合并 R1/R2/R3)
- CR: `out/simpleMemory_MAB/results/conflict_res_results_{i}.json` (合并 R1/R2/R3)
- LR: `out/simpleMemory_MAB/results/long_range_results_{i}.json` (合并 R1/R2/R3)
- TTL: `out/ttl_results_{i}.json` (合并 R1/R2/R3) ← 注意不在 simpleMemory_MAB/results/ 下

### Mem0
- R1: `out/mem0/mem0_{dataset}_results_{i}.json`
- R2: `out/mem0/mem0_{dataset}_results_{i}_r2.json` ← 规范文件
- R3: `out/mem0/mem0_{dataset}_results_{i}_r3.json` ← 规范文件
- ⚠️ 部分 R1 文件内嵌了旧版 R2/R3（高错误率），以独立 `_r2/_r3` 文件为准

### Mem0G
- 合并文件: `out/mem0g/mem0g_{dataset}_results_{i}.json` (含 R1/R2/R3 在 results dict 中)

### A-MEM
- 合并文件: `out/amem/amem_{task_short}_{i}.json` (含 R1/R2/R3 在 results dict 中)
- 进度文件: `out/amem/ingest_progress/{task_short}_{i}.json`
- ChromaDB: `out/amem/chroma/{task_short}_{i}/` + `memories.pkl`

---

## 五、服务依赖状态 (2026-03-12)

| 服务 | 容器名 | 端口 | 状态 |
|------|--------|------|------|
| Qdrant (主) | mem0g_qdrant | 6333-6334 | ✅ 运行中 |
| Qdrant (旧) | mem0_qdrant | 6335 | ✅ 运行中 (空，可忽略) |
| Neo4j (实例1) | mem0g_neo4j | 7474/7687 | ✅ 运行中 |
| Neo4j (实例2) | mem0g_neo4j_2 | 7688 | ❌ 已停止 |
| Neo4j (实例3) | mem0g_neo4j_3 | 7689 | ❌ 已停止 |
| PostgreSQL | pg | 5432 | ✅ 运行中 |
| PostgreSQL (MIRIX) | mirix_pgvector | 5433 | ✅ 运行中 |
| Letta Server | - | 8283 | ✅ 运行中 |
| Embedding Proxy | - | 8284 | ✅ 运行中 |

---

## 六、待补齐清单

### 高优先级 — 结构性缺失

| # | 任务 | 说明 |
|---|------|------|
| 1 | mem0 TTL inst0 re-ingest | Qdrant 仅 25% 数据，需 `--force` 全量重跑 (~5h) |
| 2 | mem0 TTL inst0 R1/R2/R3 | 依赖 #1 完成后才能跑 (~3h) |
| 3 | mem0g AR ingest | 缺 inst 10,12,14,16,18,20 (inst8 进行中) |
| 4 | mem0g AR R2/R3 | 大部分未跑 (仅 3 个 R2, 0 个 R3) |
| 5 | mem0g CR R3 | 缺 inst 5,6,7 |
| 6 | mem0g TTL R3 | 全部 0/6 未跑 |
| 7 | mem0g TTL ingest inst0 | 缺失 |

### 低优先级 — 429 限流导致的空答案

以下错误率较高的数据集，如需更高质量结果可选择性重跑：

| 系统 | 数据集 | Adaptor | 错误率 | 空答案数 |
|------|--------|---------|--------|---------|
| mem0 | CR | R3 | 37.2% | 298/800 |
| mem0 | LR | R3 | 31.0% | 53/171 |
| mem0 | TTL | R3 | 27.8% | 139/500 |
| mem0g | TTL | R2 | 20.2% | 101/500 |
| mem0 | LR | R2 | 16.4% | 28/171 |
| mem0 | AR | R3 | 14.5% | 290/2000 |
| simpleMem | TTL | R2 | 12.4% | 88/710 |
| mem0g | CR | R1 | 10.5% | 84/800 |

---

## 七、错误根因分析

主要错误类型：
1. **429 TPM 限流** (RateLimitExceeded.EndpointTPMExceeded) — 早期高并行跑出，占绝大多数
2. **Connection reset by peer** — Qdrant 容器意外重启/OOM 导致
3. **408 Request Timeout / SSL EOF** — 网络层偶发抖动，占比极小

3/11-12 的补跑采用了受控并行（≤4 路），429 错误率显著降低。
