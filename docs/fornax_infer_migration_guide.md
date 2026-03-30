# 从 Ark 迁移到 Fornax 做 Infer 的经验总结

> 写于 2026-03-14，适用于 memoRaxis 项目，目标读者：接手本项目或类似项目的 Claude Code session。

---

## 背景

本项目使用多种记忆系统（SimpleRAG、mem0、mem0g、MIRIX、RAPTOR、memGPT）在 MemoryAgentBench 上做基准测试。
Infer 阶段（用 LLM 根据检索到的记忆回答问题）原本使用 Ark（字节内部 LLM 服务），
因 Ark TPM 被其他任务占满，迁移到 **Fornax**（字节内部另一套 LLM 服务）。

---

## 1. 配置方式

### config/config.yaml 改法

```yaml
llm:
  provider: fornax                          # 关键：改这里
  fornax_ak: abe95a393a7a4bd7bf0eb89b2051d981
  fornax_sk: 053a5782cdd64a439587805579d34532
  fornax_prompt_key: lark.memo.main
  # Ark 配置保留但不生效（provider=fornax 时忽略）
  base_url: https://ark-cn-beijing.bytedance.net/api/v3
  model: ep-20251113195357-4gftp
  api_key: your_ark_api_key_here
```

### src/llm_interface.py 支持情况

`OpenAIClient` 已支持 `provider: fornax`，自动读取 config.yaml。
无需改脚本，改完 config.yaml 即生效。

### 环境变量（Fornax 必须）

Fornax SDK 需要以下环境变量，`llm_interface.py` 已自动设置：
```
SERVICE_ENV=boe
RUNTIME_IDC_NAME=boe
FORNAX_CUSTOM_REGION=CN
```

### 安装

```bash
pip install bytedance-fornax -i https://bytedpypi.byted.org/simple/
```

---

## 2. Fornax 速率限制实测

| 指标 | 值 |
|------|-----|
| TPM 上限 | ~250K-300K（存储估算值，实测更高，见下） |
| 限流错误形式 | `FornaxAPIError` 含 "请求 Token 数量超过模型 TPM 限制"（非 HTTP 429） |
| RPM | 无硬限制，受 TPM 约束 |

**实测并行压测结果（lark.memo.main, BOE, MIRIX Accurate_Retrieval task）：**

| 并行路数 | 延迟范围 | tokens/问题 | TPM 报错 |
|---------|---------|------------|---------|
| 1路 | ~5s | ~15K | 无 |
| 4路 | 6–16s | 8–15K | 无 |
| 8路 R1 | 12–35s | 6–72K | 无 |
| 1路 R2 | ~25s | ~228K | 无 |
| 1路 R3 | ~50s | ~400-800K | 无 |

**结论**：实际 TPM 上限远高于 250K，6-8 路并行 R1/R2 无问题。
R3 tokens 极高（400-800K/问题），6路 R3 同时跑理论上超 250K，但实测无报错——
推测 Fornax 实际限额更高，或限流窗口宽松。

**错误重试**：`llm_interface.py` 已实现指数退避重试，匹配关键词 "TPM"、"超过模型"。

---

## 3. 注意事项

### 3.1 MIRIX 内部仍用 Ark

MIRIX Docker 服务（`external/mirix_repo`）有自己的 LLM 配置（`config/mirix_config.yaml`），
使用 `model_endpoint_type: openai`（标准 HTTP 格式）。
Fornax 使用自研 SDK，**没有标准 HTTP 端点**，无法直接填入 mirix_config.yaml。

因此：
- **Ingest**（MIRIX 内部记忆提取）→ 继续用 Ark
- **Retrieve**（MIRIX 内部 topic 提取）→ 继续用 Ark（每次 retrieve 约 1K tokens，量小）
- **Infer 答案生成**（我们的脚本）→ Fornax

两者不竞争 TPM，可以同时跑 ingest 和 infer。

### 3.2 venv 路径

本项目使用 `.venv`，Fornax 和 mirix 包都装在里面：
```bash
/data00/home/ziqian/proj/memoRaxis/.venv/bin/python
```
用系统 python 会报 `ModuleNotFoundError: No module named 'fornax'`。

### 3.3 Embedding 仍用 Ark

`config/config.yaml` 的 `embedding` 配置仍用 Ark（`ark_multimodal`），
Fornax 不提供 embedding 服务，无需改动。

---

## 4. 并行调度

全量 infer 使用 `scripts/MIRIX/infer_dispatcher.py`：
- 74 个 job（AR×22 + CR×8 + LR×40 + TTL×4）
- 6 路并行，每个 job 跑 R1+R2+R3 三个 adaptor
- 日志：`logs/mirix_infer/{task}_{idx}.log`
- 调度日志：`logs/mirix_infer_dispatch.log`

查看进度：
```bash
grep -c "DONE" logs/mirix_infer_dispatch.log   # 已完成数
grep "FAIL" logs/mirix_infer_dispatch.log       # 失败列表
tail -f logs/mirix_infer_dispatch.log           # 实时跟踪
```

---

## 5. 各 adaptor 实测性能（MIRIX + Fornax，6路并行）

| Adaptor | 延迟/问题 | tokens/问题 | 说明 |
|---------|---------|------------|------|
| R1（SingleTurn） | ~20s | ~15K | retrieve 1次，生成1次 |
| R2（Iterative） | ~25s | ~228K | 5次迭代，每次 retrieve+生成 |
| R3（PlanAndAct） | ~50s | ~400-800K | 规划+多步执行，token 消耗最高 |

R2 token 量远超 R1，但延迟相近，因为 MIRIX retrieve 延迟（向量检索+Ark topic提取）
是瓶颈，不是 Fornax 生成速度。

---

## 6. 全量耗时估算（6路并行）

| 任务 | instances | q/inst | 6路耗时 |
|------|-----------|--------|--------|
| AR | 22 | 100 | ~10.5h |
| CR | 8 | 100 | ~5.3h |
| LR | 40 | 1 | ~30min |
| TTL | 4 | ~100 | ~2.2h |
| **合计** | **74** | | **~18h** |

---

## 7. 已知数据质量问题（infer 前需了解）

详见 `audit/mirix_ingest_audit_report.md`，关键点：

- **TTL inst 4**：ingest 失败（仅 1 条空 memory），infer 结果必为 0 分，已跳过
- **TTL inst 0**：context 5.6MB 过大，未 ingest，已跳过
- **CR inst 0-3**：memory 膨胀（多轮重复 ingest），冗余但不影响正确性
- **AR inst 7-13**：partial ingest（81-94%），约 3.5% 问题可能失分
- **LR inst 16, 33 等**：叙事文本，MIRIX 提取 memory 极少，这是系统固有限制

在分析结果时需对上述 instance 做标注。
