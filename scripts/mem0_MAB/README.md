# MemoryAgentBench Evaluation with Mem0

本目录包含了一套完整的工具链，用于使用 Mem0 对 MemoryAgentBench (MAB) 数据集进行摄取、推理和评估。

## 目录结构

- `ingest.py`: 数据摄取脚本，将 MAB 数据处理并存入 Mem0/Qdrant。
- `infer.py`: 推理脚本，运行不同的 Agent/Adaptor (R1, R2, R3) 进行问答。
- `analyze.py`: 分析脚本，评估推理结果并生成报告。
- `debug.py`: 简单的调试脚本，用于验证 Mem0 和 Qdrant 的连接。

---

## 1. 环境配置 (Configuration)

在使用本工具链之前，请确保已满足以下条件：

1.  **Python 依赖**: 安装项目根目录 `requirements.txt` 中的依赖，特别是 `mem0ai` 和 `qdrant-client`。
2.  **Qdrant 服务**: 确保 Qdrant 正在运行。可以使用 Docker Compose 启动：
    ```bash
    # 在本目录下运行
    docker-compose up -d
    ```
    这将启动 Qdrant 服务，默认监听 `localhost:6333`。

3.  **配置 (Config)**: 
    *   复制 `config/config.example.yaml` 到 `config/config.yaml`。
    *   在 `config.yaml` 中配置 LLM 和 Embedding 提供商。
    *   `database` 部分的 `qdrant_url` 默认为 `http://localhost:6333`。如果使用默认 Docker 配置，此项可以直接使用或省略（省略则默认为 `localhost:6333`）。

4.  **OpenAI API Key**: 确保在配置文件或环境变量中设置了正确的 API Key。

---

## 2. 调试 (Debug)

在开始大规模任务之前，建议运行 `debug.py` 来验证 Mem0 配置和数据库连接是否正常。

```bash
python debug.py
```

该脚本会尝试：
1. 加载 Mem0 配置。
2. 初始化 Memory 对象。
3. 连接到 Qdrant (默认 localhost:6333)。
4. 检查是否存在测试数据。

如果所有步骤都显示 "✓"，则说明环境配置正确。

---

## 3. 数据摄取 (Ingest)

将 MAB 数据集导入 Qdrant。

### 用法

```bash
python ingest.py --dataset <DATASET_NAME> --instance_idx <INDEX> [OPTIONS]
```

### 参数

*   `--dataset`: **(必选)** 数据集名称。可选值：
    *   `accurate_retrieval`
    *   `conflict_resolution`
    *   `long_range_understanding`
    *   `test_time_learning`
*   `--instance_idx`: **(可选, 默认 "0")** 实例索引。支持单个数字 (`0`)，范围 (`0-5`)，或列表 (`1,3,5`)。
*   `--force`: 强制覆盖已存在的数据。
*   `--qdrant_host`: Qdrant 主机地址 (默认 `localhost`)。

### 示例

```bash
# 摄取 Conflict Resolution 的第 0 到 5 个实例
python ingest.py --dataset conflict_resolution --instance_idx 0-5

# 强制重新摄取 Test Time Learning 的第 10 个实例
python ingest.py --dataset test_time_learning --instance_idx 10 --force
```

---

## 4. 推理 (Infer)

使用 Mem0 中的数据运行 Agent 进行问答。

### 用法

```bash
python infer.py --task <TASK_NAME> --dataset <DATASET_NAME> [OPTIONS]
```

### 参数

*   `--task`: **(必选)** 任务名称。注意这里需要使用**帕斯卡命名法** (PascalCase)：
    *   `Accurate_Retrieval`
    *   `Conflict_Resolution`
    *   `Long_Range_Understanding`
    *   `Test_Time_Learning`
*   `--adaptor`: 指定运行的 Adaptor 类型。可选 `R1` (单轮), `R2` (迭代), `R3` (规划执行), 或 `all` (默认)。
*   `--limit`: 限制运行的问题数量。默认为 5，设为 -1 运行所有问题。
*   `--instance_idx`: 实例索引 (需与摄取时一致)。

### 示例

```bash
# 在 Conflict Resolution 任务上运行所有 Adaptor (R1, R2, R3)
python infer.py --task Conflict_Resolution --instance_idx 0-5

# 仅运行 R1 Adaptor，限制处理前 10 个问题
python infer.py --task Accurate_Retrieval --adaptor R1 --limit 10
```

结果将保存在 `out/mem0` 目录下 (路径可能根据配置有所不同)。

---

## 5. 分析结果 (Analyze)

评估推理生成的 JSON 结果文件，计算准确率、F1 分数等指标。

### 用法

```bash
python analyze.py --task <TASK_NAME> [OPTIONS]
```

### 参数

*   `--task`: **(必选)** 任务名称 (同 Infer 阶段，如 `Conflict_Resolution`)。
*   `--input`: 输入目录或文件模式。默认为 `out/mem0`，会自动查找对应任务的结果文件。
*   `--output`: (可选) 输出分析报告的文件路径。

### 示例

```bash
# 分析 Conflict Resolution 的所有结果
python analyze.py --task Conflict_Resolution

# 指定特定的结果文件进行分析
python analyze.py --task Long_Range_Understanding --input "custom_out/results_*.json"
```

脚本会输出每个实例的详细评分以及汇总报告。

