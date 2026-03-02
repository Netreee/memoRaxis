# MemoryAgentBench Evaluation with Mem0G (Graph)

本目录包含了一套用于使用 Mem0 Graph (Mem0G) 功能的工具链，支持通过 Neo4j 图数据库增强记忆检索能力。

## 目录结构

- `ingest/`: 包含用于将数据摄取到图数据库和向量数据库的脚本。
- `infer/`: 包含用于执行推理任务的脚本。
- `docker-compose.yml`: 用于启动 Neo4j 和 Qdrant 服务的 Docker 配置文件。
- `debug_mem0g.py`: 调试脚本，用于验证 Neo4j 连接、数据插入及图检索功能。

---

## 1. 环境配置 (Configuration)

在使用 Mem0G 之前，需配置 Neo4j 图数据库。

### 1.1 启动 Neo4j 服务

本项目使用 Docker Compose 快速启动 Neo4j 和 Qdrant。

1.  切换到 `scripts/mem0g_MAB/` 目录：
    ```bash
    cd scripts/mem0g_MAB
    ```

2.  启动服务：
    ```bash
    docker-compose up -d
    ```
    此命令将启动：
    - **Neo4j**: 端口 `7474` (HTTP) 和 `7687` (Bolt)。默认用户名 `neo4j`，密码 `password`。
    - **Qdrant**: 端口 `6333`。

### 1.2 配置项目设置

在项目根目录下的 `config/config.yaml` 文件中，添加或修改 `database` 部分以包含 Neo4j 的连接信息。

> **注意**: 如果 `config/config.yaml` 不存在，请复制 `config/config.example.yaml` 并重命名。

```yaml
database:
  # ... 其他数据库配置 ...
  
  # Neo4j 配置 (用于 Mem0G)
  neo4j_url: bolt://localhost:7687
  neo4j_username: neo4j
  neo4j_password: password
```

### 1.3 安装 Python 依赖

确保已安装 `neo4j` 驱动和其他依赖：

```bash
pip install neo4j
# 或确保 requirements.txt 中包含 neo4j
```

---

## 2. 验证配置 (Verification)

使用 `debug_mem0g.py` 脚本验证 Neo4j 连接及基本功能。

### 运行调试脚本

在项目根目录下运行：

```bash
python scripts/mem0g_MAB/debug_mem0g.py
```

### 调试脚本参数说明

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--collection` | Qdrant 集合名称 | `mem0g_debug_collection` |
| `--user` | 用户标识符 | `mem0g_debug_user` |
| `--skip-add` | 跳过添加样本记忆 | `False` |
| `--skip-search` | 跳过搜索验证 | `False` |
| `--reset` | 运行前重置用户命名空间 | `False` |
| `--sleep` | 插入和搜索之间的等待时间(秒) | `1.0` |
| `--verbose` | 启用详细日志 | `False` |

例如，要重置数据并查看详细日志：

```bash
python scripts/mem0g_MAB/debug_mem0g.py --reset --verbose
```

### 预期输出

脚本执行成功应包含以下步骤的确认信息：
1.  **Step 1: Build configuration**: 加载配置。
2.  **Step 2: Initialize Mem0G**: 初始化内存对象。
3.  **Step 3.5: Inspect Graph Connection**:
    - 显示 `✓ Connected to Neo4j at bolt://localhost:7687`
    - 列出当前数据库列表。
    - 打印目标数据库中的节点样本和总数。
4.  **Step 4 & 5**: 插入样本数据并验证。
5.  **Step 6: Search checks**: 执行检索测试，此时应能看到图关系的输出。

如果遇到 `Graph store configuration missing` 或 `Graph connection failed` 错误，请检查 `config.yaml` 中的 Neo4j 配置是否正确，以及 Docker 容器是否正常运行。

---

## 3. 数据摄取 (Ingestion)

使用 `ingest.py` 脚本将 MemoryAgentBench 数据集摄取到 Mem0G (Qdrant + Neo4j)。

### 运行摄取脚本

```bash
python scripts/mem0g_MAB/ingest.py --dataset <DATASET_NAME> [Create Options]
```

### 摄取脚本参数说明

| 参数 | 必选 | 说明 | 可选值/示例 |
| :--- | :--- | :--- | :--- |
| `--dataset` | 是 | 目标数据集名称 | `accurate_retrieval`, `conflict_resolution`, `long_range_understanding`, `test_time_learning` |
| `--instance_idx` | 否 | 实例索引范围 | `0` (默认), `0-5`, `1,3` |
| `--force` | 否 | 强制重新摄取，即使数据已存在 | Flag |
| `--chunk_size` | 否 | 文本分块大小 (用于Accurate_Retrieval, Long_Range) | `int` |
| `--overlap` | 否 | 分块重叠大小 (用于Long_Range) | `int` |
| `--min_chars` | 否 | 最小字符数 (用于Conflict_Resolution, TTL) | `int` |
| `--neo4j_db` | 否 | Neo4j 数据库名称覆盖 | 默认为集合名称 |

**示例:**

```bash
# 摄取 Long_Range_Understanding 的第 0 到 5 个实例
python scripts/mem0g_MAB/ingest.py --dataset long_range_understanding --instance_idx 0-5
```

---

## 4. 推理评估 (Inference)

使用 `infer.py` 脚本运行评估任务。

### 运行推理脚本

```bash
python scripts/mem0g_MAB/infer.py --task <TASK_NAME> [Options]
```

### 推理脚本参数说明

| 参数 | 必选 | 说明 | 可选值/示例 |
| :--- | :--- | :--- | :--- |
| `--task` | 是 | 评估任务名称 | `Accurate_Retrieval`, `Conflict_Resolution`, `Long_Range_Understanding`, `Test_Time_Learning` |
| `--adaptor` | 否 | 要运行的 Adaptor | `R1`, `R2`, `R3`, `all` (默认 `all`) |
| `--limit` | 否 | 运行的问题数量限制 (-1 表示全部) | `5` (默认) |
| `--instance_idx` | 否 | 实例索引范围 | `0` (默认), `0-5`, `1,3` |
| `--output_suffix` | 否 | 输出文件名的后缀 | `string` |
| `--neo4j_db` | 否 | Neo4j 数据库名称覆盖 | 默认为集合名称 |

**示例:**

```bash
# 在 Long_Range_Understanding 任务的第 0 个实例上运行所有 Adaptor，限制前 10 个问题
python scripts/mem0g_MAB/infer.py --task Long_Range_Understanding --instance_idx 0 --limit 10
```

---

## TODO

- [ ] **Graph Query Template Confirmation**: 目前关于图查询(Graph Query)是否使用模板还不确定，这可能会牵扯到 `src.adaptor` 中的代码更改。需要会议确认讨论。
