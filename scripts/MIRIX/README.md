# MemoryAgentBench Evaluation with MIRIX

本目录包含了一套完整的工具链，用于使用 MIRIX 系统对 MemoryAgentBench (MAB) 数据集进行摄取、推理和评估。

## 目录结构

- `ingest.py`: 数据摄取脚本，将 MAB 数据处理并发送给 MIRIX 服务进行记忆存储。
- `infer.py`: 推理脚本，使用 MIRIX 的检索和记忆能力运行 Agent 进行问答。
- `analyze.py`: 分析脚本，评估推理结果并生成报告。
- `clear_memory_queue.py`: 维护脚本，用于清理数据库中卡住的记忆队列任务。

---

## 1. 环境配置 (Configuration)

在使用本工具链之前，请确保已满足以下条件：

### 1.1 Python 依赖

安装项目根目录 `requirements.txt` 中的依赖。

### 1.2 启动 MIRIX 服务 (Docker)

MIRIX 依赖于 PostgreSQL, Redis 和后端 API 服务。请按照以下步骤启动服务：

**服务列表:**

| 服务 | 端口 | 用途 |
|---------|------|---------|
| **PostgreSQL** | 5432 | 向量数据库 (pgvector) |
| **Redis Stack** | 6379 | 高性能缓存 & 向量搜索 |
| **Mirix API** | 8531 | REST API 后端服务 |
| **Dashboard** | 5173 | React Web 界面 |

**快速启动:**

```bash
# 1. 复制环境文件并配置 API Keys
cp docker/env.example .env
# 编辑 .env 文件，至少设置 OPENAI_API_KEY

# 2. 启动所有服务 (首次运行会自动构建镜像)
docker-compose up -d

# 3. 验证服务状态
docker-compose ps
```

**访问入口:**
- Dashboard: http://localhost:5173
- API 文档: http://localhost:8531/docs

### 1.3 客户端配置

脚本会默认读取 `config/mirix_config.yaml` (如果存在) 或通过命令行参数获取连接信息。
也可以在运行脚本时通过 `--api_key` 和 `--base_url` 参数指定，或设置环境变量 `MIRIX_API_KEY`。

---

## 2. 数据摄取 (Ingest)

将 MAB 数据集摄取到 MIRIX 系统中。

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
*   `--force`: 强制重新摄取（会清除该 User ID 下的旧记忆）。

### 示例

```bash
# 摄取 Conflict Resolution 的第 0 到 5 个实例
python ingest.py --dataset conflict_resolution --instance_idx 0-5
```

---

## 3. 推理 (Infer)

使用 MIRIX 运行 Agent 进行问答。

### 用法

```bash
python infer.py --task <TASK_NAME> --dataset <DATASET_NAME> [OPTIONS]
```

### 参数

*   `--task`: **(必选)** 任务名称 (PascalCase)：
    *   `Accurate_Retrieval`
    *   `Conflict_Resolution`
    *   `Long_Range_Understanding`
    *   `Test_Time_Learning`
*   `--adaptor`: 指定 Adaptor 类型。可选 `R1` (单轮), `R2` (迭代), `R3` (规划), `all` (默认)。
*   `--limit`: 限制运行的问题数量。默认为 5，设为 -1 运行所有。
*   `--instance_idx`: 实例索引 (需与摄取时一致)。

### 示例

```bash
# 在 Conflict Resolution 任务上运行所有 Adaptor
python infer.py --task Conflict_Resolution --instance_idx 0-5

# 仅运行 R1 Adaptor
python infer.py --task Accurate_Retrieval --adaptor R1 --limit 10
```

结果将保存在 `out/mirix` 目录下。

---

## 4. 分析结果 (Analyze)

评估推理生成的 JSON 结果文件的准确率。

### 用法

```bash
python analyze.py --task <TASK_NAME> [OPTIONS]
```

### 参数

*   `--task`: **(必选)** 任务名称。
*   `--input`: 输入目录或文件模式。默认为 `out/mirix`。
*   `--output`: (可选) 输出报告路径。

### 示例

```bash
python analyze.py --task Conflict_Resolution
```

---

## 5. 维护与调试 (Maintenance)

### 常用 Docker 命令

```bash
# 查看日志
docker-compose logs -f mirix_api

# 重启服务
docker-compose restart mirix_api

# 停止服务
docker-compose down

# 重置所有数据 (警告：不可恢复)
docker-compose down -v
rm -rf .persist/
docker-compose up -d
```

### 清理记忆队列

如果发现任务长时间卡在 `queued` 或 `processing` 状态，可以使用此脚本清理：

```bash
python clear_memory_queue.py
```
该脚本会扫描数据库中的异常状态记录，并询问是否将其标记为 `failed` 以恢复队列处理。
