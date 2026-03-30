# MemOS Memory Agent Bench (MAB) Scripts

本目录包含基于 MemOS 的 MemoryAgentBench (MAB) 数据 ingest 与 infer 脚本。

---

## 启动 MemOS 服务（新流程）

### 1. 克隆仓库

  git clone https://github.com/nuoxiichn/MemOS.git

### 2. 配置环境变量文件

在 MemOS 仓库根目录执行：

  cp .env.example .env

然后按你的环境修改 .env 中配置（例如数据库、模型、密钥等）。

### 3. 确保依赖服务运行中

启动并确认以下服务可用：

- qdrant
- neo4j（企业版）

### 4. 在 MemOS 仓库启动 API 服务

在 MemOS 仓库根目录执行：

  PYTHONPATH=src poetry run uvicorn memos.api.server_api:app --host 0.0.0.0 --port 8000 --reload

服务启动后默认可访问：

- http://localhost:8000
- http://localhost:8000/docs

---

## 在本仓库运行脚本

确认上面的 MemOS API 已启动后，再回到本仓库执行 ingest / infer。

### Ingest

示例：

  python scripts/MemOS_MAB/ingest.py --dataset conflict_resolution --instance_idx 0

参数说明：

- --dataset（必填）：数据集，取值为 accurate_retrieval / conflict_resolution / long_range_understanding / test_time_learning
- --instance_idx：样本索引，支持单值或范围，如 0、0-5、1,3
- --force：强制重跑 ingest（当前 MemOS 脚本中该参数仅保留兼容语义）
- --chunk_size：分块大小（适用于 accurate_retrieval、long_range_understanding）
- --overlap：分块重叠（适用于 long_range_understanding）
- --min_chars：最小分块字符数（适用于 conflict_resolution、test_time_learning）

### Infer

示例：

  python scripts/MemOS_MAB/infer.py --task conflict_resolution --instance_idx 0 --adaptor all --limit 5

参数说明：

- --task（必填）：任务/数据集，取值为 accurate_retrieval / conflict_resolution / long_range_understanding / test_time_learning
- --instance_idx：样本索引，支持单值或范围，如 0、0-5、1,3
- --adaptor：选择评测 adaptor，支持 R1、R2、R3、all（可多选，如 --adaptor R1 R3）
- --limit：每个实例评测的问题数量，-1 表示全部
- --output_suffix：输出文件名后缀

结果默认输出到：

- out/memOS/
