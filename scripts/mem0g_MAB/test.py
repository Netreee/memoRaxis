import os
from mem0 import Memory
# 1. 强制指定认证信息
NEO4J_URL = "bolt://localhost:7687"  # 必须是 bolt，不要用 neo4j://
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"
# 2. 构建 Config
# 注意：Mem0 0.x 版本通常直接把参数传给 GraphStore
mem0_config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USER,
            "password": NEO4J_PASSWORD,
            # mem0ai==1.0.3 + langchain-neo4j>=0.8.0 存在参数位不兼容：
            # 这里若显式传 database，会被下游误当成 bearer token。
            # 省略后会使用 Neo4j 默认数据库（neo4j）。
        }
    },
    # 如果你使用了向量存储，也要配置 vector_store，否则 Mem0 可能会报错
    # 这里假设你只测图，或者有其他配置
    # "vector_store": { ... } 
}

# 3. 实例化
# 这步会调用 Neo4jGraph(url=..., username=..., password=...)
# 只要这三个参数传进去了，它绝对不会发送 Token。
print("正在尝试连接 Neo4j...")
try:
    mem0 = Memory.from_config(mem0_config)
    print("✅ Mem0 初始化成功！连接已建立。")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
