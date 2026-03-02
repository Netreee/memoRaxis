#!/usr/bin/env python3
"""
独立的Mem0调试脚本
"""
from mem0 import Memory
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.mem0_utils import get_mem0_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fail(step_name, error):
    print(f"\n✗ {step_name} 失败: {error}")
    sys.exit(1)

def test_mem0_basic():
    """测试mem0的基本功能"""
    
    collection_name = "test_debug_collection"
    user_id = "test_user"
    
    print(f"--- 开始 Mem0 调试 (Collection: {collection_name}, User: {user_id}) ---")

    # 步骤 1: 配置
    try:
        print("1. 加载配置...", end=" ", flush=True)
        mem0_config = get_mem0_config(collection_name)
        print("✓")
    except Exception as e:
        fail("配置加载", e)

    # 步骤 2: 初始化
    try:
        print("2. 初始化 Memory...", end=" ", flush=True)
        memory = Memory.from_config(mem0_config)
        print("✓")
    except Exception as e:
        fail("Memory 初始化", e)
    
    # 步骤 3: 检查现有数据
    try:
        print("3. 检查现有数据...", end=" ", flush=True)
        existing = memory.get_all(user_id=user_id)
        
        count = 0
        if isinstance(existing, dict):
            if "results" in existing:
                count = len(existing["results"])
        elif isinstance(existing, list):
            count = len(existing)
            
        print(f"✓ (找到 {count} 条)")
    except Exception as e:
        fail("获取现有数据", e)
    
    # 步骤 4: 添加测试数据
    test_messages = [
        "Alice is my best friend",
        "Bob likes to play basketball"
    ]
    
    try:
        print(f"4. 添加 {len(test_messages)} 条测试消息...", end=" ", flush=True)
        for msg in test_messages:
            memory.add(msg, user_id=user_id)
        print("✓")
    except Exception as e:
        fail("添加数据", e)
    
    # 步骤 5: 验证添加
    try:
        print("5. 验证添加结果...", end=" ", flush=True)
        all_memories = memory.get_all(user_id=user_id)
        
        count = 0
        if isinstance(all_memories, dict):
            if "results" in all_memories:
                count = len(all_memories["results"])
        elif isinstance(all_memories, list):
            count = len(all_memories)
            
        print(f"✓ (当前共 {count} 条)")
    except Exception as e:
        fail("验证数据", e)
    
    # 步骤 6: 搜索测试
    try:
        print("6. 测试搜索 'Alice'...", end=" ", flush=True)
        search_result = memory.search("Alice", user_id=user_id, limit=1)
        
        found = False
        if isinstance(search_result, dict) and search_result.get("results"):
             found = True
        elif isinstance(search_result, list) and len(search_result) > 0:
             found = True
        # Check hasattr just in case
        elif hasattr(search_result, 'results') and search_result.results:
             found = True

        print(f"✓ ({'找到相关' if found else '未找到'})")
    except Exception as e:
        fail("搜索测试", e)
    
    print("\n--- 调试完成: 所有步骤通过 ---")

if __name__ == "__main__":
    test_mem0_basic()
