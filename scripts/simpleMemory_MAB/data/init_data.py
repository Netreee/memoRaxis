
from src.simple_memory import SimpleRAGMemory

def init_data():
    # 手动连接并 Drop 表，以确保 Schema 更新
    import psycopg2
    from src.config import get_config
    conf = get_config().database
    conn = psycopg2.connect(conf.get("url"))
    conn.autocommit = True
    with conn.cursor() as cur:
        print("Dropping old table...")
        cur.execute("DROP TABLE IF EXISTS memory_records;")
    conn.close()

    memory = SimpleRAGMemory()
    print("Resetting database...")
    memory.reset()
    
    default_data = [
        {
            "content": "Python 是一种解释型、高级编程语言，由 Guido van Rossum 于 1991 年创建。",
            "metadata": {"source": "编程语言百科", "topic": "Python"},
        },
        {
            "content": "Python 的设计哲学强调代码可读性，使用缩进来表示代码块。",
            "metadata": {"source": "编程语言百科", "topic": "Python"},
        },
        {
            "content": "机器学习是人工智能的一个子领域，通过数据训练模型来做出预测或决策。",
            "metadata": {"source": "AI 基础教程", "topic": "机器学习"},
        },
        {
            "content": "深度学习是机器学习的一种方法，使用多层神经网络来学习数据表示。",
            "metadata": {"source": "AI 基础教程", "topic": "深度学习"},
        },
        {
            "content": "Transformer 架构是一种基于注意力机制的神经网络架构，广泛用于自然语言处理。",
            "metadata": {"source": "深度学习进阶", "topic": "Transformer"},
        },
        {
            "content": "BERT 是 Google 提出的预训练语言模型，使用双向 Transformer 编码器。",
            "metadata": {"source": "NLP 模型介绍", "topic": "BERT"},
        },
        {
            "content": "GPT 系列模型由 OpenAI 开发，使用自回归方式生成文本。",
            "metadata": {"source": "NLP 模型介绍", "topic": "GPT"},
        },
        {
            "content": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术，用于增强 LLM 的知识能力。",
            "metadata": {"source": "LLM 应用技术", "topic": "RAG"},
        },
    ]

    print(f"Injecting {len(default_data)} records...")
    for item in default_data:
        memory.add_memory(item["content"], item["metadata"])
        print(f"Added: {item['content'][:20]}...")
    
    print("Done!")

if __name__ == "__main__":
    init_data()
