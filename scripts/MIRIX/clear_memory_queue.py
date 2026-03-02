import sys
import os
from datetime import datetime, timezone

# 确保脚本可以找到 mirix 包
sys.path.append(os.getcwd())

try:
    from mirix.server.server import SessionLocal
    from mirix.orm.memory_queue_trace import MemoryQueueTrace
    from sqlalchemy import or_
except ImportError:
    print("错误: 无法导入 mirix 模块，请确保在项目根目录下运行此脚本。")
    sys.exit(1)

def clear_pending_queue_traces():
    print("正在连接数据库...")
    session = SessionLocal()
    try:
        # 查找所有状态为 queued 或 processing 的记录
        pending_traces = session.query(MemoryQueueTrace).filter(
            or_(
                MemoryQueueTrace.status == 'queued',
                MemoryQueueTrace.status == 'processing'
            )
        ).all()
        
        count = len(pending_traces)
        print(f"发现了 {count} 条待处理或卡住的记忆队列记录。")
        
        if count == 0:
            print("没有需要清理的记录。")
            return

        confirm = input("是否将这些记录标记为失败(Failed)以清除它们？(y/n): ")
        if confirm.lower() != 'y':
            print("操作已取消。")
            return

        for trace in pending_traces:
            trace.status = 'failed'
            trace.error_message = 'Manually cleared by user script'
            # 设置完成时间
            trace.completed_at = datetime.now(timezone.utc)
            
        session.commit()
        print(f"成功清理了 {count} 条记录。")

    except Exception as e:
        print(f"发生错误: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    clear_pending_queue_traces()