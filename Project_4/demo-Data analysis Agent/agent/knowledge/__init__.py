"""知识库模块"""


if __name__ == "__main__":
    # 测试用例：验证知识库可初始化并加载默认指标
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.knowledge.knowledge_base import MetricKnowledgeBase
    kb = MetricKnowledgeBase()
    assert kb.metrics, "知识库指标为空"
    print(f"测试通过: 知识库加载 {len(kb.metrics)} 个指标")
